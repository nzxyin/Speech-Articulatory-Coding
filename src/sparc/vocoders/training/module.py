"""LightningModule for GAN training of the articulatory vocoders (contract section 5).

Optimization is manual and driven by the module's own counters ``g_step``, ``d_step`` and ``samples_consumed``, which
are saved in the checkpoint; ``trainer.global_step`` (it counts both optimizers) is never used for a schedule.
"""

from itertools import chain
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.utils import clip_grad_norm_

from sparc.vocoders.constants import HOP, SAMPLE_RATE
from sparc.vocoders.losses.losses import ADVERSARIAL_LOSSES, feature_matching_loss, mr_stft_distance
from sparc.vocoders.models.frontend import load_stats
from sparc.vocoders.models.speaker import PitchCode, StatsCode, gaussian_kl, total_correlation
from sparc.vocoders.training.callbacks import EVAL_SEED, capture_rng_state, fixed_torch_rng, restore_rng_state
from sparc.vocoders.training.schedules import WarmupCosine
from sparc.vocoders.training.speaker_terms import (
    CropSpeakerStats,
    CropSpeakerVerifier,
    build_speaker,
    cond_stat_vector,
    cond_stats_config,
    frozen_code,
    identity_loss,
    mix_codes,
    relative_f0_features,
    relative_loudness_features,
    select_fmaps,
    swap_batch,
)


def render_mel(log_mel: torch.Tensor, vmin: float, vmax: float) -> np.ndarray:
    """Colour image ``[3, n_mels, frames]`` (uint8, low frequencies at the bottom) of a log-mel ``[n_mels, frames]``."""
    from matplotlib import colormaps

    scaled = ((log_mel.float() - vmin) / max(vmax - vmin, 1e-6)).clamp(0.0, 1.0).flip(0).cpu().numpy()
    rgb = colormaps["viridis"](scaled)[..., :3]
    return (rgb * 255).astype(np.uint8).transpose(2, 0, 1)


class VocoderGANModule(pl.LightningModule):
    """Generator, speaker FFN and discriminators with the staged recipe of the shared GAN configuration.

    While ``g_step < train.mel_warmup_steps`` the generator trains on ``mel_weight * mel`` alone. Afterwards each batch
    makes one discriminator update (generator output detached; per-set loss weighted by ``adv_weight``) and one
    generator update with the discriminators frozen: ``mel_weight * mel + sum_d (adv_weight_d * adv_d + fm_weight_d *
    fm_d) + mss_weight * mss``. ``stats`` is the training statistics dict, read from ``cfg.stats_path`` if omitted.
    """

    def __init__(self, cfg: DictConfig | dict, stats: dict | None = None):
        super().__init__()
        cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})
        self.cfg = cfg
        self.automatic_optimization = False
        stats = load_stats(cfg.stats_path if stats is None else stats)

        self.generator = instantiate(cfg.vocoder.generator, stats=stats, _convert_="all")
        self.speaker = build_speaker(cfg, stats)
        spk = cfg.speaker
        self.vae = bool(spk.get("vae", {}).get("enabled", False))
        self.mix_p = float(spk.get("mix", {}).get("p", 0.0))
        self.cycle_weight = float(spk.get("mix", {}).get("cycle_weight", 0.0))
        self.tc_weight = float(spk.get("tc", {}).get("weight", 0.0))
        self.crop_stats = CropSpeakerStats() if self.cycle_weight > 0 else None
        # Phase 4 (speaker-embedding cloning): swap-in-training, SV identity loss, speaker-relative F0
        self.swap_p = float(spk.get("swap", {}).get("p", 0.0))
        self.identity_weight = float(spk.get("identity", {}).get("weight", 0.0))
        self.rel_f0 = bool(spk.get("rel_f0", {}).get("enabled", False))
        ident_cfg = spk.get("identity", {})
        self.grad_probe_every = int(ident_cfg.get("grad_probe_every", 0))
        self.verifier = (
            CropSpeakerVerifier(ident_cfg.get("sv_dir") or None) if self.identity_weight > 0 else None
        )
        self.pitch_code = PitchCode(stats["logf0_mean"], stats["logf0_std"]) if self.rel_f0 else None
        self.f0_fallback_hz = float(np.exp(stats["logf0_mean"]))
        # Phase 8 conditioning study: utterance-level statistics added to the speaker code (off by default)
        self.cond_f0_range, self.cond_loudness = cond_stats_config(spk)
        self.loud_log_mean, self.loud_log_std = float(stats["loud_log_mean"]), float(stats["loud_log_std"])
        n_stats = int(self.cond_f0_range) + int(self.cond_loudness != "off")
        self.stats_code = StatsCode(n_stats) if n_stats else None
        if n_stats and (self.swap_p > 0 or self.mix_p > 0):
            raise ValueError("speaker.cond_stats cannot be combined with swap-in-training or code mixing")
        if self.rel_f0 and cfg.vocoder.generator.get("pitch_mode") != "log_relative":
            raise ValueError("speaker.rel_f0.enabled needs vocoder.generator.pitch_mode=log_relative")
        self._tc_pool: torch.Tensor | None = None
        self._tc_dataset_size = 0
        self.discriminators = nn.ModuleDict(
            {name: instantiate(spec.module, _convert_="all") for name, spec in cfg.loss.discriminators.items()}
        )
        self.adv_weights = {name: float(spec.adv_weight) for name, spec in cfg.loss.discriminators.items()}
        self.fm_weights = {name: float(spec.fm_weight) for name, spec in cfg.loss.discriminators.items()}
        self.d_loss_fn, self.g_loss_fn = ADVERSARIAL_LOSSES[cfg.loss.adversarial]
        self.mel_loss = instantiate(cfg.loss.mel, _convert_="all")
        self.mss_loss = instantiate(cfg.loss.mss, _convert_="all")

        self.g_step = 0
        self.d_step = 0
        self.samples_consumed = 0
        self.last_validated_g_step = 0
        self.sched_g = WarmupCosine(cfg.optim.lr, cfg.optim.warmup_steps, cfg.train.max_g_steps)
        self.sched_d = WarmupCosine(
            cfg.optim.lr, cfg.optim.warmup_steps, max(cfg.train.max_g_steps - cfg.train.mel_warmup_steps, 1)
        )
        self.last_losses: dict[str, torch.Tensor] = {}
        self.last_validation: dict[str, float] = {}
        self._val_loader = None
        self._reference_logged: set[str] = set()

    @property
    def generator_parameters(self) -> list[nn.Parameter]:
        extra = [] if self.pitch_code is None else list(self.pitch_code.parameters())
        if self.stats_code is not None:
            extra += list(self.stats_code.parameters())
        return list(chain(self.generator.parameters(), self.speaker.parameters(), extra))

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        optim = self.cfg.optim
        kwargs = {"lr": optim.lr, "betas": tuple(optim.betas), "weight_decay": optim.weight_decay}
        opt_g = torch.optim.AdamW(self.generator_parameters, **kwargs)
        opt_d = torch.optim.AdamW(self.discriminators.parameters(), **kwargs)
        return [opt_g, opt_d]

    @property
    def adversarial_phase(self) -> bool:
        return self.g_step >= self.cfg.train.mel_warmup_steps

    def speaker_code(self, batch: dict[str, Any], sample: bool = False) -> tuple[torch.Tensor, dict]:
        """``(z, info)`` of the batch's references. With speaker-relative F0 (``speaker.rel_f0``) the code carries the
        reference utterances' pitch level (``batch["ref_f0"]``, Hz): ``z = encoder(refs) + PitchCode(ref_f0)``."""
        z, info = self.speaker.embed(batch["spk_raw"], batch.get("ref_mask"), sample=sample)
        if self.pitch_code is not None:
            z = z + self.pitch_code(batch["ref_f0"].to(z.device))
        if self.stats_code is not None:
            self._require_cond_stats(batch)
            vec = cond_stat_vector(
                batch, self.cond_f0_range, self.cond_loudness, self.loud_log_mean, self.loud_log_std, z.device
            )
            z = z + self.stats_code(vec).to(z.dtype)
        return z, info

    def stats_term(self, ref_f0_std: float, utt_loud: float) -> torch.Tensor:
        """``[64]`` conditioning-statistics term of the code for one utterance (the code the training batch would get
        is ``encoder(refs) + pitch term + stats_term``); NaN statistics count as the training mean."""
        device = next(self.stats_code.parameters()).device
        batch = {"ref_f0_std": torch.tensor([ref_f0_std]), "utt_loud": torch.tensor([utt_loud])}
        vec = cond_stat_vector(
            batch, self.cond_f0_range, self.cond_loudness, self.loud_log_mean, self.loud_log_std, device
        )
        return self.stats_code(vec)[0]

    def _require_cond_stats(self, batch: dict[str, Any]) -> None:
        need = (["ref_f0_std"] if self.cond_f0_range else []) + (["utt_loud"] if self.cond_loudness != "off" else [])
        missing = [k for k in need if k not in batch]
        if missing:
            raise KeyError(
                f"conditioning-statistics checkpoint: the batch needs {missing} (ref_f0_std: ln-F0 std of the code's "
                "references, utt_loud: mean ln(loudness + eps) over the source utterance's voiced frames); see "
                "data.dataset.voiced_logf0_std / voiced_log_loudness"
            )

    def synthesize(
        self, batch: dict[str, Any], z: torch.Tensor | None = None, features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Runs speaker encoder and generator on a batch and checks the output shape.

        ``z`` overrides the speaker code; otherwise the encoder's deterministic code (VAE: posterior mean) is used.
        ``features`` overrides ``batch["features"]`` (absolute F0 in Hz); with speaker-relative F0 the F0 channel is
        divided by ``batch["utt_f0"]`` (median voiced F0 of the source utterance) before the generator, so a
        conversion only needs the source batch with the target's ``ref_f0``.
        """
        spk = self.speaker_code(batch)[0] if z is None else z
        feats = batch["features"] if features is None else features
        if self.rel_f0:
            if "utt_f0" not in batch:
                raise KeyError(
                    "speaker-relative F0 checkpoint: the batch needs 'utt_f0' (source median voiced F0, Hz) and "
                    "'ref_f0' (target reference median, Hz); build it with f0_conditioning, and take the code from "
                    "module.speaker_code(batch), not module.speaker.embed (which drops the pitch term)"
                )
            feats = relative_f0_features(feats, batch["utt_f0"].to(feats.device), self.f0_fallback_hz)
        if self.cond_loudness == "moved":
            self._require_cond_stats(batch)
            feats = relative_loudness_features(feats, batch["utt_loud"].to(feats.device), self.loud_log_mean)
        wav = self.generator(feats, spk)
        expected = (batch["features"].shape[0], 1, batch["features"].shape[-1] * HOP)
        if tuple(wav.shape) != expected:
            raise ValueError(f"generator returned {tuple(wav.shape)}, expected {expected}")
        return wav

    def _step_grads(self, opt, loss: torch.Tensor, params: list[nn.Parameter]) -> torch.Tensor:
        opt.zero_grad()
        self.manual_backward(loss)
        norm = clip_grad_norm_(params, float("inf"))
        opt.step()
        return norm.detach()

    def _discriminator_update(self, opt_d, audio: torch.Tensor, wav_hat: torch.Tensor) -> dict[str, torch.Tensor]:
        self.toggle_optimizer(opt_d)
        total, logs = 0.0, {}
        fake = wav_hat.detach()
        for name, disc in self.discriminators.items():
            real_logits, fake_logits, _, _ = disc(audio, fake)
            loss, _ = self.d_loss_fn(real_logits, fake_logits)
            total = total + self.adv_weights[name] * loss
            logs[f"train/d_loss/{name}"] = loss.detach()
            logs[f"train/d_real/{name}"] = torch.stack([r.detach().mean() for r in real_logits]).mean()
            logs[f"train/d_fake/{name}"] = torch.stack([f.detach().mean() for f in fake_logits]).mean()
        self.sched_d.apply(opt_d, self.d_step)
        logs["train/grad_norm_d"] = self._step_grads(opt_d, total, list(self.discriminators.parameters()))
        logs["train/d_total"] = total.detach()
        logs["train/lr_d"] = torch.tensor(opt_d.param_groups[0]["lr"])
        self.untoggle_optimizer(opt_d)
        return logs

    def _generator_update(
        self, opt_g, audio: torch.Tensor, wav_hat: torch.Tensor, aux: dict | None = None
    ) -> dict[str, torch.Tensor]:
        """Generator step. ``aux`` (Phase 3) holds ``recon`` (elements with a target), the codes and the batch;
        mixed / prior elements get the adversarial loss only."""
        self.toggle_optimizer(opt_g)
        weights = self.cfg.loss
        aux = aux or {}
        recon = aux.get("recon")
        idx = None if recon is None or bool(recon.all()) else torch.nonzero(recon).flatten()
        target, output = (audio, wav_hat) if idx is None else (audio[idx], wav_hat[idx])
        mel = self.mel_loss(output, target)
        total = weights.mel_weight * mel
        logs = {"train/mel": mel.detach()}
        if self.adversarial_phase:
            for name, disc in self.discriminators.items():
                real_logits, fake_logits, real_fmaps, fake_fmaps = disc(audio, wav_hat)
                adv, _ = self.g_loss_fn(fake_logits)
                if idx is not None:
                    real_fmaps, fake_fmaps = select_fmaps(real_fmaps, idx), select_fmaps(fake_fmaps, idx)
                fm = feature_matching_loss(real_fmaps, fake_fmaps)
                total = total + self.adv_weights[name] * adv + self.fm_weights[name] * fm
                logs[f"train/g_adv/{name}"] = adv.detach()
                logs[f"train/g_fm/{name}"] = fm.detach()
            if weights.mss_weight > 0:
                mss = self.mss_loss(output, target)
                total = total + weights.mss_weight * mss
                logs["train/mss"] = mss.detach()
        speaker_total = self._speaker_terms(aux, audio, wav_hat, logs)
        total = total + speaker_total
        # g_step is incremented after this update and logging tests the incremented value, so probe on g_step + 1
        probe_due = self.grad_probe_every > 0 and (self.g_step + 1) % self.grad_probe_every == 0
        if aux.get("identity_term") is not None and probe_due:
            logs["train/identity_grad_ratio"] = self._identity_grad_ratio(weights.mel_weight * mel, aux["identity_term"])
        self.sched_g.apply(opt_g, self.g_step)
        logs["train/grad_norm_g"] = self._step_grads(opt_g, total, self.generator_parameters)
        logs["train/g_total"] = total.detach()
        logs["train/lr_g"] = torch.tensor(opt_g.param_groups[0]["lr"])
        self.untoggle_optimizer(opt_g)
        return logs

    def _identity_grad_ratio(self, mel_term: torch.Tensor, identity_term: torch.Tensor) -> torch.Tensor:
        """``||grad of the weighted identity term|| / ||grad of the weighted mel term||`` over the generator parameters:
        the diagnostic for choosing ``speaker.identity.weight`` (aim for roughly 0.05-0.2). Graphs are kept for the
        real backward pass; the cost is two extra backward passes, paid only every ``grad_probe_every`` steps."""
        params = self.generator_parameters

        def norm(term: torch.Tensor) -> torch.Tensor:
            grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            return torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads if g is not None))

        return (norm(identity_term) / norm(mel_term).clamp_min(1e-12)).detach()

    def _speaker_terms(self, aux: dict, audio: torch.Tensor, wav_hat: torch.Tensor, logs: dict) -> torch.Tensor:
        """Phase 3 terms: capacity-controlled KL, beta-TC on a side batch, latent cycle. 0 for the comparison arms."""
        total = torch.zeros((), device=audio.device)
        info = aux.get("info", {})
        spk = self.cfg.speaker
        if self.vae and "mu" in info:
            kl = gaussian_kl(info["mu"], info["logvar"]).mean()
            vae = spk.vae
            capacity = float(vae.capacity_max) * min(1.0, self.g_step / max(int(vae.capacity_steps), 1))
            total = total + float(vae.gamma) * (kl - capacity).abs()
            logs["train/kl"], logs["train/kl_capacity"] = kl.detach(), torch.tensor(capacity)
        if self.tc_weight > 0:
            pool = self._side_pool(audio.device)
            pick = torch.randint(len(pool), (int(spk.tc.batch),), device=audio.device)
            mu, logvar = self.speaker.posterior(pool[pick].float())
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            tc = total_correlation(z, mu, logvar, self._tc_dataset_size)
            total = total + self.tc_weight * tc
            logs["train/tc"] = tc.detach()
        if self.verifier is not None and self.adversarial_phase:
            with torch.no_grad():
                real_emb = self.verifier(audio)
            fake_emb = self.verifier(wav_hat)
            target = aux.get("target_index")
            target = torch.arange(len(audio), device=audio.device) if target is None else target
            ok = aux.get("identity_ok")  # mixed / prior codes have no speaker to compare with
            idx = torch.arange(len(audio), device=audio.device) if ok is None else torch.nonzero(ok).flatten()
            if len(idx):
                ident = identity_loss(fake_emb[idx], real_emb, target[idx])
                total = total + self.identity_weight * ident
                aux["identity_term"] = self.identity_weight * ident
                logs["train/identity"] = ident.detach()
        if self.crop_stats is not None and self.adversarial_phase:
            features = aux["features"]
            with torch.no_grad():
                code_real = frozen_code(self.speaker, self.crop_stats(audio, features))
            code_fake = frozen_code(self.speaker, self.crop_stats(wav_hat, features))
            shift = (aux["z_used"] - aux["z"]).detach()
            cycle = ((code_fake - code_real - shift) ** 2).mean()
            total = total + self.cycle_weight * cycle
            logs["train/cycle"] = cycle.detach()
        return total

    def _side_pool(self, device: torch.device) -> torch.Tensor:
        """Random training references (all-layer statistics, fp16) on the GPU for the beta-TC side batch."""
        if self._tc_pool is None:
            store = self.trainer.datamodule.train_dataset().store
            rng = np.random.default_rng([int(self.cfg.seed), 7])
            size = min(int(self.cfg.speaker.tc.pool_size), len(store))
            rows = np.sort(rng.choice(len(store), size=size, replace=False))
            pool = np.stack([store.speaker_vector(int(r)) for r in rows]).astype(np.float16)
            self._tc_pool = torch.from_numpy(pool).to(device)
            self._tc_dataset_size = len(store)
        return self._tc_pool

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()
        audio = batch["audio"]
        z, info = self.speaker_code(batch, sample=self.vae)
        z_used, recon, features = z, None, batch["features"]
        if self.mix_p > 0 and self.adversarial_phase:
            gen = torch.Generator().manual_seed(int(self.cfg.seed) * 1_000_003 + self.g_step)
            mix = self.cfg.speaker.mix
            z_used, recon = mix_codes(z, self.mix_p, float(mix.prior_frac), gen)
        identity_ok, target_index, n_swapped = recon, None, 0
        if self.swap_p > 0 and self.adversarial_phase:
            gen = torch.Generator().manual_seed(int(self.cfg.seed) * 1_000_003 + self.g_step + 7_000_001)
            z_used, features, recon, target_index, n_swapped = swap_batch(
                z_used, features, batch["speaker_id"], batch.get("speaker_f0"), self.swap_p, gen,
                recon=recon, partner_z=z, scale_f0=not self.rel_f0,
            )  # fmt: skip
        wav_hat = self.synthesize(batch, z=z_used, features=features)
        aux = {
            "recon": recon, "info": info, "z": z, "z_used": z_used, "features": batch["features"],
            "target_index": target_index, "identity_ok": identity_ok,
        }  # fmt: skip
        logs = {}
        if self.swap_p > 0:
            logs["train/swap_frac"] = torch.tensor(n_swapped / audio.shape[0])
        if self.adversarial_phase:
            logs.update(self._discriminator_update(opt_d, audio, wav_hat))
        logs.update(self._generator_update(opt_g, audio, wav_hat, aux))
        self.g_step += 1
        self.d_step += int("train/d_total" in logs)
        self.samples_consumed += audio.shape[0] * self.trainer.world_size
        self.last_losses = logs
        if self.g_step % self.cfg.train.log_every_g_steps == 0:
            self.log_scalars({**logs, "train/d_step": self.d_step, "train/samples_consumed": self.samples_consumed})

    def on_train_batch_start(self, batch: dict[str, Any], batch_idx: int) -> int | None:
        return -1 if self.g_step >= self.cfg.train.max_g_steps else None

    def validation_due(self) -> bool:
        """True at every ``val_every_g_steps`` generator steps that have not been validated yet."""
        return (
            self.g_step > 0
            and self.g_step % self.cfg.train.val_every_g_steps == 0
            and self.g_step != self.last_validated_g_step
        )

    def on_train_start(self) -> None:
        """A checkpoint is written before the validation of its own step, so a resumed run validates that step first."""
        if self.g_step < self.cfg.train.max_g_steps and self.validation_due():
            self.run_validation()

    def on_train_batch_end(self, outputs: Any, batch: dict[str, Any], batch_idx: int) -> None:
        if self.validation_due() and not self.trainer.should_stop:
            self.run_validation()
        if self.g_step >= self.cfg.train.max_g_steps:
            self.trainer.should_stop = True

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["vocoder_counters"] = {
            "g_step": self.g_step,
            "d_step": self.d_step,
            "samples_consumed": self.samples_consumed,
            "last_validated_g_step": self.last_validated_g_step,
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        counters = checkpoint["vocoder_counters"]
        self.g_step = int(counters["g_step"])
        self.d_step = int(counters["d_step"])
        self.samples_consumed = int(counters["samples_consumed"])
        self.last_validated_g_step = int(counters.get("last_validated_g_step", 0))
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None:
            datamodule.set_samples_consumed(self.samples_consumed)

    def log_scalars(self, scalars: dict[str, Any]) -> None:
        """Writes scalars to every logger with ``g_step`` on the x axis; raises on a non-finite value."""
        values = {k: float(v) for k, v in scalars.items()}
        bad = [k for k, v in values.items() if not np.isfinite(v)]
        if bad:
            raise FloatingPointError(f"non-finite values {bad} at g_step {self.g_step}")
        for logger in self.loggers:
            logger.log_metrics(values, step=self.g_step)

    def _log_media(self, tag: str, wav: torch.Tensor, image: np.ndarray) -> None:
        for logger in self.loggers:
            name = type(logger).__name__
            if name == "TensorBoardLogger":
                logger.experiment.add_audio(f"{tag}/audio", wav.reshape(1, -1), self.g_step, sample_rate=SAMPLE_RATE)
                logger.experiment.add_image(f"{tag}/mel", image, self.g_step, dataformats="CHW")
            elif name == "WandbLogger":
                import wandb

                logger.experiment.log(
                    {
                        f"{tag}/audio": wandb.Audio(wav.reshape(-1).numpy(), sample_rate=SAMPLE_RATE),
                        f"{tag}/mel": wandb.Image(image.transpose(1, 2, 0)),
                    },
                    step=self.g_step,
                )

    def _validation_loader(self):
        if self._val_loader is None:
            loader = self.trainer.datamodule.val_dataloader()
            self._val_loader = loader[0] if isinstance(loader, (list, tuple)) else loader
        return self._val_loader

    @torch.no_grad()
    def run_validation(self) -> dict[str, float]:
        """Mel L1 and MR-STFT distance over the fixed dev subset; audio and mel images for the logging subset.

        The first items of the validation loader are the logging subset (``datamodule.logging_ids``, else
        ``data.logging_subset_size`` of them); rank 0 handles those and the remaining items are split over the ranks.
        Every synthesis runs under ``fixed_torch_rng`` (the DDSP noise branch draws from the global torch generator), so
        the validation numbers do not depend on the training RNG stream. The global RNG states are restored afterwards:
        iterating a validation loader draws from the torch generator, which would otherwise make a run that validated
        differ from one that resumed from a checkpoint written before the validation.
        """
        rng_state = capture_rng_state(self.device)
        was_training = self.training
        try:
            return self._validate()
        finally:
            self.train(was_training)
            restore_rng_state(rng_state, self.device)

    def _validate(self) -> dict[str, float]:
        trainer = self.trainer
        logging_ids = getattr(trainer.datamodule, "logging_ids", None)
        n_media = len(logging_ids) if logging_ids is not None else int(self.cfg.data.logging_subset_size)
        rank, world = trainer.global_rank, trainer.world_size
        mss = self.cfg.loss.mss
        self.eval()
        sums = torch.zeros(3, device=self.device)
        with trainer.precision_plugin.forward_context():
            for i, batch in enumerate(self._validation_loader()):
                if (rank != 0) if i < n_media else ((i - n_media) % world != rank):
                    continue
                batch = self.transfer_batch_to_device(batch, self.device, 0)
                audio = batch["audio"]
                with fixed_torch_rng(self.device, EVAL_SEED):
                    wav_hat = self.synthesize(batch).float()
                sums += torch.stack(
                    [
                        self.mel_loss(wav_hat, audio),
                        mr_stft_distance(wav_hat, audio, fft_sizes=list(mss.fft_sizes), hop_ratio=mss.hop_ratio),
                        torch.ones((), device=self.device),
                    ]
                )
                if i < n_media:
                    self._log_validation_media(batch["id"][0], wav_hat, audio)
        if world > 1:
            sums = trainer.strategy.reduce(sums, reduce_op="sum")
        self.last_validated_g_step = self.g_step
        count = max(float(sums[2]), 1.0)
        self.last_validation = {"val/mel_l1": float(sums[0]) / count, "val/mr_stft": float(sums[1]) / count}
        if trainer.is_global_zero:
            self.log_scalars(self.last_validation)
        return self.last_validation

    def _log_validation_media(self, utt_id: str, wav_hat: torch.Tensor, audio: torch.Tensor) -> None:
        if not self.trainer.is_global_zero or not self.loggers:
            return
        ref_mel = self.mel_loss.log_mel(audio)[0]
        vmin, vmax = float(ref_mel.min()), float(ref_mel.max())
        if utt_id not in self._reference_logged:
            self._log_media(f"ref/{utt_id}", audio.cpu(), render_mel(ref_mel, vmin, vmax))
            self._reference_logged.add(utt_id)
        self._log_media(f"val/{utt_id}", wav_hat.cpu(), render_mel(self.mel_loss.log_mel(wav_hat)[0], vmin, vmax))

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        with fixed_torch_rng(self.device, EVAL_SEED):
            wav = self.synthesize(batch)
        return {"wav": wav.float().cpu(), "id": list(batch["id"]), "condition": list(batch["condition"])}
