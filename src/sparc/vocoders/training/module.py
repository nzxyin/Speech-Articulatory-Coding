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
from sparc.vocoders.models.speaker import SpeakerFFN
from sparc.vocoders.training.callbacks import EVAL_SEED, capture_rng_state, fixed_torch_rng, restore_rng_state
from sparc.vocoders.training.schedules import WarmupCosine


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
        layer = cfg.speaker.layer
        self.speaker = SpeakerFFN(
            dropout=cfg.speaker.dropout, mean=stats[f"spk_{layer}_mean"], std=stats[f"spk_{layer}_std"]
        )
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
        return list(chain(self.generator.parameters(), self.speaker.parameters()))

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        optim = self.cfg.optim
        kwargs = {"lr": optim.lr, "betas": tuple(optim.betas), "weight_decay": optim.weight_decay}
        opt_g = torch.optim.AdamW(self.generator_parameters, **kwargs)
        opt_d = torch.optim.AdamW(self.discriminators.parameters(), **kwargs)
        return [opt_g, opt_d]

    @property
    def adversarial_phase(self) -> bool:
        return self.g_step >= self.cfg.train.mel_warmup_steps

    def synthesize(self, batch: dict[str, Any]) -> torch.Tensor:
        """Runs speaker FFN and generator on a batch and checks the output shape."""
        spk = self.speaker(batch["spk_raw"])
        wav = self.generator(batch["features"], spk)
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

    def _generator_update(self, opt_g, audio: torch.Tensor, wav_hat: torch.Tensor) -> dict[str, torch.Tensor]:
        self.toggle_optimizer(opt_g)
        weights = self.cfg.loss
        mel = self.mel_loss(wav_hat, audio)
        total = weights.mel_weight * mel
        logs = {"train/mel": mel.detach()}
        if self.adversarial_phase:
            for name, disc in self.discriminators.items():
                real_logits, fake_logits, real_fmaps, fake_fmaps = disc(audio, wav_hat)
                adv, _ = self.g_loss_fn(fake_logits)
                fm = feature_matching_loss(real_fmaps, fake_fmaps)
                total = total + self.adv_weights[name] * adv + self.fm_weights[name] * fm
                logs[f"train/g_adv/{name}"] = adv.detach()
                logs[f"train/g_fm/{name}"] = fm.detach()
            if weights.mss_weight > 0:
                mss = self.mss_loss(wav_hat, audio)
                total = total + weights.mss_weight * mss
                logs["train/mss"] = mss.detach()
        self.sched_g.apply(opt_g, self.g_step)
        logs["train/grad_norm_g"] = self._step_grads(opt_g, total, self.generator_parameters)
        logs["train/g_total"] = total.detach()
        logs["train/lr_g"] = torch.tensor(opt_g.param_groups[0]["lr"])
        self.untoggle_optimizer(opt_g)
        return logs

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()
        audio = batch["audio"]
        wav_hat = self.synthesize(batch)
        logs = {}
        if self.adversarial_phase:
            logs.update(self._discriminator_update(opt_d, audio, wav_hat))
        logs.update(self._generator_update(opt_g, audio, wav_hat))
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
