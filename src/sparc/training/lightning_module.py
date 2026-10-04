# PyTorch Lightning reproduction of the SPARC paper's vocoder + speaker-
# encoder training (arXiv:2406.12998, Section III-B / Appendix B.3-B.7).
#
# Trainable: HiFiGANGenerator (fresh weights) + SpeakerEncodingLayer FFN
# (fresh weights) + MPD/MSD discriminators (fresh weights, training-only).
# Frozen (not touched here at all): WavLM, the linear EMA-inversion head,
# CREPE pitch tracking, loudness -- those targets/conditioning come
# pre-computed from the dataset (see dataset.py).

import random
import warnings

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from ..generator import HiFiGANGenerator
from ..spk_encoder import SpeakerEncodingLayer
from .discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from .losses import discriminator_loss, feature_loss, generator_adv_loss
from .mel import MelSpectrogram

# Matches the shipped en+ checkpoint's generator_configs (see
# model_englishplus_2M.yaml), which is consistent with the paper's
# description of the HiFi-GAN generator with FiLM speaker conditioning.
DEFAULT_GENERATOR_CONFIG = dict(
    in_channels=14,
    out_channels=1,
    channels=512,
    kernel_size=7,
    upsample_scales=[8, 5, 4, 2],
    upsample_kernel_sizes=[16, 10, 8, 4],
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    use_additional_convs=True,
    bias=True,
    nonlinear_activation="LeakyReLU",
    nonlinear_activation_params={"negative_slope": 0.1},
    use_weight_norm=True,
    use_tanh=True,
    use_spk=True,
    spk_emb_size=64,
    pitch_offset=50,
    pitch_rescale=0.01,
    pitch_axis=12,
)


# With manual optimization Lightning's trainer.global_step advances once per
# optimizer.step() call, i.e. twice per batch (discriminator + generator).
# Config values (max_steps, checkpoint_every_n_steps) are expressed in batches;
# use batches_to_global_steps() when handing them to Lightning.
OPTIMIZERS_PER_BATCH = 2


def batches_to_global_steps(n_batches):
    return None if n_batches is None else n_batches * OPTIMIZERS_PER_BATCH


def lr_lambda(step, halve_every=200000, static_after=None):
    # Step-halving multiplier (MultiStepLR(gamma=0.5) with a milestone every
    # `halve_every` steps, as in the ParallelWaveGAN HiFi-GAN recipe).
    # `static_after` optionally freezes the multiplier from that step on.
    if static_after is not None:
        step = min(step, static_after)
    return 0.5 ** (step // halve_every)


def lr_decay_warning(max_steps, halve_every, static_after=None, floor=1e-3):
    """Message if the lr multiplier at `max_steps` is below `floor`, else None."""
    if max_steps is None or max_steps <= 0:
        return None
    mult = lr_lambda(max_steps, halve_every, static_after)
    if mult >= floor:
        return None
    return (
        f"lr schedule decays to {mult:.2e} x base lr by max_steps={max_steps} "
        f"(halve every {halve_every} steps, static after {static_after}); "
        f"training will effectively stall. Increase lr_halve_every."
    )


class SparcVocoderTraining(pl.LightningModule):
    def __init__(
        self,
        generator_config=None,
        spk_ft_size=1024,
        spk_emb_size=64,
        lr=1e-4,
        betas=(0.5, 0.9),
        lr_halve_every=200000,
        lr_static_after=None,
        mel_weight=45.0,
        fm_weight=2.0,
        gan_weight=1.0,
        log_audio_every_n_steps=500,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        gen_cfg = dict(DEFAULT_GENERATOR_CONFIG)
        if generator_config:
            gen_cfg.update(generator_config)
        self.generator = HiFiGANGenerator(**gen_cfg)

        self.speaker_ffn = SpeakerEncodingLayer(spk_ft_size=spk_ft_size, spk_emb_size=spk_emb_size)
        # paper specifies Dropout(0.2); the existing inference-only class hardcodes 0.0.
        self.speaker_ffn.spk_fc[2] = torch.nn.Dropout(0.2)

        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.mel = MelSpectrogram()

    def setup(self, stage=None):
        if stage != "fit":
            return
        # The schedulers step once per batch, while trainer.max_steps counts
        # both optimizers' steps.
        max_steps = self.trainer.max_steps
        max_batches = max_steps // OPTIMIZERS_PER_BATCH if max_steps and max_steps > 0 else max_steps
        msg = lr_decay_warning(max_batches, self.hparams.lr_halve_every, self.hparams.lr_static_after)
        if msg:
            warnings.warn(msg)

    @property
    def batch_step(self):
        """Number of completed batches (each = one discriminator + one generator update)."""
        return self.global_step // OPTIMIZERS_PER_BATCH

    def on_save_checkpoint(self, checkpoint):
        # Lightning does not checkpoint RNG state; store it so a resumed run
        # continues the same random stream (dropout, DataLoader shuffling and
        # worker seeds are all derived from these generators).
        # (numpy's state is converted to plain ints/tensors so the checkpoint
        # still loads under torch.load(weights_only=True).)
        name, keys, pos, has_gauss, cached = np.random.get_state()
        checkpoint["sparc_rng_state"] = {
            "python": random.getstate(),
            "numpy": (name, torch.from_numpy(keys.astype(np.int64)), int(pos), int(has_gauss), float(cached)),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("sparc_rng_state")
        # The checkpoint holds rank 0's state only; restoring it on every rank
        # would give all DDP ranks identical dropout/crops, so only restore
        # in single-process runs.
        if state is None or self.trainer.world_size > 1:
            return
        random.setstate(state["python"])
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
        torch.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            if len(state["cuda"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

    def forward(self, art, spk_raw):
        spk_emb = self.speaker_ffn(spk_raw)
        c = art.transpose(1, 2)  # (B, T, 14) -> (B, 14, T)
        wav_hat = self.generator(c, spk_emb)  # (B, 1, T*320)
        return wav_hat, spk_emb

    def training_step(self, batch, batch_idx):
        art, spk_raw, audio = batch["art"], batch["spk_raw"], batch["audio"]
        opt_g, opt_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        wav_hat, spk_emb = self(art, spk_raw)
        wav_hat = wav_hat[:, 0]
        min_len = min(wav_hat.shape[-1], audio.shape[-1])
        wav_hat = wav_hat[..., :min_len]
        audio = audio[..., :min_len]

        # --- discriminator step ---
        # toggle_optimizer freezes every parameter outside this optimizer, so
        # the generator step below doesn't compute gradients for the
        # discriminators (and vice versa).
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        y_dp_r, y_dp_g, _, _ = self.mpd(audio.unsqueeze(1), wav_hat.detach().unsqueeze(1))
        loss_d_p, _, _ = discriminator_loss(y_dp_r, y_dp_g)
        y_ds_r, y_ds_g, _, _ = self.msd(audio.unsqueeze(1), wav_hat.detach().unsqueeze(1))
        loss_d_s, _, _ = discriminator_loss(y_ds_r, y_ds_g)
        loss_d = loss_d_p + loss_d_s
        self.manual_backward(loss_d)
        opt_d.step()
        sched_d.step()
        self.untoggle_optimizer(opt_d)

        # --- generator + speaker-FFN step ---
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        mel_real = self.mel(audio)
        mel_fake = self.mel(wav_hat)
        loss_mel = F.l1_loss(mel_fake, mel_real)

        y_dp_r, y_dp_g, fmap_p_r, fmap_p_g = self.mpd(audio.unsqueeze(1), wav_hat.unsqueeze(1))
        y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = self.msd(audio.unsqueeze(1), wav_hat.unsqueeze(1))
        loss_fm = feature_loss(fmap_p_r, fmap_p_g) + feature_loss(fmap_s_r, fmap_s_g)
        loss_gan_g, _ = generator_adv_loss(y_dp_g)
        loss_gan_g_s, _ = generator_adv_loss(y_ds_g)
        loss_gan_g = loss_gan_g + loss_gan_g_s

        loss_g = (
            self.hparams.gan_weight * loss_gan_g
            + self.hparams.mel_weight * loss_mel
            + self.hparams.fm_weight * loss_fm
        )
        self.manual_backward(loss_g)
        opt_g.step()
        sched_g.step()
        self.untoggle_optimizer(opt_g)

        self.log_dict(
            {
                "loss/disc": loss_d,
                "loss/gen_total": loss_g,
                "loss/mel": loss_mel,
                "loss/fm": loss_fm,
                "loss/gan_g": loss_gan_g,
                "lr": opt_g.param_groups[0]["lr"],
            },
            prog_bar=True,
            on_step=True,
        )
        # monitored by ModelCheckpoint (save_top_k>1 requires a ranked
        # quantity; there's no validation loss here, so rank by recency
        # instead -- see cli/train.py).
        self.log("step_metric", float(self.global_step), prog_bar=False, on_step=True)

        if batch_idx % self.hparams.log_audio_every_n_steps == 0:
            self._log_media(
                real_audio=audio[0].detach().cpu(),
                gen_audio=wav_hat[0].detach().cpu().clamp(-1, 1),
                mel_real_img=_mel_to_image(mel_real[0]),
                mel_gen_img=_mel_to_image(mel_fake[0]),
            )

        return loss_g

    def _log_media(self, real_audio, gen_audio, mel_real_img, mel_gen_img):
        """Log audio/mel-spectrogram samples to every attached logger.

        `self.log`/`self.log_dict` already dispatch scalars to all loggers
        via Lightning's common Logger interface -- this only exists because
        audio/image logging has no such common interface across backends,
        so each logger's own native API has to be called directly.
        """
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                exp = logger.experiment
                exp.add_audio("train/real", real_audio, self.batch_step, sample_rate=16000)
                exp.add_audio("train/generated", gen_audio, self.batch_step, sample_rate=16000)
                exp.add_image("train/mel_real", mel_real_img, self.batch_step, dataformats="HW")
                exp.add_image("train/mel_generated", mel_gen_img, self.batch_step, dataformats="HW")
            elif isinstance(logger, WandbLogger):
                import wandb

                logger.experiment.log(
                    {
                        "train/real": wandb.Audio(real_audio.numpy(), sample_rate=16000),
                        "train/generated": wandb.Audio(gen_audio.numpy(), sample_rate=16000),
                        "train/mel_real": wandb.Image(mel_real_img.numpy()),
                        "train/mel_generated": wandb.Image(mel_gen_img.numpy()),
                    },
                    step=self.batch_step,
                )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.speaker_ffn.parameters()),
            lr=self.hparams.lr, betas=self.hparams.betas,
        )
        opt_d = torch.optim.Adam(
            list(self.mpd.parameters()) + list(self.msd.parameters()),
            lr=self.hparams.lr, betas=self.hparams.betas,
        )
        sched_g = torch.optim.lr_scheduler.LambdaLR(
            opt_g, lambda step: lr_lambda(step, self.hparams.lr_halve_every, self.hparams.lr_static_after)
        )
        sched_d = torch.optim.lr_scheduler.LambdaLR(
            opt_d, lambda step: lr_lambda(step, self.hparams.lr_halve_every, self.hparams.lr_static_after)
        )
        return [opt_g, opt_d], [sched_g, sched_d]


def _mel_to_image(mel):
    mel = mel.detach().cpu()
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    return mel.flip(0)
