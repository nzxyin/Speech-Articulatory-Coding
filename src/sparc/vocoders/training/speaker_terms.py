"""Speaker-embedding Phase 3 pieces of the GAN recipe: encoder construction, code mixing and the extra loss terms.

Interpolation training decodes part of each batch from codes that have no target audio: a point on the line between
two batch speakers' codes, or a draw from the prior. Those crops get only the adversarial loss; the latent cycle
(``CropSpeakerStats`` + the encoder with frozen weights) asks that the encoding of the output move by the same vector
as the code: ``E(G(x, z_used)) - E(x) = z_used - z``.
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.func import functional_call

from sparc.vocoders.constants import PERIODICITY_CHANNEL, SAMPLE_RATE
from sparc.vocoders.models.speaker import LayerPoolSpeakerEncoder, SpeakerFFN

WAVLM_DIR = (
    Path("/data/user_data/xoy/sparc-vocoders/hf_hub/models--microsoft--wavlm-large/snapshots")
    / "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
)


def build_speaker(cfg: DictConfig, stats: dict) -> nn.Module:
    """Speaker encoder of ``cfg.speaker``: the comparison's FFN (``type: ffn``) or the layer-pool encoder."""
    spk = cfg.speaker
    kind = spk.get("type", "ffn")
    if kind == "ffn":
        layer = spk.layer
        return SpeakerFFN(dropout=spk.dropout, mean=stats[f"spk_{layer}_mean"], std=stats[f"spk_{layer}_std"])
    if kind != "layerpool":
        raise ValueError(f"unknown speaker.type {kind!r}")
    if spk.layer != "layers" or not spk.get("layer_stats_root"):
        raise ValueError("speaker.type=layerpool needs speaker.layer=layers and speaker.layer_stats_root")
    with np.load(Path(spk.layer_stats_root) / "stats.npz") as npz:
        stat_mean = np.stack([npz["mean_mean"], npz["std_mean"]])
        stat_std = np.stack([npz["mean_std"], npz["std_std"]])
    return LayerPoolSpeakerEncoder(
        n_layers=stat_mean.shape[1],
        hidden_dim=int(spk.get("hidden_dim", 1024)),
        dropout=spk.dropout,
        use_std=bool(spk.get("use_std", True)),
        vae=bool(spk.vae.enabled),
        stat_mean=torch.from_numpy(stat_mean),
        stat_std=torch.from_numpy(stat_std),
    )


def mix_codes(z: torch.Tensor, p: float, prior_frac: float, generator: torch.Generator):
    """Replaces ``round(p B)`` codes of ``z [B, D]``: a ``prior_frac`` share by N(0, I) draws, the rest by
    ``(1 - a) z_i + a z_j`` with ``a ~ U(0, 1)`` and another batch element ``j``.

    Returns ``(z_used, recon)``; ``recon [B]`` marks the untouched elements (the ones with a target).
    """
    b, d = z.shape
    n = int(round(p * b))
    recon = torch.ones(b, dtype=torch.bool, device=z.device)
    if n == 0 or b < 2:
        return z, recon
    picked = torch.randperm(b, generator=generator)[:n]
    u = torch.rand(n, generator=generator)
    alpha = torch.rand(n, 1, generator=generator)
    partner = (picked + torch.randint(1, b, (n,), generator=generator)) % b
    prior = torch.randn(n, d, generator=generator)
    picked, partner = picked.to(z.device), partner.to(z.device)
    mixed = (1 - alpha.to(z)) * z[picked] + alpha.to(z) * z[partner]
    mixed = torch.where((u < prior_frac).to(z.device)[:, None], prior.to(z), mixed)
    z_used = z.clone()
    z_used[picked] = mixed
    recon[picked] = False
    return z_used, recon


def select_fmaps(fmaps: list[list[torch.Tensor]], index: torch.Tensor) -> list[list[torch.Tensor]]:
    """Discriminator feature maps restricted to batch elements ``index``."""
    return [[f[index] for f in per_disc] for per_disc in fmaps]


def resample_24k_to_16k(wav: torch.Tensor) -> torch.Tensor:
    """Differentiable 3:2 polyphase resampling ``[B, N] -> [B, ceil(2N/3)]`` (windowed sinc, as torchaudio)."""
    import torchaudio.functional as AF

    return AF.resample(wav, SAMPLE_RATE, 16000)


class CropSpeakerStats(nn.Module):
    """All-layer periodicity-weighted mean / std of frozen WavLM Large on audio crops, differentiable in the audio.

    Mirrors ``sparc.vocoders.features.spk_layers`` (z-scored 16 kHz input, weights = periodicity channel), except that
    resampling is torchaudio's sinc resampler. WavLM is kept outside the module tree, so it is neither saved in
    checkpoints nor seen by the optimizers.
    """

    def __init__(self):
        super().__init__()
        self._wavlm: list = []

    def wavlm(self, device: torch.device):
        if not self._wavlm:
            from transformers import WavLMModel

            model = WavLMModel.from_pretrained(str(WAVLM_DIR)).eval()
            model.requires_grad_(False)
            self._wavlm.append(model)
        model = self._wavlm[0]
        param = next(model.parameters(), None)
        if param is not None and param.device != device:
            model.to(device)
        return model

    def forward(self, audio: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """``audio [B, 1, N]`` at 24 kHz and ``features [B, 15, T]`` -> statistics ``[B, 2, 25, 1024]``."""
        x = resample_24k_to_16k(audio[:, 0].float())
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-5)
        states = self.wavlm(x.device)(x, output_hidden_states=True).hidden_states
        h = torch.stack(states, 1)  # [B, 25, F, 1024]
        frames = min(h.shape[2], features.shape[-1])
        h = h[:, :, :frames]
        w = features[:, PERIODICITY_CHANNEL, :frames].float().clamp_min(0.0)
        w = torch.where(w.sum(-1, keepdim=True) > 0, w, torch.ones_like(w))
        w = (w / w.sum(-1, keepdim=True))[:, None, :, None]
        mean = (w * h).sum(2)
        std = torch.sqrt((w * (h - mean[:, :, None]) ** 2).sum(2) + 1e-8)
        return torch.stack([mean, std], 1)


def frozen_code(encoder: nn.Module, spk_raw: torch.Tensor) -> torch.Tensor:
    """Encoder output for ``spk_raw`` with the encoder's current weights detached and dropout off."""
    params = {k: v.detach() for k, v in encoder.named_parameters()}
    buffers = dict(encoder.named_buffers())
    was_training = encoder.training
    encoder.eval()
    try:
        return functional_call(encoder, {**params, **buffers}, (spk_raw,))
    finally:
        encoder.train(was_training)
