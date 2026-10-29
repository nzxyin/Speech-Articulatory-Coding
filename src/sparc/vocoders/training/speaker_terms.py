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

from sparc.vocoders.constants import F0_CHANNEL, LOUDNESS_CHANNEL, PERIODICITY_CHANNEL, SAMPLE_RATE
from sparc.vocoders.features.spk_phase6 import INPUT_DIMS, P6_LAYERS, input_stats
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
        if layer in P6_LAYERS:  # Phase 6: wider input, same hidden width (1024) and output
            mean, std = input_stats(layer, stats, spk.get("input_stats"))
            return SpeakerFFN(in_dim=INPUT_DIMS[layer], dropout=spk.dropout, mean=mean, std=std)
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


SV_DIR = (
    Path("/data/user_data/xoy/sparc-vocoders/hf_hub/models--microsoft--wavlm-base-plus-sv/snapshots")
    / "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
)  # the revision pinned in cfg.eval_metrics.speaker.wavlm_sv


# ---- Phase 4: swap-in-training, identity loss, speaker-relative F0 --------------------------------------------------


def pick_swaps(speaker_ids: torch.Tensor, p: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Elements to swap and their partners: ``round(p B)`` distinct elements ``i`` of the batch, each paired with a
    random element ``j`` of a different speaker (``speaker_ids [B]``).

    An element whose batch holds no other speaker is left out, so the pairs never have equal speakers. Returns
    ``(picked, partner)``, both ``[n]`` long tensors on the CPU (``n`` may be below ``round(p B)``).
    """
    ids = speaker_ids.detach().cpu()
    b = len(ids)
    picked, partner = [], []
    for i in torch.randperm(b, generator=generator)[: int(round(p * b))].tolist():
        candidates = torch.nonzero(ids != ids[i]).flatten()
        if len(candidates) == 0:
            continue
        picked.append(i)
        partner.append(int(candidates[torch.randint(len(candidates), (1,), generator=generator)]))
    return torch.tensor(picked, dtype=torch.long), torch.tensor(partner, dtype=torch.long)


def swap_f0_factor(speaker_f0: torch.Tensor, picked: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
    """F0 ratio ``median(partner's speaker) / median(own speaker)`` per swapped element; 1 where either is unknown."""
    ratio = speaker_f0[partner].float() / speaker_f0[picked].float()
    return torch.where(torch.isfinite(ratio) & (ratio > 0), ratio, torch.ones_like(ratio))


def scale_voiced_f0(features: torch.Tensor, index: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Copy of ``features [B, 15, T]`` whose elements ``index`` have the F0 of voiced frames (periodicity > 0)
    multiplied by ``factor`` (one per element); unvoiced frames, as in ``eval/interp.py``, keep their value."""
    out = features.clone()
    if len(index) == 0:
        return out
    voiced = out[index, PERIODICITY_CHANNEL] > 0
    f0 = out[index, F0_CHANNEL]
    out[index, F0_CHANNEL] = torch.where(voiced, f0 * factor.to(f0)[:, None], f0)
    return out


def swap_batch(
    z: torch.Tensor,
    features: torch.Tensor,
    speaker_ids: torch.Tensor,
    speaker_f0: torch.Tensor | None,
    p: float,
    generator: torch.Generator,
    recon: torch.Tensor | None = None,
    partner_z: torch.Tensor | None = None,
    scale_f0: bool = True,
):
    """Swap-in-training: element ``i`` is decoded with the code of an element ``j`` of another speaker, and (with
    ``scale_f0``) its voiced F0 is scaled by ``median F0 of j's speaker / median F0 of i's speaker``, so the pitch fits
    the borrowed voice. Only elements still marked in ``recon`` (not already mixed) are swapped. ``z`` is the code to
    start from (it may hold mixed codes) and ``partner_z`` (default ``z``) the codes that are borrowed.

    Returns ``(z_used, features_used, recon, target_index, n_swapped)``: ``recon`` is False for swapped elements (they
    have no target audio) and ``target_index [B]`` names, per element, the batch element whose real crop shows the
    speaker of the code used (itself unless swapped).
    """
    b = z.shape[0]
    recon = torch.ones(b, dtype=torch.bool, device=z.device) if recon is None else recon.clone()
    target = torch.arange(b, device=z.device)
    picked, partner = pick_swaps(speaker_ids, p, generator)
    keep = recon[picked.to(z.device)].cpu()
    picked, partner = picked[keep].to(z.device), partner[keep].to(z.device)
    z_used, features_used = z.clone(), features
    if len(picked):
        z_used[picked] = (z if partner_z is None else partner_z)[partner]
        recon[picked] = False
        target[picked] = partner
        if scale_f0 and speaker_f0 is not None:
            features_used = scale_voiced_f0(features, picked, swap_f0_factor(speaker_f0, picked, partner))
    return z_used, features_used, recon, target, len(picked)


def relative_f0_features(features: torch.Tensor, utt_f0: torch.Tensor, fallback_hz: float) -> torch.Tensor:
    """Copy of ``features [B, 15, T]`` whose F0 channel is the ratio ``f0 / utt_f0`` (every frame; unvoiced frames
    keep their filled-in contour). ``utt_f0 [B]`` is the median voiced F0 of the whole source utterance in Hz; an
    unknown (NaN or non-positive) median is replaced by ``fallback_hz`` (the frontend's mean F0).

    Inverse: ``f0 = ratio * utt_f0``. The pitch level is then carried only by the speaker code.
    """
    out = features.clone()
    median = torch.where(torch.isfinite(utt_f0) & (utt_f0 > 0), utt_f0.float(), torch.full_like(utt_f0, fallback_hz))
    out[:, F0_CHANNEL] = features[:, F0_CHANNEL] / median.to(features)[:, None]
    return out


LOUDNESS_MODES = ("off", "added", "moved")
# ln(f0 std of an utterance / REL_F0_RANGE) is the pitch-range statistic; 0.19 is the pooled std of ln(f0 / utterance
# median) of the Phase 4 speaker-relative F0 input (models.frontend.REL_LOGF0_STD), so a typical speaker maps near 0.
REL_F0_RANGE = 0.19


def cond_stats_config(speaker) -> tuple[bool, str]:
    """``(f0_range, loudness)`` of ``speaker.cond_stats``: pitch range of the references added to the code (bool) and the
    source utterance's loudness level: ``"off"``, ``"added"`` (code only; the frame input keeps absolute loudness) or
    ``"moved"`` (code, and the frame loudness input becomes relative to the utterance level)."""
    cfg = speaker.get("cond_stats", {}) or {}
    f0_range = bool(cfg.get("f0_range", False))
    loudness = str(cfg.get("loudness", "off")) if cfg.get("loudness", "off") is not False else "off"
    if loudness not in LOUDNESS_MODES:
        raise ValueError(f"speaker.cond_stats.loudness must be one of {LOUDNESS_MODES}, got {loudness!r}")
    return f0_range, loudness


def cond_stats_enabled(speaker) -> bool:
    f0_range, loudness = cond_stats_config(speaker)
    return f0_range or loudness != "off"


def cond_stat_vector(
    batch: dict, f0_range: bool, loudness: str, loud_log_mean: float, loud_log_std: float, device
) -> torch.Tensor:
    """``[B, k]`` normalized statistics for :class:`StatsCode`: ``ln(ref_f0_std / 0.19)`` and
    ``(utt_loud - loud_log_mean) / loud_log_std``; NaN (no voiced frames) maps to 0, the training mean."""
    cols = []
    if f0_range:
        r = batch["ref_f0_std"].to(device).float()
        cols.append(torch.where(r > 0, torch.log(r.clamp_min(1e-4) / REL_F0_RANGE), torch.zeros_like(r)))
    if loudness != "off":
        level = batch["utt_loud"].to(device).float()
        cols.append(torch.where(torch.isfinite(level), (level - loud_log_mean) / loud_log_std, torch.zeros_like(level)))
    return torch.stack(cols, dim=1)


def relative_loudness_features(features: torch.Tensor, utt_loud: torch.Tensor, loud_log_mean: float) -> torch.Tensor:
    """Copy of ``features [B, 15, T]`` whose loudness channel is scaled by ``exp(loud_log_mean - utt_loud)`` so that the
    frontend's ``(ln(loudness + eps) - loud_log_mean) / std`` becomes ``(ln loudness - utt_loud) / std`` (up to the
    additive ``eps``): the utterance's loudness level leaves the frame input and travels through the conditioning
    path. ``utt_loud [B]`` is the mean ``ln(loudness + eps)`` over the utterance's voiced frames (NaN: unchanged)."""
    out = features.clone()
    shift = torch.where(torch.isfinite(utt_loud), loud_log_mean - utt_loud.float(), torch.zeros_like(utt_loud).float())
    out[:, LOUDNESS_CHANNEL] = features[:, LOUDNESS_CHANNEL] * torch.exp(shift).to(features)[:, None]
    return out


def identity_loss(emb_fake: torch.Tensor, emb_real: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    """``mean_i (1 - cos(emb_fake[i], emb_real[target_index[i]]))`` for ``[B, D]`` embeddings.

    ``emb_real`` is the WavLM-SV embedding of the REAL CROP of the batch element ``target_index[i]`` (the item's own
    crop for arm V; the swap partner's crop for S+V), not of the reference utterance the code was computed from. This
    deviates from the plan's wording ("WavLM-SV embedding of the reference"): the reference audio is not loaded in the
    training batch. Consequence for V: the target shares content with the output, so the term partly overlaps the mel
    loss; for S+V the partner crop is another utterance of the code's speaker, a valid identity target.
    """
    cos = torch.nn.functional.cosine_similarity(emb_fake, emb_real[target_index], dim=-1)
    return (1.0 - cos).mean()


class CropSpeakerVerifier(nn.Module):
    """Frozen WavLM-base-plus-SV (``WavLMForXVector``, 512-d) on audio crops, differentiable in the audio.

    Same model and revision as the evaluation's WavLM-SV similarity. Input: 24 kHz crops resampled to 16 kHz by
    torchaudio's sinc resampler; the checkpoint's feature extractor has ``do_normalize: false``, so the waveform goes in
    unnormalized (as ``eval/metrics/speaker.py`` feeds it). Kept outside the module tree (like ``CropSpeakerStats``):
    not saved in checkpoints, not seen by the optimizers; always in eval mode (no spec-augment), fp32.
    """

    def __init__(self, sv_dir: str | Path | None = None):
        super().__init__()
        self.sv_dir = Path(sv_dir) if sv_dir else SV_DIR  # config: speaker.identity.sv_dir
        self._sv: list = []

    def model(self, device: torch.device):
        if not self._sv:
            from transformers import WavLMForXVector

            model = WavLMForXVector.from_pretrained(str(self.sv_dir)).float().eval()
            model.requires_grad_(False)
            self._sv.append(model)
        model = self._sv[0]
        param = next(model.parameters(), None)
        if param is not None and param.device != device:
            model.to(device)
        return model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio [B, 1, N]`` at 24 kHz -> raw x-vector embeddings ``[B, D]``."""
        x = resample_24k_to_16k(audio[:, 0].float())
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.model(x.device)(input_values=x).embeddings


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
