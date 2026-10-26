"""Segment-aware pooling variants of the speaker-encoder screen (hypotheses H1, H2, H3, H11).

Every variant keeps the 64-d output and the baseline head (Linear(in, 1024), GELU, dropout 0.2, Linear(1024, 64));
only the pooled input differs. Hyperparameters are fixed here, before the first run, from the stated conventions:

- voiced frame: cached SPARC periodicity >= 0.5 (the midpoint of its 0-1 range);
- silence frame: periodicity < 0.5 and z-scored loudness < 0.1 (the 20th percentile of loudness over cached training
  frames is 0.106; frames below it are the pauses and the noise floor); unvoiced speech: the other frames;
- a class with fewer than ``MIN_FRAMES`` = 3 frames (60 ms) in the crop backs off to the baseline pooled vector;
- k-means codebooks use K = 6 (the broad sound classes: vowel, nasal, approximant, fricative, stop, silence) and K = 16
  (a finer codebook), 300k random training frames of the layer-6 features, 25 Lloyd iterations, seed 0, training
  frames only; each cluster mean passes a shared Linear(1024, 128) before the head;
- the spectral branch uses 23 cepstral coefficients (DCT-II of the log-mel, c1..c23; c0 dropped as the level);
- soft gates: a 2-layer MLP with 16 hidden units, softmax over three segments, smoothing mass of one frame.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from sparc.vocoders.constants import SPEAKER_DIM
from sparc.vocoders.spkscreen.data import FrameBatch, FrameSet
from sparc.vocoders.spkscreen.pooling import PoolingModule, pool_weights, register, weighted_mean

VOICED_PER = 0.5
SILENCE_LOUD = 0.1
MIN_FRAMES = 3.0


def segment_masks(b: FrameBatch) -> torch.Tensor:
    """``[B, T, 3]`` float one-hot over (voiced, unvoiced speech, silence) on real frames."""
    m = b.mask
    voiced = (b.periodicity >= VOICED_PER) & m
    silence = (b.periodicity < VOICED_PER) & (b.zloud < SILENCE_LOUD) & m
    unv = m & ~voiced & ~silence
    return torch.stack([voiced, unv, silence], -1).float()


def class_means(x: torch.Tensor, w: torch.Tensor, backoff: torch.Tensor, min_frames: float = MIN_FRAMES):
    """Per-class weighted means ``[B, C, D]`` of ``x [B, T, D]`` for weights ``w [B, T, C]``; classes with weight
    below ``min_frames`` take ``backoff [B, D]``. Also returns the class weight sums ``[B, C]``."""
    s = w.sum(1)
    mean = torch.einsum("btc,btd->bcd", w, x) / s.clamp_min(1e-6)[..., None]
    return torch.where((s >= min_frames)[..., None], mean, backoff[:, None, :]), s


def dct_matrix(n_mel: int = 80, n_cep: int = 23) -> torch.Tensor:
    """Orthonormal DCT-II rows 1..n_cep ``[n_cep, n_mel]``."""
    k = torch.arange(1, n_cep + 1)[:, None].float()
    n = torch.arange(n_mel)[None].float()
    return math.sqrt(2.0 / n_mel) * torch.cos(math.pi * k * (n + 0.5) / n_mel)


class _SegmentBase(PoolingModule):
    """Fixed pooled feature vector (``features``), per-dimension z-score from training utterances, baseline head."""

    layers = (6,)

    def _init_head(self, norm_dim: int, in_dim: int | None = None, dropout: float = 0.2):
        self.register_buffer("mean", torch.zeros(norm_dim))
        self.register_buffer("std", torch.ones(norm_dim))
        self.head = nn.Sequential(nn.Linear(in_dim or norm_dim, 1024), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(1024, SPEAKER_DIM))

    def base_vec(self, b: FrameBatch) -> torch.Tensor:
        return weighted_mean(b.hidden[:, 0], pool_weights(b.periodicity, b.mask))

    def features(self, b: FrameBatch) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def _train_stats(self, train: FrameSet, fn) -> tuple[torch.Tensor, torch.Tensor]:
        self.to(train.device)
        v = torch.cat([fn(train.full(range(i, min(i + 128, train.n)), self.layers, self.needs_mel)).cpu()
                       for i in range(0, train.n, 128)])
        return v.mean(0), v.std(0).clamp_min(1e-6)

    @torch.no_grad()
    def setup(self, train: FrameSet) -> None:
        m, s = self._train_stats(train, self.features)
        self.mean.copy_(m)
        self.std.copy_(s)

    def forward(self, b: FrameBatch) -> torch.Tensor:
        return self.head((self.features(b) - self.mean) / self.std)


def _vuv_two(b: FrameBatch) -> torch.Tensor:
    w = segment_masks(b)
    return torch.stack([w[..., 0], w[..., 1] + w[..., 2]], -1)


@register("seg_vuv2")
class VoicedUnvoiced2(_SegmentBase):
    """H1, two-way: voiced mean and mean of the non-voiced frames (speech and silence together), concatenated."""

    def __init__(self):
        super().__init__()
        self._init_head(2048)

    def features(self, b):
        m, _ = class_means(b.hidden[:, 0], _vuv_two(b), self.base_vec(b))
        return m.flatten(1)


@register("seg_vus3")
class VoicedUnvoicedSilence3(_SegmentBase):
    """H1, three-way: voiced, unvoiced speech and silence means, concatenated."""

    def __init__(self):
        super().__init__()
        self._init_head(3072)

    def features(self, b):
        m, _ = class_means(b.hidden[:, 0], segment_masks(b), self.base_vec(b))
        return m.flatten(1)


@register("seg_vu_soft")
class VoicedUnvoicedSoft(_SegmentBase):
    """H1, soft two-way without silence: periodicity-weighted voiced mean (the baseline pool) and the mean of unvoiced
    speech weighted by (1 - periodicity); silence frames (loudness below the threshold) carry no weight in either."""

    def __init__(self):
        super().__init__()
        self._init_head(2048)

    def features(self, b):
        speech = (b.zloud >= SILENCE_LOUD) & b.mask
        w = torch.stack([b.periodicity * b.mask, (1 - b.periodicity) * speech], -1)
        m, _ = class_means(b.hidden[:, 0], w, self.base_vec(b))
        return m.flatten(1)


class _GateBase(_SegmentBase):
    """H2: three softmax gates per frame from a small MLP on (periodicity, log loudness[, frame features]); each
    segment is the gate-weighted mean, smoothed towards the baseline pooled vector with a mass of one frame."""

    use_feat = False

    def __init__(self):
        super().__init__()
        self._init_head(3072)
        n_in = 2 + (16 if self.use_feat else 0)
        if self.use_feat:
            self.norm = nn.LayerNorm(1024)
            self.fproj = nn.Linear(1024, 16)
        self.gate = nn.Sequential(nn.Linear(n_in, 16), nn.GELU(), nn.Linear(16, 3))

    def gates(self, b):
        z = [b.periodicity[..., None], torch.log(b.zloud + 0.05)[..., None]]
        if self.use_feat:
            z.append(self.fproj(self.norm(b.hidden[:, 0])))
        return torch.softmax(self.gate(torch.cat(z, -1)), -1) * b.mask[..., None]

    def features(self, b):
        g = self.gates(b)
        base = self.base_vec(b)
        m = (torch.einsum("btc,btd->bcd", g, b.hidden[:, 0]) + base[:, None, :]) / (g.sum(1) + 1.0)[..., None]
        return m.flatten(1)

    @torch.no_grad()
    def setup(self, train: FrameSet) -> None:
        # gated means live on the scale of the baseline pooled vector: its statistics, tiled over the three segments
        m, s = self._train_stats(train, self.base_vec)
        self.mean.copy_(m.repeat(3))
        self.std.copy_(s.repeat(3))


@register("seg_gate_basic")
class GateBasic(_GateBase):
    use_feat = False


@register("seg_gate_feat")
class GateFeat(_GateBase):
    use_feat = True


class _Codebook(_SegmentBase):
    """H3: fixed k-means codebook on training layer-6 frames; mean per cluster (back-off to the baseline pool), z-score,
    shared Linear(1024, 128) per cluster block, concatenated, baseline head."""

    K = 6
    PROJ = 128

    def __init__(self):
        super().__init__()
        self._init_head(self.K * 1024, self.K * self.PROJ)
        self.register_buffer("centroids", torch.zeros(self.K, 1024))
        self.proj = nn.Linear(1024, self.PROJ)

    @torch.no_grad()
    def setup(self, train: FrameSet) -> None:
        dev = train.device
        self.to(dev)
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(train.hidden[6].shape[0], generator=g)[:300_000].to(dev)
        x = train.hidden[6][idx].float()
        c = x[torch.randperm(x.shape[0], generator=g)[: self.K].to(dev)].clone()
        for _ in range(25):
            a = torch.cdist(x, c).argmin(1)
            for k in range(self.K):
                sel = a == k
                if sel.any():
                    c[k] = x[sel].mean(0)
        self.centroids.copy_(c)
        self.cluster_sizes = torch.bincount(torch.cdist(x, c).argmin(1), minlength=self.K).cpu().tolist()
        super().setup(train)

    def features(self, b):
        x = b.hidden[:, 0]
        B, T, D = x.shape
        a = torch.cdist(x.reshape(-1, D), self.centroids).argmin(1).reshape(B, T)
        w = F.one_hot(a, self.K).float() * b.mask[..., None]
        m, _ = class_means(x, w, self.base_vec(b))
        return m.flatten(1)

    def forward(self, b):
        z = ((self.features(b) - self.mean) / self.std).reshape(-1, self.K, 1024)
        return self.head(self.proj(z).flatten(1))


@register("seg_codebook6")
class Codebook6(_Codebook):
    K = 6


@register("seg_codebook16")
class Codebook16(_Codebook):
    K = 16


class _MelBranch(_SegmentBase):
    """H11: layer-6 pooled vector (baseline, or the two-way voiced/non-voiced vector) plus cepstrally smoothed log-mel
    mean and standard deviation of voiced and unvoiced-speech frames (back-off to all frames)."""

    needs_mel = True
    n_cep = 23
    two_way = False

    def __init__(self):
        super().__init__()
        self._init_head((2048 if self.two_way else 1024) + 2 * 2 * self.n_cep)
        self.register_buffer("dct", dct_matrix(80, self.n_cep))

    def features(self, b):
        base = self.base_vec(b)
        w = segment_masks(b)[..., :2]
        s = w.sum(1)
        cep = torch.einsum("btm,cm->btc", b.mel, self.dct)
        n = b.mask.sum(1, keepdim=True).clamp_min(1)
        mu_all = (cep * b.mask[..., None]).sum(1) / n
        sd_all = (((cep - mu_all[:, None]) ** 2 * b.mask[..., None]).sum(1) / n).sqrt()
        mu = torch.einsum("btk,btc->bkc", w, cep) / s.clamp_min(1e-6)[..., None]
        sq = torch.stack([((cep - mu[:, k, None]) ** 2 * w[..., k, None]).sum(1) for k in range(2)], 1)
        sd = (sq / s.clamp_min(1e-6)[..., None]).sqrt()
        ok = (s >= MIN_FRAMES)[..., None]
        mu = torch.where(ok, mu, mu_all[:, None])
        sd = torch.where(ok, sd, sd_all[:, None])
        if self.two_way:
            m, _ = class_means(b.hidden[:, 0], _vuv_two(b), base)
            parts = [m.flatten(1)]
        else:
            parts = [base]
        return torch.cat(parts + [mu.flatten(1), sd.flatten(1)], 1)


@register("seg_mel_on_base")
class MelOnBase(_MelBranch):
    two_way = False


@register("seg_mel_on_vuv2")
class MelOnVUV2(_MelBranch):
    two_way = True
