"""Multi-layer, multi-scale and frame-network variants of the speaker-encoder screen (hypotheses H9, H10, H12, H14).

H13 (multi-reference aggregation) is an inference-time study of the trained baseline encoders: see
``sparc.vocoders.spkscreen.multiref``.

Every variant keeps the 64-d output, the baseline head (z-score, Linear(in, 1024), GELU, dropout 0.2, Linear(1024, 64);
the z-score is fitted on the training utterances in ``setup``) and the shared trainer. Hyperparameters are fixed here
from the literature before the first run:

* H9 ``ms_l1_l6``: periodicity-weighted means of WavLM layers 1 and 6 concatenated (2048-d); ``ms_layerw``: layers 1, 3,
  6 and 12 combined by a learned softmax over layers (SUPERB-style weighted sum, logits start at zero) of per-layer
  z-scored frames, then the periodicity-weighted mean.
* H10 ``ms_seg05``: segments of 25 frames (0.5 s); each segment's periodicity-weighted mean passes a shared
  Linear(1024, 256) + GELU; the segment embeddings are pooled again by a mean and a standard deviation weighted by the
  number of real frames; the head reads [utterance mean (1024), segment mean (256), segment std (256)].
  ``ms_mfa``: multi-layer feature aggregation before pooling (MFA-Conformer, Zhang 2022): layers 1, 6 and 12 z-scored,
  concatenated per frame, Linear(3072, 512) + GELU, then periodicity-weighted mean and standard deviation (1024-d).
* H12 ``ms_tdnn``: layer 6 z-scored, Linear(1024, 512) + GELU, three residual dilated 1-d convolution blocks (kernel 3,
  dilations 2, 3, 4 as in ECAPA-TDNN, pre-LayerNorm, GELU, 0.1 dropout), periodicity-weighted mean and standard
  deviation (1024-d). ``ms_xformer``: layer 6 z-scored, Linear(1024, 256), two pre-norm transformer encoder layers
  (4 heads, feed-forward 1024, dropout 0.1, no positional encoding), mean and standard deviation (512-d).
* H14 ``ms_grl_content`` / ``ms_grl_f0`` / ``ms_grl_both``: the baseline encoder plus adversarial heads behind a gradient
  reversal layer (Ganin 2016) with constant strength 0.1. Content adversary: Linear(64, 256), GELU, Linear(256, 32)
  predicting the crop's histogram over a 32-word k-means codebook of layer-6 frames (fit on 150k training frames, 25
  Lloyd iterations, seed 0) by soft cross entropy. F0 adversary: the same head predicting the crop's periodicity-weighted
  mean log F0 (standardized with the training utterances' median log F0 mean and SD) by mean squared error. The
  adversarial loss is added to the shared objective (the pre-registration allows this for H14).
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sparc.vocoders.models.speaker import SpeakerFFN
from sparc.vocoders.spkscreen.data import FrameBatch, FrameSet
from sparc.vocoders.spkscreen.pooling import PoolingModule, pool_weights, register, weighted_mean

DIM = 1024


def mean_std(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = weighted_mean(x, w)
    var = weighted_mean((x - mu[:, None]) ** 2, w)
    return torch.cat([mu, (var + eps).sqrt()], 1)


class MSPool(PoolingModule):
    """Base: z-scored multi-layer frames, ``stats(b)`` to a ``[B, in_dim]`` vector, then the baseline head."""

    def __init__(self, layers: tuple[int, ...], in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.layers = tuple(layers)
        self.ffn = SpeakerFFN(in_dim=in_dim, hidden_dim=DIM, dropout=dropout)
        self.register_buffer("fmean", torch.zeros(len(layers), DIM))
        self.register_buffer("fstd", torch.ones(len(layers), DIM))

    def znorm(self, b: FrameBatch) -> torch.Tensor:
        """``[B, L, T, D]`` frames z-scored per layer with training-frame statistics, zero on padding."""
        return (b.hidden - self.fmean[None, :, None]) / self.fstd[None, :, None] * b.mask[:, None, :, None]

    def stats(self, b: FrameBatch) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def setup_extra(self, train: FrameSet) -> None:
        """Hook run after the frame statistics are fitted (codebooks, target statistics)."""

    @torch.no_grad()
    def setup(self, train: FrameSet) -> None:
        for li, k in enumerate(self.layers):
            h = train.hidden[k]
            n = h.shape[0]
            s = torch.zeros(DIM, dtype=torch.float64, device=h.device)
            s2 = torch.zeros_like(s)
            for i in range(0, n, 500_000):
                c = h[i : i + 500_000].double()
                s += c.sum(0)
                s2 += (c**2).sum(0)
            mean = s / n
            self.fmean[li] = mean.float().to(self.fmean.device)
            self.fstd[li] = (s2 / n - mean**2).clamp_min(1e-8).sqrt().float().to(self.fstd.device)
        self.to(train.hidden[self.layers[0]].device)
        self.setup_extra(train)
        was = self.training
        self.eval()
        order = np.argsort(train.lens)
        vecs = []
        for i in range(0, train.n, 64):
            vecs.append(self.stats(train.full(order[i : i + 64], self.layers)).float().cpu())
        v = torch.cat(vecs)
        self.ffn.mean.copy_(v.mean(0))
        self.ffn.std.copy_(v.std(0).clamp_min(1e-6))
        self.train(was)

    def forward(self, b: FrameBatch) -> torch.Tensor:
        return self.ffn(self.stats(b))


# ---------------------------------------------------------------------------------------------------- H9


@register("ms_l1_l6")
class ShallowPlusMid(MSPool):
    """Periodicity-weighted means of layers 1 and 6, concatenated."""

    def __init__(self, shallow: int = 1, dropout: float = 0.2):
        super().__init__((shallow, 6), 2 * DIM, dropout)

    def stats(self, b: FrameBatch) -> torch.Tensor:
        w = pool_weights(b.periodicity, b.mask)
        return torch.cat([weighted_mean(b.hidden[:, i], w) for i in range(len(self.layers))], 1)


@register("ms_layerw")
class LearnedLayerWeights(MSPool):
    """Learned softmax weights over layers 1, 3, 6, 12 applied to z-scored frames before the weighted mean."""

    def __init__(self, layers: tuple[int, ...] = (1, 3, 6, 12), dropout: float = 0.2):
        super().__init__(tuple(layers), DIM, dropout)
        self.logits = nn.Parameter(torch.zeros(len(layers)))

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, 0)

    def stats(self, b: FrameBatch) -> torch.Tensor:
        x = (self.znorm(b) * self.layer_weights()[None, :, None, None]).sum(1)
        return weighted_mean(x, pool_weights(b.periodicity, b.mask))


# ---------------------------------------------------------------------------------------------------- H10


@register("ms_seg05")
class SegmentPooling(MSPool):
    """Utterance mean plus mean and std over 0.5 s segment embeddings."""

    def __init__(self, seg: int = 25, seg_dim: int = 256, dropout: float = 0.2):
        super().__init__((6,), DIM + 2 * seg_dim, dropout)
        self.seg = seg
        self.proj = nn.Sequential(nn.Linear(DIM, seg_dim), nn.GELU())

    def stats(self, b: FrameBatch) -> torch.Tensor:
        x = self.znorm(b)[:, 0]
        w = pool_weights(b.periodicity, b.mask)
        utt = weighted_mean(x, w)
        B, T, D = x.shape
        pad = (-T) % self.seg
        S = (T + pad) // self.seg
        x = F.pad(x, (0, 0, 0, pad)).view(B, S, self.seg, D)
        m = F.pad(b.mask.to(x.dtype), (0, pad)).view(B, S, self.seg)
        wv = F.pad(w, (0, pad)).view(B, S, self.seg)
        mass = wv.sum(2, keepdim=True)
        wv = torch.where(mass > 1e-6, wv, m)  # segments without periodic frames fall back to uniform
        sm = (x * wv[..., None]).sum(2) / wv.sum(2, keepdim=True).clamp_min(1e-12)  # [B, S, D]
        emb = self.proj(sm)
        sw = m.sum(2)  # real frames per segment
        mu = (emb * sw[..., None]).sum(1) / sw.sum(1, keepdim=True)
        var = ((emb - mu[:, None]) ** 2 * sw[..., None]).sum(1) / sw.sum(1, keepdim=True)
        return torch.cat([utt, mu, (var + 1e-6).sqrt()], 1)


@register("ms_mfa")
class MultiLayerAggregation(MSPool):
    """Layers 1, 6, 12 concatenated per frame, projected, then periodicity-weighted mean and std."""

    def __init__(self, layers: tuple[int, ...] = (1, 6, 12), width: int = 512, dropout: float = 0.2):
        super().__init__(tuple(layers), 2 * width, dropout)
        self.proj = nn.Sequential(nn.Linear(len(layers) * DIM, width), nn.GELU())

    def stats(self, b: FrameBatch) -> torch.Tensor:
        x = self.znorm(b)  # [B, L, T, D]
        x = x.permute(0, 2, 1, 3).flatten(2)
        y = self.proj(x) * b.mask[..., None]
        return mean_std(y, pool_weights(b.periodicity, b.mask))


# ---------------------------------------------------------------------------------------------------- H12


class DilatedBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.out = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x) * mask[..., None]
        h = F.gelu(self.conv(h.transpose(1, 2)).transpose(1, 2))
        return (x + self.drop(self.out(h))) * mask[..., None]


@register("ms_tdnn")
class FrameTDNN(MSPool):
    """Three residual dilated convolution blocks on projected layer-6 frames, then mean and std."""

    def __init__(self, width: int = 512, dilations: tuple[int, ...] = (2, 3, 4), dropout: float = 0.2):
        super().__init__((6,), 2 * width, dropout)
        self.inp = nn.Sequential(nn.Linear(DIM, width), nn.GELU())
        self.blocks = nn.ModuleList([DilatedBlock(width, d, 0.1) for d in dilations])

    def stats(self, b: FrameBatch) -> torch.Tensor:
        m = b.mask.to(b.hidden.dtype)
        x = self.inp(self.znorm(b)[:, 0]) * m[..., None]
        for blk in self.blocks:
            x = blk(x, m)
        return mean_std(x, pool_weights(b.periodicity, b.mask))


@register("ms_xformer")
class FrameTransformer(MSPool):
    """Two pre-norm transformer encoder layers on projected layer-6 frames, then mean and std."""

    def __init__(self, width: int = 256, layers: int = 2, heads: int = 4, ff: int = 1024, dropout: float = 0.2):
        super().__init__((6,), 2 * width, dropout)
        self.inp = nn.Sequential(nn.Linear(DIM, width), nn.GELU())
        layer = nn.TransformerEncoderLayer(width, heads, ff, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def stats(self, b: FrameBatch) -> torch.Tensor:
        m = b.mask.to(b.hidden.dtype)
        x = self.inp(self.znorm(b)[:, 0]) * m[..., None]
        x = self.enc(x, src_key_padding_mask=~b.mask) * m[..., None]
        return mean_std(x, pool_weights(b.periodicity, b.mask))


# ---------------------------------------------------------------------------------------------------- H14


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class GRLBaseline(MSPool):
    """Baseline encoder (layer 6, periodicity-weighted mean, FFN) plus gradient-reversed content / F0 adversaries."""

    def __init__(self, content: bool, f0: bool, lam: float = 0.1, clusters: int = 32, n_fit: int = 150_000,
                 iters: int = 25, dropout: float = 0.2):
        super().__init__((6,), DIM, dropout)
        self.content, self.f0, self.lam, self.k, self.n_fit, self.iters = content, f0, lam, clusters, n_fit, iters
        self.register_buffer("codebook", torch.zeros(clusters, DIM))
        self.register_buffer("f0_stat", torch.tensor([5.0, 0.3]))
        if content:
            self.adv_c = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, clusters))
        if f0:
            self.adv_f = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 1))
        self._adv = None

    def stats(self, b: FrameBatch) -> torch.Tensor:
        return weighted_mean(b.hidden[:, 0], pool_weights(b.periodicity, b.mask))

    @torch.no_grad()
    def setup_extra(self, train: FrameSet) -> None:
        h = train.hidden[6]
        g = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.randperm(h.shape[0], generator=g)[: self.n_fit].to(h.device)
        x = (h[idx].float() - self.fmean[0]) / self.fstd[0]
        c = x[torch.randperm(x.shape[0], generator=g)[: self.k].to(h.device)].clone()
        for _ in range(self.iters):
            a = ((x**2).sum(-1, keepdim=True) - 2 * x @ c.T + (c**2).sum(-1)).argmin(-1)
            for j in range(self.k):
                sel = a == j
                c[j] = x[sel].mean(0) if sel.any() else x[int(torch.randint(0, x.shape[0], (1,), generator=g))]
        self.codebook.copy_(c)
        lf = np.asarray(train.logf0)
        lf = lf[np.isfinite(lf)]
        self.f0_stat.copy_(torch.tensor([lf.mean(), lf.std() + 1e-6], dtype=torch.float32))

    def forward(self, b: FrameBatch) -> torch.Tensor:
        emb = self.ffn(self.stats(b))
        self._adv = None
        if self.training:
            rev = GradReverse.apply(emb, self.lam)
            loss = emb.new_zeros(())
            if self.content:
                x = (b.hidden[:, 0] - self.fmean[0]) / self.fstd[0]
                d = (x**2).sum(-1, keepdim=True) - 2 * x @ self.codebook.T + (self.codebook**2).sum(-1)
                oh = F.one_hot(d.argmin(-1), self.k).to(x.dtype) * b.mask[..., None]
                hist = oh.sum(1) / b.mask.sum(1, keepdim=True).clamp_min(1)
                loss = loss - (hist * F.log_softmax(self.adv_c(rev), -1)).sum(-1).mean()
            if self.f0:
                w = pool_weights(b.periodicity, b.mask)
                lf = weighted_mean(torch.log(b.f0.clamp_min(1.0))[..., None], w)[:, 0]
                loss = loss + F.mse_loss(self.adv_f(rev)[:, 0], (lf - self.f0_stat[0]) / self.f0_stat[1])
            self._adv = loss
        return emb

    def extra_loss(self) -> torch.Tensor:
        return self._adv if self._adv is not None else 0.0


register("ms_grl_content")(lambda **kw: GRLBaseline(content=True, f0=False, **kw))
register("ms_grl_f0")(lambda **kw: GRLBaseline(content=False, f0=True, **kw))
register("ms_grl_both")(lambda **kw: GRLBaseline(content=True, f0=True, **kw))
