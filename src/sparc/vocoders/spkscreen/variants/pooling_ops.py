"""Pooling-operator family of the speaker-encoder screen (hypotheses H4 to H8): what replaces the periodicity-weighted
mean of WavLM layer 6 before the shipped-size FFN.

Every variant reads layer 6 only and keeps the shared trainer, objective and 64-d output. Each one turns the frames
into a fixed-size statistics vector of size ``D`` and feeds it to ``SpeakerFFN(in_dim=D, hidden_dim=1024)``; the FFN's
input z-score is fitted in ``setup`` from the training utterances with the module as initialized. Hyperparameters are
fixed here from the literature, before the first run (docs/spkemb_encoder_screen.md):

* H4 attentive statistics: ECAPA-TDNN channel-wise attention, bottleneck 128 (Desplanques 2020); the ``prior`` form adds
  log periodicity weight to the attention logits so that the attention starts from SPARC's weighting.
* H5 multi-head: 4 heads (Okabe 2018), each head attends and pools its own 256-d slice (mean and std); multi-query:
  4 queries pooling all 1024 dims (mean only).
* H6 NetVLAD (Arandjelovic 2016) with 8 clusters on a 128-d projection, plus 2 ghost clusters (Zhong 2018) whose
  residuals are dropped; and a fixed k-means codebook of 8 centroids (fit on training frames, seed 0, 25 iterations)
  with per-cluster means projected 1024 to 128 and concatenated. Neither uses periodicity: absorbing silence and
  noise is the point of the ghost clusters.
* H7 weighted mean+std, mean+std+skewness, and mean plus the covariance of a learned 16-d projection (upper triangle).
* H8 weighted trimmed mean (10 percent per side) and weighted quantiles (0.1, 0.5, 0.9).
"""

import numpy as np
import torch
from torch import nn

from sparc.vocoders.models.speaker import SpeakerFFN
from sparc.vocoders.spkscreen.data import FrameBatch, FrameSet
from sparc.vocoders.spkscreen.pooling import PoolingModule, pool_weights, register, weighted_mean

DIM = 1024


def wstats(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean and standard deviation over time of ``x [B, T, D]`` with weights ``w [B, T]``."""
    mu = weighted_mean(x, w)
    var = weighted_mean((x - mu[:, None]) ** 2, w)
    return mu, (var + eps).sqrt()


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over time (dim 1) of ``[B, T, C]`` logits restricted to real frames."""
    return torch.softmax(logits.masked_fill(~mask[..., None], float("-inf")), dim=1)


class StatsPool(PoolingModule):
    """Frames to a ``[B, D]`` statistics vector (``stats``), then the FFN. Subclasses set ``D`` and implement ``stats``."""

    layers = (6,)

    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.ffn = SpeakerFFN(in_dim=in_dim, hidden_dim=DIM, dropout=dropout)
        self.register_buffer("fmean", torch.zeros(DIM))
        self.register_buffer("fstd", torch.ones(DIM))

    def frames(self, b: FrameBatch) -> torch.Tensor:
        """Layer 6 frames ``[B, T, 1024]`` (raw)."""
        return b.hidden[:, 0]

    def znorm(self, b: FrameBatch) -> torch.Tensor:
        """Frames z-scored with training-frame statistics, zero on padding."""
        return (self.frames(b) - self.fmean) / self.fstd * b.mask[..., None]

    def stats(self, b: FrameBatch) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def setup_extra(self, train: FrameSet) -> None:
        """Hook run after the frame statistics are fitted and before the FFN input z-score (codebooks)."""

    @torch.no_grad()
    def setup(self, train: FrameSet) -> None:
        h = train.hidden[self.layers[0]]
        n = h.shape[0]
        s = torch.zeros(DIM, dtype=torch.float64, device=h.device)
        s2 = torch.zeros_like(s)
        for i in range(0, n, 500_000):
            c = h[i : i + 500_000].double()
            s += c.sum(0)
            s2 += (c**2).sum(0)
        mean = s / n
        self.fmean.copy_(mean.float())
        self.fstd.copy_(((s2 / n - mean**2).clamp_min(1e-8).sqrt()).float())
        self.to(h.device)
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


# ---------------------------------------------------------------------------------------------------- H4, H5


class AttentiveStats(StatsPool):
    """ECAPA-style channel-wise attentive statistics: softmax over time per channel, weighted mean and std."""

    def __init__(self, prior: bool = False, bottleneck: int = 128, dropout: float = 0.2):
        super().__init__(2 * DIM, dropout)
        self.prior = prior
        self.att = nn.Sequential(nn.Linear(DIM, bottleneck), nn.Tanh(), nn.Linear(bottleneck, DIM))

    def stats(self, b: FrameBatch) -> torch.Tensor:
        x = self.znorm(b)
        logits = self.att(x)
        if self.prior:
            logits = logits + torch.log(pool_weights(b.periodicity, b.mask) + 1e-4)[..., None]
        a = masked_softmax(logits, b.mask)
        mu = (a * x).sum(1)
        var = (a * x * x).sum(1) - mu**2
        return torch.cat([mu, var.clamp_min(1e-5).sqrt()], 1)


register("pool_asp")(AttentiveStats)
register("pool_asp_prior")(lambda **kw: AttentiveStats(prior=True, **kw))


@register("pool_mhap")
class MultiHeadAttentive(StatsPool):
    """Multi-head attentive statistics: each of ``heads`` heads attends over time and pools its own slice of channels."""

    def __init__(self, heads: int = 4, bottleneck: int = 128, dropout: float = 0.2):
        super().__init__(2 * DIM, dropout)
        self.heads = heads
        self.att = nn.Sequential(nn.Linear(DIM, bottleneck), nn.Tanh(), nn.Linear(bottleneck, heads))

    def stats(self, b: FrameBatch) -> torch.Tensor:
        x = self.znorm(b)
        a = masked_softmax(self.att(x), b.mask)  # [B, T, H]
        xh = x.view(*x.shape[:2], self.heads, DIM // self.heads)
        mu = (a[..., None] * xh).sum(1)
        var = (a[..., None] * xh * xh).sum(1) - mu**2
        return torch.cat([mu.flatten(1), var.clamp_min(1e-5).sqrt().flatten(1)], 1)


@register("pool_mqap")
class MultiQueryAttentive(StatsPool):
    """Multi-query attentive pooling: ``queries`` attention distributions, each pooling all channels (mean only)."""

    def __init__(self, queries: int = 4, bottleneck: int = 128, dropout: float = 0.2):
        super().__init__(queries * DIM, dropout)
        self.att = nn.Sequential(nn.Linear(DIM, bottleneck), nn.Tanh(), nn.Linear(bottleneck, queries))

    def stats(self, b: FrameBatch) -> torch.Tensor:
        x = self.znorm(b)
        a = masked_softmax(self.att(x), b.mask)  # [B, T, Q]
        return torch.einsum("btq,btd->bqd", a, x).flatten(1)


# ---------------------------------------------------------------------------------------------------- H6


class NetVLAD(StatsPool):
    """NetVLAD over a 128-d projection of the frames; ``ghost`` extra clusters absorb frames and are dropped.

    ``V_k = sum_t a_tk (x_t - c_k) / T`` with soft assignments ``a`` over real and ghost clusters; output is the
    ``clusters * 128`` concatenation of the real clusters.
    """

    def __init__(self, clusters: int = 8, ghost: int = 0, proj: int = 128, dropout: float = 0.2):
        super().__init__(clusters * proj, dropout)
        self.k, self.g = clusters, ghost
        self.proj = nn.Linear(DIM, proj, bias=False)
        self.assign = nn.Linear(proj, clusters + ghost)
        self.centroids = nn.Parameter(torch.randn(clusters + ghost, proj) * 0.1)

    def stats(self, b: FrameBatch) -> torch.Tensor:
        x = self.proj(self.znorm(b))  # [B, T, P]
        a = torch.softmax(self.assign(x), -1) * b.mask[..., None]  # [B, T, K+G]
        n = b.mask.sum(1).clamp_min(1)[:, None, None]
        v = (torch.einsum("btk,btp->bkp", a, x) - a.sum(1)[..., None] * self.centroids) / n
        return v[:, : self.k].flatten(1)


register("pool_netvlad")(NetVLAD)
register("pool_ghostvlad")(lambda **kw: NetVLAD(ghost=2, **kw))


@register("pool_kmeans")
class KMeansCodebook(StatsPool):
    """Fixed k-means codebook: per-cluster mean of the z-scored frames, shared 1024 to 128 projection, concatenation.

    A cluster with no frame in the reference backs off to the utterance mean over all frames.
    """

    def __init__(self, clusters: int = 8, proj: int = 128, iters: int = 25, n_fit: int = 150_000, dropout: float = 0.2):
        super().__init__(clusters * proj, dropout)
        self.k, self.iters, self.n_fit = clusters, iters, n_fit
        self.register_buffer("codebook", torch.zeros(clusters, DIM))
        self.proj = nn.Linear(DIM, proj)

    def assign(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest centroid of z-scored frames ``[B, T, D]``: ``[B, T]``."""
        d = (x**2).sum(-1, keepdim=True) - 2 * x @ self.codebook.T + (self.codebook**2).sum(-1)
        return d.argmin(-1)

    @torch.no_grad()
    def setup_extra(self, train: FrameSet) -> None:
        h = train.hidden[self.layers[0]]
        g = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.randperm(h.shape[0], generator=g)[: self.n_fit].to(h.device)
        x = (h[idx].float() - self.fmean) / self.fstd
        c = x[torch.randperm(x.shape[0], generator=g)[: self.k].to(h.device)].clone()
        for _ in range(self.iters):
            d = (x**2).sum(-1, keepdim=True) - 2 * x @ c.T + (c**2).sum(-1)
            a = d.argmin(-1)
            for j in range(self.k):
                sel = a == j
                c[j] = x[sel].mean(0) if sel.any() else x[int(torch.randint(0, x.shape[0], (1,), generator=g))]
        self.codebook.copy_(c)

    def stats(self, b: FrameBatch) -> torch.Tensor:
        x = self.znorm(b)
        oh = torch.nn.functional.one_hot(self.assign(x), self.k).to(x.dtype) * b.mask[..., None]  # [B, T, K]
        cnt = oh.sum(1)  # [B, K]
        mean_all = x.sum(1) / b.mask.sum(1, keepdim=True).clamp_min(1)
        mu = torch.einsum("btk,btd->bkd", oh, x) / cnt.clamp_min(1)[..., None]
        mu = torch.where((cnt > 0)[..., None], mu, mean_all[:, None])
        return self.proj(mu).flatten(1)


# ---------------------------------------------------------------------------------------------------- H7


@register("pool_meanstd")
class MeanStd(StatsPool):
    """Periodicity-weighted mean and standard deviation (the base of the higher-order arms)."""

    def __init__(self, dropout: float = 0.2):
        super().__init__(2 * DIM, dropout)

    def stats(self, b: FrameBatch) -> torch.Tensor:
        mu, sd = wstats(self.frames(b), pool_weights(b.periodicity, b.mask))
        return torch.cat([mu, sd], 1)


@register("pool_skew")
class MeanStdSkew(StatsPool):
    """Periodicity-weighted mean, standard deviation and skewness (standardized third moment, clipped to +-5)."""

    def __init__(self, dropout: float = 0.2):
        super().__init__(3 * DIM, dropout)

    def stats(self, b: FrameBatch) -> torch.Tensor:
        x, w = self.frames(b), pool_weights(b.periodicity, b.mask)
        mu, sd = wstats(x, w)
        sd = sd.clamp_min(1e-3)
        skew = weighted_mean(((x - mu[:, None]) / sd[:, None]) ** 3, w).clamp(-5, 5)
        return torch.cat([mu, sd, skew], 1)


@register("pool_lowrank_cov")
class LowRankCovariance(StatsPool):
    """Weighted mean plus the covariance of a learned ``rank``-d projection of the z-scored frames (upper triangle)."""

    def __init__(self, rank: int = 16, dropout: float = 0.2):
        super().__init__(DIM + rank * (rank + 1) // 2, dropout)
        self.proj = nn.Linear(DIM, rank, bias=False)
        self.register_buffer("tri", torch.triu_indices(rank, rank))

    def stats(self, b: FrameBatch) -> torch.Tensor:
        w = pool_weights(b.periodicity, b.mask)
        mu = weighted_mean(self.frames(b), w)
        z = self.proj(self.znorm(b))
        zc = z - weighted_mean(z, w)[:, None]
        cov = torch.einsum("bt,bti,btj->bij", w, zc, zc) / w.sum(1).clamp_min(1e-12)[:, None, None]
        return torch.cat([mu, cov[:, self.tri[0], self.tri[1]]], 1)


# ---------------------------------------------------------------------------------------------------- H8


def sorted_weighted(x: torch.Tensor, w: torch.Tensor):
    """Sort ``x [B, T, D]`` along time per channel; returns sorted values, matching weights and normalized cumulative weights."""
    v, idx = torch.sort(x, dim=1)
    ws = torch.gather(w[..., None].expand_as(x), 1, idx)
    c = ws.cumsum(1) / ws.sum(1, keepdim=True).clamp_min(1e-12)
    return v, ws, c


@register("pool_trimmed")
class TrimmedMean(StatsPool):
    """Periodicity-weighted trimmed mean per channel: weight mass below ``trim`` and above ``1 - trim`` is dropped."""

    def __init__(self, trim: float = 0.1, dropout: float = 0.2):
        super().__init__(DIM, dropout)
        self.trim = trim

    def stats(self, b: FrameBatch) -> torch.Tensor:
        v, ws, c = sorted_weighted(self.frames(b), pool_weights(b.periodicity, b.mask))
        lo = c - ws / ws.sum(1, keepdim=True).clamp_min(1e-12)
        keep = (torch.clamp(c, max=1 - self.trim) - torch.clamp(lo, min=self.trim)).clamp_min(0)
        return (keep * v).sum(1) / keep.sum(1).clamp_min(1e-12)


@register("pool_quantile")
class Quantiles(StatsPool):
    """Periodicity-weighted per-channel quantiles (0.1, 0.5, 0.9), concatenated."""

    def __init__(self, qs: tuple[float, ...] = (0.1, 0.5, 0.9), dropout: float = 0.2):
        super().__init__(len(qs) * DIM, dropout)
        self.qs = tuple(qs)

    def stats(self, b: FrameBatch) -> torch.Tensor:
        v, _, c = sorted_weighted(self.frames(b), pool_weights(b.periodicity, b.mask))
        out = []
        for q in self.qs:
            i = (c < q - 1e-6).sum(1, keepdim=True).clamp(max=v.shape[1] - 1)
            out.append(torch.gather(v, 1, i)[:, 0])
        return torch.cat(out, 1)
