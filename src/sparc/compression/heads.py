"""EMA prediction heads on top of a (subset) SSL encoder.

    linear -- frame-wise D -> 12 (the probe; can be initialized from a ridge solution)
    conv   -- D -> H, temporal Conv1d of `kernel` frames, -> 12; causal (past frames only) or centered
    attn   -- D -> H, one Transformer layer whose attention is limited to a `window` of frames
              (past-only when causal, symmetric otherwise; window=0 means unlimited), -> 12

Layer pooling (LayerPool) replaces the last hidden state by a weighted sum of all retained layers'
outputs, each first layer-normalized (no affine) so that layers with a larger residual-stream scale
do not dominate. It adds no encoder compute. Modes:
    static -- one softmax weight per layer, shared by all frames (SUPERB-style weighted sum)
    attn   -- frame-wise weights: score_l(t) = b_l + v . tanh(W LN(h_l(t))), softmax over layers l,
              so each frame can draw on different layers (zero-initialized: starts uniform)
With per_articulator, every articulator (TD, TB, TT, LI, UL, LL) gets its own weights (static logits,
or its own v and b for attn) and its own pooled representation, and the head predicts that
articulator's x/y pair from it: articulator-specific input/output projections, with the temporal
convolution shared. Per-articulator pooling is supported for the linear and conv heads.

All heads predict standardized EMA. `smooth` applies the FIR equivalent of the zero-phase 10 Hz
Butterworth low-pass that the ridge probe applies to its input features (linear and time-invariant,
so filtering a linear head's output is the same as filtering its input). It looks 1 s ahead,
so offline only. Note that the encoders themselves use bidirectional self-attention: "causal"
here restricts only the head's temporal aggregation, not the encoder.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.inversion import butter_bandpass

FT_SR = 50
N_ARTICULATORS = 6  # channel pairs (0,1), (2,3), ... in mngu0.CHANNELS order
POOL_MODES = ("none", "static", "attn")


def lowpass_kernel(cut=10, fs=FT_SR, half=50, order=5):
    """Impulse response of filtfilt(butter(order, cut)) truncated to 2*half+1 taps (symmetric)."""
    from scipy.signal import filtfilt

    b, a = butter_bandpass(cut, fs, order=order)
    delta = np.zeros(8 * half + 1)
    delta[4 * half] = 1.0
    h = filtfilt(b, a, delta)[3 * half : 5 * half + 1]
    return torch.tensor(h / h.sum(), dtype=torch.float32)


class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        k = lowpass_kernel()
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = (len(k) - 1) // 2

    def forward(self, y):  # (B, T, C)
        B, T, C = y.shape
        x = y.transpose(1, 2).reshape(B * C, 1, T)
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        return F.conv1d(x, self.kernel.to(x.dtype)).reshape(B, C, T).transpose(1, 2)


class LayerPool(nn.Module):
    """Weighted sum over layers with G independent weightings; returns (B, G, T, D)."""

    def __init__(self, n_layers, dim, mode="static", groups=1, attn_dim=64, norm=True):
        super().__init__()
        if mode not in ("static", "attn"):
            raise ValueError(mode)
        self.mode, self.groups, self.norm = mode, groups, norm
        self.logits = nn.Parameter(torch.zeros(groups, n_layers))
        if mode == "attn":
            self.proj = nn.Linear(dim, attn_dim)
            self.v = nn.Parameter(torch.zeros(groups, attn_dim))  # zero: uniform weights at init
        self.weight_sum = None  # running sum of weights (G, L) and frame count, for reporting
        self.weight_n = 0

    def reset_stats(self):
        self.weight_sum, self.weight_n = None, 0

    def mean_weights(self):
        return None if self.weight_sum is None else (self.weight_sum / max(self.weight_n, 1)).tolist()

    def forward(self, states, pad_mask=None):  # list of L tensors (B, T, D)
        Hn = torch.stack([F.layer_norm(s, s.shape[-1:]) for s in states], 2)  # (B, T, L, D)
        H = Hn if self.norm else torch.stack(states, 2)  # values pooled: normalized (default) or raw
        if self.mode == "static":
            w = torch.softmax(self.logits, -1)  # (G, L)
            out = torch.einsum("gl,btld->bgtd", w.to(H.dtype), H)
            w_frames = w[None, :, None, :].expand(H.shape[0], -1, H.shape[1], -1)
        else:
            k = torch.tanh(self.proj(Hn))  # (B, T, L, A); scores always from normalized layers
            scores = torch.einsum("btla,ga->bgtl", k, self.v.to(k.dtype)) + self.logits[None, :, None, :].to(k.dtype)
            w_frames = torch.softmax(scores.float(), -1)  # (B, G, T, L)
            out = torch.einsum("bgtl,btld->bgtd", w_frames.to(H.dtype), H)
        with torch.no_grad():
            valid = torch.ones(w_frames.shape[0], w_frames.shape[2], device=w_frames.device) if pad_mask is None \
                else (~pad_mask).float()
            s = torch.einsum("bgtl,bt->gl", w_frames.float(), valid)
            self.weight_sum = s if self.weight_sum is None else self.weight_sum + s
            self.weight_n += float(valid.sum())
        return out


class GroupLinear(nn.Module):
    """G independent Linear(d_in, d_out) applied to x: (B, G, T, d_in) -> (B, G, T, d_out)."""

    def __init__(self, groups, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, d_in, d_out))
        self.bias = nn.Parameter(torch.empty(groups, d_out))
        bound = 1 / math.sqrt(d_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.einsum("bgti,gio->bgto", x, self.weight.to(x.dtype)) + self.bias[None, :, None, :].to(x.dtype)


class Head(nn.Module):
    def __init__(self, dim, kind="linear", hidden=256, kernel=9, window=0, causal=False, n_out=12,
                 smooth=True, n_pool_layers=0, pool="static", per_articulator=False, pool_norm=True):
        super().__init__()
        self.kind, self.causal, self.window, self.kernel = kind, causal, window, kernel
        if per_articulator and not n_pool_layers:
            raise ValueError("per_articulator needs layer pooling")
        if per_articulator and kind not in ("linear", "conv"):
            raise ValueError("per_articulator supports the linear and conv heads")
        self.groups = N_ARTICULATORS if per_articulator else 1
        self.per_articulator = per_articulator
        self.pool = LayerPool(n_pool_layers, dim, pool, self.groups, norm=pool_norm) if n_pool_layers else None
        self.norm = nn.LayerNorm(dim) if kind != "linear" else nn.Identity()
        # Per-channel standardization of the linear head's input (set_input_stats), one row per group.
        # Raw SSL features have very uneven channel scales, so without it the ridge weights are tiny and
        # an Adam step of ~lr per weight throws the head off the probe solution at once. With pooling the
        # input distribution moves as the pool weights train, so pooled linear heads re-standardize from
        # running statistics every epoch (track_input_stats / restandardize), preserving the function.
        self.register_buffer("in_mean", torch.zeros(self.groups, dim))
        self.register_buffer("in_std", torch.ones(self.groups, dim))
        self.track_input_stats = kind == "linear" and n_pool_layers > 0
        self._stat_sum = self._stat_sq = None
        self._stat_n = None
        G, per_out = self.groups, n_out // self.groups
        if kind == "linear":
            self.out = GroupLinear(G, dim, per_out) if per_articulator else nn.Linear(dim, n_out)
        elif kind == "conv":
            self.inp = GroupLinear(G, dim, hidden) if per_articulator else nn.Linear(dim, hidden)
            self.conv = nn.Conv1d(hidden, hidden, kernel)
            self.out = GroupLinear(G, hidden, per_out) if per_articulator else nn.Linear(hidden, n_out)
        elif kind == "attn":
            self.inp = nn.Linear(dim, hidden)
            self.block = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout=0.1, batch_first=True,
                                                    norm_first=True)
            self.out = nn.Linear(hidden, n_out)
        else:
            raise ValueError(kind)
        self.smooth = Smooth() if smooth else None

    def set_input_stats(self, mean, std):
        """mean/std: (D,) for every group, or (G, D)."""
        with torch.no_grad():
            self.in_mean.copy_(torch.as_tensor(mean).expand_as(self.in_mean))
            self.in_std.copy_(torch.as_tensor(std).clamp_min(1e-6).expand_as(self.in_std))

    def init_linear(self, W, b):
        """Load a ridge solution (W: D x 12, b: 12) mapping raw features to standardized EMA,
        re-expressed on the standardized input: W' = diag(std) W, b' = b + mean @ W."""
        assert self.kind == "linear" and not self.per_articulator
        W, b = torch.as_tensor(W, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)
        with torch.no_grad():
            self.out.weight.copy_((W * self.in_std[0].cpu()[:, None]).T)
            self.out.bias.copy_(b + self.in_mean[0].cpu() @ W)

    def _accumulate_stats(self, h, pad_mask):
        """Running per-group sums of the (pre-standardization) linear-head input over valid frames."""
        with torch.no_grad():
            hg = h if h.dim() == 4 else h[:, None]  # (B, G, T, D)
            valid = torch.ones(hg.shape[0], hg.shape[2], device=h.device) if pad_mask is None else (~pad_mask).float()
            x = hg.float()
            s = torch.einsum("bgtd,bt->gd", x, valid)
            q = torch.einsum("bgtd,bt->gd", x * x, valid)
            n = valid.sum()
            if self._stat_sum is None:
                self._stat_sum, self._stat_sq, self._stat_n = s, q, n
            else:
                self._stat_sum, self._stat_sq, self._stat_n = self._stat_sum + s, self._stat_sq + q, self._stat_n + n

    @torch.no_grad()
    def restandardize(self):
        """Move the input standardization to the running statistics and re-express the linear map so
        the head computes exactly the same function: W' = W * (s'/s), b' = b + ((m' - m)/s) @ W."""
        if self._stat_sum is None or float(self._stat_n) < 2:
            return
        m_new = self._stat_sum / self._stat_n
        s_new = (self._stat_sq / self._stat_n - m_new**2).clamp_min(0).sqrt().clamp_min(1e-6)
        m_old, s_old = self.in_mean.float(), self.in_std.float()
        if self.per_articulator:  # out.weight (G, D, o), out.bias (G, o)
            W = self.out.weight.float()
            self.out.bias.add_(torch.einsum("gd,gdo->go", (m_new - m_old) / s_old, W).to(self.out.bias.dtype))
            self.out.weight.copy_(W * (s_new / s_old)[:, :, None])
        else:  # nn.Linear: weight (o, D), bias (o,)
            W = self.out.weight.float()
            self.out.bias.add_((W @ ((m_new[0] - m_old[0]) / s_old[0])).to(self.out.bias.dtype))
            self.out.weight.copy_(W * (s_new[0] / s_old[0])[None, :])
        self.in_mean.copy_(m_new)
        self.in_std.copy_(s_new)
        self._stat_sum = self._stat_sq = self._stat_n = None

    def _attn_mask(self, T, device):
        i = torch.arange(T, device=device)
        d = i[None, :] - i[:, None]  # key - query
        allowed = torch.ones(T, T, dtype=torch.bool, device=device)
        if self.causal:
            allowed &= d <= 0
        if self.window:
            allowed &= d.abs() <= self.window
        return ~allowed  # True = masked

    def _conv(self, x):  # (N, T, H) -> (N, T, H)
        x = x.transpose(1, 2)
        pad = (self.kernel - 1, 0) if self.causal else ((self.kernel - 1) // 2, self.kernel // 2)
        return F.gelu(self.conv(F.pad(x, pad))).transpose(1, 2)

    def pooled(self, h, pad_mask=None):
        """Head input before the per-kind layers: (B, T, D), or (B, G, T, D) for per-articulator pooling."""
        if self.pool is not None:
            h = self.pool(h, pad_mask)
            if not self.per_articulator:
                h = h[:, 0]
        return h

    def forward(self, h, pad_mask=None):
        """h: (B, T, D) or list of per-layer (B, T, D) when pooling; pad_mask: (B, T) True on padding."""
        h = self.pooled(h, pad_mask)
        if self.kind == "linear":
            if self.training and self.track_input_stats:
                self._accumulate_stats(h, pad_mask)
            if h.dim() == 4:  # (B, G, T, D), per-group statistics
                h = (h - self.in_mean[None, :, None, :]) / self.in_std[None, :, None, :]
            else:
                h = (h - self.in_mean[0]) / self.in_std[0]
        h = self.norm(h)
        if self.per_articulator:  # h: (B, G, T, D)
            if self.kind == "conv":
                x = F.gelu(self.inp(h))  # (B, G, T, H)
                B, G, T, Hd = x.shape
                x = self._conv(x.reshape(B * G, T, Hd)).reshape(B, G, T, Hd)
                h = x
            y = self.out(h)  # (B, G, T, 2)
            y = y.permute(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], -1)  # articulator pairs in channel order
        else:
            if self.kind == "conv":
                h = self._conv(F.gelu(self.inp(h)))
            elif self.kind == "attn":
                x = self.inp(h)
                h = self.block(x, src_mask=self._attn_mask(x.shape[1], x.device), src_key_padding_mask=pad_mask)
            y = self.out(h)
        return self.smooth(y) if self.smooth is not None else y
