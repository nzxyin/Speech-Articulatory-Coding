"""EMA prediction heads on top of a (subset) SSL encoder.

    linear -- frame-wise D -> 12 (the probe; can be initialized from a ridge solution)
    conv   -- D -> H, temporal Conv1d of `kernel` frames, -> 12; causal (past frames only) or centered
    attn   -- D -> H, one Transformer layer whose attention is limited to a `window` of frames
              (past-only when causal, symmetric otherwise; window=0 means unlimited), -> 12

Optional layer pooling replaces the last hidden state by a softmax-weighted sum of the retained
layers' outputs (SUPERB-style), which adds no encoder compute.

All heads predict standardized EMA. `smooth` applies the FIR equivalent of the zero-phase 10 Hz
Butterworth low-pass that the ridge probe applies to its input features (linear and time-invariant,
so filtering a linear head's output is the same as filtering its input). It looks 1 s ahead,
so offline only. Note that the encoders themselves use bidirectional self-attention: "causal"
here restricts only the head's temporal aggregation, not the encoder.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.inversion import butter_bandpass

FT_SR = 50


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
    def __init__(self, n):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n))

    def forward(self, states):  # list of (B, T, D)
        w = torch.softmax(self.logits, 0)
        return sum(wi * s for wi, s in zip(w, states))


class Head(nn.Module):
    def __init__(self, dim, kind="linear", hidden=256, kernel=9, window=0, causal=False, n_out=12,
                 smooth=True, n_pool_layers=0):
        super().__init__()
        self.kind, self.causal, self.window, self.kernel = kind, causal, window, kernel
        self.pool = LayerPool(n_pool_layers) if n_pool_layers else None
        self.norm = nn.LayerNorm(dim) if kind != "linear" else nn.Identity()
        # Fixed per-channel standardization of the linear head's input (set_input_stats). Raw SSL
        # features have very uneven channel scales, so without it the ridge weights are tiny and an
        # Adam step of ~lr per weight throws the head off the probe solution at once.
        self.register_buffer("in_mean", torch.zeros(dim))
        self.register_buffer("in_std", torch.ones(dim))
        if kind == "linear":
            self.out = nn.Linear(dim, n_out)
        elif kind == "conv":
            self.inp = nn.Linear(dim, hidden)
            self.conv = nn.Conv1d(hidden, hidden, kernel)
            self.out = nn.Linear(hidden, n_out)
        elif kind == "attn":
            self.inp = nn.Linear(dim, hidden)
            self.block = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout=0.1, batch_first=True,
                                                    norm_first=True)
            self.out = nn.Linear(hidden, n_out)
        else:
            raise ValueError(kind)
        self.smooth = Smooth() if smooth else None

    def set_input_stats(self, mean, std):
        with torch.no_grad():
            self.in_mean.copy_(torch.as_tensor(mean))
            self.in_std.copy_(torch.as_tensor(std).clamp_min(1e-6))

    def init_linear(self, W, b):
        """Load a ridge solution (W: D x 12, b: 12) mapping raw features to standardized EMA,
        re-expressed on the standardized input: W' = diag(std) W, b' = b + mean @ W."""
        assert self.kind == "linear"
        W, b = torch.as_tensor(W, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)
        with torch.no_grad():
            self.out.weight.copy_((W * self.in_std.cpu()[:, None]).T)
            self.out.bias.copy_(b + self.in_mean.cpu() @ W)

    def _attn_mask(self, T, device):
        i = torch.arange(T, device=device)
        d = i[None, :] - i[:, None]  # key - query
        allowed = torch.ones(T, T, dtype=torch.bool, device=device)
        if self.causal:
            allowed &= d <= 0
        if self.window:
            allowed &= d.abs() <= self.window
        return ~allowed  # True = masked

    def forward(self, h, pad_mask=None):
        """h: (B, T, D) or list of per-layer (B, T, D) when pooling; pad_mask: (B, T) True on padding."""
        if self.pool is not None:
            h = self.pool(h)
        if self.kind == "linear":
            h = (h - self.in_mean) / self.in_std
        h = self.norm(h)
        if self.kind == "conv":
            x = F.gelu(self.inp(h)).transpose(1, 2)
            pad = (self.kernel - 1, 0) if self.causal else ((self.kernel - 1) // 2, self.kernel // 2)
            h = F.gelu(self.conv(F.pad(x, pad))).transpose(1, 2)
        elif self.kind == "attn":
            x = self.inp(h)
            h = self.block(x, src_mask=self._attn_mask(x.shape[1], x.device), src_key_padding_mask=pad_mask)
        y = self.out(h)
        return self.smooth(y) if self.smooth is not None else y
