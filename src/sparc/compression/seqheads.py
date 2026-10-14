"""Sequence heads with inductive biases a transformer encoder does not already provide.

All share the conv head's front end (LayerNorm -> Linear D->H -> GELU) and its optional ±1 s output smoother
(heads.Smooth), so only the temporal module differs:
    gru    -- bidirectional GRU (H/2 units per direction): a nonlinear dynamical system with continuous state;
    ssm    -- bidirectional diagonal linear state-space layer (S4D-Lin initialization, N states per channel,
              applied as a long convolution): smooth linear dynamics, the classical model of articulator motion;
    mamba  -- bidirectional selective state-space block (Mamba; input-dependent step size, B and C), reference
              PyTorch scan (sequences are ~250 frames, so the sequential scan is cheap).
Each predicts standardized EMA from (B, T, D) encoder features; pad_mask (B, T) is True on padding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import Smooth


class _Front(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.inp = nn.Linear(dim, hidden)

    def forward(self, h):
        return F.gelu(self.inp(self.norm(h)))


class GRUHead(nn.Module):
    def __init__(self, dim, hidden=256, n_out=12, smooth=True):
        super().__init__()
        self.front = _Front(dim, hidden)
        self.rnn = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden, n_out)
        self.smooth = Smooth() if smooth else None

    def forward(self, h, pad_mask=None):
        x = self.front(h)
        if pad_mask is not None:
            lens = (~pad_mask).sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
            y, _ = self.rnn(packed)
            y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
        else:
            y, _ = self.rnn(x)
        y = self.out(y)
        return self.smooth(y, pad_mask) if self.smooth is not None else y


class S4DLayer(nn.Module):
    """Diagonal linear SSM per channel (S4D-Lin init), one direction; y = conv(x, K) + D x."""

    def __init__(self, hidden, n_state=16, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        H, N = hidden, n_state // 2  # complex conjugate pairs
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(N).float().repeat(H, 1))
        self.C = nn.Parameter(torch.randn(H, N, 2) * 0.5 ** 0.5)
        self.D = nn.Parameter(torch.randn(H))

    def kernel(self, T):
        dt = torch.exp(self.log_dt)[:, None]  # (H, 1)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N)
        C = torch.view_as_complex(self.C)
        dtA = A * dt
        Cb = C * (torch.exp(dtA) - 1.0) / A  # zero-order hold
        t = torch.arange(T, device=A.device)
        K = 2 * torch.einsum("hn,hnt->ht", Cb, torch.exp(dtA[:, :, None] * t)).real  # (H, T)
        return K

    def forward(self, x):  # (B, T, H) -> (B, T, H)
        B, T, H = x.shape
        K = self.kernel(T)
        n = 2 * T
        y = torch.fft.irfft(torch.fft.rfft(x.transpose(1, 2), n=n) * torch.fft.rfft(K, n=n), n=n)[..., :T]
        return y.transpose(1, 2) + x * self.D


class SSMHead(nn.Module):
    def __init__(self, dim, hidden=256, n_out=12, smooth=True, n_state=16):
        super().__init__()
        self.front = _Front(dim, hidden)
        self.fwd = S4DLayer(hidden, n_state)
        self.bwd = S4DLayer(hidden, n_state)
        self.mix = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, n_out)
        self.smooth = Smooth() if smooth else None

    def forward(self, h, pad_mask=None):
        x = self.front(h)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask[..., None], 0.0)
        f = self.fwd(x)
        b = self._flip(self.bwd(self._flip(x, pad_mask)), pad_mask)
        y = self.out(F.gelu(self.mix(torch.cat([f, b], -1))))
        return self.smooth(y, pad_mask) if self.smooth is not None else y

    @staticmethod
    def _flip(x, pad_mask):
        """Reverse each sequence over its valid frames (padding stays at the end)."""
        if pad_mask is None:
            return x.flip(1)
        out = torch.zeros_like(x)
        lens = (~pad_mask).sum(1)
        for i, n in enumerate(lens.tolist()):
            out[i, :n] = x[i, :n].flip(0)
        return out


def seq_scan(a, b):
    """h_t = a_t * h_{t-1} + b_t over dim 1 (h_0 = 0), sequential reference."""
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, 1)


def pscan(a, b):
    """Same recurrence as seq_scan with a Hillis-Steele associative scan: ceil(log2 T) vectorized steps.
    Composition of (a1, b1) then (a2, b2) is (a2 a1, a2 b1 + b2); out-of-range positions use the identity (1, 0).
    Stable because every a is a decay in (0, 1]."""
    T, step = a.shape[1], 1
    while step < T:
        a_prev = torch.cat([torch.ones_like(a[:, :step]), a[:, :-step]], 1)
        b_prev = torch.cat([torch.zeros_like(b[:, :step]), b[:, :-step]], 1)
        b = a * b_prev + b
        a = a * a_prev
        step *= 2
    return b


class MambaBlock(nn.Module):
    """One-direction selective SSM block (Mamba); parallel associative scan (sequential reference via parallel=False)."""

    def __init__(self, hidden, n_state=16, expand=2, d_conv=4, parallel=True):
        super().__init__()
        self.parallel = parallel
        E = expand * hidden
        self.E, self.N = E, n_state
        self.dt_rank = math.ceil(hidden / 16)
        self.in_proj = nn.Linear(hidden, 2 * E)
        self.conv = nn.Conv1d(E, E, d_conv, groups=E, padding=d_conv - 1)
        self.x_proj = nn.Linear(E, self.dt_rank + 2 * n_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, E)
        A = torch.arange(1, n_state + 1).float().repeat(E, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, hidden)
        with torch.no_grad():  # dt in [1e-3, 1e-1] at init, as in Mamba
            dt = torch.exp(torch.rand(E) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):  # (B, T, H)
        B, T, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :T].transpose(1, 2))
        dbc = self.x_proj(u)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.N, self.N], -1)
        dt = F.softplus(self.dt_proj(dt))  # (B, T, E)
        A = -torch.exp(self.A_log)  # (E, N)
        dA = torch.exp(dt[..., None] * A)  # (B, T, E, N)
        dBu = dt[..., None] * Bm[:, :, None, :] * u[..., None]  # (B, T, E, N)
        hs = pscan(dA, dBu) if self.parallel else seq_scan(dA, dBu)  # (B, T, E, N)
        y = torch.einsum("bten,btn->bte", hs, Cm) + u * self.D
        return self.out_proj(y * F.silu(z))


class MambaHead(nn.Module):
    def __init__(self, dim, hidden=256, n_out=12, smooth=True, n_state=16):
        super().__init__()
        self.front = _Front(dim, hidden)
        self.fwd = MambaBlock(hidden, n_state)
        self.bwd = MambaBlock(hidden, n_state)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, n_out)
        self.smooth = Smooth() if smooth else None

    def forward(self, h, pad_mask=None):
        x = self.front(h)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask[..., None], 0.0)
        y = x + self.fwd(x) + SSMHead._flip(self.bwd(SSMHead._flip(x, pad_mask)), pad_mask)
        y = self.out(self.norm(y))
        return self.smooth(y, pad_mask) if self.smooth is not None else y
