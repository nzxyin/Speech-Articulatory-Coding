"""Feature-wise linear modulation from the speaker embedding, shared by all three vocoders."""

import torch
from torch import nn


class FiLM(nn.Module):
    """Per-channel scale and shift predicted from a conditioning vector: ``x * (1 + gamma) + beta``.

    The projection is zero-initialized, so the layer is the identity at initialization. Call
    :meth:`reset_parameters` again after any global weight initialization that touches ``nn.Linear`` modules.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.channels = channels
        self.proj = nn.Linear(cond_dim, 2 * channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        # The projection runs in float32 even under autocast: in bf16, (1 + gamma) would be quantized to steps of
        # about 0.004, which would round away small speaker modulations.
        dtype = torch.promote_types(self.proj.weight.dtype, torch.float32)
        with torch.autocast(device_type=cond.device.type, enabled=False):
            gamma, beta = nn.functional.linear(
                cond.to(dtype), self.proj.weight.to(dtype), self.proj.bias.to(dtype)
            ).chunk(2, dim=-1)  # [B, C] each
        shape = [x.shape[0]] + [1] * (x.dim() - 1)
        shape[channel_dim] = self.channels
        return x * (1 + gamma.reshape(shape)).to(x.dtype) + beta.reshape(shape).to(x.dtype)


class FiLMLayerNorm(nn.Module):
    """LayerNorm over the last dimension without affine parameters, followed by :class:`FiLM`.

    Generalizes Vocos' ``AdaLayerNorm`` (a lookup table of scale and shift) to a continuous speaker embedding.
    Expects channel-last input ``[B, T, C]``.
    """

    def __init__(self, cond_dim: int, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.film = FiLM(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = nn.functional.layer_norm(x, (self.channels,), eps=self.eps)
        return self.film(x, cond, channel_dim=-1)


COND_TYPES = ("film", "adaln_zero", "film_gate")


class AdaLNZero(nn.Module):
    """Adaptive layer norm with a zero-initialized gate (the modulation of DiT, Peebles and Xie 2023), for conv features.

    Returns ``(h, gate)`` for channel-first ``x [B, C, T]``: ``h = LN(x) * (1 + scale) + shift`` and ``gate [B, C, 1]``,
    where ``scale, shift, gate`` are linear in the conditioning vector and the projection is zero-initialized (so at
    initialization ``h = LN(x)`` and ``gate = 0``). The caller applies it as ``x + gate * f(h)``, which makes the
    residual unit the identity at initialization whatever the conditioning is. ``LN`` normalizes over the channels of each
    frame (no affine parameters), not over time, so a frame's output does not depend on the utterance length (training
    crops are 64 frames, inference runs on whole utterances). The projection runs in float32 under autocast, as in
    :class:`FiLM`.
    """

    def __init__(self, cond_dim: int, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.proj = nn.Linear(cond_dim, 3 * channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.promote_types(self.proj.weight.dtype, torch.float32)
        with torch.autocast(device_type=cond.device.type, enabled=False):
            shift, scale, gate = nn.functional.linear(
                cond.to(dtype), self.proj.weight.to(dtype), self.proj.bias.to(dtype)
            ).chunk(3, dim=-1)  # [B, C] each
        normed = nn.functional.layer_norm(x.transpose(1, 2), (self.channels,), eps=self.eps).transpose(1, 2)
        h = normed * (1 + scale[:, :, None]).to(x.dtype) + shift[:, :, None].to(x.dtype)
        return h, gate[:, :, None].to(x.dtype)


class AdaLNZeroSite(nn.Module):
    """Stage-level modulation ``x + gate * (LN(x) * (1 + scale) + shift)``; the identity at initialization."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.ada = AdaLNZero(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h, gate = self.ada(x, cond)
        return x + gate * h


class GatedFiLM(nn.Module):
    """FiLM plus a zero-initialized per-channel gate (Phase 8 conditioning screen, arm ``film_gate``).

    Returns ``(film(x), gate [B, C, 1])``; the caller applies the residual update ``x + gate * film(f(x))``. The
    modulation position (the output of the unit's convolutions) and the absence of a normalization layer are those of
    :class:`FiLM`; only the gate is new, so a unit starts as the identity (like AdaLN-zero), which the plain FiLM unit
    does not (its residual branch is on at initialization). Projection in float32 under autocast, as in :class:`FiLM`.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.channels = channels
        self.proj = nn.Linear(cond_dim, 3 * channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.promote_types(self.proj.weight.dtype, torch.float32)
        with torch.autocast(device_type=cond.device.type, enabled=False):
            gamma, beta, gate = nn.functional.linear(
                cond.to(dtype), self.proj.weight.to(dtype), self.proj.bias.to(dtype)
            ).chunk(3, dim=-1)
        h = x * (1 + gamma[:, :, None]).to(x.dtype) + beta[:, :, None].to(x.dtype)
        return h, gate[:, :, None].to(x.dtype)
