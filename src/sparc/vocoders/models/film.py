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
