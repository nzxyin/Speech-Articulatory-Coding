"""Reflect padding and centred STFT built from slices, flips and concatenation.

``F.pad(..., mode="reflect")`` (and therefore ``torch.stft(center=True)``) has no deterministic CUDA backward, so a
run with ``trainer.deterministic: true`` fails as soon as a gradient reaches the generated waveform. These helpers
give the same values with a backward that is deterministic.
"""

import torch


def reflect_pad_1d(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Reflect-pads the last axis of ``x``; equal to ``F.pad(x, (left, right), "reflect")``."""
    length = x.shape[-1]
    if left >= length or right >= length:
        raise ValueError(f"reflect padding ({left}, {right}) needs more than {max(left, right)} samples, got {length}")
    parts = []
    if left:
        parts.append(x[..., 1 : left + 1].flip(-1))
    parts.append(x)
    if right:
        parts.append(x[..., length - right - 1 : length - 1].flip(-1))
    return torch.cat(parts, dim=-1) if len(parts) > 1 else x


def stft(
    x: torch.Tensor, n_fft: int, hop_length: int, win_length: int, window: torch.Tensor, center: bool = True
) -> torch.Tensor:
    """Complex STFT ``[B, n_fft // 2 + 1, frames]`` of ``x[B, L]`` with reflect centring."""
    if center:
        x = reflect_pad_1d(x, n_fft // 2, n_fft // 2)
    return torch.stft(
        x, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False, return_complex=True
    )
