"""Pluggable pooling-module interface of the speaker-encoder screen and the variant registry.

A variant is a ``PoolingModule``: padded frame features plus periodicity, loudness and mask in, a 64-d embedding out.
It declares which cached WavLM layers it reads (``layers``) and whether it needs the log-mel (``needs_mel``). Register
it with ``@register("name")`` in a file under ``spkscreen/variants/`` (imported automatically); the trainer builds it
with ``build(name, **kwargs)`` and calls ``setup(train_set)`` once, where it may compute training-set statistics.
"""

import importlib
import pkgutil
from collections.abc import Callable

import torch
from torch import nn

from sparc.vocoders.constants import SPEAKER_DIM
from sparc.vocoders.spkscreen.data import FrameBatch, FrameSet

REGISTRY: dict[str, Callable[..., "PoolingModule"]] = {}


class PoolingModule(nn.Module):
    """Base class: ``forward(batch) -> [B, 64]``."""

    layers: tuple[int, ...] = (6,)
    needs_mel: bool = False
    out_dim: int = SPEAKER_DIM

    def setup(self, train: FrameSet) -> None:
        """Called once before training with the training frames (statistics for normalization, codebooks)."""

    def forward(self, batch: FrameBatch) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def register(name: str):
    def deco(fn):
        if name in REGISTRY:
            raise ValueError(f"variant {name!r} is registered twice")
        REGISTRY[name] = fn
        return fn

    return deco


def _load_variants() -> None:
    from sparc.vocoders.spkscreen import variants

    for m in pkgutil.iter_modules(variants.__path__):
        importlib.import_module(f"{variants.__name__}.{m.name}")


def build(name: str, **kwargs) -> PoolingModule:
    _load_variants()
    if name not in REGISTRY:
        raise KeyError(f"unknown variant {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean over time of ``x [B, T, D]`` with ``w [B, T]`` (use :func:`pool_weights` for the fallback)."""
    return (x * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)


def pool_weights(periodicity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """SPARC's pooling weights: the cached periodicity on real frames; uniform over real frames where it sums to 0."""
    w = periodicity * mask
    empty = w.sum(1, keepdim=True) <= 0
    return torch.where(empty, mask.to(w.dtype), w)
