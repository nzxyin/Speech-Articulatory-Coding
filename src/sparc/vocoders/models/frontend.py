"""Normalization of the raw cached features, shared by all three vocoders."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from sparc.vocoders.constants import (
    F0_CHANNEL,
    LOUDNESS_CHANNEL,
    LOUDNESS_EPS,
    N_EMA,
    N_FEATURES,
    PERIODICITY_CHANNEL,
)

PITCH_MODES = ("log", "sparc_linear", "linear500")


@dataclass
class FrontendOut:
    x: torch.Tensor  # [B, C_in, T] normalized network input
    f0_hz: torch.Tensor  # [B, 1, T] raw F0 in Hz
    periodicity: torch.Tensor  # [B, 1, T] raw periodicity
    loudness: torch.Tensor  # [B, 1, T] raw (gained) linear loudness


def load_stats(stats: dict | str | Path) -> dict:
    """Returns the statistics dictionary, reading it from a JSON file if a path is given."""
    if isinstance(stats, (str, Path)):
        with open(stats) as f:
            return json.load(f)
    return dict(stats)


class FeatureFrontend(nn.Module):
    """Turns raw features ``[B, 15, T]`` into the normalized network input.

    EMA channels are z-scored per channel; pitch is ``(ln f0 - mean) / std`` (``pitch_mode="log"``), SPARC's
    ``(f0 - 50) * 0.01`` (``"sparc_linear"``) or ``f0 / 500`` (``"linear500"``); loudness is
    ``(ln(loudness + 1e-4) - mean) / std`` with global constants, so the gain stays visible; periodicity is
    z-scored. With ``voiced_flag=True`` a 16th channel ``1[periodicity > 0]`` is appended. The input tensor is never
    modified. Statistics are buffers, so they are saved with the model.
    """

    def __init__(self, stats: dict | str | Path, pitch_mode: str = "log", voiced_flag: bool = False):
        super().__init__()
        if pitch_mode not in PITCH_MODES:
            raise ValueError(f"pitch_mode must be one of {PITCH_MODES}, got {pitch_mode!r}")
        stats = load_stats(stats)
        self.pitch_mode = pitch_mode
        self.voiced_flag = voiced_flag
        self.out_channels = N_FEATURES + int(voiced_flag)

        def buf(name, value, shape):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).reshape(shape))

        buf("ema_mean", stats["ema_mean"], (1, N_EMA, 1))
        buf("ema_std", torch.as_tensor(stats["ema_std"], dtype=torch.float32).clamp_min(1e-6), (1, N_EMA, 1))
        buf("logf0_mean", stats["logf0_mean"], ())
        buf("logf0_std", max(float(stats["logf0_std"]), 1e-6), ())
        buf("loud_log_mean", stats["loud_log_mean"], ())
        buf("loud_log_std", max(float(stats["loud_log_std"]), 1e-6), ())
        buf("per_mean", stats["per_mean"], ())
        buf("per_std", max(float(stats["per_std"]), 1e-6), ())

    def forward(self, features: torch.Tensor) -> FrontendOut:
        if features.dim() != 3 or features.shape[1] != N_FEATURES:
            raise ValueError(f"expected features of shape [B, {N_FEATURES}, T], got {tuple(features.shape)}")
        ema = features[:, :N_EMA]
        f0 = features[:, F0_CHANNEL : F0_CHANNEL + 1]
        loud = features[:, LOUDNESS_CHANNEL : LOUDNESS_CHANNEL + 1]
        per = features[:, PERIODICITY_CHANNEL : PERIODICITY_CHANNEL + 1]

        ema_n = (ema - self.ema_mean) / self.ema_std
        if self.pitch_mode == "log":
            f0_n = (torch.log(f0.clamp_min(1.0)) - self.logf0_mean) / self.logf0_std
        elif self.pitch_mode == "sparc_linear":
            f0_n = (f0 - 50.0) * 0.01
        else:
            f0_n = f0 / 500.0
        loud_n = (torch.log(loud.clamp_min(0.0) + LOUDNESS_EPS) - self.loud_log_mean) / self.loud_log_std
        per_n = (per - self.per_mean) / self.per_std

        parts = [ema_n, f0_n, loud_n, per_n]
        if self.voiced_flag:
            parts.append((per > 0).to(features.dtype))
        return FrontendOut(x=torch.cat(parts, dim=1), f0_hz=f0, periodicity=per, loudness=loud)
