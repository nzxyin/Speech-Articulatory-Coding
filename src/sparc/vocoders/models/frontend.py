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

PITCH_MODES = ("log", "sparc_linear", "linear500", "log_relative")
# Phase 4 ``log_relative``: the F0 channel holds the ratio f0 / (median voiced f0 of the utterance), so the input is
# ln(ratio) / REL_LOGF0_STD. The constant is the pooled std of ln(f0 / utterance median) over all frames of 1500
# LibriTTS-R train.clean.100 utterances (0.178 in one sample, 0.195 in a review sample of 300; voiced fraction 0.56),
# rounded to 0.19; the mean (~0.00-0.01) is left alone.
REL_LOGF0_STD = 0.19
# Binary voicing gate (``binary_periodicity=True``): the periodicity channel becomes 1[periodicity > VOICING_THRESHOLD].
# The cached periodicity is CREPE periodicity zeroed below 0.4 (and where the loudness is below 0.05), i.e. 0 in unvoiced
# frames, so the extractor's own voiced / unvoiced decision is ``periodicity > 0`` (the cache holds no value in (0, 0.4)
# except 3e-7 of the frames, a few utterances with fewer than 5 sub-threshold frames). On 300 LibriTTS-R train.clean.100
# utterances (2-12 s, 200 speakers) the frame agreement with pYIN is 0.827 for tau = 0 (= the extractor's 0.4 cut, voiced
# fraction 0.550 over the whole split) and falls monotonically for every higher cut (0.804 at 0.5, 0.771 at 0.6, 0.725
# at 0.7, 0.495 at 0.9), so no stricter threshold is supported; lower cuts cannot be tested on the cache and would
# coincide with 0 anyway. tau = 0 is also the decision every other piece of code uses (``voiced_flag``, the voiced F0
# statistics, the speaker-pooling weights).
VOICING_THRESHOLD = 0.0


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
    ``(f0 - 50) * 0.01`` (``"sparc_linear"``), ``f0 / 500`` (``"linear500"``) or, for a speaker-relative F0 channel
    (a ratio to the utterance's median voiced F0, written by the training module), ``ln(ratio) / REL_LOGF0_STD``
    (``"log_relative"``; ``FrontendOut.f0_hz`` then holds the ratio, not Hz); loudness is
    ``(ln(loudness + 1e-4) - mean) / std`` with global constants, so the gain stays visible; periodicity is
    z-scored. With ``voiced_flag=True`` a 16th channel ``1[periodicity > 0]`` is appended. The input tensor is never
    modified. Statistics are buffers, so they are saved with the model.

    ``drop_periodicity=True`` leaves the periodicity channel out of the network input (the original SPARC vocoder input
    set: 12 EMA, pitch, loudness, in that order, so ``out_channels`` is 14); the raw periodicity is still returned in
    ``FrontendOut.periodicity`` for models that use it outside the input. It cannot be combined with ``voiced_flag``,
    which is itself derived from periodicity.

    ``binary_periodicity=True`` replaces the continuous periodicity channel by the raw 0 / 1 voicing flag
    ``1[periodicity > VOICING_THRESHOLD]`` in the same position (15 channels, same order). The flag is fed unstandardized:
    its std over the training frames is 0.50, the same order as the unit-variance z-scored channels, and z-scoring it with
    the continuous channel's statistics (mean 0.437, std 0.414) would not describe it. The raw periodicity is still
    returned in ``FrontendOut.periodicity``. It cannot be combined with ``drop_periodicity`` or ``voiced_flag``.
    """

    def __init__(
        self,
        stats: dict | str | Path,
        pitch_mode: str = "log",
        voiced_flag: bool = False,
        drop_periodicity: bool = False,
        binary_periodicity: bool = False,
    ):
        super().__init__()
        if pitch_mode not in PITCH_MODES:
            raise ValueError(f"pitch_mode must be one of {PITCH_MODES}, got {pitch_mode!r}")
        if drop_periodicity and voiced_flag:
            raise ValueError("voiced_flag is derived from periodicity and cannot be combined with drop_periodicity")
        if binary_periodicity and (drop_periodicity or voiced_flag):
            raise ValueError("binary_periodicity cannot be combined with drop_periodicity or voiced_flag")
        stats = load_stats(stats)
        self.binary_periodicity = binary_periodicity
        self.pitch_mode = pitch_mode
        self.voiced_flag = voiced_flag
        self.drop_periodicity = drop_periodicity
        self.out_channels = N_FEATURES - int(drop_periodicity) + int(voiced_flag)

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
        elif self.pitch_mode == "log_relative":
            f0_n = torch.log(f0.clamp_min(1e-2)) / REL_LOGF0_STD
        elif self.pitch_mode == "sparc_linear":
            f0_n = (f0 - 50.0) * 0.01
        else:
            f0_n = f0 / 500.0
        loud_n = (torch.log(loud.clamp_min(0.0) + LOUDNESS_EPS) - self.loud_log_mean) / self.loud_log_std
        if self.binary_periodicity:
            per_n = (per > VOICING_THRESHOLD).to(features.dtype)
        else:
            per_n = (per - self.per_mean) / self.per_std

        parts = [ema_n, f0_n, loud_n] if self.drop_periodicity else [ema_n, f0_n, loud_n, per_n]
        if self.voiced_flag:
            parts.append((per > 0).to(features.dtype))
        return FrontendOut(x=torch.cat(parts, dim=1), f0_hz=f0, periodicity=per, loudness=loud)
