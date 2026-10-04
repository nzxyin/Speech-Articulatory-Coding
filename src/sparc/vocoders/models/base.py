"""Common interface for the articulatory vocoders."""

import torch
from torch import nn

from sparc.vocoders.constants import HOP, N_FEATURES, SPEAKER_DIM


class Vocoder(nn.Module):
    """Maps raw features ``[B, 15, T]`` and a speaker embedding ``[B, 64]`` to a waveform ``[B, 1, 480 T]``.

    Subclasses build a :class:`~sparc.vocoders.models.frontend.FeatureFrontend` from the training statistics and
    must never modify ``features`` in place.
    """

    hop: int = HOP
    n_features: int = N_FEATURES
    spk_dim: int = SPEAKER_DIM

    def forward(self, features: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def check_io(self, features: torch.Tensor, spk: torch.Tensor, wav: torch.Tensor) -> None:
        """Raises if the shapes break the interface contract."""
        batch, channels, frames = features.shape
        if channels != self.n_features:
            raise ValueError(f"expected {self.n_features} feature channels, got {channels}")
        if spk.shape != (batch, self.spk_dim):
            raise ValueError(f"expected speaker embedding of shape {(batch, self.spk_dim)}, got {tuple(spk.shape)}")
        if wav.shape != (batch, 1, frames * self.hop):
            raise ValueError(f"expected output of shape {(batch, 1, frames * self.hop)}, got {tuple(wav.shape)}")
