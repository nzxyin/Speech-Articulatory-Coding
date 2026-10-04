"""Speaker encoder FFN trained jointly with each vocoder."""

import torch
from torch import nn

from sparc.vocoders.constants import SPEAKER_DIM, SPEAKER_RAW_DIM


class SpeakerFFN(nn.Module):
    """Maps a pooled 1024-d WavLM vector to a 64-d speaker embedding.

    Same layers as ``sparc.spk_encoder.SpeakerEncodingLayer``, preceded by a fixed per-dimension z-score whose
    statistics come from the training cache and are stored as buffers.
    """

    def __init__(
        self,
        in_dim: int = SPEAKER_RAW_DIM,
        hidden_dim: int = SPEAKER_RAW_DIM,
        out_dim: int = SPEAKER_DIM,
        dropout: float = 0.2,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ):
        super().__init__()
        mean = torch.zeros(in_dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        std = torch.ones(in_dim) if std is None else torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean", mean.reshape(in_dim).clone())
        self.register_buffer("std", std.reshape(in_dim).clamp_min(1e-6).clone())
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, spk_raw: torch.Tensor) -> torch.Tensor:
        return self.net((spk_raw - self.mean) / self.std)
