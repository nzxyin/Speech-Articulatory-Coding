"""Baseline: WavLM layer 6, periodicity-weighted mean, z-score, shipped FFN (Linear-GELU-dropout-Linear to 64-d)."""

import torch

from sparc.vocoders.models.speaker import SpeakerFFN
from sparc.vocoders.spkscreen.data import FrameBatch, FrameSet
from sparc.vocoders.spkscreen.pooling import PoolingModule, pool_weights, register, weighted_mean


@register("baseline")
class PeriodicityMeanFFN(PoolingModule):
    layers = (6,)

    def __init__(self, layer: int = 6, dropout: float = 0.2):
        super().__init__()
        self.layers = (layer,)
        self.ffn = SpeakerFFN(dropout=dropout)

    def pooled(self, batch: FrameBatch) -> torch.Tensor:
        return weighted_mean(batch.hidden[:, 0], pool_weights(batch.periodicity, batch.mask))

    @torch.no_grad()
    def setup(self, train: FrameSet) -> None:
        """Per-dimension statistics of the pooled training utterances (as the shipped FFN's input z-score)."""
        vecs = []
        for i in range(0, train.n, 256):
            vecs.append(self.pooled(train.full(range(i, min(i + 256, train.n)), self.layers)).cpu())
        v = torch.cat(vecs)
        self.ffn.mean.copy_(v.mean(0))
        self.ffn.std.copy_(v.std(0).clamp_min(1e-6))

    def forward(self, batch: FrameBatch) -> torch.Tensor:
        return self.ffn(self.pooled(batch))
