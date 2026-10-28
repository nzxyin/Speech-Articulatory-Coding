"""Variants of the out-of-domain encoder screen (docs/spkemb_ood_screen.md); hyperparameters fixed there before any run.

All read WavLM layer 6, keep the 64-d output and the shipped FFN as the final map (``SpeakerFFN`` with 1024 hidden units,
dropout 0.2, input z-score fitted in ``setup``).

- ``ood_gruconv`` (1): frame head on the z-scored layer-6 frames, then the baseline periodicity-weighted mean. Head:
  LayerNorm, Linear(1024, 256), GELU, dropout 0.1, bidirectional GRU (128 per direction), conv1d (256 to 1024, kernel 9,
  centered, zero-initialised) added residually to the z-scored frames, so training starts from the baseline's pooled
  vector. GRU sequences are packed, so padding never reaches the backward direction.
- ``ood_gruconv_asp`` (2): same head, then ECAPA-style channel-wise attentive statistics (bottleneck 128; mean and std,
  2048-d) instead of the periodicity-weighted mean.
- ``ood_framemlp``: control for (1) and (2) without temporal context: the same front end, then Linear(256, 256), GELU
  and Linear(256, 1024) per frame (residual, zero-initialised), then the periodicity-weighted mean.
- ``ood_sv`` (3): frozen WeSpeaker ResNet34 embedding of the same crop alone (256-d) into the FFN.
- ``ood_sv_l6`` (4): the layer-6 periodicity-weighted mean (1024-d) concatenated with that embedding (1280-d).
- ``ood_width_ctl`` (7): width control for (4): the layer-6 mean concatenated with its own first 256 dimensions
  (1280-d, no new information; same input width and first-layer weight count).
"""

import torch
from torch import nn

from sparc.vocoders.models.speaker import SpeakerFFN
from sparc.vocoders.spkscreen.data import FrameBatch, FrameSet
from sparc.vocoders.spkscreen.pooling import PoolingModule, pool_weights, register, weighted_mean
from sparc.vocoders.spkscreen.variants.pooling_ops import DIM, StatsPool, masked_softmax

HID = 256


class _FrontEnd(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.proj = nn.Linear(DIM, HID)
        self.drop = nn.Dropout(dropout)


class GRUConvFrames(_FrontEnd):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(HID, HID // 2, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(HID, DIM, 9, padding=4)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.drop(torch.nn.functional.gelu(self.proj(self.norm(z)))) * mask[..., None]
        lens = mask.sum(1).clamp(min=1).cpu()
        pk = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(pk)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=z.shape[1])
        d = self.conv((out * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        return (z + d) * mask[..., None]


class MLPFrames(_FrontEnd):
    def __init__(self):
        super().__init__()
        self.mid = nn.Linear(HID, HID)
        self.out = nn.Linear(HID, DIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.drop(torch.nn.functional.gelu(self.proj(self.norm(z))))
        d = self.out(torch.nn.functional.gelu(self.mid(h)))
        return (z + d) * mask[..., None]


class _FrameHeadPool(StatsPool):
    """``StatsPool`` whose frames pass through a frame head (``self.head``) before pooling."""

    def __init__(self, in_dim: int, head: nn.Module):
        super().__init__(in_dim)
        self.head = head

    def head_frames(self, b: FrameBatch) -> torch.Tensor:
        return self.head(self.znorm(b), b.mask)


@register("ood_gruconv")
class GRUConvMean(_FrameHeadPool):
    def __init__(self):
        super().__init__(DIM, GRUConvFrames())

    def stats(self, b):
        return weighted_mean(self.head_frames(b), pool_weights(b.periodicity, b.mask))


@register("ood_framemlp")
class FrameMLPMean(_FrameHeadPool):
    def __init__(self):
        super().__init__(DIM, MLPFrames())

    def stats(self, b):
        return weighted_mean(self.head_frames(b), pool_weights(b.periodicity, b.mask))


@register("ood_gruconv_asp")
class GRUConvASP(_FrameHeadPool):
    def __init__(self, bottleneck: int = 128):
        super().__init__(2 * DIM, GRUConvFrames())
        self.att = nn.Sequential(nn.Linear(DIM, bottleneck), nn.Tanh(), nn.Linear(bottleneck, DIM))

    def stats(self, b):
        x = self.head_frames(b)
        a = masked_softmax(self.att(x), b.mask)
        mu = (a * x).sum(1)
        var = (a * x * x).sum(1) - mu**2
        return torch.cat([mu, var.clamp_min(1e-5).sqrt()], 1)


class _VecFFN(PoolingModule):
    """A fixed pooled vector (``vec``), per-dimension z-score from training utterances, shipped-size FFN."""

    layers = (6,)

    def __init__(self, in_dim: int):
        super().__init__()
        self.ffn = SpeakerFFN(in_dim=in_dim, hidden_dim=DIM)

    def l6(self, b: FrameBatch) -> torch.Tensor:
        return weighted_mean(b.hidden[:, 0], pool_weights(b.periodicity, b.mask))

    def vec(self, b: FrameBatch) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    @torch.no_grad()
    def setup(self, train: FrameSet) -> None:
        self.to(train.device)
        was = self.training
        self.eval()
        v = torch.cat([self.vec(train.full(range(i, min(i + 128, train.n)), self.layers)).float().cpu()
                       for i in range(0, train.n, 128)])
        self.ffn.mean.copy_(v.mean(0))
        self.ffn.std.copy_(v.std(0).clamp_min(1e-6))
        self.train(was)

    def forward(self, b: FrameBatch) -> torch.Tensor:
        return self.ffn(self.vec(b))


class _SVBase(_VecFFN):
    def __init__(self, in_dim: int):
        super().__init__(in_dim)
        from sparc.vocoders.spkscreen.ood_sv import SVEmbedder

        self.sv = SVEmbedder()


@register("ood_sv")
class SVOnly(_SVBase):
    def __init__(self):
        super().__init__(256)

    def vec(self, b):
        return self.sv(b)


@register("ood_sv_l6")
class SVPlusL6(_SVBase):
    def __init__(self):
        super().__init__(1024 + 256)

    def vec(self, b):
        return torch.cat([self.l6(b), self.sv(b)], 1)


@register("ood_width_ctl")
class WidthControl(_VecFFN):
    def __init__(self):
        super().__init__(1024 + 256)

    def vec(self, b):
        m = self.l6(b)
        return torch.cat([m, m[:, :256]], 1)
