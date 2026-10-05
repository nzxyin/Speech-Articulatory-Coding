"""Discriminators for the 24 kHz vocoder comparison: multi-period, multi-resolution and multi-scale.

``MultiPeriodDiscriminator`` and ``MultiResolutionDiscriminator`` are ported from Vocos
(https://github.com/gemelo-ai/vocos, commit eb39abfc42c1dee4854d9b10d44dd7d4fd3b0e53, MIT License,
Copyright (c) 2023 Charactr Inc.), which in turn adapt HiFi-GAN (Kong et al., 2020) and the Descript audio codec.
Changes: ``einops.rearrange`` replaced by ``permute``; inputs are ``[B, 1, L]``; the optional conditioning embedding
is removed; ``torch.nn.utils.parametrizations.weight_norm`` replaces the deprecated ``weight_norm``.

``MultiScaleDiscriminator`` follows HiFi-GAN and the fork's ``sparc.training.discriminators`` with the pooling bug
fixed: the fork gave the pooling layers strides (2, 4) and applied them cumulatively, so the third scale was 1/8
instead of 1/4 (fork issue #9). Here the stride of each pooling layer is the ratio of consecutive scales.

Every discriminator set maps ``(y[B, 1, L], y_hat[B, 1, L])`` to ``(real_logits, fake_logits, real_fmaps,
fake_fmaps)``: two lists with one tensor per sub-discriminator and two lists of per-sub-discriminator lists of
feature maps.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from sparc.vocoders.losses.ops import reflect_pad_1d, stft

DiscOutput = tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]

VOCOS_BAND_EDGES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _pair(disc: nn.Module, y: torch.Tensor, y_hat: torch.Tensor):
    """Runs one sub-discriminator on the real and generated waveform, which must have the same shape."""
    if y.shape != y_hat.shape:
        raise ValueError(f"real {tuple(y.shape)} and generated {tuple(y_hat.shape)} waveforms differ in shape")
    return disc(y), disc(y_hat)


class PeriodDiscriminator(nn.Module):
    """One period of the multi-period discriminator; the first convolution's feature map is not returned."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, lrelu_slope: float = 0.1):
        super().__init__()
        self.period = period
        self.lrelu_slope = lrelu_slope
        pad = (kernel_size // 2, 0)
        widths = (1, 32, 128, 512, 1024, 1024)
        strides = (stride, stride, stride, stride, 1)
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(widths[i], widths[i + 1], (kernel_size, 1), (strides[i], 1), padding=pad))
            for i in range(5)
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            x = reflect_pad_1d(x, 0, self.period - t % self.period)
            t = x.shape[-1]
        x = x.reshape(b, c, t // self.period, self.period)
        for i, conv in enumerate(self.convs):
            x = F.leaky_relu(conv(x), self.lrelu_slope)
            if i > 0:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator (periods 2, 3, 5, 7, 11 by default), 41,105,770 parameters."""

    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(PeriodDiscriminator(p) for p in periods)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> DiscOutput:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for disc in self.discriminators:
            (y_d_r, fmap_r), (y_d_g, fmap_g) = _pair(disc, y, y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ResolutionDiscriminator(nn.Module):
    """One STFT resolution of the multi-resolution discriminator.

    The waveform is mean-removed and peak-normalized to 0.8 per example, so the discriminator is blind to level.
    The complex STFT (real and imaginary parts as two channels) is split into frequency bands that share the time
    axis, each band has its own convolution stack, and the band outputs are concatenated along frequency.
    """

    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        band_edges: Sequence[float] = VOCOS_BAND_EDGES,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_length = int(window_length * hop_factor)
        self.lrelu_slope = lrelu_slope
        self.register_buffer("window", torch.hann_window(window_length), persistent=False)
        n_bins = window_length // 2 + 1
        self.bands = [(int(lo * n_bins), int(hi * n_bins)) for lo, hi in zip(band_edges[:-1], band_edges[1:])]
        self.band_convs = nn.ModuleList(self._conv_stack(channels) for _ in self.bands)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    @staticmethod
    def _conv_stack(channels: int) -> nn.ModuleList:
        return nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
            ]
        )

    def spectrogram(self, x: torch.Tensor) -> list[torch.Tensor]:
        """``x[B, L]`` to a list of ``[B, 2, frames, band_bins]`` tensors, one per band."""
        x = x - x.mean(dim=-1, keepdim=True)
        x = 0.8 * x / (x.abs().amax(dim=-1, keepdim=True) + 1e-9)
        spec = stft(x, self.window_length, self.hop_length, self.window_length, self.window.to(x.dtype))
        spec = torch.view_as_real(spec).permute(0, 3, 2, 1)  # b f t c -> b c t f
        return [spec[..., lo:hi] for lo, hi in self.bands]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            bands = self.spectrogram(x.squeeze(1).float())
        fmap, outs = [], []
        for band, stack in zip(bands, self.band_convs):
            for i, conv in enumerate(stack):
                band = F.leaky_relu(conv(band), self.lrelu_slope)
                if i > 0:
                    fmap.append(band)
            outs.append(band)
        x = self.conv_post(torch.cat(outs, dim=-1))
        fmap.append(x)
        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    """Multi-resolution discriminator (windows 2048, 1024, 512 by default), 1,413,990 parameters.

    ``num_bands`` frequency bands per resolution; five bands use the Vocos edges (0, .1, .25, .5, .75, 1) of the
    Nyquist range, any other count splits it evenly. The time axis needs at least 13 STFT frames at every resolution.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (2048, 1024, 512),
        num_bands: int = 5,
        channels: int = 32,
        hop_factor: float = 0.25,
    ):
        super().__init__()
        if num_bands == len(VOCOS_BAND_EDGES) - 1:
            edges = VOCOS_BAND_EDGES
        else:
            edges = tuple(i / num_bands for i in range(num_bands + 1))
        self.discriminators = nn.ModuleList(
            ResolutionDiscriminator(w, channels=channels, hop_factor=hop_factor, band_edges=edges) for w in fft_sizes
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> DiscOutput:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for disc in self.discriminators:
            (y_d_r, fmap_r), (y_d_g, fmap_g) = _pair(disc, y, y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleDiscriminator(nn.Module):
    """One scale of the multi-scale discriminator; returns the feature map of every convolution."""

    def __init__(self, use_spectral_norm: bool = False, lrelu_slope: float = 0.1):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator on the waveform at scales 1, 1/2, 1/4 (29,618,821 parameters).

    Scale ``s`` sees the input average-pooled by a factor ``s``; the pooling layers are applied one after another with
    stride ``scales[i] / scales[i - 1]``, as in HiFi-GAN. The first scale uses spectral norm, the others weight norm.
    """

    def __init__(self, scales: Sequence[int] = (1, 2, 4)):
        super().__init__()
        if scales[0] != 1 or any(b % a for a, b in zip(scales[:-1], scales[1:])):
            raise ValueError(f"scales must start at 1 and each divide the next, got {tuple(scales)}")
        self.scales = tuple(scales)
        self.discriminators = nn.ModuleList(ScaleDiscriminator(use_spectral_norm=(i == 0)) for i in range(len(scales)))
        self.meanpools = nn.ModuleList(
            nn.AvgPool1d(4, b // a, padding=2) for a, b in zip(self.scales[:-1], self.scales[1:])
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> DiscOutput:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            (y_d_r, fmap_r), (y_d_g, fmap_g) = _pair(disc, y, y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
