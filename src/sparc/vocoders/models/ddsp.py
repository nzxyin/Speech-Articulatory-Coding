"""DDSP vocoder driven by SPARC articulatory features, 24 kHz output (see docs/vocoders/INTERFACES.md, section 3).

The synthesis chain (sin and cos harmonic bank with a softmax over harmonics and a hard Nyquist mask, filtered noise
built by overlap-adding frame-wise FIR-filtered uniform noise, learned post filter, scaled-sigmoid parameter heads,
loudness FiLM on the trunk output, dilated residual trunk) is ported from DDSP-Articulatory-Vocoder
(https://github.com/Louis0324/DDSP-Articulatory-Vocoder, commit dc6df4ca0ed7bdd50630b7b26882291c5cee8a57, MIT License,
Copyright (c) 2024 Louis Liu, Drake Lin), itself based on sweetcocoa/ddsp-pytorch (MIT) and the intro2ddsp tutorial.
RT-VC (no license) is not used as a code source; only its idea of conditioning the heads on the speaker is reused.

Changes from the reference, all motivated in the Phase 1 report (section 6):

- 24 kHz and a 50 Hz feature stream: a learned upsampler (linear x2 plus convolution and a residual block per stage)
  brings the trunk output to the control rate, with F0, loudness and periodicity re-injected at the last stage.
- BatchNorm becomes a channel LayerNorm; the shared :class:`FiLM` (speaker) sits after the second norm of every
  residual block and in the first layer of both heads; the post filter is initialized to the identity.
- F0 is interpolated linearly in log-Hz to the sample rate with frame ``j`` anchored at ``480 j + f0_anchor_offset``
  (the measured F0 centre of SPARC), clamped to [50, 550] Hz; the fundamental phase is a float64 cumulative sum, wrapped
  before it is multiplied by the harmonic number.
- Harmonic gains are upsampled with a symmetric Hann window (exactly overlap-add constant) and the last frame is held,
  so crops do not fade out at their end; the filtered noise is delay-compensated and covers the full signal.
- All synthesis runs in float32 with autocast disabled.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from sparc.vocoders.constants import (
    F0_CENTRE_OFFSET,
    F0_MAX_HZ,
    F0_MIN_HZ,
    HOP,
    LOUDNESS_CENTRE_OFFSET,
    LOUDNESS_CHANNEL,
    SAMPLE_RATE,
    SPEAKER_DIM,
)
from sparc.vocoders.models.base import Vocoder
from sparc.vocoders.models.film import FiLM
from sparc.vocoders.models.frontend import FeatureFrontend

CONTROL_RATES = (100, 200, 400)
BYPASS_CHANNELS = slice(12, 15)  # normalized ln F0, loudness and periodicity
MASK_FILL = -1.0e4


def sample_linear(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Linearly interpolates ``x[B, C, T]`` at fractional frame ``positions[N]``; the edge values are held."""
    batch, channels, frames = x.shape
    if frames == 1:
        return x.expand(batch, channels, positions.numel())
    pos = positions.clamp(0, frames - 1)
    lo = pos.floor().long().clamp(max=frames - 2)
    w = (pos - lo).to(x.dtype)
    return x.index_select(-1, lo) * (1 - w) + x.index_select(-1, lo + 1) * w


def frame_positions(n: int, spacing: int, offset: float, device: torch.device) -> torch.Tensor:
    """Frame-grid positions of samples ``spacing * i`` when frame ``t`` sits at sample ``HOP * t + offset``."""
    return (torch.arange(n, device=device, dtype=torch.float32) * spacing - offset) / HOP


def anchored_f0(f0_hz: torch.Tensor, anchor_offset: float, f0_min: float = F0_MIN_HZ, f0_max: float = F0_MAX_HZ):
    """F0 in Hz at every sample, ``[B, 1, T] -> [B, 480 T]``.

    The track is clamped to ``[f0_min, f0_max]``, interpolated linearly in ln Hz with frame ``j`` at sample
    ``480 j + anchor_offset`` and held constant before the first and after the last anchor.
    """
    log_f0 = torch.log(f0_hz.float().clamp(f0_min, f0_max))
    pos = frame_positions(f0_hz.shape[-1] * HOP, 1, anchor_offset, f0_hz.device)
    return torch.exp(sample_linear(log_f0, pos))[:, 0].clamp(f0_min, f0_max)


def fundamental_phase(f0: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Fractional phase in cycles of the fundamental, ``[B, N] -> [B, N]`` (float32), accumulated in float64."""
    cycles = torch.cumsum(f0.double() / sample_rate, dim=1)
    return torch.remainder(cycles, 1.0).float()


def scaled_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Scaled sigmoid of the original DDSP paper, range (0, 2]."""
    return 2.0 * torch.sigmoid(x) ** math.log(10) + 1e-7


def hann_upsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Upsamples ``x[B, C, T]`` to ``[B, C, T * factor]`` with a symmetric Hann(2 factor + 1) kernel.

    Frame ``j`` peaks at sample ``factor * j`` and neighbouring kernels sum to one, so a constant input stays constant.
    The last frame is held for one more frame so that the output does not fade towards its end.
    """
    batch, channels, frames = x.shape
    x = F.pad(x, (0, 1), mode="replicate")
    window = torch.hann_window(2 * factor + 1, periodic=False, device=x.device, dtype=x.dtype)
    kernel = window.view(1, 1, -1).expand(channels, 1, -1).contiguous()
    y = F.conv_transpose1d(x, kernel, stride=factor, padding=factor, output_padding=factor - 1, groups=channels)
    return y[..., : frames * factor]


def fft_conv_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolves ``x[B, N]`` with a 1-D ``kernel`` by FFT and returns the centred ``N`` samples."""
    n = x.shape[-1] + kernel.numel() - 1
    nfft = 1 << (n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(kernel, nfft), nfft)
    start = (kernel.numel() - 1) // 2
    return y[..., start : start + x.shape[-1]]


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of ``[B, C, T]``."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResBlock(nn.Module):
    """Dilated residual block of the reference encoder with channel LayerNorm and a speaker FiLM after the second norm."""

    def __init__(self, channels: int, dilation: int, spk_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = ChannelLayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = ChannelLayerNorm(channels)
        self.film = FiLM(spk_dim, channels)

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.film(self.norm2(self.conv2(h)), spk)
        return F.relu(x + h)


class UpsampleStage(nn.Module):
    """Linear x2 upsampling (frame ``i`` at ``2 i``), optional concatenated bypass channels, conv, norm, residual block."""

    def __init__(
        self, in_channels: int, out_channels: int, spk_dim: int, extra_channels: int = 0, kernel_size: int = 5
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels + extra_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = ChannelLayerNorm(out_channels)
        self.res = ResBlock(out_channels, 1, spk_dim)

    def forward(self, x: torch.Tensor, spk: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        pos = torch.arange(2 * x.shape[-1], device=x.device, dtype=torch.float32) / 2
        x = sample_linear(x, pos)
        if skip is not None:
            x = torch.cat([x, skip.to(x.dtype)], dim=1)
        return self.res(F.relu(self.norm(self.conv(x))), spk)


class HeadMLP(nn.Module):
    """Three 1x1 convolutions with the speaker FiLM after the first one (the head layout of the reference encoder)."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int, spk_dim: int, film: bool = True):
        super().__init__()
        self.l1 = nn.Conv1d(in_channels, hidden, 1)
        self.film = FiLM(spk_dim, hidden) if film else None
        self.l2 = nn.Conv1d(hidden, hidden, 1)
        self.l3 = nn.Conv1d(hidden, out_channels, 1)

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        h = self.l1(x)
        if self.film is not None:
            h = self.film(h, spk)
        return self.l3(F.relu(self.l2(F.relu(h))))


class DDSPVocoder(Vocoder):
    """DDSP24: dilated residual trunk at 50 Hz, learned upsampler to ``control_rate``, harmonic plus filtered-noise synthesis.

    Args:
        stats: training statistics (dict or JSON path) for the shared :class:`FeatureFrontend`.
        channels: trunk width.
        n_harmonics: number of harmonics ``K`` of the sin and cos oscillator banks.
        n_noise_bands: number of filter-bank magnitudes per frame (FIR length ``2 n - 1``).
        control_rate: rate of the parameter frames in Hz (100, 200 or 400); a frame holds ``24000 / rate`` samples.
        noise_gain_cap: maximum noise filter magnitude is ``2 * noise_gain_cap``.
        post_filter_taps: length of the learned post filter (identity at initialization).
        f0_anchor_offset: sample offset of F0 frame ``j`` from ``480 j`` when F0 is interpolated to the sample rate.
        spk_dim: speaker embedding size.
        trunk_stacks: number of repeated dilation stacks in the 50 Hz trunk (the half receptive field is about 35 frames
            with one stack and 66 with two).
        trunk_dilations: dilations of one stack.
        up_channels: output width of each upsampling stage; one stage per doubling from 50 Hz to ``control_rate``.
        head_hidden: hidden width of the two parameter heads.
        periodicity_gate: multiply the harmonic branch by the voiced flag ``periodicity > 0``, linearly interpolated with
            the same anchoring as F0 (periodicity comes from the same pitch frame).
        loudness_film: keep the reference loudness conditioner on the trunk output.
        head_film: apply the speaker FiLM in the heads.
        noise_bias_init: initial bias of the last noise-head layer (a quiet noise branch at the start).
        taper_hz: width of the cosine taper below Nyquist that silences harmonics approaching it.
        pitch_mode, voiced_flag: forwarded to the :class:`FeatureFrontend`.
    """

    def __init__(
        self,
        stats: dict | str | Path,
        channels: int = 256,
        n_harmonics: int = 100,
        n_noise_bands: int = 129,
        control_rate: int = 200,
        noise_gain_cap: float = 0.25,
        post_filter_taps: int = 1537,
        f0_anchor_offset: float = F0_CENTRE_OFFSET,
        spk_dim: int = SPEAKER_DIM,
        trunk_stacks: int = 2,
        trunk_dilations: Sequence[int] = (1, 2, 4, 8),
        up_channels: Sequence[int] = (192, 128),
        head_hidden: int = 512,
        periodicity_gate: bool = False,
        loudness_film: bool = True,
        head_film: bool = True,
        noise_bias_init: float = -3.0,
        taper_hz: float = 1000.0,
        pitch_mode: str = "log",
        voiced_flag: bool = False,
    ):
        super().__init__()
        if control_rate not in CONTROL_RATES:
            raise ValueError(f"control_rate must be one of {CONTROL_RATES}, got {control_rate}")
        if trunk_stacks < 1:
            raise ValueError(f"trunk_stacks must be at least 1, got {trunk_stacks}")
        up_channels = tuple(int(c) for c in up_channels)
        n_stages = int(math.log2(control_rate // 50))
        if len(up_channels) != n_stages:
            raise ValueError(f"control_rate {control_rate} needs {n_stages} up_channels entries, got {up_channels}")
        if post_filter_taps < 1:
            raise ValueError(f"post_filter_taps must be positive, got {post_filter_taps}")
        self.spk_dim = spk_dim
        self.frontend = FeatureFrontend(stats, pitch_mode=pitch_mode, voiced_flag=voiced_flag)
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.control_rate = control_rate
        self.frame_hop = SAMPLE_RATE // control_rate  # samples per control frame
        self.up_total = control_rate // 50
        self.noise_gain_cap = noise_gain_cap
        self.f0_anchor_offset = f0_anchor_offset
        self.periodicity_gate = periodicity_gate
        self.taper_hz = taper_hz
        self.noise_margin = -(-(2 * (n_noise_bands - 1) + self.frame_hop) // (2 * self.frame_hop))

        self.in_conv = nn.Conv1d(self.frontend.out_channels, channels, 3, padding=1)
        self.trunk = nn.ModuleList(
            [ResBlock(channels, int(d), spk_dim) for _ in range(trunk_stacks) for d in trunk_dilations]
        )
        self.out_conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.loudness_cond = (
            nn.Sequential(
                nn.Conv1d(1, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(channels, 2 * channels, 3, padding=1),
            )
            if loudness_film
            else None
        )
        widths = (channels,) + up_channels
        self.up = nn.ModuleList(
            [
                UpsampleStage(widths[i], widths[i + 1], spk_dim, extra_channels=3 if i == n_stages - 1 else 0)
                for i in range(n_stages)
            ]
        )
        self.head_amp = HeadMLP(widths[-1], head_hidden, 2 * (n_harmonics + 1), spk_dim, head_film)
        self.head_noise = HeadMLP(widths[-1], head_hidden, n_noise_bands, spk_dim, head_film)
        nn.init.constant_(self.head_noise.l3.bias, noise_bias_init)
        self.post_filter = nn.Parameter(torch.zeros(post_filter_taps))
        self.post_filter.data[(post_filter_taps - 1) // 2] = 1.0

        self.register_buffer("fir_window", torch.hann_window(2 * n_noise_bands - 1, periodic=True), persistent=False)
        self.register_buffer(
            "harmonic_numbers", torch.arange(1, n_harmonics + 1, dtype=torch.float32).view(1, -1, 1), persistent=False
        )

    def bypass(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized ln F0, loudness and periodicity at the control rate, each stream sampled at its own centre."""
        n_fine = x.shape[-1] * self.up_total
        offsets = (F0_CENTRE_OFFSET, LOUDNESS_CENTRE_OFFSET, F0_CENTRE_OFFSET)
        streams = x[:, BYPASS_CHANNELS]
        out = [
            sample_linear(streams[:, i : i + 1], frame_positions(n_fine, self.frame_hop, off, x.device))
            for i, off in enumerate(offsets)
        ]
        return torch.cat(out, dim=1)

    def controls(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """Trunk and upsampler: normalized input ``[B, C_in, T]`` to features ``[B, C_f, T * control_rate / 50]``."""
        h = self.in_conv(x)
        for block in self.trunk:
            h = block(h, spk)
        h = self.out_conv(h)
        if self.loudness_cond is not None:
            gain, shift = self.loudness_cond(x[:, LOUDNESS_CHANNEL : LOUDNESS_CHANNEL + 1]).chunk(2, dim=1)
            h = h * gain + shift
        skip = self.bypass(x)
        for i, stage in enumerate(self.up):
            h = stage(h, spk, skip if i == len(self.up) - 1 else None)
        return h

    def harmonics(self, f0: torch.Tensor, amp: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Harmonic branch. ``f0[B, N]`` Hz at every sample, ``amp[B, 2 (K + 1), N / frame_hop]`` raw head output."""
        k, hop = self.n_harmonics, self.frame_hop
        gain_sin = scaled_sigmoid(amp[:, :1])
        gain_cos = scaled_sigmoid(amp[:, k + 1 : k + 2])
        logit_sin, logit_cos = amp[:, 1 : k + 1], amp[:, k + 2 :]
        mask = self.harmonic_numbers * f0[:, ::hop].unsqueeze(1) < SAMPLE_RATE / 2
        dist_sin = torch.softmax(logit_sin.masked_fill(~mask, MASK_FILL), dim=1)
        dist_cos = torch.softmax(logit_cos.masked_fill(~mask, MASK_FILL), dim=1)
        weights = hann_upsample(torch.cat([dist_sin * gain_sin, dist_cos * gain_cos], dim=1), hop)
        w_sin, w_cos = weights.chunk(2, dim=1)
        with torch.no_grad():
            frac = fundamental_phase(f0).unsqueeze(1)
            angle = 2 * math.pi * torch.remainder(self.harmonic_numbers * frac, 1.0)
            freq = self.harmonic_numbers * f0.unsqueeze(1)
            ramp = ((freq - (SAMPLE_RATE / 2 - self.taper_hz)) / self.taper_hz).clamp(0, 1)
            taper = 0.5 * (1 + torch.cos(math.pi * ramp))
            sin, cos = torch.sin(angle) * taper, torch.cos(angle) * taper
        out = (w_sin * sin).sum(1) + (w_cos * cos).sum(1)
        return out if gate is None else out * gate

    def noise(self, raw: torch.Tensor) -> torch.Tensor:
        """Filtered-noise branch from raw head output ``raw[B, n_bands, Tf]``; returns ``[B, Tf * frame_hop]``."""
        hop, nb = self.frame_hop, self.n_noise_bands
        length, margin = 2 * nb - 1, self.noise_margin
        n_out = raw.shape[-1] * hop
        mag = F.pad(scaled_sigmoid(raw) * self.noise_gain_cap, (margin, margin), mode="replicate")
        frames = mag.shape[-1]
        zero_phase = torch.fft.irfft(mag.transpose(1, 2), n=length, dim=-1)
        fir = zero_phase.roll(nb - 1, -1) * self.fir_window
        noise = torch.rand(mag.shape[0], frames, hop, device=raw.device) * 2 - 1
        nfft = 1 << (hop + length - 2).bit_length()
        filtered = torch.fft.irfft(torch.fft.rfft(noise, nfft) * torch.fft.rfft(fir, nfft), nfft)[
            ..., : hop + length - 1
        ]
        ola = F.fold(
            filtered.transpose(1, 2),
            output_size=(1, (frames - 1) * hop + hop + length - 1),
            kernel_size=(1, hop + length - 1),
            stride=(1, hop),
        ).flatten(1)
        start = (nb - 1) + margin * hop + hop // 2
        return ola[:, start : start + n_out]

    def synthesize(
        self, f0_hz: torch.Tensor, periodicity: torch.Tensor, amp: torch.Tensor, noise_raw: torch.Tensor
    ) -> torch.Tensor:
        """Float32 synthesis from F0 ``[B, 1, T]`` (Hz), periodicity ``[B, 1, T]`` and raw head outputs."""
        with torch.autocast(device_type=f0_hz.device.type, enabled=False):
            f0 = anchored_f0(f0_hz.float(), self.f0_anchor_offset)
            gate = None
            if self.periodicity_gate:
                voiced = (periodicity.float() > 0).float()
                pos = frame_positions(f0.shape[-1], 1, self.f0_anchor_offset, f0.device)
                gate = sample_linear(voiced, pos)[:, 0]
            wav = self.harmonics(f0, amp.float(), gate) + self.noise(noise_raw.float())
            return fft_conv_same(wav, self.post_filter.float())

    def forward(self, features: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        front = self.frontend(features)
        h = self.controls(front.x, spk)
        wav = self.synthesize(front.f0_hz, front.periodicity, self.head_amp(h, spk), self.head_noise(h, spk))
        wav = wav.unsqueeze(1)
        self.check_io(features, spk, wav)
        return wav
