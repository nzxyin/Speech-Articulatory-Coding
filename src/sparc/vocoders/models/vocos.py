"""Vocos (ConvNeXt backbone + iSTFT head) driven by SPARC features and conditioned on a speaker embedding.

Ported from Vocos (https://github.com/gemelo-ai/vocos, commit eb39abf, MIT License, Copyright (c) 2023 Charactr Inc.):
``ConvNeXtBlock`` (modules.py), ``VocosBackbone`` (models.py), ``ISTFTHead`` (heads.py) and the ``same``-padding
``ISTFT`` (spectral_ops.py). Changes: the LayerNorm in the embedding and in every block is a
:class:`~sparc.vocoders.models.film.FiLMLayerNorm` driven by the 64-d speaker embedding (the final LayerNorm stays
unconditional); the FiLM projections are re-zeroed after the backbone initialization; the magnitude clip scales with
``n_fft`` and acts on the log-magnitude before ``exp`` (no overflow, finite gradients); the head and the iSTFT run in
float32 with autocast disabled; the input is optionally upsampled by linear interpolation; the window envelope is
recomputed without a host synchronization.

Options (decision D8): ``A`` runs the backbone at 50 Hz (n_fft 1920, hop 480), ``B`` at 100 Hz (960, 240) and ``C`` at
200 Hz (480, 120). With ``same`` padding the output has exactly ``480 T`` samples.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from sparc.vocoders.constants import HOP
from sparc.vocoders.models.base import Vocoder
from sparc.vocoders.models.film import FiLM, FiLMLayerNorm
from sparc.vocoders.models.frontend import FeatureFrontend

OPTION_UPSAMPLE = {"A": 1, "B": 2, "C": 4}


class ISTFT(nn.Module):
    """Inverse STFT with ``same`` padding: ``T`` frames of hop ``hop_length`` give exactly ``T * hop_length`` samples.

    ``torch.istft`` cannot do this because its NOLA check fails at the edges; here the edges are trimmed after
    overlap-add and the signal is divided by the trimmed squared-window envelope. Requires ``n_fft == win_length``.
    """

    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        if (n_fft - hop_length) % 2:
            raise ValueError("n_fft - hop_length must be even for same padding")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad = (n_fft - hop_length) // 2
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def _overlap_add(self, frames: torch.Tensor, length: int) -> torch.Tensor:
        out = F.fold(
            frames,
            output_size=(1, length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        )
        return out[:, 0, 0, self.pad : length - self.pad]

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """``spec[B, n_fft // 2 + 1, T]`` complex to ``wav[B, T * hop_length]``."""
        if spec.dim() != 3:
            raise ValueError(f"expected a 3D spectrogram, got {tuple(spec.shape)}")
        frames = spec.shape[-1]
        length = (frames - 1) * self.hop_length + self.n_fft
        windowed = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward") * self.window[None, :, None]
        wav = self._overlap_add(windowed, length)
        window_sq = self.window.square()[None, :, None].expand(1, self.n_fft, frames)
        envelope = self._overlap_add(window_sq, length)
        return wav / envelope


class ISTFTHead(nn.Module):
    """Predicts log-magnitude and phase per bin from ``[B, T, dim]`` and synthesizes the waveform in float32."""

    def __init__(self, dim: int, n_fft: int, hop_length: int, mag_clip_ratio: float = 0.5):
        super().__init__()
        self.n_fft = n_fft
        self.mag_clip = mag_clip_ratio * n_fft
        self.log_mag_clip = math.log(self.mag_clip)
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft, hop_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.out(x.float()).transpose(1, 2)
            log_mag, phase = x.chunk(2, dim=1)
            mag = torch.exp(log_mag.clamp(max=self.log_mag_clip))
            spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
            return self.istft(spec)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for 1D signals with a speaker-conditioned LayerNorm and layer scale."""

    def __init__(
        self, dim: int, intermediate_dim: int, layer_scale_init_value: float, spk_dim: int, kernel_size: int = 7
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = FiLMLayerNorm(spk_dim, dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """``x[B, dim, T]`` to ``[B, dim, T]``."""
        h = self.dwconv(x).transpose(1, 2)
        h = self.norm(h, spk)
        h = self.pwconv2(self.act(self.pwconv1(h)))
        return x + (self.gamma * h).transpose(1, 2)


class VocosBackbone(nn.Module):
    """Embedding convolution, FiLM-LayerNorm, ``num_layers`` ConvNeXt blocks and an unconditional final LayerNorm."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        spk_dim: int,
        kernel_size: int = 7,
        layer_scale_init_value: float | None = None,
    ):
        super().__init__()
        if layer_scale_init_value is None:
            layer_scale_init_value = 1 / num_layers
        self.embed = nn.Conv1d(in_channels, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = FiLMLayerNorm(spk_dim, dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(dim, intermediate_dim, layer_scale_init_value, spk_dim, kernel_size)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)
        for module in self.modules():
            if isinstance(module, FiLM):
                module.reset_parameters()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """``x[B, in_channels, T]`` to ``[B, T, dim]``."""
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2), spk).transpose(1, 2)
        for block in self.blocks:
            x = block(x, spk)
        return self.final_norm(x.transpose(1, 2))


class VocosVocoder(Vocoder):
    """Vocos vocoder: ``features[B, 15, T]``, ``spk[B, 64]`` to ``wav[B, 1, 480 T]``.

    ``option`` picks the backbone rate: ``A`` (no upsampling), ``B`` (2x) or ``C`` (4x linear interpolation of the
    normalized features). The iSTFT uses ``hop = 480 / upsample`` and ``n_fft = n_fft_per_hop * hop``; magnitudes
    are clipped at ``mag_clip_ratio * n_fft``, the sum of the Hann window and so an upper bound on any bin of a
    full-scale signal.
    """

    def __init__(
        self,
        stats: dict | str,
        option: str = "B",
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        spk_dim: int = 64,
        kernel_size: int = 7,
        layer_scale_init_value: float | None = None,
        n_fft_per_hop: int = 4,
        mag_clip_ratio: float = 0.5,
        pitch_mode: str = "log",
        voiced_flag: bool = False,
    ):
        super().__init__()
        if option not in OPTION_UPSAMPLE:
            raise ValueError(f"option must be one of {tuple(OPTION_UPSAMPLE)}, got {option!r}")
        if spk_dim != self.spk_dim:
            raise ValueError(f"spk_dim must be {self.spk_dim}, got {spk_dim}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd to keep the frame count, got {kernel_size}")
        self.option = option
        self.upsample = OPTION_UPSAMPLE[option]
        hop = HOP // self.upsample
        self.frontend = FeatureFrontend(stats, pitch_mode=pitch_mode, voiced_flag=voiced_flag)
        self.backbone = VocosBackbone(
            self.frontend.out_channels,
            dim,
            intermediate_dim,
            num_layers,
            spk_dim,
            kernel_size,
            layer_scale_init_value,
        )
        self.head = ISTFTHead(dim, n_fft_per_hop * hop, hop, mag_clip_ratio)

    def forward(self, features: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        expected = (features.shape[0], self.spk_dim)
        if spk.shape != expected:
            raise ValueError(f"expected speaker embedding of shape {expected}, got {tuple(spk.shape)}")
        x = self.frontend(features).x
        if self.upsample > 1:
            x = F.interpolate(x, size=x.shape[-1] * self.upsample, mode="linear", align_corners=False)
        wav = self.head(self.backbone(x, spk)).unsqueeze(1)
        self.check_io(features, spk, wav)
        return wav
