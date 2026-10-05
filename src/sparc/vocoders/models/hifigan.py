"""HiFi-GAN generator driven by SPARC articulatory features, 24 kHz output (see docs/vocoders/INTERFACES.md, section 3).

Ported from ``src/sparc/generator.py`` and ``src/sparc/block.py`` of this repository at commit 603c58b (the SPARC code
of Cho et al., 2024, used here with the authors' permission, see LICENSE), which follow jik876/hifi-gan (MIT, Copyright
(c) 2020 Jungil Kong) through kan-bayashi/ParallelWaveGAN (MIT, Copyright (c) 2019 Tomoki Hayashi).

Changes from the SPARC generator: the shared :class:`FeatureFrontend` is the first operation (no in-place pitch
rescale), the speaker FiLM is the shared ``x * (1 + gamma) + beta`` module with zero-initialized projections instead
of a per-unit MLP with a soft clamp, convolutions keep PyTorch's default initialization (the fork's N(0, 0.01) reset
ran after weight norm and never took effect), weight norm uses ``torch.nn.utils.parametrizations``, and the number of
input channels and the upsampling stack are configurable.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils import parametrizations, parametrize

from sparc.vocoders.constants import N_FEATURES, SPEAKER_DIM
from sparc.vocoders.models.base import Vocoder
from sparc.vocoders.models.film import FiLM
from sparc.vocoders.models.frontend import FeatureFrontend


def _weight_norm(conv: nn.Module) -> nn.Module:
    return parametrizations.weight_norm(conv)


class ResidualFiLMBlock(nn.Module):
    """HiFi-GAN residual block (one kernel size, several dilations) with a speaker FiLM on every residual unit.

    Each unit is ``x + FiLM(conv2(lrelu(conv1(lrelu(x)))))`` with ``conv1`` dilated and ``conv2`` undilated.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        spk_dim: int,
        negative_slope: float,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel size must be odd, got {kernel_size}")
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.films = nn.ModuleList()
        for dilation in dilations:
            self.convs1.append(
                nn.Sequential(
                    nn.LeakyReLU(negative_slope),
                    _weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=dilation,
                            padding=(kernel_size - 1) // 2 * dilation,
                        )
                    ),
                )
            )
            self.convs2.append(
                nn.Sequential(
                    nn.LeakyReLU(negative_slope),
                    _weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)),
                )
            )
            self.films.append(FiLM(spk_dim, channels))

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, film in zip(self.convs1, self.convs2, self.films):
            x = x + film(conv2(conv1(x)), spk)
        return x


class HiFiGANVocoder(Vocoder):
    """HiFi-GAN generator: ``features[B, 15, T]`` and ``spk[B, 64]`` to ``wav[B, 1, 480 T]``.

    Stage ``i`` upsamples by ``upsample_scales[i]`` with a transposed convolution (halving the channel count), then
    averages the outputs of the multi-receptive-field residual blocks, one per entry of ``resblock_kernel_sizes``.
    The product of ``upsample_scales`` must equal the hop of 480 samples. ``stats`` are the training statistics (a
    dict or a JSON path) consumed by the feature frontend; ``kernel_size`` is that of the input and output
    convolutions, ``negative_slope`` the LeakyReLU slope in the stages and ``output_negative_slope`` the one before
    the output convolution (the reference implementation uses PyTorch's default there). Under autocast the stages run
    in reduced precision, but the output convolution and the tanh run at least in float32, so the waveform never comes
    out in half precision.
    """

    def __init__(
        self,
        stats,
        in_channels: int = N_FEATURES,
        channels: int = 512,
        upsample_scales: Sequence[int] = (8, 5, 4, 3),
        upsample_kernel_sizes: Sequence[int] = (16, 10, 8, 6),
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        resblock_dilations: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        spk_dim: int = SPEAKER_DIM,
        pitch_mode: str = "log",
        voiced_flag: bool = False,
        kernel_size: int = 7,
        negative_slope: float = 0.1,
        output_negative_slope: float = 0.01,
    ):
        super().__init__()
        scales = tuple(int(s) for s in upsample_scales)
        kernels = tuple(int(k) for k in upsample_kernel_sizes)
        block_kernels = tuple(int(k) for k in resblock_kernel_sizes)
        dilations = tuple(tuple(int(d) for d in ds) for ds in resblock_dilations)
        if len(scales) != len(kernels):
            raise ValueError(f"got {len(scales)} upsample scales but {len(kernels)} kernel sizes")
        if len(block_kernels) != len(dilations):
            raise ValueError(f"got {len(block_kernels)} residual kernel sizes but {len(dilations)} dilation lists")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel size must be odd, got {kernel_size}")
        hop = 1
        for scale in scales:
            hop *= scale
        if hop != self.hop:
            raise ValueError(f"upsample scales {scales} multiply to {hop}, expected the hop {self.hop}")
        if spk_dim != self.spk_dim:
            raise ValueError(f"spk_dim must be {self.spk_dim}, got {spk_dim}")
        if channels % (2 ** len(scales)) != 0:
            raise ValueError(f"channels ({channels}) must be divisible by 2 ** {len(scales)}")

        self.frontend = FeatureFrontend(stats, pitch_mode=pitch_mode, voiced_flag=voiced_flag)
        if in_channels != self.frontend.out_channels:
            raise ValueError(f"in_channels={in_channels} but the frontend produces {self.frontend.out_channels}")
        self.num_blocks = len(block_kernels)

        self.input_conv = _weight_norm(nn.Conv1d(in_channels, channels, kernel_size, padding=(kernel_size - 1) // 2))
        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i, (scale, kernel) in enumerate(zip(scales, kernels)):
            if kernel < scale:
                raise ValueError(f"upsample kernel {kernel} is smaller than its scale {scale}")
            output_padding = (kernel - scale) % 2
            padding = (kernel - scale + output_padding) // 2
            c_in, c_out = channels // 2**i, channels // 2 ** (i + 1)
            self.upsamples.append(
                nn.Sequential(
                    nn.LeakyReLU(negative_slope),
                    _weight_norm(
                        nn.ConvTranspose1d(c_in, c_out, kernel, scale, padding=padding, output_padding=output_padding)
                    ),
                )
            )
            for block_kernel, block_dilations in zip(block_kernels, dilations):
                self.blocks.append(ResidualFiLMBlock(c_out, block_kernel, block_dilations, spk_dim, negative_slope))
        self.output_conv = nn.Sequential(
            nn.LeakyReLU(output_negative_slope),
            _weight_norm(nn.Conv1d(c_out, 1, kernel_size, padding=(kernel_size - 1) // 2)),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(self.frontend(features).x)
        for i, upsample in enumerate(self.upsamples):
            x = upsample(x)
            blocks = self.blocks[i * self.num_blocks : (i + 1) * self.num_blocks]
            out = blocks[0](x, spk)
            for block in blocks[1:]:
                out = out + block(x, spk)
            x = out / self.num_blocks
        with torch.autocast(device_type=x.device.type, enabled=False):
            wav = self.output_conv(x.to(torch.promote_types(x.dtype, torch.float32)))
        self.check_io(features, spk, wav)
        return wav

    def remove_weight_norm(self) -> None:
        """Folds weight norm into plain weights for inference (irreversible; the state dict keys change)."""
        for module in list(self.modules()):
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(module, "weight")
