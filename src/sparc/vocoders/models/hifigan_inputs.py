"""HiFi-GAN generator with the original SPARC input set (12 EMA, pitch, loudness: 14 channels, periodicity left out).

The fork's vocoders take 15 frame inputs (the original 14 plus periodicity, channel 14 of the cached features). This
class is the generator of ``sparc.vocoders.models.hifigan`` with the periodicity channel removed from the network input
by the feature frontend (``FeatureFrontend(drop_periodicity=True)``) and an input convolution with 14 input channels.
Everything else (stages, FiLM, speaker code, losses, discriminators, data, cached features with 15 columns) is
untouched; periodicity is not used anywhere else in the generator or the losses, and the speaker-embedding pooling
keeps reading the cached periodicity weights outside the model. The channel order is that of the original generator
input (``sparc.sparc``: ``[ema, pitch, loudness]``).

Built on top of :class:`HiFiGANVocoder` without touching it: the parent is constructed as usual and its frontend and
input convolution are replaced.
"""

from torch import nn

from sparc.vocoders.models.frontend import FeatureFrontend
from sparc.vocoders.models.hifigan import HiFiGANVocoder, _weight_norm


class HiFiGANOriginalInputsVocoder(HiFiGANVocoder):
    """``HiFiGANVocoder`` whose network input is the 14 original SPARC channels (``in_channels`` must be 14)."""

    def __init__(self, stats, in_channels: int = 14, channels: int = 512, kernel_size: int = 7, **kwargs):
        if in_channels != 14:
            raise ValueError(f"the original SPARC input set has 14 channels, got in_channels={in_channels}")
        if kwargs.get("voiced_flag", False):
            raise ValueError("voiced_flag needs periodicity and is not available with the original input set")
        super().__init__(stats, in_channels=15, channels=channels, kernel_size=kernel_size, **kwargs)
        self.frontend = FeatureFrontend(stats, pitch_mode=kwargs.get("pitch_mode", "log"), drop_periodicity=True)
        assert self.frontend.out_channels == in_channels
        self.input_conv = _weight_norm(nn.Conv1d(in_channels, channels, kernel_size, padding=(kernel_size - 1) // 2))
