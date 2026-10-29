"""Tests of the original-input-set HiFi-GAN (no periodicity channel) and of the frontend switch behind it."""

import pytest
import torch

from sparc.vocoders.models.frontend import FeatureFrontend
from sparc.vocoders.models.hifigan import HiFiGANVocoder
from sparc.vocoders.models.hifigan_inputs import HiFiGANOriginalInputsVocoder

STATS = {
    "ema_mean": [0.1 * i for i in range(12)],
    "ema_std": [1.0 + 0.1 * i for i in range(12)],
    "logf0_mean": 5.0,
    "logf0_std": 0.3,
    "loud_log_mean": -5.0,
    "loud_log_std": 1.5,
    "per_mean": 0.5,
    "per_std": 0.4,
}


def features(batch=2, frames=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(batch, 15, frames, generator=g)
    f[:, 12] = 80.0 + 200.0 * torch.rand(batch, frames, generator=g)
    f[:, 13] = 0.2 * torch.rand(batch, frames, generator=g)
    f[:, 14] = torch.where(
        torch.rand(batch, frames, generator=g) < 0.3, 0.0, 0.5 + 0.4 * torch.rand(batch, frames, generator=g)
    )
    return f


def test_frontend_default_unchanged():
    f = features()
    default = FeatureFrontend(STATS)
    explicit = FeatureFrontend(STATS, drop_periodicity=False)
    assert default.out_channels == 15
    assert torch.equal(default(f).x, explicit(f).x)


def test_frontend_drop_is_the_first_14_channels_in_original_order():
    f = features()
    full = FeatureFrontend(STATS)(f)
    dropped_fe = FeatureFrontend(STATS, drop_periodicity=True)
    dropped = dropped_fe(f)
    assert dropped_fe.out_channels == 14
    assert dropped.x.shape[1] == 14
    assert torch.equal(dropped.x, full.x[:, :14])  # 12 EMA, pitch, loudness
    assert torch.equal(dropped.periodicity, f[:, 14:15])  # raw periodicity still returned for outside users
    with pytest.raises(ValueError):
        FeatureFrontend(STATS, drop_periodicity=True, voiced_flag=True)


def test_generator_ignores_periodicity_and_default_generator_does_not():
    small = dict(channels=32)
    g14 = HiFiGANOriginalInputsVocoder(STATS, in_channels=14, **small).eval()
    g15 = HiFiGANVocoder(STATS, in_channels=15, **small).eval()
    assert g14.input_conv.weight.shape[1] == 14 and g15.input_conv.weight.shape[1] == 15
    f, spk = features(), torch.randn(2, 64)
    other = f.clone()
    other[:, 14] = 0.0
    with torch.no_grad():
        assert torch.equal(g14(f, spk), g14(other, spk))
        assert not torch.equal(g15(f, spk), g15(other, spk))
    with pytest.raises(ValueError):
        HiFiGANOriginalInputsVocoder(STATS, in_channels=15, **small)
