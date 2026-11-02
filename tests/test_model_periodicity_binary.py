"""Tests of the binary voicing gate (periodicity channel replaced by 1[periodicity > 0]) in the frontend and HiFi-GAN."""

import pytest
import torch

from sparc.vocoders.models.frontend import VOICING_THRESHOLD, FeatureFrontend
from sparc.vocoders.models.hifigan import HiFiGANVocoder
from sparc.vocoders.models.hifigan_inputs import HiFiGANOriginalInputsVocoder
from tests.test_model_hifigan_inputs import STATS, features


def test_default_is_unchanged_and_explicit_false_is_identical():
    f = features()
    default, explicit = FeatureFrontend(STATS), FeatureFrontend(STATS, binary_periodicity=False)
    assert default.out_channels == explicit.out_channels == 15
    assert torch.equal(default(f).x, explicit(f).x)
    per_z = (f[:, 14] - STATS["per_mean"]) / STATS["per_std"]
    assert torch.equal(default(f).x[:, 14], per_z)  # the continuous channel is still the z-scored periodicity


def test_binary_channel_is_exactly_zero_one_in_the_same_position():
    f = features()
    f[0, 14, :4] = torch.tensor([0.0, 1e-9, 0.4, 0.9])  # unvoiced, just above the cut, extractor floor, high
    f[1, 14, 0] = -0.0
    fe = FeatureFrontend(STATS, binary_periodicity=True)
    out = fe(f)
    assert fe.out_channels == out.x.shape[1] == 15
    assert VOICING_THRESHOLD == 0.0
    assert torch.equal(out.x[:, 14], (f[:, 14] > VOICING_THRESHOLD).to(out.x.dtype))
    assert set(torch.unique(out.x[:, 14]).tolist()) <= {0.0, 1.0}
    assert out.x[0, 14, :4].tolist() == [0.0, 1.0, 1.0, 1.0]  # strictly greater than the threshold
    assert out.x[1, 14, 0] == 0.0
    ref = FeatureFrontend(STATS)(f)
    assert torch.equal(out.x[:, :14], ref.x[:, :14])  # EMA, pitch, loudness untouched
    assert torch.equal(out.periodicity, f[:, 14:15])  # raw periodicity still returned for outside users


def test_binary_input_does_not_depend_on_the_continuous_value():
    f = features()
    g = f.clone()
    g[:, 14] = torch.where(f[:, 14] > 0, 0.123, 0.0)
    fe = FeatureFrontend(STATS, binary_periodicity=True)
    assert torch.equal(fe(f).x, fe(g).x)


def test_incompatible_combinations_are_rejected():
    with pytest.raises(ValueError):
        FeatureFrontend(STATS, binary_periodicity=True, drop_periodicity=True)
    with pytest.raises(ValueError):
        FeatureFrontend(STATS, binary_periodicity=True, voiced_flag=True)
    with pytest.raises(ValueError):
        HiFiGANOriginalInputsVocoder(STATS, in_channels=14, channels=32, binary_periodicity=True)
    with pytest.raises(ValueError):  # 15 channels either way; a 16-channel input does not exist for the gate
        HiFiGANVocoder(STATS, in_channels=16, channels=32, binary_periodicity=True)


def test_generator_state_dict_is_shared_with_the_reference():
    small = dict(channels=32)
    ref = HiFiGANVocoder(STATS, in_channels=15, **small).eval()
    gate = HiFiGANVocoder(STATS, in_channels=15, binary_periodicity=True, **small).eval()
    assert ref.state_dict().keys() == gate.state_dict().keys()
    gate.load_state_dict(ref.state_dict(), strict=True)
    f, spk = features(), torch.randn(2, 64)
    other = f.clone()
    other[:, 14] = torch.where(f[:, 14] > 0, 0.31, 0.0)  # same voicing, other periodicity values
    with torch.no_grad():
        assert torch.equal(gate(f, spk), gate(other, spk))
        assert not torch.equal(ref(f, spk), ref(other, spk))
        assert not torch.equal(gate(f, spk), ref(f, spk))
