"""Tests of the HiFi-GAN vocoder (sparc.vocoders.models.hifigan) and its Hydra configs."""

from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sparc.generator import HiFiGANGenerator
from sparc.vocoders.models.film import FiLM
from sparc.vocoders.models.hifigan import HiFiGANVocoder

CONF_DIR = Path(__file__).resolve().parents[1] / "src" / "sparc" / "conf" / "vocoder"
VARIANTS = ("hifigan", "hifigan_c320", "hifigan_c384", "hifigan_c256")
SMALL = dict(channels=32)


@pytest.fixture(scope="module")
def stats() -> dict:
    return {
        "ema_mean": [0.1 * i for i in range(12)],
        "ema_std": [1.0 + 0.1 * i for i in range(12)],
        "logf0_mean": 5.0,
        "logf0_std": 0.3,
        "loud_log_mean": -5.0,
        "loud_log_std": 1.5,
        "per_mean": 0.5,
        "per_std": 0.4,
    }


def make_features(batch: int, frames: int, seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(batch, 15, frames, generator=g)
    feats[:, 12] = 80.0 + 200.0 * torch.rand(batch, frames, generator=g)
    feats[:, 13] = 0.2 * torch.rand(batch, frames, generator=g)
    feats[:, 14] = torch.where(
        torch.rand(batch, frames, generator=g) < 0.3, 0.0, 0.5 + 0.4 * torch.rand(batch, frames, generator=g)
    )
    return feats.to(dtype)


@pytest.fixture(scope="module")
def small_model(stats) -> HiFiGANVocoder:
    torch.manual_seed(0)
    return HiFiGANVocoder(stats, **SMALL).eval()


@pytest.mark.parametrize("frames", [1, 7, 64, 123])
def test_output_length(small_model, frames):
    feats = make_features(2, frames)
    spk = torch.randn(2, 64)
    with torch.no_grad():
        wav = small_model(feats, spk)
    assert wav.shape == (2, 1, 480 * frames)
    assert wav.dtype == torch.float32
    assert torch.isfinite(wav).all()
    assert wav.abs().max() <= 1.0


def test_default_parameter_count_and_film_units(stats):
    model = HiFiGANVocoder(stats)
    assert sum(p.numel() for p in model.parameters()) == 14_105_026
    films = [m for m in model.modules() if isinstance(m, FiLM)]
    assert len(films) == 36
    assert sorted({f.channels for f in films}) == [32, 64, 128, 256]
    with torch.no_grad():
        wav = model.eval()(make_features(1, 3), torch.randn(1, 64))
    assert wav.shape == (1, 1, 1440)


def test_film_is_identity_at_init(small_model):
    feats = make_features(2, 12)
    spk_a, spk_b = torch.randn(2, 64), 5 * torch.randn(2, 64)
    with torch.no_grad():
        assert torch.equal(small_model(feats, spk_a), small_model(feats, spk_b))


def test_speaker_changes_output_once_film_is_nonzero(stats):
    torch.manual_seed(0)
    model = HiFiGANVocoder(stats, **SMALL).eval()
    feats = make_features(1, 12)
    spk_a, spk_b = torch.randn(1, 64), torch.randn(1, 64)
    for m in model.modules():
        if isinstance(m, FiLM):
            torch.nn.init.normal_(m.proj.weight, std=0.1)
    with torch.no_grad():
        assert not torch.allclose(model(feats, spk_a), model(feats, spk_b))


def test_features_not_modified(small_model):
    feats = make_features(2, 16)
    before = feats.clone()
    small_model(feats, torch.randn(2, 64))
    small_model(feats, torch.randn(2, 64))
    assert torch.equal(feats, before)


def test_batch_independence(small_model):
    feats = make_features(3, 10)
    spk = torch.randn(3, 64)
    with torch.no_grad():
        batched = small_model(feats, spk)
        single = small_model(feats[1:2], spk[1:2])
    assert torch.allclose(batched[1:2], single, atol=1e-5)


def test_default_conv_initialization(stats):
    torch.manual_seed(0)
    model = HiFiGANVocoder(stats)
    v = model.input_conv.parametrizations.weight.original1
    expected_std = (1.0 / (15 * 7) ** 0.5) / 3**0.5
    assert v.std().item() == pytest.approx(expected_std, rel=0.1)


def test_all_parameters_receive_gradients(stats):
    torch.manual_seed(0)
    model = HiFiGANVocoder(stats, **SMALL)
    model(make_features(1, 8), torch.randn(1, 64)).square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_remove_weight_norm_keeps_output(stats):
    torch.manual_seed(0)
    model = HiFiGANVocoder(stats, **SMALL).eval()
    for m in model.modules():
        if isinstance(m, FiLM):
            torch.nn.init.normal_(m.proj.weight, std=0.1)
    feats, spk = make_features(1, 9), torch.randn(1, 64)
    with torch.no_grad():
        before = model(feats, spk)
    model.remove_weight_norm()
    assert not any("parametrizations" in n for n, _ in model.named_parameters())
    with torch.no_grad():
        after = model(feats, spk)
    assert torch.allclose(before, after, atol=1e-5)


def test_receptive_field(stats):
    torch.manual_seed(0)
    model = HiFiGANVocoder(stats, **SMALL).double().eval()
    model.remove_weight_norm()
    frames, centre = 81, 40
    spk = torch.randn(1, 64, dtype=torch.float64)
    for offset in (0, 479):
        feats = make_features(1, frames, dtype=torch.float64).requires_grad_(True)
        model(feats, spk)[0, 0, 480 * centre + offset].backward()
        support = (feats.grad[0].abs().sum(dim=0) > 0).nonzero().flatten() - centre
        lo, hi = support.min().item(), support.max().item()
        assert -15 <= lo <= -13 and 13 <= hi <= 15
        assert 27 <= hi - lo + 1 <= 29


def test_invalid_configs(stats):
    with pytest.raises(ValueError):
        HiFiGANVocoder(stats, upsample_scales=(8, 5, 4, 2), upsample_kernel_sizes=(16, 10, 8, 4), **SMALL)
    with pytest.raises(ValueError):
        HiFiGANVocoder(stats, voiced_flag=True, **SMALL)
    with pytest.raises(ValueError):
        HiFiGANVocoder(stats, upsample_kernel_sizes=(16, 10, 8), **SMALL)
    wide = HiFiGANVocoder(stats, voiced_flag=True, in_channels=16, **SMALL)
    assert wide(make_features(1, 3), torch.randn(1, 64)).shape == (1, 1, 1440)


def test_even_and_odd_strides_give_exact_length(stats):
    model = HiFiGANVocoder(
        stats,
        channels=16,
        upsample_scales=(10, 6, 4, 2),
        upsample_kernel_sizes=(20, 12, 8, 4),
        resblock_kernel_sizes=(3,),
        resblock_dilations=((1, 3),),
    )
    assert model(make_features(1, 5), torch.randn(1, 64)).shape == (1, 1, 2400)


@pytest.mark.parametrize("name", VARIANTS)
def test_config_instantiates(name, stats):
    cfg = OmegaConf.load(CONF_DIR / f"{name}.yaml")
    assert cfg.name == name
    model = instantiate(cfg.generator, stats=stats)
    assert isinstance(model, HiFiGANVocoder)
    assert model.frontend.pitch_mode == "log"
    with torch.no_grad():
        assert model.eval()(make_features(1, 2), torch.randn(1, 64)).shape == (1, 1, 960)


def test_variants_differ_only_in_channels():
    base = OmegaConf.to_container(OmegaConf.load(CONF_DIR / "hifigan.yaml"))
    assert base["generator"]["channels"] == 512
    for name, channels in (("hifigan_c320", 320), ("hifigan_c384", 384), ("hifigan_c256", 256)):
        other = OmegaConf.to_container(OmegaConf.load(CONF_DIR / f"{name}.yaml"))
        assert other["generator"]["channels"] == channels
        assert other["name"] == name
        for cfg in (base, other):
            cfg["name"] = None
            cfg["generator"]["channels"] = None
        assert base == other


def test_size_variants_are_ordered(stats):
    counts = {}
    for name in VARIANTS:
        cfg = OmegaConf.load(CONF_DIR / f"{name}.yaml")
        counts[name] = sum(p.numel() for p in instantiate(cfg.generator, stats=stats).parameters())
    assert counts["hifigan_c256"] < counts["hifigan_c320"] < counts["hifigan_c384"] < counts["hifigan"]


def closed_form_parameters(channels: int, spk_dim: int = 64, in_channels: int = 15, kernel_size: int = 7) -> int:
    """Parameter count from the layer shapes: weight-normed convolutions carry weight, bias and a gain per dim 0."""
    total = in_channels * channels * kernel_size + 2 * channels
    c = channels
    for scale, kernel in zip((8, 5, 4, 3), (16, 10, 8, 6)):
        half = c // 2
        total += c * half * kernel + half + c
        for res_kernel in (3, 7, 11):
            for _ in range(3):
                total += 2 * (half * half * res_kernel + 2 * half) + spk_dim * 2 * half + 2 * half
        c = half
    return total + c * kernel_size + 2


@pytest.mark.parametrize("channels", [32, 256, 320, 384, 512])
def test_parameter_count_matches_closed_form(stats, channels):
    model = HiFiGANVocoder(stats, channels=channels)
    assert sum(p.numel() for p in model.parameters()) == closed_form_parameters(channels)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_matches_sparc_generator_when_film_is_identity(stats):
    """The port equals the fork's HiFiGANGenerator without speaker conditioning, weight for weight."""
    torch.manual_seed(0)
    model = HiFiGANVocoder(stats, channels=32).double().eval()
    reference = HiFiGANGenerator(
        in_channels=15,
        channels=32,
        kernel_size=7,
        upsample_scales=(8, 5, 4, 3),
        upsample_kernel_sizes=(16, 10, 8, 6),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5)] * 3,
        use_spk=False,
        pitch_offset=0.0,
        pitch_rescale=1.0,
    ).double().eval()
    state = model.state_dict()
    renamed = {
        key: state[
            key.replace("weight_g", "parametrizations.weight.original0").replace(
                "weight_v", "parametrizations.weight.original1"
            )
        ]
        for key in reference.state_dict()
    }
    reference.load_state_dict(renamed, strict=True)
    feats = make_features(2, 11, dtype=torch.float64)
    with torch.no_grad():
        expected = reference(model.frontend(feats).x.clone())
        actual = model(feats, torch.randn(2, 64, dtype=torch.float64))
    assert expected.abs().max() > 1e-3
    assert torch.allclose(actual, expected, atol=1e-10)


def test_autocast_keeps_float32_output(stats):
    torch.manual_seed(0)
    model = HiFiGANVocoder(stats, **SMALL)
    feats, spk = make_features(2, 9), torch.randn(2, 64)
    with torch.no_grad():
        reference = model.eval()(feats, spk)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        wav = model.train()(feats, spk)
    assert wav.dtype == torch.float32
    assert wav.shape == reference.shape
    assert torch.isfinite(wav).all()
    wav.square().mean().backward()
    assert all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert (wav.detach() - reference).abs().max() < 0.1


@pytest.mark.parametrize("pitch_mode", ["log", "sparc_linear", "linear500"])
def test_pitch_modes(stats, pitch_mode):
    model = HiFiGANVocoder(stats, pitch_mode=pitch_mode, **SMALL).eval()
    assert model.frontend.pitch_mode == pitch_mode
    with torch.no_grad():
        assert model(make_features(1, 4), torch.randn(1, 64)).shape == (1, 1, 1920)


def test_degenerate_features_stay_finite(small_model):
    feats = make_features(2, 10)
    feats[0, 12:15] = 0.0
    feats[1, 13] = 1e6
    feats[1, 12] = 1e5
    with torch.no_grad():
        wav = small_model(feats, torch.randn(2, 64))
    assert torch.isfinite(wav).all()
    assert wav.abs().max() <= 1.0


def test_state_dict_round_trip_restores_output_and_statistics(stats):
    torch.manual_seed(0)
    model = HiFiGANVocoder(stats, **SMALL).eval()
    for m in model.modules():
        if isinstance(m, FiLM):
            torch.nn.init.normal_(m.proj.weight, std=0.1)
    other_stats = dict(stats, logf0_mean=6.0, per_std=0.9)
    clone = HiFiGANVocoder(other_stats, **SMALL).eval()
    result = clone.load_state_dict(model.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert clone.frontend.logf0_mean.item() == pytest.approx(5.0)
    feats, spk = make_features(1, 8), torch.randn(1, 64)
    with torch.no_grad():
        assert torch.equal(model(feats, spk), clone(feats, spk))
