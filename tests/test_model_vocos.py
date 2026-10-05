"""Tests for the Vocos vocoder (sparc.vocoders.models.vocos)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

import sparc
from sparc.vocoders.constants import HOP, N_EMA, N_FEATURES
from sparc.vocoders.models.film import FiLM
from sparc.vocoders.models.vocos import ISTFT, VocosVocoder

CONF_DIR = Path(sparc.__file__).parent / "conf" / "vocoder"
PARAMS_FULL = {"A": 14_276_482, "B": 13_784_002, "C": 13_537_762}
SMALL = dict(dim=32, intermediate_dim=96, num_layers=2)


def make_stats() -> dict:
    return {
        "ema_mean": [0.0] * N_EMA,
        "ema_std": [1.0] * N_EMA,
        "logf0_mean": 5.0,
        "logf0_std": 0.3,
        "loud_log_mean": -5.0,
        "loud_log_std": 1.5,
        "per_mean": 0.5,
        "per_std": 0.4,
    }


def make_features(batch: int, frames: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(batch, N_FEATURES, frames, generator=g)
    feats[:, 12] = 100.0 + 50.0 * torch.rand(batch, frames, generator=g)
    feats[:, 13] = 0.01 + 0.05 * torch.rand(batch, frames, generator=g)
    feats[:, 14] = torch.rand(batch, frames, generator=g)
    return feats


def small_model(option: str = "B", **kwargs) -> VocosVocoder:
    torch.manual_seed(0)
    return VocosVocoder(make_stats(), option=option, **{**SMALL, **kwargs}).eval()


@pytest.mark.parametrize("option", ["A", "B", "C"])
@pytest.mark.parametrize("frames", [1, 7, 64, 123])
def test_output_length(option, frames):
    model = small_model(option)
    feats = make_features(2, frames)
    spk = torch.randn(2, 64)
    with torch.no_grad():
        wav = model(feats, spk)
    assert wav.shape == (2, 1, HOP * frames)
    assert wav.dtype == torch.float32
    assert torch.isfinite(wav).all()


@pytest.mark.parametrize("option", ["A", "B", "C"])
def test_parameter_count(option):
    model = VocosVocoder(make_stats(), option=option)
    assert sum(p.numel() for p in model.parameters()) == PARAMS_FULL[option]


def test_full_size_forward():
    model = VocosVocoder(make_stats(), option="B").eval()
    with torch.no_grad():
        wav = model(make_features(1, 7), torch.randn(1, 64))
    assert wav.shape == (1, 1, HOP * 7)


@pytest.mark.parametrize("option,n_fft,hop", [("A", 1920, 480), ("B", 960, 240), ("C", 480, 120)])
def test_istft_geometry(option, n_fft, hop):
    head = small_model(option).head
    assert head.istft.n_fft == n_fft and head.istft.hop_length == hop
    assert head.out.out_features == n_fft + 2
    assert head.mag_clip == n_fft / 2


@pytest.mark.parametrize("n_fft,hop", [(1920, 480), (960, 240), (480, 120)])
@pytest.mark.parametrize("frames", [1, 2, 25])
def test_istft_perfect_reconstruction(n_fft, hop, frames):
    g = torch.Generator().manual_seed(1)
    signal = torch.randn(2, frames * hop, generator=g)
    istft = ISTFT(n_fft, hop)
    pad = (n_fft - hop) // 2
    padded = nn.functional.pad(signal[:, None], (pad, pad), mode="constant")[:, 0]
    spec = torch.stft(padded, n_fft, hop, n_fft, istft.window, center=False, return_complex=True)
    assert spec.shape[-1] == frames
    recon = istft(spec)
    assert recon.shape == signal.shape
    assert (recon - signal).abs().max() < 1e-5


@pytest.mark.parametrize("option", ["A", "B", "C"])
def test_film_identity_at_init(option):
    model = small_model(option)
    films = [m for m in model.modules() if isinstance(m, FiLM)]
    assert len(films) == 1 + SMALL["num_layers"]
    for film in films:
        assert film.proj.weight.abs().sum() == 0 and film.proj.bias.abs().sum() == 0
    feats = make_features(1, 9)
    with torch.no_grad():
        y1 = model(feats, torch.randn(1, 64, generator=torch.Generator().manual_seed(1)))
        y2 = model(feats, torch.randn(1, 64, generator=torch.Generator().manual_seed(2)))
    assert torch.equal(y1, y2)


def test_final_norm_is_unconditional():
    model = small_model()
    assert isinstance(model.backbone.final_norm, nn.LayerNorm)
    assert sum(isinstance(m, FiLM) for m in model.backbone.final_norm.modules()) == 0


def test_film_gradient_nonzero_and_speaker_matters_after_update():
    model = small_model()
    feats, spk = make_features(2, 9), torch.randn(2, 64)
    model(feats, spk).square().mean().backward()
    films = [m for m in model.modules() if isinstance(m, FiLM)]
    assert all(f.proj.weight.grad.abs().sum() > 0 for f in films)
    with torch.no_grad():
        for f in films:
            f.proj.weight.normal_(std=0.1)
        y1 = model(feats, spk)
        y2 = model(feats, torch.randn(2, 64))
    assert (y1 - y2).abs().max() > 1e-4


def test_features_not_modified():
    model = small_model()
    feats, spk = make_features(2, 11), torch.randn(2, 64)
    before = feats.clone()
    model(feats, spk)
    assert torch.equal(feats, before)


def test_head_float32_under_bf16_autocast():
    model = small_model()
    seen = {}
    model.backbone.blocks[0].pwconv1.register_forward_hook(lambda m, i, o: seen.update(backbone=o.dtype))
    model.head.out.register_forward_hook(lambda m, i, o: seen.update(head_in=i[0].dtype, head_out=o.dtype))
    model.head.istft.register_forward_hook(lambda m, i, o: seen.update(istft_in=i[0].dtype, istft_out=o.dtype))
    feats, spk = make_features(1, 9), torch.randn(1, 64)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        wav = model(feats, spk)
        ref = model(feats, spk)
    assert seen["backbone"] == torch.bfloat16
    assert seen["head_in"] == seen["head_out"] == torch.float32
    assert seen["istft_in"] == torch.complex64 and seen["istft_out"] == torch.float32
    assert wav.dtype == torch.float32 and torch.isfinite(wav).all()
    assert torch.equal(wav, ref)


def test_magnitude_clip_bounds_output():
    model = small_model()
    with torch.no_grad():
        model.head.out.weight.zero_()
        model.head.out.bias.zero_()
        model.head.out.bias[: model.head.n_fft // 2 + 1] = 200.0  # exp overflows float32
        wav = model(make_features(1, 5), torch.randn(1, 64))
    assert torch.isfinite(wav).all()
    bound = model.head.mag_clip * (model.head.n_fft // 2 + 1) * 2 / model.head.n_fft
    assert wav.abs().max() <= bound


def test_layer_scale_default_and_override():
    assert torch.allclose(small_model().backbone.blocks[0].gamma, torch.full((32,), 0.5))
    model = small_model(layer_scale_init_value=0.25)
    assert torch.allclose(model.backbone.blocks[0].gamma, torch.full((32,), 0.25))


def test_stats_are_buffers_in_state_dict():
    state = small_model().state_dict()
    assert "frontend.ema_mean" in state and "frontend.loud_log_mean" in state
    assert not any(k.endswith("window") for k in state)


def test_voiced_flag_changes_input_channels():
    model = small_model(voiced_flag=True)
    assert model.backbone.embed.in_channels == N_FEATURES + 1
    with torch.no_grad():
        wav = model(make_features(1, 6), torch.randn(1, 64))
    assert wav.shape == (1, 1, 6 * HOP)


def test_invalid_option_and_shapes():
    with pytest.raises(ValueError):
        VocosVocoder(make_stats(), option="D")
    model = small_model()
    with pytest.raises(ValueError):
        model(make_features(2, 5), torch.randn(3, 64))


@pytest.mark.parametrize(
    "name,option,dim,params",
    [
        ("vocos", "B", 512, PARAMS_FULL["B"]),
        ("vocos_a", "A", 512, PARAMS_FULL["A"]),
        ("vocos_c", "C", 512, PARAMS_FULL["C"]),
        ("vocos_d336", "B", 336, None),
    ],
)
def test_configs_instantiate(name, option, dim, params):
    cfg = OmegaConf.load(CONF_DIR / f"{name}.yaml")
    assert cfg.name == name
    model = instantiate(cfg.generator, stats=make_stats())
    assert isinstance(model, VocosVocoder)
    assert model.option == option and model.backbone.embed.out_channels == dim
    count = sum(p.numel() for p in model.parameters())
    if params is not None:
        assert count == params
    else:
        assert 5_500_000 < count < 6_500_000
    with torch.no_grad():
        assert model.eval()(make_features(1, 3), torch.randn(1, 64)).shape == (1, 1, 3 * HOP)


def test_log_magnitude_overflow_keeps_gradients_finite():
    model = small_model().train()
    with torch.no_grad():
        model.head.out.bias[: model.head.n_fft // 2 + 1] = 200.0
    model(make_features(1, 5), torch.randn(1, 64)).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_log_domain_clip_equals_clip_after_exp():
    torch.manual_seed(3)
    head = small_model().head
    with torch.no_grad():
        head.out.bias[: head.n_fft // 4] = head.log_mag_clip + 1.0
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        log_mag, phase = head.out(x).transpose(1, 2).chunk(2, dim=1)
        mag = torch.exp(log_mag).clamp(max=head.mag_clip)
        expected = head.istft(torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)))
        assert (torch.exp(log_mag) > head.mag_clip).any()
        assert torch.allclose(head(x), expected, rtol=1e-4, atol=1e-4)


def test_layer_scale_zero_is_respected():
    model = small_model(layer_scale_init_value=0.0)
    assert model.backbone.blocks[0].gamma.abs().sum() == 0


def test_even_kernel_size_rejected():
    with pytest.raises(ValueError):
        VocosVocoder(make_stats(), kernel_size=6, **SMALL)


@pytest.mark.parametrize("option,upsample", [("A", 1), ("B", 2), ("C", 4)])
def test_upsampling_is_cell_centred(option, upsample):
    model = small_model(option)
    captured = {}
    model.backbone.embed.register_forward_pre_hook(lambda m, args: captured.update(x=args[0].detach().clone()))
    frames = 12
    with torch.no_grad():
        model(make_features(1, frames), torch.randn(1, 64))
        base = model.frontend(make_features(1, frames)).x[0]
    up = captured["x"][0]
    assert up.shape == (base.shape[0], frames * upsample)
    coords = (np.arange(frames * upsample) + 0.5) / upsample - 0.5
    expected = np.stack([np.interp(coords, np.arange(frames), row.numpy()) for row in base])
    assert np.abs(up.numpy() - expected).max() < 1e-5


@pytest.mark.parametrize("option,n_fft,hop", [("A", 1920, 480), ("B", 960, 240), ("C", 480, 120)])
def test_istft_frame_centres_match_feature_cells(option, n_fft, hop):
    upsample = HOP // hop
    istft = ISTFT(n_fft, hop)
    frame = 3
    bins = torch.arange(n_fft // 2 + 1)
    spec = torch.zeros(1, n_fft // 2 + 1, 8, dtype=torch.complex64)
    spec[0, :, frame] = torch.exp(-2j * torch.pi * bins * (n_fft // 2) / n_fft)  # impulse at the window centre
    wav = istft(spec)[0]
    peak = int(wav.abs().argmax())
    assert peak == frame * hop + hop // 2
    cell_coordinate = (frame + 0.5) / upsample - 0.5  # interpolation position of the same sub-frame, in feature frames
    assert abs((peak + 0.5) / HOP - 0.5 - cell_coordinate) <= 0.5 / HOP + 1e-9


def test_state_dict_round_trip_reproduces_output():
    torch.manual_seed(1)
    model = small_model()
    for p in model.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    other = small_model(**{})
    other.load_state_dict(model.state_dict())
    feats, spk = make_features(1, 9), torch.randn(1, 64)
    with torch.no_grad():
        assert torch.equal(model(feats, spk), other(feats, spk))


def test_batch_and_crop_consistency():
    torch.manual_seed(2)
    model = small_model()
    for p in model.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    feats, spk = make_features(3, 100), torch.randn(3, 64)
    with torch.no_grad():
        full = model(feats, spk)
        single = model(feats[1:2], spk[1:2])
        crop = model(feats[:, :, 20:84], spk)
    assert (full[1:2] - single).abs().max() < 1e-5
    # the receptive field is about 4.5 frames, so frames 30..74 of the full signal equal frames 10..54 of the crop
    assert (full[..., 30 * HOP : 74 * HOP] - crop[..., 10 * HOP : 54 * HOP]).abs().max() < 1e-5
    assert full.abs().max() > 1e-3


def test_backward_finite_under_bf16_autocast():
    model = small_model().train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        wav = model(make_features(2, 9), torch.randn(2, 64))
    wav.float().square().mean().backward()
    assert wav.dtype == torch.float32
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
