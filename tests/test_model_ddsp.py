"""Tests of the DDSP vocoder (sparc.vocoders.models.ddsp) and its Hydra configs."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sparc.vocoders.models.ddsp import (
    DDSPVocoder,
    anchored_f0,
    fft_conv_same,
    fundamental_phase,
    hann_upsample,
)
from sparc.vocoders.models.film import FiLM

CONF_DIR = Path(__file__).resolve().parents[1] / "src" / "sparc" / "conf" / "vocoder"
VARIANTS = ("ddsp", "ddsp_c448", "ddsp_rate100", "ddsp_1stack")
SMALL = dict(channels=24, n_harmonics=12, n_noise_bands=17, post_filter_taps=65, head_hidden=24, up_channels=(16, 12))
UP_CHANNELS = {100: (12,), 200: (16, 12), 400: (16, 12, 8)}
FS = 24000


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


def small(stats, rate=200, **kw) -> DDSPVocoder:
    torch.manual_seed(0)
    args = {**SMALL, "control_rate": rate, "up_channels": UP_CHANNELS[rate], **kw}
    return DDSPVocoder(stats, **args).eval()


def make_features(batch: int, frames: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(batch, 15, frames, generator=g)
    feats[:, 12] = 80.0 + 200.0 * torch.rand(batch, frames, generator=g)
    feats[:, 13] = 0.2 * torch.rand(batch, frames, generator=g)
    feats[:, 14] = torch.where(
        torch.rand(batch, frames, generator=g) < 0.3, 0.0, 0.5 + 0.4 * torch.rand(batch, frames, generator=g)
    )
    return feats


def run(model, feats, spk, seed=1):
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(feats, spk)


@pytest.mark.parametrize("frames", [1, 7, 64, 123])
@pytest.mark.parametrize(
    "rate,kw", [(100, {}), (200, {}), (400, {}), (200, {"trunk_stacks": 1}), (200, {"periodicity_gate": True})]
)
def test_output_length(stats, rate, kw, frames):
    model = small(stats, rate, **kw)
    wav = run(model, make_features(2, frames), torch.randn(2, 64))
    assert wav.shape == (2, 1, 480 * frames)
    assert wav.dtype == torch.float32
    assert torch.isfinite(wav).all()


def test_default_parameter_count(stats):
    model = DDSPVocoder(stats)
    assert sum(p.numel() for p in model.parameters()) == 5_920_908
    film = sum(p.numel() for m in model.modules() if isinstance(m, FiLM) for p in m.parameters())
    assert film == 440_960
    assert not any("frontend" in n for n, _ in model.named_parameters())
    assert "frontend.ema_mean" in model.state_dict()


@pytest.mark.parametrize("rate", [100, 200])
def test_speaker_film_is_identity_at_init(stats, rate):
    model = small(stats, rate)
    feats = make_features(2, 9)
    out_a = run(model, feats, torch.randn(2, 64, generator=torch.Generator().manual_seed(1)))
    out_b = run(model, feats, 5 * torch.randn(2, 64, generator=torch.Generator().manual_seed(2)))
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, FiLM):
                m.proj.weight.normal_(0, 0.5)
    out_c = run(model, feats, torch.randn(2, 64, generator=torch.Generator().manual_seed(1)))
    out_d = run(model, feats, 5 * torch.randn(2, 64, generator=torch.Generator().manual_seed(2)))
    assert (out_c - out_d).abs().max() > 1e-3


def test_features_not_modified(stats):
    model = small(stats)
    feats = make_features(2, 9)
    before = feats.clone()
    run(model, feats, torch.randn(2, 64))
    assert torch.equal(feats, before)


def test_every_parameter_gets_gradient(stats):
    model = small(stats, 400).train()
    wav = model(make_features(2, 12), torch.randn(2, 64))
    wav.pow(2).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing


@pytest.mark.parametrize("stacks,low,high", [(1, 33, 38), (2, 64, 69)])
def test_receptive_field(stats, stacks, low, high):
    model = small(stats, trunk_stacks=stacks)
    frames, centre = 241, 120
    feats = make_features(1, frames)
    perturbed = feats.clone()
    perturbed[:, 0, centre] += 5.0
    spk = torch.randn(1, 64)
    with torch.no_grad():
        a = model.controls(model.frontend(feats).x, spk)
        b = model.controls(model.frontend(perturbed).x, spk)
    changed = torch.nonzero((a - b).abs().amax(dim=(0, 1)) > 0).flatten()
    per_frame = model.up_total
    left = centre - changed.min().item() / per_frame
    right = changed.max().item() / per_frame - centre
    assert low <= max(left, right) <= high


def test_anchored_f0_grid_and_clamp():
    f0 = torch.tensor([[[100.0, 100.0, 200.0, 200.0, 20.0, 900.0]]])
    out = anchored_f0(f0, 120)
    assert out.shape == (1, 6 * 480)
    assert torch.allclose(out[0, :120], torch.full((120,), 100.0))
    for j, value in enumerate([100.0, 100.0, 200.0, 200.0, 50.0, 550.0]):
        assert out[0, 480 * j + 120].item() == pytest.approx(value, rel=1e-5)
    assert out.min() >= 50.0 and out.max() <= 550.0
    assert torch.allclose(out[0, 480 * 5 + 120 :], torch.full((480 - 120,), 550.0))
    flat = anchored_f0(torch.full((2, 1, 5), 133.0), 0)
    assert torch.allclose(flat, torch.full_like(flat, 133.0))


@pytest.mark.parametrize("offset", [0, 120])
def test_f0_step_crosses_log_midpoint_at_anchor(offset):
    frames, j = 40, 20  # frames < j are 100 Hz, frames >= j are 200 Hz
    f0 = torch.full((1, 1, frames), 100.0)
    f0[..., j:] = 200.0
    out = anchored_f0(f0, offset)[0]
    mid = math.sqrt(100.0 * 200.0)
    crossing = int((out < mid).sum())
    assert abs(crossing - (480 * (j - 0.5) + offset)) <= 1
    assert out[480 * j + offset].item() == pytest.approx(200.0, rel=1e-5)
    assert out[480 * (j - 1) + offset].item() == pytest.approx(100.0, rel=1e-5)


def test_phase_is_continuous_across_f0_step():
    f0 = torch.cat([torch.full((1, 4000), 100.0), torch.full((1, 4000), 200.0)], dim=1)  # abrupt step
    phase = fundamental_phase(f0)
    wave = torch.sin(2 * math.pi * phase)
    assert (wave[:, 1:] - wave[:, :-1]).abs().max() <= 2 * math.pi * 200.0 / FS * 1.001
    cycles = torch.cumsum(f0.double() / FS, 1)
    err = torch.remainder(phase.double() - cycles + 0.5, 1.0) - 0.5
    assert err.abs().max() < 1e-6


def test_phase_accuracy_over_long_signal():
    n = FS * 40
    t = torch.arange(n, dtype=torch.float64) / FS
    f0 = (150 + 30 * torch.sin(2 * math.pi * 0.3 * t)).float().unsqueeze(0)
    phase = fundamental_phase(f0)
    truth = torch.cumsum(f0.double() / FS, 1)
    err = torch.remainder(phase.double() - truth + 0.5, 1.0) - 0.5
    assert err.abs().max() < 1e-4


def instantaneous_frequency(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    spec = torch.fft.fft(x)
    h = torch.zeros(n)
    h[0] = 1
    h[1 : n // 2] = 2
    h[n // 2] = 1
    analytic = torch.fft.ifft(spec * h)
    phase = torch.from_numpy(np.unwrap(torch.angle(analytic).double().numpy()))
    return (phase[1:] - phase[:-1]) * FS / (2 * math.pi)


@pytest.mark.parametrize("offset", [0, 120])
def test_f0_step_appears_in_synthesized_frequency(stats, offset):
    model = small(stats, f0_anchor_offset=offset, n_harmonics=4)
    frames, j, k = 40, 20, 4
    f0_frames = torch.full((1, 1, frames), 100.0)
    f0_frames[..., j:] = 200.0
    amp = torch.full((1, 2 * (k + 1), frames * 4), -50.0)
    amp[:, 0] = 20.0
    amp[:, k + 1] = 20.0
    amp[:, 1] = 0.0
    amp[:, k + 2] = 0.0
    gain_only = model.harmonics(anchored_f0(f0_frames, offset), amp)
    freq = instantaneous_frequency(gain_only[0])
    mid = math.sqrt(100.0 * 200.0)
    window = slice(480 * (j - 3), 480 * (j + 3))
    crossing = window.start + int((freq[window] < mid).sum())
    assert abs(crossing - (480 * (j - 0.5) + offset)) <= 4
    assert freq[480 * (j + 1)].item() == pytest.approx(200.0, rel=0.01)
    assert freq[480 * (j - 2)].item() == pytest.approx(100.0, rel=0.01)
    assert gain_only.abs().max() == pytest.approx(2 * math.sqrt(2), rel=1e-3)


def test_hann_upsample_holds_constants_and_peaks_at_frames():
    x = torch.full((1, 3, 5), 0.7)
    y = hann_upsample(x, 120)
    assert y.shape == (1, 3, 600)
    torch.testing.assert_close(y, torch.full_like(y, 0.7), atol=1e-6, rtol=0)
    delta = torch.zeros(1, 1, 8)
    delta[..., 3] = 1.0
    y = hann_upsample(delta, 60)[0, 0]
    assert y.argmax().item() == 180
    centroid = (y * torch.arange(480)).sum() / y.sum()
    assert centroid.item() == pytest.approx(180.0, abs=1e-3)


@pytest.mark.parametrize("rate", [100, 200, 400])
def test_noise_is_centred_on_frames_and_covers_the_edges(stats, rate):
    model = small(stats, rate)
    hop, frames = model.frame_hop, 20
    raw = torch.full((64, model.n_noise_bands, frames), -20.0)
    raw[:, :, 10] = 8.0
    torch.manual_seed(0)
    noise = model.noise(raw)
    assert noise.shape == (64, frames * hop)
    energy = noise.pow(2).mean(0)
    centroid = (energy * torch.arange(frames * hop)).sum() / energy.sum()
    assert centroid.item() == pytest.approx(10 * hop, abs=3.0)
    torch.manual_seed(0)
    edges = model.noise(torch.full((64, model.n_noise_bands, frames), 0.0))
    power = edges.pow(2).mean(0)
    middle = power[frames * hop // 2 - 200 : frames * hop // 2 + 200].mean()
    assert power[:20].mean() > 0.5 * middle
    assert power[-20:].mean() > 0.5 * middle


@pytest.mark.parametrize("taps", [65, 64, 1537])
def test_post_filter_is_identity_at_init(stats, taps):
    model = small(stats, post_filter_taps=taps)
    x = torch.randn(2, 1000)
    torch.testing.assert_close(fft_conv_same(x, model.post_filter), x, atol=1e-5, rtol=0)


def test_synthesis_is_float32_with_autocast_disabled(stats):
    model = small(stats)
    seen = []
    harmonics, noise = model.harmonics, model.noise

    def spy(fn, name):
        def wrapped(*args, **kwargs):
            seen.append((name, torch.is_autocast_enabled("cpu"), [a.dtype for a in args if torch.is_tensor(a)]))
            return fn(*args, **kwargs)

        return wrapped

    model.harmonics, model.noise = spy(harmonics, "harmonics"), spy(noise, "noise")
    feats = make_features(2, 9)
    spk = torch.randn(2, 64)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        controls = model.controls(model.frontend(feats).x, spk)
        assert controls.dtype == torch.bfloat16
        wav = model(feats, spk)
    assert wav.dtype == torch.float32 and torch.isfinite(wav).all()
    assert {name for name, _, _ in seen} == {"harmonics", "noise"}
    for _, enabled, dtypes in seen:
        assert not enabled
        assert all(d == torch.float32 for d in dtypes)


def test_periodicity_gate_silences_harmonics_when_unvoiced(stats):
    model = small(stats, periodicity_gate=True)
    model.noise = lambda raw: torch.zeros(raw.shape[0], raw.shape[-1] * model.frame_hop)
    frames = 10
    f0 = torch.full((1, 1, frames), 150.0)
    amp = torch.randn(1, 2 * 13, frames * 4)
    voiced = torch.ones(1, 1, frames)
    silent = model.synthesize(f0, torch.zeros_like(voiced), amp, torch.zeros(1, 17, frames * 4))
    assert silent.abs().max() == 0
    active = model.synthesize(f0, voiced, amp, torch.zeros(1, 17, frames * 4))
    assert active.abs().max() > 1e-3
    half = voiced.clone()
    half[..., 5:] = 0
    mixed = model.synthesize(f0, half, amp, torch.zeros(1, 17, frames * 4))[0]
    assert mixed[: 480 * 4].abs().max() > 1e-3
    assert mixed[480 * 5 + 120 :].abs().max() < 1e-5


@pytest.mark.parametrize("offset", [0, 120, 183])
def test_periodicity_gate_follows_the_f0_anchor(stats, offset):
    model = small(stats, periodicity_gate=True, f0_anchor_offset=offset)
    model.noise = lambda raw: torch.zeros(raw.shape[0], raw.shape[-1] * model.frame_hop)
    frames, j = 12, 6
    f0 = torch.full((1, 1, frames), 150.0)
    amp = torch.randn(1, 2 * 13, frames * 4, generator=torch.Generator().manual_seed(0))
    voiced = torch.ones(1, 1, frames)
    voiced[..., j:] = 0
    wav = model.synthesize(f0, voiced, amp, torch.zeros(1, 17, frames * 4))[0]
    assert wav[: 480 * (j - 1) + offset].abs().max() > 1e-3
    assert wav[480 * j + offset + 2 :].abs().max() < 1e-5


def test_bypass_streams_sit_at_their_own_centres(stats):
    model = small(stats)
    x = torch.zeros(1, 15, 6)
    x[:, 12, 3] = x[:, 13, 3] = x[:, 14, 3] = 1.0
    out = model.bypass(x)[0]
    per_frame = model.up_total
    assert out.argmax(-1).tolist() == [3 * per_frame + 1, 3 * per_frame, 3 * per_frame + 1]


@pytest.mark.parametrize("offset", [0, 120])
def test_synthesized_frequency_reaches_the_new_f0_at_the_anchor(stats, offset):
    model = small(stats, f0_anchor_offset=offset, n_harmonics=4).eval()
    frames, j, k = 40, 20, 4
    f0 = torch.full((1, 1, frames), 100.0)
    f0[..., j:] = 200.0
    amp = torch.full((1, 2 * (k + 1), frames * 4), -50.0)
    amp[:, 0] = amp[:, k + 1] = 20.0
    amp[:, 1] = amp[:, k + 2] = 0.0
    with torch.no_grad():
        wav = model.synthesize(f0, torch.ones(1, 1, frames), amp, torch.full((1, 17, frames * 4), -50.0))[0]
    freq = instantaneous_frequency(wav)
    assert freq[480 * (j - 1) + offset - 20].item() == pytest.approx(100.0, rel=0.02)
    assert freq[480 * j + offset + 20].item() == pytest.approx(200.0, rel=0.02)
    assert freq[480 * (j - 1) + offset + 240].item() == pytest.approx(math.sqrt(100.0 * 200.0), rel=0.03)


def test_batch_elements_are_independent(stats):
    model = small(stats)
    model.noise = lambda raw: torch.zeros(raw.shape[0], raw.shape[-1] * model.frame_hop)
    feats, spk = make_features(3, 11), torch.randn(3, 64)
    with torch.no_grad():
        batched = model(feats, spk)
        single = torch.cat([model(feats[i : i + 1], spk[i : i + 1]) for i in range(3)])
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=0)


def test_output_is_reproducible_under_a_seed_and_only_noise_is_random(stats):
    model = small(stats)
    feats, spk = make_features(1, 9), torch.randn(1, 64)
    assert torch.equal(run(model, feats, spk, seed=3), run(model, feats, spk, seed=3))
    assert not torch.equal(run(model, feats, spk, seed=3), run(model, feats, spk, seed=4))
    model.noise = lambda raw: torch.zeros(raw.shape[0], raw.shape[-1] * model.frame_hop)
    assert torch.equal(run(model, feats, spk, seed=3), run(model, feats, spk, seed=4))


def test_degenerate_features_stay_finite(stats):
    model = small(stats)
    feats = make_features(1, 10)
    feats[:, 12, :3] = 0.0
    feats[:, 12, 5] = 5000.0
    feats[:, 13, :2] = 0.0
    feats[:, 14] = 0.0
    wav = run(model, feats, torch.randn(1, 64))
    assert torch.isfinite(wav).all()


def test_state_dict_round_trip_is_strict_and_exact(stats):
    model = small(stats)
    other = DDSPVocoder(stats, **{**SMALL, "up_channels": UP_CHANNELS[200]}).eval()
    other.load_state_dict(model.state_dict(), strict=True)
    assert "fir_window" not in model.state_dict() and "harmonic_numbers" not in model.state_dict()
    feats, spk = make_features(1, 9), torch.randn(1, 64)
    torch.testing.assert_close(run(model, feats, spk), run(other, feats, spk), rtol=0, atol=0)


def test_invalid_options(stats):
    with pytest.raises(ValueError):
        DDSPVocoder(stats, control_rate=300)
    with pytest.raises(ValueError):
        DDSPVocoder(stats, control_rate=100)  # default up_channels has two stages
    with pytest.raises(ValueError):
        DDSPVocoder(stats, trunk_stacks=0)


@pytest.mark.parametrize("name", VARIANTS)
def test_hydra_config_instantiates(stats, name):
    cfg = OmegaConf.load(CONF_DIR / f"{name}.yaml")
    assert cfg.name == name
    assert cfg.generator._target_ == "sparc.vocoders.models.ddsp.DDSPVocoder"
    assert "stats" not in cfg.generator
    model = instantiate(cfg.generator, stats=stats).eval()
    n_params = sum(p.numel() for p in model.parameters())
    expected = {"ddsp": 5_920_908, "ddsp_c448": 14_434_572, "ddsp_rate100": 5_748_428, "ddsp_1stack": 4_208_780}
    assert n_params == expected[name]
    wav = run(model, make_features(1, 5), torch.randn(1, 64))
    assert wav.shape == (1, 1, 2400)
    if name == "ddsp_rate100":
        assert model.control_rate == 100
    if name == "ddsp_1stack":
        assert len(model.trunk) == 4
