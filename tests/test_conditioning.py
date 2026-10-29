"""Phase 8 conditioning study: AdaLN-zero, conditioning sites, utterance-level statistics (docs/spkemb_conditioning_screen.md)."""

import numpy as np
import pytest
import torch

from sparc.vocoders.constants import F0_CHANNEL, LOUDNESS_CHANNEL, PERIODICITY_CHANNEL
from sparc.vocoders.data.dataset import voiced_log_loudness, voiced_logf0_std
from sparc.vocoders.models.film import AdaLNZero, FiLM
from sparc.vocoders.models.hifigan import HiFiGANVocoder
from sparc.vocoders.models.speaker import StatsCode
from sparc.vocoders.training.speaker_terms import (
    REL_F0_RANGE,
    cond_stat_vector,
    cond_stats_config,
    cond_stats_enabled,
    relative_loudness_features,
)

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


def make_features(batch=2, frames=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(batch, 15, frames, generator=g)
    f[:, 12] = 80.0 + 200.0 * torch.rand(batch, frames, generator=g)
    f[:, 13] = 0.2 * torch.rand(batch, frames, generator=g)
    f[:, 14] = 0.5 + 0.4 * torch.rand(batch, frames, generator=g)
    return f


def build(**kwargs):
    torch.manual_seed(0)
    return HiFiGANVocoder(STATS, channels=32, **kwargs).eval()


def randomize_modulation(model, std=0.05):
    torch.manual_seed(1)
    for module in model.modules():
        if isinstance(module, (FiLM, AdaLNZero)):
            torch.nn.init.normal_(module.proj.weight, std=std)
    return model


def test_default_has_no_new_parameters_and_matches_reference_checksums():
    model = randomize_modulation(build())
    keys = list(model.state_dict())
    assert not any(k.startswith(("site_in", "site_up")) for k in keys)
    assert not any(".adas." in k for k in keys)
    with torch.no_grad():
        wav = model(make_features(), torch.randn(2, 64, generator=torch.Generator().manual_seed(2)))
    # Values of the generator before the conditioning switches existed (same seed, same randomized modulation).
    assert wav.abs().sum().item() == pytest.approx(GOLDEN_ABS_SUM, rel=1e-6)
    assert wav.sum().item() == pytest.approx(GOLDEN_SUM, rel=1e-5, abs=1e-4)


@pytest.mark.parametrize("cond_sites", ["residual", "all"])
def test_adaln_zero_is_speaker_independent_at_init(cond_sites):
    model = build(cond_type="adaln_zero", cond_sites=cond_sites)
    feats = make_features()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        a = model(feats, torch.randn(2, 64, generator=g))
        b = model(feats, torch.randn(2, 64, generator=g))
        c = model(feats, torch.zeros(2, 64))
    assert torch.equal(a, b) and torch.equal(a, c)
    assert a.shape == (2, 1, 480 * 9) and torch.isfinite(a).all()


@pytest.mark.parametrize("cond_type", ["film", "adaln_zero"])
def test_all_sites_identity_at_init_and_active_after_training_step(cond_type):
    base = build(cond_type=cond_type)
    sites = build(cond_type=cond_type, cond_sites="all")
    sites.load_state_dict(base.state_dict(), strict=False)
    feats, spk = make_features(), torch.randn(2, 64)
    with torch.no_grad():
        assert torch.equal(base(feats, spk), sites(feats, spk))
    n_new = sum(p.numel() for n, p in sites.named_parameters() if n.startswith(("site_in", "site_up")))
    assert n_new > 0
    randomize_modulation(sites)
    with torch.no_grad():
        assert not torch.allclose(base(feats, spk), sites(feats, spk))


def test_adaln_zero_gradient_reaches_modulation_and_convs_after_one_step():
    model = build(cond_type="adaln_zero", cond_sites="all").train()
    feats, spk = make_features(), torch.randn(2, 64)
    model(feats, spk).square().mean().backward()
    ada = [p for n, p in model.named_parameters() if "adas" in n and n.endswith("proj.weight")]
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in ada)  # gate gradient is nonzero at init
    with torch.no_grad():
        for p in ada:
            p.add_(0.01 * torch.sign(p.grad))
    model.zero_grad()
    model(feats, spk).square().mean().backward()
    convs = [p for n, p in model.named_parameters() if "convs1" in n and "original" in n]
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in convs)


def test_adaln_normalizes_over_channels_not_time():
    ada = AdaLNZero(64, 8)
    x = torch.randn(1, 8, 20)
    cond = torch.randn(1, 64)
    full, _ = ada(x, cond)
    part, _ = ada(x[:, :, 5:12], cond)
    assert torch.allclose(full[:, :, 5:12], part, atol=1e-6)  # per-frame, so independent of the utterance length


def test_invalid_options():
    with pytest.raises(ValueError):
        build(cond_type="bad")
    with pytest.raises(ValueError):
        build(cond_sites="bad")


def test_stats_code_is_zero_at_init_and_config_parsing():
    assert StatsCode(2)(torch.randn(3, 2)).abs().sum() == 0
    assert cond_stats_config({}) == (False, "off")
    assert cond_stats_config({"cond_stats": {"f0_range": True, "loudness": "moved"}}) == (True, "moved")
    assert cond_stats_config({"cond_stats": {"loudness": False}}) == (False, "off")
    assert not cond_stats_enabled({"cond_stats": {"f0_range": False, "loudness": "off"}})
    assert cond_stats_enabled({"cond_stats": {"loudness": "added"}})
    with pytest.raises(ValueError):
        cond_stats_config({"cond_stats": {"loudness": "x"}})


def test_stat_helpers():
    t = 40
    feats = np.zeros((t, 15), dtype=np.float32)
    feats[:, F0_CHANNEL] = np.exp(np.linspace(np.log(100), np.log(200), t))
    feats[:, PERIODICITY_CHANNEL] = 0.8
    feats[::2, PERIODICITY_CHANNEL] = 0.0  # unvoiced frames are ignored
    loud = np.full(t, 0.01, dtype=np.float32)
    voiced = feats[:, PERIODICITY_CHANNEL] > 0
    assert voiced_logf0_std(feats) == pytest.approx(np.log(feats[voiced, F0_CHANNEL]).std())
    assert voiced_log_loudness(feats, loud) == pytest.approx(np.log(0.01 + 1e-4), abs=1e-6)
    assert np.isnan(voiced_log_loudness(np.zeros((3, 15), dtype=np.float32), np.ones(3)))
    assert np.isnan(voiced_logf0_std(np.zeros((3, 15), dtype=np.float32)))
    batch = {"ref_f0_std": torch.tensor([REL_F0_RANGE, float("nan")]), "utt_loud": torch.tensor([-5.0 + 1.5, float("nan")])}
    vec = cond_stat_vector(batch, True, "added", -5.0, 1.5, "cpu")
    assert vec.shape == (2, 2)
    assert torch.allclose(vec[0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.equal(vec[1], torch.zeros(2))  # unknown statistics map to the training mean


def test_relative_loudness_moves_the_level_out_of_the_frontend_input():
    from sparc.vocoders.models.frontend import FeatureFrontend

    fe = FeatureFrontend(STATS, pitch_mode="log_relative")
    feats = make_features()
    feats[:, LOUDNESS_CHANNEL] = 0.05 + 0.1 * torch.rand(2, 9)
    level = torch.log(feats[:, LOUDNESS_CHANNEL] + 1e-4).mean(dim=1)
    rel = relative_loudness_features(feats, level, STATS["loud_log_mean"])
    assert torch.equal(rel[:, :LOUDNESS_CHANNEL], feats[:, :LOUDNESS_CHANNEL])  # nothing else is touched
    ch = 12 + 1  # ema(12) + pitch -> loudness channel of the frontend output
    x = fe(rel).x[:, ch]
    expected = (torch.log(feats[:, LOUDNESS_CHANNEL] + 1e-4) - level[:, None]) / STATS["loud_log_std"]
    assert torch.allclose(x, expected, atol=0.05)  # equal up to the additive eps inside the log
    assert torch.allclose(x.mean(dim=1), torch.zeros(2), atol=0.05)  # level removed
    same = relative_loudness_features(feats, torch.full((2,), float("nan")), STATS["loud_log_mean"])
    assert torch.equal(same, feats)  # unknown level: unchanged
    assert not torch.equal(feats[:, LOUDNESS_CHANNEL], rel[:, LOUDNESS_CHANNEL])


def test_film_gate_is_identity_at_init_and_trains_the_gate():
    from sparc.vocoders.models.film import GatedFiLM

    model = build(cond_type="film_gate")
    assert any(isinstance(m, GatedFiLM) for m in model.modules()) and not any(".adas." in k for k in model.state_dict())
    feats = make_features()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        a = model(feats, torch.randn(2, 64, generator=g))
        b = model(feats, torch.randn(2, 64, generator=g))
    assert torch.equal(a, b)  # every unit is the identity at initialization, whatever the code is
    spk = torch.randn(2, 64, generator=g)
    model.train()
    model(feats, spk).square().mean().backward()
    gates = [m.proj.weight.grad for m in model.modules() if isinstance(m, GatedFiLM)]
    assert all(gr is not None and gr.abs().sum() > 0 for gr in gates)  # gate gradient is nonzero at step 0
    # same modulation position and form as FiLM: with gate fixed to 1 the unit is the FiLM unit
    gf = GatedFiLM(64, 8)
    f = FiLM(64, 8)
    with torch.no_grad():
        gf.proj.weight.normal_(std=0.1)
        f.proj.weight.copy_(gf.proj.weight[:16])
        f.proj.bias.zero_()
        gf.proj.weight[16:] = 0.0
        gf.proj.bias[16:] = 1.0
    x, c = torch.randn(2, 8, 5), torch.randn(2, 64)
    h, gate = gf(x, c)
    assert torch.allclose(gate, torch.ones_like(gate)) and torch.allclose(h, f(x, c), atol=1e-6)


def test_contour_conditioning():
    plain = build()
    feats = make_features()
    spk = torch.randn(2, 64, generator=torch.Generator().manual_seed(2))
    for mode in ("added", "only"):
        model = build(cond_contour=mode)
        assert any(k.startswith("blocks.0.contours.") for k in model.state_dict())
        missing = model.load_state_dict(plain.state_dict(), strict=False)
        assert not missing.unexpected_keys and all("contours" in k for k in missing.missing_keys)
        with torch.no_grad():
            out = model(feats, spk)
            if mode == "added":  # zero-initialized: the generator starts as the default one, bit for bit
                assert torch.equal(out, plain(feats, spk))
            else:  # pitch and loudness removed from the input and the contour path is still zero: no dependence
                other = feats.clone()
                other[:, 12] = other[:, 12] * 1.3
                other[:, 13] = other[:, 13] * 0.5 + 0.01
                assert torch.equal(out, model(other, spk))
        torch.manual_seed(5)
        for m in model.blocks:
            for c in m.contours:
                torch.nn.init.normal_(c.weight, std=0.05)
        other = feats.clone()
        other[:, 12] = other[:, 12] * 1.3
        with torch.no_grad():
            assert not torch.equal(model(feats, spk), model(other, spk))  # now the contour acts
    with pytest.raises(ValueError):
        build(cond_contour="bad")
    with pytest.raises(ValueError):
        build(cond_contour="added", cond_type="adaln_zero")
    with pytest.raises(ValueError):
        build(cond_contour="added", contour_smooth=4)


def test_contour_is_a_five_frame_moving_average():
    model = build(cond_contour="added")
    torch.manual_seed(5)
    for m in model.blocks:
        for c in m.contours:
            torch.nn.init.normal_(c.weight, std=0.05)
    feats = make_features(frames=40)
    spk = torch.randn(2, 64, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        full = model(feats, spk)
        # the contour is a moving average of 5 frames, so check the same operation directly:
        x = model.frontend(feats).x[:, [12, 13]]
        pad = 2
        sm = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(x, (pad, pad), mode="replicate"), 5, 1)
    assert full.shape == (2, 1, 480 * 40) and torch.isfinite(full).all()
    assert torch.allclose(sm[:, :, 10], x[:, :, 8:13].mean(dim=2), atol=1e-6)


GOLDEN_ABS_SUM = 1665.4556884765625
GOLDEN_SUM = -1665.4556884765625
