"""Paired X-Y EMA probes: edit geometry, moments and metrics, config grid, resumable driver (CPU, fast)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import sparc
from sparc.vocoders.constants import EMA_NAMES, LOUDNESS_CHANNEL, N_EMA
from sparc.vocoders.eval import probes_xy
from sparc.vocoders.eval.probes_xy import (
    SENSORS,
    apply_xy_edit,
    bump_centers,
    bump_window,
    measure_xy,
    metrics_from_moments,
    regression_in_std_units,
    sensor_directions,
    speech_mask,
    speech_weight,
    xy_edit_name,
    xy_edits,
    xy_moments,
)

CONF = Path(sparc.__file__).parent / "conf"
STD = np.array([0.95, 0.85, 0.93, 0.9, 0.96, 0.9, 0.92, 0.87, 0.87, 0.86, 0.92, 0.85])
MEAN = np.linspace(-0.2, 0.2, N_EMA)
STATS = {"ema_std": STD.tolist(), "ema_mean": MEAN.tolist()}


def cfg_with(**overrides):
    group = OmegaConf.load(CONF / "eval_features" / "default.yaml")
    cfg = OmegaConf.create({"eval_features": group})
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def make_features(batch: int = 2, frames: int = 120, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 15, frames, generator=g)
    x[:, :N_EMA] = x[:, :N_EMA] * torch.tensor(STD, dtype=torch.float32)[None, :, None]
    t = torch.arange(frames)
    speech = ((t > 20) & (t < 100)).float()  # silence at both ends, speech in the middle
    x[:, LOUDNESS_CHANNEL] = 0.01 + 1.0 * speech[None] * (0.8 + 0.2 * torch.rand(batch, frames, generator=g))
    return x


def directions():
    rng = np.random.default_rng(0)
    ema = rng.normal(size=(5000, N_EMA)) * STD + MEAN
    ema[:, 5] += 0.8 * (ema[:, 4] - MEAN[4])  # TT is correlated, so its principal direction is oblique
    return sensor_directions(ema, MEAN, STD)


def edit(sensor="TD", direction="x", profile="const", sign=1.0):
    amp = 0.5 * sign if profile == "dyn" else sign
    e = {"type": "xy", "sensor": sensor, "direction": direction, "profile": profile, "sign": sign, "amp": amp}
    e["name"] = xy_edit_name(e)
    return e


SPEC = cfg_with().eval_features.probes_xy


# ---------------------------------------------------------------------------------------------------------------------
# sensors and directions


def test_sensor_channel_pairs_follow_the_ema_order():
    assert SENSORS == {"TD": (0, 1), "TB": (2, 3), "TT": (4, 5), "LI": (6, 7), "UL": (8, 9), "LL": (10, 11)}
    for name, (cx, cy) in SENSORS.items():
        assert EMA_NAMES[cx] == name + "X" and EMA_NAMES[cy] == name + "Y"


def test_directions_are_orthonormal_and_major_is_the_principal_axis():
    dirs = directions()
    for sensor, d in dirs.items():
        for v in d.values():
            assert np.linalg.norm(v) == pytest.approx(1.0)
        assert d["major"] @ d["minor"] == pytest.approx(0.0, abs=1e-12)
        assert d["x"].tolist() == [1.0, 0.0] and d["y"].tolist() == [0.0, 1.0]
        assert max(abs(d["major"])) == pytest.approx(abs(d["major"]).max()) and d["major"][np.argmax(abs(d["major"]))] > 0
    tt = dirs["TT"]["major"]  # correlated X and Y: the principal axis is oblique with equal signs
    assert tt[0] > 0.5 and tt[1] > 0.5
    assert dirs["TD"]["major"].shape == (2,)


# ---------------------------------------------------------------------------------------------------------------------
# edit functions


@pytest.mark.parametrize("profile", ["const", "speech", "dyn", "bump"])
@pytest.mark.parametrize("direction", ["x", "y", "major", "minor"])
def test_edit_changes_only_the_sensor_channels(profile, direction):
    x = make_features()
    before = x.clone()
    dirs = directions()
    y = apply_xy_edit(x, edit("LI", direction, profile, -1.0), STATS, dirs, SPEC)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    other = [c for c in range(15) if c not in (6, 7)]
    assert torch.equal(x[:, other], y[:, other])
    assert (y[:, [6, 7]] != x[:, [6, 7]]).any()


@pytest.mark.parametrize("direction", ["x", "y", "major", "minor"])
def test_constant_edits_have_equal_euclidean_norm_in_std_units(direction):
    x = make_features()
    dirs = directions()
    for sensor, (cx, cy) in SENSORS.items():
        y = apply_xy_edit(x, edit(sensor, direction, "const", 1.0), STATS, dirs, SPEC)
        dz = (y[:, [cx, cy]] - x[:, [cx, cy]]).numpy() / STD[[cx, cy]][None, :, None]
        norms = np.linalg.norm(dz, axis=1)  # (B, T)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
        mean_dir = dz.mean(axis=2)
        np.testing.assert_allclose(mean_dir, np.broadcast_to(dirs[sensor][direction], mean_dir.shape), atol=1e-5)


def test_x_only_edit_equals_the_single_channel_probe_protocol():
    x = make_features()
    y = apply_xy_edit(x, edit("TD", "x", "const", -1.0), STATS, directions(), SPEC)
    torch.testing.assert_close(y[:, 0] - x[:, 0], torch.full_like(x[:, 0], -STD[0]), atol=1e-6, rtol=0)
    assert torch.equal(y[:, 1], x[:, 1])


def test_speech_only_edit_is_zero_in_silence_and_full_in_speech():
    x = make_features(1)
    y = apply_xy_edit(x, edit("TB", "y", "speech", 1.0), STATS, directions(), SPEC)
    d = (y[0, 3] - x[0, 3]).numpy() / STD[3]
    assert np.all(d[:10] == 0) and np.all(d[-10:] == 0)  # far inside the silence
    np.testing.assert_allclose(d[40:80], 1.0, atol=1e-5)
    assert np.all((d >= -1e-6) & (d <= 1 + 1e-6))
    assert 0 < d[18] < 1 or 0 < d[19] < 1 or 0 < d[20] < 1 or d[21] > 0  # a ramp at the onset


def test_dynamic_scaling_scales_the_component_and_keeps_the_utterance_mean():
    x = make_features(2)
    dirs = directions()
    for alpha in (0.5, -0.5):
        for direction in ("x", "major", "minor"):
            e = edit("TT", direction, "dyn", np.sign(alpha))
            y = apply_xy_edit(x, e, STATS, dirs, SPEC)
            u = dirs["TT"][direction]
            for b in range(2):
                z0 = x[b, [4, 5]].numpy() / STD[[4, 5]][:, None]
                z1 = y[b, [4, 5]].numpy() / STD[[4, 5]][:, None]
                q0, q1 = u @ z0, u @ z1
                np.testing.assert_allclose(q1.mean(), q0.mean(), atol=1e-5)  # the mean along u is kept
                np.testing.assert_allclose(q1 - q1.mean(), (1 + alpha) * (q0 - q0.mean()), atol=1e-5)
                v = np.array([-u[1], u[0]])
                np.testing.assert_allclose(v @ z1, v @ z0, atol=1e-5)  # the orthogonal component is untouched
                np.testing.assert_allclose((z1 - z0).mean(1), 0.0, atol=1e-5)


def test_bumps_are_three_hann_windows_on_speech_frames():
    x = make_features(1, 200)
    x[0, LOUDNESS_CHANNEL] = 0.01
    x[0, LOUDNESS_CHANNEL, 30:190] = 1.0
    y = apply_xy_edit(x, edit("UL", "x", "bump", 1.0), STATS, directions(), SPEC)
    d = (y[0, 8] - x[0, 8]).numpy() / STD[8]
    assert d.max() == pytest.approx(1.0, abs=1e-5) and d.min() >= 0
    assert len(bump_window(300)) == 15 and bump_window(300).max() == pytest.approx(1.0)
    nz = np.flatnonzero(d > 0)
    assert nz.min() >= 30 - 7 and nz.max() <= 190 + 7  # centred on speech frames
    peaks = [i for i in range(1, 199) if d[i] >= d[i - 1] and d[i] > d[i + 1] and d[i] > 0.99]
    assert len(peaks) == 3
    mask = speech_mask(x[0, LOUDNESS_CHANNEL].numpy())
    assert bump_centers(mask, [0.2, 0.5, 0.8], 7) == peaks or len(peaks) == 3
    assert (np.diff(peaks) > 15).all()  # no overlap
    assert np.count_nonzero(d[:23] > 0) == 0  # nothing before the first window


def test_short_utterance_bump_stays_inside_the_utterance():
    x = make_features(1, 12)
    y = apply_xy_edit(x, edit("LL", "y", "bump", 1.0), STATS, directions(), SPEC)
    assert y.shape == x.shape and np.isfinite(y.numpy()).all()


def test_speech_mask_and_weight():
    loud = np.concatenate([np.full(20, 0.01), np.full(30, 1.0), np.full(3, 0.01), np.full(30, 1.0), np.full(20, 0.01)])
    mask = speech_mask(loud, 20.0, 5)
    assert mask[20:83].all() and not mask[:20].any() and not mask[83:].any()  # the 3-frame gap is closed
    assert not speech_mask(loud, 20.0, 2)[50:53].any()  # a gap longer than gap_frames stays open
    w = speech_weight(mask, 5)
    assert w.min() >= 0 and w.max() <= 1 and w[40] == pytest.approx(1.0) and w[5] == 0
    assert speech_weight(mask, 1).tolist() == mask.astype(float).tolist()


def test_edit_rejects_bad_input():
    x = make_features()
    with pytest.raises(ValueError):
        apply_xy_edit(x[:, :14], edit(), STATS, directions(), SPEC)


def test_edit_grid_and_names():
    edits = xy_edits(cfg_with())
    names = [e["name"] for e in edits]
    assert len(names) == len(set(names)) == 6 * 4 * 4 * 2 == 192
    assert "xy_TD_major_dyn_m" in names and "xy_LL_minor_bump_p" in names and "xy_TT_x_const_p" in names
    assert {e["amp"] for e in edits if e["profile"] == "dyn"} == {-0.5, 0.5}
    assert {e["amp"] for e in edits if e["profile"] != "dyn"} == {-1.0, 1.0}
    small = xy_edits(cfg_with(**{"eval_features.probes_xy.only": ["xy_TD_x_*", "*_minor_dyn_p"]}))
    assert len(small) == 8 + 6 and all(e["name"].startswith("xy_TD_x_") or e["name"].endswith("minor_dyn_p") for e in small)


# ---------------------------------------------------------------------------------------------------------------------
# metrics


def synthetic_response(dt: np.ndarray, cols, gain=0.5, rotation_deg=0.0, leak=0.0, rng=None):
    """dm (n, 12) = rotated and scaled dt in the sensor channels plus leakage noise in the other channels."""
    rng = rng or np.random.default_rng(0)
    th = np.radians(rotation_deg)
    rot = gain * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dm = np.zeros((len(dt), N_EMA))
    dm[:, list(cols)] = dt @ rot.T
    others = [c for c in range(N_EMA) if c not in cols]
    dm[:, others] = leak * rng.normal(size=(len(dt), len(others)))
    return dm


def test_gain_formula_recovers_a_known_gain_angle_and_leakage():
    rng = np.random.default_rng(1)
    t = 400
    cols = SENSORS["TB"]
    dt = np.outer(rng.normal(size=t), [0.6, 0.8])  # varying amplitude along a fixed direction
    mask = rng.random(t) > 0.3
    for gain, rot in [(0.1, 0.0), (0.5, 30.0), (1.0, -20.0), (-0.2, 0.0)]:
        dm = synthetic_response(dt, cols, gain, rot, leak=0.0)
        sp, ot = xy_moments(dt, dm, cols, mask)
        m = metrics_from_moments(sp + ot, cols)
        assert m["gain"] == pytest.approx(gain * np.cos(np.radians(rot)), abs=1e-9)
        assert m["gain_perp"] == pytest.approx(gain * np.sin(np.radians(rot)), abs=1e-9)
        assert m["leakage"] == pytest.approx(0.0, abs=1e-12)
        assert m["own_rms"] == pytest.approx(abs(gain), abs=1e-9)
        if gain > 0:
            assert m["angle_deg"] == pytest.approx(rot, abs=1e-6)
    dm = synthetic_response(dt, cols, 1.0, 0.0, leak=0.2, rng=rng)
    sp, ot = xy_moments(dt, dm, cols, mask)
    m = metrics_from_moments(sp + ot, cols)
    assert m["leakage"] == pytest.approx(0.2 / np.sqrt((dt**2).sum(1).mean()), rel=0.05)  # RMS over 10 channels / RMS target


def test_moments_split_by_mask_and_add_up():
    rng = np.random.default_rng(2)
    cols = SENSORS["TT"]
    dt = rng.normal(size=(100, 2))
    dm = synthetic_response(dt, cols, 0.3, 0.0, leak=0.1, rng=rng)
    mask = np.arange(100) < 40
    sp, ot = xy_moments(dt, dm, cols, mask)
    assert sp[0] == 40 and ot[0] == 60
    full, _ = xy_moments(dt, dm, cols, np.ones(100, dtype=bool))
    np.testing.assert_allclose(sp + ot, full, atol=1e-9)
    assert metrics_from_moments(sp, cols)["gain"] == pytest.approx(0.3)
    assert np.isnan(metrics_from_moments(xy_moments(dt, dm, cols, np.ones(100, dtype=bool))[1], cols)["gain"])  # no frames


def test_regression_prediction_from_moments_matches_applying_the_matrix():
    rng = np.random.default_rng(3)
    cols = SENSORS["LI"]
    B = np.eye(N_EMA) * 0.9 + 0.05 * rng.normal(size=(N_EMA, N_EMA))
    M = regression_in_std_units(B, STD)
    dt_raw = rng.normal(size=(200, 2)) * STD[list(cols)]
    dt = dt_raw / STD[list(cols)]
    full_raw = np.zeros((200, N_EMA))
    full_raw[:, list(cols)] = dt_raw
    dm_raw = full_raw @ B.T  # y_c' = sum_c B[c', c] x_c
    dm = dm_raw / STD[None, :]
    mask = rng.random(200) > 0.5
    sp, ot = xy_moments(dt, dm, cols, mask)
    direct = metrics_from_moments(sp + ot, cols)
    pred = metrics_from_moments(sp + ot, cols, pred_matrix=M)
    for k in ("gain", "gain_perp", "angle_deg", "leakage", "own_rms"):
        assert pred[k] == pytest.approx(direct[k], rel=1e-9, abs=1e-12)


def test_measure_xy_truncates_to_the_common_length_and_reports_all_columns():
    rng = np.random.default_rng(4)
    cols = SENSORS["UL"]
    t = 60
    dt_raw = rng.normal(size=(t, 2)) * STD[list(cols)]
    base = rng.normal(size=(t - 1, 15)).astype(np.float32)  # re-extraction is one frame shorter than the input
    edited = base.copy()
    edited[:, list(cols)] += (0.4 * dt_raw[: t - 1]).astype(np.float32)
    mask = np.arange(t) > 10
    row = measure_xy(base, edited, dt_raw, mask, edit("UL", "x", "const"), STD)
    assert row["n_frames"] == t - 1 and row["gain"] == pytest.approx(0.4, abs=1e-5)
    assert row["leakage"] == pytest.approx(0.0, abs=1e-5) and row["angle_deg"] == pytest.approx(0.0, abs=1e-3)
    assert len(row["mom_speech"]) == len(row["mom_other"]) == probes_xy.MOMENT_LEN
    assert row["gain_speech"] == pytest.approx(0.4, abs=1e-5) and row["gain_other"] == pytest.approx(0.4, abs=1e-5)


# ---------------------------------------------------------------------------------------------------------------------
# driver: toy vocoder that carries the EMA through the waveform with a gain of 0.5


class ToyModule(nn.Module):
    """First 12 samples of every 480-sample frame hold the EMA channels; the rest is silent."""

    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(1, 1)
        self.speaker = nn.Linear(1, 1)

    def synthesize(self, batch):
        f = batch["features"]  # (1, 15, T)
        wav = torch.zeros(f.shape[0], f.shape[2], 480)
        wav[:, :, :N_EMA] = f[:, :N_EMA].transpose(1, 2)
        return wav.reshape(f.shape[0], 1, -1)


class ToyExtractor:
    def __init__(self, cfg, head, device):
        self.head = head

    def describe(self):
        return {"head": self.head}

    def extract(self, wav, sr):
        frames = len(wav) // 480 - 1  # one frame shorter, like the real re-extraction
        feats = np.zeros((frames, 15), dtype=np.float32)
        feats[:, :N_EMA] = 0.5 * wav[: frames * 480].reshape(frames, 480)[:, :N_EMA]
        return {"feats": feats, "loud_raw": np.ones(frames, dtype=np.float32), "T": frames}


@pytest.fixture
def driver_cfg(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(STATS))
    vcfg = OmegaConf.create({"stats_path": str(stats_path)})
    ids = [f"spk{i}_utt" for i in range(3)]
    batches = [
        {"id": uid, "features": make_features(1, 80, seed=i), "spk_raw": torch.zeros(1, 1024), "condition": "T1", "ref_id": uid}
        for i, uid in enumerate(ids)
    ]
    loads = []

    def load_vocoder(experiment, vocoder, ckpt, device, overrides=()):
        loads.append(vocoder)
        module = ToyModule()
        module.checkpoint_path, module.checkpoint_step = "toy.ckpt", 7
        return module, vcfg

    monkeypatch.setattr("sparc.vocoders.eval.vocoder_loader.load_vocoder", load_vocoder)
    monkeypatch.setattr("sparc.vocoders.eval.reextract.ReExtractor", ToyExtractor)
    from sparc.vocoders.eval import probes

    monkeypatch.setattr(probes, "probe_subset", lambda cfg, split="test.clean": ids)
    monkeypatch.setattr(probes, "probe_batches", lambda vcfg, ids, device, split: [b for b in batches if b["id"] in ids])
    ema = np.random.default_rng(0).normal(size=(3000, N_EMA)) * STD + MEAN
    monkeypatch.setattr(probes_xy, "training_ema", lambda cfg: ema)
    cfg = cfg_with(
        **{
            "paths.cache_root": str(tmp_path),
            "eval.systems": {"toy": {"kind": "vocoder", "experiment": "main", "vocoder": "toy"}},
            "eval_features.probes_xy.sensors": ["TD", "TT"],
            "eval_features.probes_xy.directions": ["x", "major"],
            "eval_features.probes_xy.profiles": ["const", "dyn"],
        }
    )
    return cfg, tmp_path / "out", loads


def test_run_probes_xy_writes_resumes_stops_and_recovers_the_toy_gain(driver_cfg):
    cfg, out, loads = driver_cfg
    n_edits = len(xy_edits(cfg))
    assert n_edits == 2 * 2 * 2 * 2
    calls = {"n": 0}

    def stop_after_three() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    probes_xy.run_probes_xy(cfg, "toy", "cpu", out, stop_after_three)
    assert len(list((out / "edits").glob("*.parquet"))) == 3 and not (out / "results.parquet").exists()
    first = {p.name: p.stat().st_mtime_ns for p in (out / "edits").glob("*.parquet")}
    probes_xy.run_probes_xy(cfg, "toy", "cpu", out, lambda: False)
    assert all((out / "edits" / n).stat().st_mtime_ns == m for n, m in first.items())
    results = pd.read_parquet(out / "results.parquet")
    assert len(results) == 3 * n_edits and list(results["edit"].unique()) == [e["name"] for e in xy_edits(cfg)]
    assert results["gain"].to_numpy() == pytest.approx(0.5, abs=1e-5)  # the toy path reproduces half of every edit
    assert results["leakage"].abs().max() < 1e-5 and results["angle_deg"].abs().max() < 1e-3
    assert (out / "wav" / "spk0_utt" / "unedited.wav").exists() and (out / "wav" / "spk0_utt" / "xy_TD_x_const_p.wav").exists()
    assert not (out / "wav" / "spk1_utt").exists()
    meta = json.loads((out / "probes_xy_meta.json").read_text())
    assert meta["checkpoint"] == {"ckpt": "toy.ckpt", "g_step": 7} and len(meta["edits"]) == n_edits
    assert set(meta["directions"]) == set(SENSORS) and len(meta["ema_std"]) == N_EMA
    assert not list(out.rglob("*.tmp"))
    loads.clear()
    probes_xy.run_probes_xy(cfg, "toy", "cpu", out, lambda: False)
    assert loads == []


def test_run_probes_xy_limit_keeps_a_subset_of_the_utterances(driver_cfg):
    cfg, out, _ = driver_cfg
    probes_xy.run_probes_xy(cfg, "toy", "cpu", out, lambda: False, limit=1)
    assert pd.read_parquet(out / "results.parquet")["id"].nunique() == 1


def test_stage_is_registered():
    from sparc.vocoders.eval import stages
    from sparc.vocoders.eval.systems import STAGE_KINDS, STAGES

    assert "probes_xy" in STAGES and STAGE_KINDS["probes_xy"] == ("vocoder",) and "probes_xy" in stages.SYSTEM_STAGES


def test_metrics_vec_matches_scalar_metrics_on_batches_and_marks_empty_cells():
    rng = np.random.default_rng(5)
    cols = SENSORS["TD"]
    moms = []
    for _ in range(4):
        dt = rng.normal(size=(50, 2))
        dm = synthetic_response(dt, cols, 0.3, 10.0, leak=0.1, rng=rng)
        sp, ot = xy_moments(dt, dm, cols, rng.random(50) > 0.5)
        moms.append(sp + ot)
    batch = np.stack(moms)  # (4, MOMENT_LEN)
    vec = probes_xy.metrics_vec(batch, cols)
    for i in range(4):
        scalar = metrics_from_moments(batch[i], cols)
        for k, v in scalar.items():
            assert vec[k][i] == pytest.approx(v)
    M = regression_in_std_units(np.eye(N_EMA) * 0.7, STD)
    pred = probes_xy.metrics_vec(batch.reshape(2, 2, -1), cols, M)
    assert pred["gain"].shape == (2, 2) and np.allclose(pred["gain"], 0.7) and np.allclose(pred["leakage"], 0.0)
    empty = probes_xy.metrics_vec(np.zeros(probes_xy.MOMENT_LEN), cols)
    assert all(np.isnan(v) for v in empty.values())


def test_sensor_variances_are_the_eigenvalues_behind_the_directions():
    rng = np.random.default_rng(6)
    ema = rng.normal(size=(4000, N_EMA)) * STD + MEAN
    ema[:, 1] = 0.6 * (ema[:, 0] - MEAN[0]) / STD[0] * STD[1] + MEAN[1] + 0.3 * rng.normal(size=4000)
    var = probes_xy.sensor_variances(ema, MEAN, STD)
    assert set(var) == set(SENSORS) and all(v[0] >= v[1] > 0 for v in var.values())
    assert var["TD"][0] > 1.2 and var["TB"][0] == pytest.approx(1.0, abs=0.1)
    z = (ema[:, [0, 1]] - MEAN[[0, 1]]) / STD[[0, 1]]
    major = sensor_directions(ema, MEAN, STD)["TD"]["major"]
    assert np.var(z @ major, ddof=1) == pytest.approx(var["TD"][0], rel=1e-9)
