"""Controllability probes: edit functions, subset selection, per-edit measures and the resumable driver (CPU)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch import nn

import sparc
from sparc.vocoders.constants import EMA_NAMES, F0_CHANNEL, LOUDNESS_CHANNEL, N_EMA, PERIODICITY_CHANNEL, SAMPLE_RATE
from sparc.vocoders.eval import probes
from sparc.vocoders.eval.probes import apply_edit, edit_name, edit_target, measure_edit, probe_edits, probe_subset

CONF = Path(sparc.__file__).parent / "conf"
STATS = {"ema_std": [0.5 + 0.1 * c for c in range(N_EMA)]}


def eval_features_cfg(**overrides):
    group = OmegaConf.load(CONF / "eval_features" / "default.yaml")
    cfg = OmegaConf.create({"eval_features": group})
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def make_features(batch: int = 2, frames: int = 30, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 15, frames, generator=g)
    x[:, F0_CHANNEL] = 100.0 + 50.0 * torch.rand(batch, frames, generator=g)
    x[:, LOUDNESS_CHANNEL] = 0.01 + 0.1 * torch.rand(batch, frames, generator=g)
    voiced = torch.rand(batch, frames, generator=g) > 0.4
    x[:, PERIODICITY_CHANNEL] = torch.where(voiced, 0.4 + 0.5 * torch.rand(batch, frames, generator=g), torch.zeros(batch, frames))
    return x


def changed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a != b)


# ---------------------------------------------------------------------------------------------------------------------
# edit functions


def test_f0_edit_changes_only_voiced_f0_frames():
    x = make_features()
    before = x.clone()
    y = apply_edit(x, {"type": "f0", "semitones": 2.0}, STATS)
    torch.testing.assert_close(x, before, rtol=0, atol=0)  # input untouched
    assert y is not x and y.data_ptr() != x.data_ptr()
    voiced = x[:, PERIODICITY_CHANNEL] > 0
    diff = changed(x, y)
    assert not diff[:, [c for c in range(15) if c != F0_CHANNEL]].any()
    assert torch.equal(diff[:, F0_CHANNEL], voiced)
    torch.testing.assert_close(y[:, F0_CHANNEL][voiced], x[:, F0_CHANNEL][voiced] * 2 ** (2 / 12))
    assert torch.equal(y[:, F0_CHANNEL][~voiced], x[:, F0_CHANNEL][~voiced])  # unvoiced frames keep their F0
    down = apply_edit(x, {"type": "f0", "semitones": -4.0}, STATS)
    torch.testing.assert_close(down[:, F0_CHANNEL][voiced], x[:, F0_CHANNEL][voiced] * 2 ** (-4 / 12))


def test_loudness_edit_scales_only_the_loudness_channel():
    x = make_features()
    y = apply_edit(x, {"type": "loudness", "scale": 0.5}, STATS)
    assert not changed(x, y)[:, [c for c in range(15) if c != LOUDNESS_CHANNEL]].any()
    torch.testing.assert_close(y[:, LOUDNESS_CHANNEL], 0.5 * x[:, LOUDNESS_CHANNEL])
    torch.testing.assert_close(apply_edit(x, {"type": "loudness", "scale": 2.0}, STATS)[:, LOUDNESS_CHANNEL], 2 * x[:, LOUDNESS_CHANNEL])


def test_periodicity_edit_zeroes_only_the_periodicity_channel():
    x = make_features()
    y = apply_edit(x, {"type": "periodicity", "value": 0.0}, STATS)
    assert not changed(x, y)[:, :PERIODICITY_CHANNEL].any()
    assert (y[:, PERIODICITY_CHANNEL] == 0).all()
    assert (x[:, PERIODICITY_CHANNEL] > 0).any()  # the test data has voiced frames


@pytest.mark.parametrize("channel", [0, 5, 11])
@pytest.mark.parametrize("multiple", [-1.0, 1.0])
def test_ema_edit_moves_one_channel_by_its_training_std(channel, multiple):
    x = make_features()
    before = x.clone()
    y = apply_edit(x, {"type": "ema", "channel": channel, "multiple": multiple}, STATS)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    other = [c for c in range(15) if c != channel]
    assert not changed(x, y)[:, other].any()
    torch.testing.assert_close(y[:, channel] - x[:, channel], torch.full_like(x[:, channel], multiple * STATS["ema_std"][channel]))


def test_edit_rejects_bad_input():
    x = make_features()
    with pytest.raises(ValueError):
        apply_edit(x[:, :14], {"type": "loudness", "scale": 2.0}, STATS)
    with pytest.raises(ValueError):
        apply_edit(x, {"type": "pitch"}, STATS)
    with pytest.raises(ValueError):
        apply_edit(x, {"type": "ema", "channel": 12, "multiple": 1.0}, STATS)


def test_edit_list_from_config_matches_the_contract():
    edits = probe_edits(eval_features_cfg())
    names = [e["name"] for e in edits]
    assert len(names) == len(set(names)) == 4 + 2 + 1 + 2 * N_EMA
    assert {"f0_m4", "f0_m2", "f0_p2", "f0_p4", "loud_x0.5", "loud_x2", "per_zero"} <= set(names)
    assert "ema_TDX_p1" in names and "ema_LLY_m1" in names
    ema = [e for e in edits if e["type"] == "ema"]
    assert sorted({e["channel"] for e in ema}) == list(range(N_EMA)) and {e["multiple"] for e in ema} == {-1.0, 1.0}
    assert edit_name({"type": "f0", "semitones": 4.0}) == "f0_p4"
    assert edit_target({"type": "f0", "semitones": -2.0}, STATS["ema_std"]) == -200.0
    assert edit_target({"type": "loudness", "scale": 2.0}, STATS["ema_std"]) == pytest.approx(6.0206, abs=1e-3)
    assert edit_target({"type": "ema", "channel": 2, "multiple": -1.0}, STATS["ema_std"]) == pytest.approx(-0.7)


# ---------------------------------------------------------------------------------------------------------------------
# subset


def write_index(root: Path, split: str = "test.clean", n_speakers: int = 12, per_speaker: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for s in range(n_speakers):
        for j in range(per_speaker):
            seconds = float(rng.uniform(1.0, 14.0))
            rows.append({"id": f"{100 + s}_{7}_{j:06d}_000000", "speaker": str(100 + s), "n24": int(seconds * SAMPLE_RATE)})
    index = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffled rows
    directory = root / "packed" / split
    directory.mkdir(parents=True)
    index.to_parquet(directory / "index.parquet")
    return index


def test_probe_subset_rules(tmp_path):
    index = write_index(tmp_path)
    cfg = eval_features_cfg(**{"paths.cache_root": str(tmp_path), "eval_features.probes.n_utterances": 20})
    ids = probe_subset(cfg)
    assert ids == sorted(ids) and len(ids) == len(set(ids)) == 20
    chosen = index.set_index("id").loc[ids]
    assert ((chosen["n24"] / SAMPLE_RATE >= 3.0) & (chosen["n24"] / SAMPLE_RATE <= 10.0)).all()
    assert chosen["speaker"].value_counts().max() <= 2
    assert probe_subset(cfg) == ids  # deterministic
    # the stated rule: sorted candidates, default_rng(0) permutation, at most 2 per speaker
    cand = index[(index["n24"] / SAMPLE_RATE >= 3.0) & (index["n24"] / SAMPLE_RATE <= 10.0)].sort_values("id")
    order = np.random.default_rng(0).permutation(len(cand))
    expected, count = [], {}
    for i in order:
        sp = cand["speaker"].iloc[i]
        if count.get(sp, 0) >= 2:
            continue
        expected.append(cand["id"].iloc[i])
        count[sp] = count.get(sp, 0) + 1
        if len(expected) == 20:
            break
    assert ids == sorted(expected)
    other = eval_features_cfg(**{"paths.cache_root": str(tmp_path), "eval_features.probes.n_utterances": 20, "eval_features.probes.seed": 1})
    assert probe_subset(other) != ids


def test_probe_subset_returns_fewer_when_the_pool_is_small(tmp_path):
    write_index(tmp_path, n_speakers=3, per_speaker=9)
    cfg = eval_features_cfg(**{"paths.cache_root": str(tmp_path)})  # 50 requested, at most 3 x 2 available
    assert len(probe_subset(cfg)) <= 6


# ---------------------------------------------------------------------------------------------------------------------
# measures


def make_reextraction(frames: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(frames, 15)).astype(np.float32)
    feats[:, F0_CHANNEL] = 150 + 20 * np.sin(np.arange(frames) / 9)
    feats[:, PERIODICITY_CHANNEL] = np.where(np.arange(frames) % 5 == 0, 0.0, 0.8)
    loud = rng.uniform(0.01, 0.1, size=frames).astype(np.float32)
    return feats, loud


def test_measure_edit_known_values():
    feats, loud = make_reextraction()
    std = np.array(STATS["ema_std"])
    shifted = feats.copy()
    shifted[:, F0_CHANNEL] *= 2 ** (200 / 1200)
    row = measure_edit(feats, loud, shifted, loud, {"type": "f0", "semitones": 2.0}, std)
    assert row["f0_shift_median_cents"] == pytest.approx(200.0, abs=1e-3) and row["f0_shift_mean_cents"] == pytest.approx(200.0, abs=1e-3)
    assert row["n_voiced_both"] == (feats[:, PERIODICITY_CHANNEL] > 0).sum()
    assert np.isnan(row["ema_gain"]) and np.isnan(row["ema_leakage"]) and row["ema_shift_all"] == 0.0
    assert row["loud_shift_db_median"] == pytest.approx(0.0, abs=1e-9)

    row = measure_edit(feats, loud, feats, loud * 2, {"type": "loudness", "scale": 2.0}, std)
    assert row["loud_shift_db_median"] == pytest.approx(6.02, abs=0.05)

    zeroed = feats.copy()
    zeroed[:, PERIODICITY_CHANNEL] = 0.0
    row = measure_edit(feats, loud, zeroed, loud, {"type": "periodicity", "value": 0.0}, std)
    assert row["voiced_frac_edit"] == 0.0 and row["voiced_frac_base"] == pytest.approx(0.8)
    assert np.isnan(row["f0_shift_median_cents"]) and row["n_voiced_both"] == 0.0


def test_measure_edit_ema_gain_and_leakage():
    feats, loud = make_reextraction()
    std = np.array(STATS["ema_std"])
    c, s = 3, -1.0
    moved = feats.copy()
    moved[:, c] += 0.8 * s * std[c]  # the vocoder reproduces 80 % of the edit
    for other in range(N_EMA):
        if other != c:
            moved[:, other] += 0.1 * std[other]  # and leaks 0.1 std everywhere else
    row = measure_edit(feats, loud, moved, loud, {"type": "ema", "channel": c, "multiple": s}, std)
    assert row["ema_gain"] == pytest.approx(0.8, abs=1e-6)
    assert row["ema_leakage"] == pytest.approx(0.1, abs=1e-6)
    assert row["ema_shift_all"] == pytest.approx((0.8 + 11 * 0.1) / 12, abs=1e-6)


def test_measure_edit_truncates_to_the_shorter_stream():
    feats, loud = make_reextraction(frames=100)
    row = measure_edit(feats, loud, feats[:98], loud[:98], {"type": "periodicity", "value": 0.0}, np.ones(12))
    assert row["n_frames"] == 98


# ---------------------------------------------------------------------------------------------------------------------
# driver: resume, stop, outputs (stubs for the vocoder, the data and the extractor)


class ToyModule(nn.Module):
    """``synthesize`` maps loudness and F0 of the features to a 480-samples-per-frame tone."""

    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(1, 1)
        self.speaker = nn.Linear(1, 1)

    def synthesize(self, batch):
        f = batch["features"]
        voiced = (f[:, PERIODICITY_CHANNEL] > 0).float()  # unvoiced frames are silent, so zeroing periodicity silences all
        wav = (voiced * f[:, LOUDNESS_CHANNEL] * torch.sin(f[:, F0_CHANNEL] / 50.0)).repeat_interleave(480, dim=-1)
        return wav[:, None]


class ToyExtractor:
    """Re-extraction stub: the features of a waveform are simple functions of it (no model)."""

    def __init__(self, cfg, head, device):
        self.head = head

    def describe(self):
        return {"head": self.head}

    def extract(self, wav, sr):
        frames = len(wav) // 480 - 1
        chunks = wav[: frames * 480].reshape(frames, 480)
        feats = np.zeros((frames, 15), dtype=np.float32)
        feats[:, :N_EMA] = np.arange(N_EMA)[None] * 0.0 + chunks.mean(1, keepdims=True)
        feats[:, F0_CHANNEL] = 100.0
        feats[:, PERIODICITY_CHANNEL] = np.where(np.abs(chunks).mean(1) > 1e-6, 0.9, 0.0)
        return {"feats": feats, "loud_raw": np.abs(chunks).mean(1).astype(np.float32), "T": frames}


@pytest.fixture
def driver_cfg(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(STATS))
    vcfg = OmegaConf.create({"stats_path": str(stats_path)})
    ids = [f"spk{i}_utt" for i in range(4)]
    batches = []
    for i, uid in enumerate(ids):
        features = make_features(1, 25, seed=i)
        features[:, LOUDNESS_CHANNEL] = features[:, LOUDNESS_CHANNEL].abs() + 0.05
        batches.append({"id": uid, "features": features, "spk_raw": torch.zeros(1, 1024), "condition": "T1", "ref_id": uid})
    loads = []

    def load_vocoder(experiment, vocoder, ckpt, device, overrides=()):
        loads.append((experiment, vocoder))
        module = ToyModule()
        module.checkpoint_path, module.checkpoint_step = "toy.ckpt", 7
        return module, vcfg

    monkeypatch.setattr("sparc.vocoders.eval.vocoder_loader.load_vocoder", load_vocoder)
    monkeypatch.setattr("sparc.vocoders.eval.reextract.ReExtractor", ToyExtractor)
    monkeypatch.setattr(probes, "probe_subset", lambda cfg, split="test.clean": ids)
    monkeypatch.setattr(probes, "probe_batches", lambda vcfg, ids, device, split: batches)
    cfg = eval_features_cfg(
        **{
            "paths.cache_root": str(tmp_path),
            "eval.systems": {"toy": {"kind": "vocoder", "experiment": "main", "vocoder": "toy"}},
            "eval_features.probes.example_utterances": 2,
        }
    )
    return cfg, tmp_path / "out", loads


def test_run_probes_writes_resumes_and_stops(driver_cfg):
    cfg, out, loads = driver_cfg
    n_edits = len(probe_edits(cfg))
    calls = {"n": 0}

    def stop_after_three() -> bool:
        calls["n"] += 1
        return calls["n"] > 3  # the stop flag is polled once per edit: three edits finish, the fourth is not started

    probes.run_probes(cfg, "toy", "cpu", out, stop_after_three)
    done = sorted(p.stem for p in (out / "edits").glob("*.parquet"))
    assert len(done) == 3 and not (out / "results.parquet").exists()
    first = {p.name: p.stat().st_mtime_ns for p in (out / "edits").glob("*.parquet")}

    probes.run_probes(cfg, "toy", "cpu", out, lambda: False)
    assert len(list((out / "edits").glob("*.parquet"))) == n_edits
    assert all((out / "edits" / name).stat().st_mtime_ns == mtime for name, mtime in first.items())  # skipped, not redone
    results = pd.read_parquet(out / "results.parquet")
    assert len(results) == 4 * n_edits and set(results["id"]) == {f"spk{i}_utt" for i in range(4)}
    assert list(results["edit"].unique()) == [e["name"] for e in probe_edits(cfg)]
    assert not results.drop(columns=["ema_gain", "ema_leakage"]).isna().all().any()
    assert (out / "wav" / "spk0_utt" / "f0_p2.wav").exists() and (out / "wav" / "spk1_utt" / "unedited.wav").exists()
    assert not (out / "wav" / "spk2_utt").exists()  # only the first example_utterances utterances
    audio, rate = sf.read(out / "wav" / "spk0_utt" / "unedited.wav")
    assert rate == SAMPLE_RATE and len(audio) == 25 * 480
    meta = json.loads((out / "probes_meta.json").read_text())
    assert meta["checkpoint"] == {"ckpt": "toy.ckpt", "g_step": 7} and len(meta["edits"]) == n_edits
    assert not list(out.rglob("*.tmp"))  # atomic writes leave no temporary files

    loads.clear()
    probes.run_probes(cfg, "toy", "cpu", out, lambda: False)  # everything done: nothing is loaded or recomputed
    assert loads == []


def test_run_probes_row_content(driver_cfg):
    cfg, out, _ = driver_cfg
    probes.run_probes(cfg, "toy", "cpu", out, lambda: False)
    results = pd.read_parquet(out / "results.parquet")
    loud = results[results["edit"] == "loud_x2"]
    assert loud["target"].to_numpy() == pytest.approx(6.0206, abs=1e-3)
    # the toy wav scales linearly with loudness, and so does its re-extracted loud_raw
    assert loud["loud_shift_db_median"].between(5.5, 6.5).all()
    per = results[results["edit"] == "per_zero"]
    assert (per["voiced_frac_edit"] == 0.0).all() and (per["voiced_frac_base"] > 0.3).all()  # toy: unvoiced input is silent
    assert set(results.columns) >= {"id", "edit", "type", "channel", "param", "target", "ema_gain", "f0_shift_median_cents"}


# ---------------------------------------------------------------------------------------------------------------------
# slow: real subset, a trained checkpoint and the real re-extraction (CPU)


@pytest.mark.slow
def test_real_probe_subset_and_a_short_probe_run_with_a_trained_vocoder(tmp_path):
    import os

    needed = ("HF_HUB_CACHE", "SPARC_REFIT_NPZ", "SPARC_VOC_CACHE", "SPARC_VOC_RUNS", "LIBRITTSR_RAW")
    if any(name not in os.environ for name in needed):
        pytest.skip("environment of the feature cache is not set")
    cache = Path(os.environ["SPARC_VOC_CACHE"])
    ckpts = list((Path(os.environ["SPARC_VOC_RUNS"]) / "main" / "hifigan" / "ckpt").glob("step*.ckpt"))
    if not (cache / "packed" / "test.clean" / "index.parquet").exists() or not ckpts:
        pytest.skip("packed test.clean features or the main/hifigan checkpoint are missing")
    cfg = eval_features_cfg(
        **{
            "paths.cache_root": str(cache),
            "eval.systems": {"hifigan": {"kind": "vocoder", "experiment": "main", "vocoder": "hifigan"}},
        }
    )
    # the full subset: 50 utterances of 3-10 s, at most 2 per speaker, deterministic, sorted
    subset = probe_subset(cfg)
    index = pd.read_parquet(cache / "packed" / "test.clean" / "index.parquet", columns=["id", "speaker", "n24"]).set_index("id")
    assert len(subset) == 50 and subset == sorted(subset) and probe_subset(cfg) == subset
    chosen = index.loc[subset]
    assert chosen["n24"].between(3.0 * SAMPLE_RATE, 10.0 * SAMPLE_RATE).all() and chosen["speaker"].value_counts().max() <= 2

    # a short run: two utterances, four edits (the frame-level effect of each edit on a trained vocoder)
    cfg.eval_features.probes.n_utterances = 2
    cfg.eval_features.probes.f0_semitones = [4]
    cfg.eval_features.probes.loudness_scales = [2.0]
    cfg.eval_features.probes.ema_std_multiples = []
    cfg.eval_features.probes.example_utterances = 1
    out = tmp_path / "hifigan"
    probes.run_probes(cfg, "hifigan", "cpu", out, lambda: False)
    results = pd.read_parquet(out / "results.parquet")
    assert sorted(results["edit"].unique()) == ["f0_p4", "loud_x2", "per_zero"] and len(results) == 6
    print(results[["id", "edit", "target", "f0_shift_median_cents", "loud_shift_db_median", "voiced_frac_base", "voiced_frac_edit"]].to_string())
    f0 = results[results["edit"] == "f0_p4"]
    assert (f0["f0_shift_median_cents"].between(200, 600)).all()  # target +400 cents
    loud = results[results["edit"] == "loud_x2"]
    assert (loud["loud_shift_db_median"].between(3.0, 9.0)).all()  # target +6.02 dB
    per = results[results["edit"] == "per_zero"]
    assert (per["voiced_frac_edit"] < per["voiced_frac_base"]).all()
    meta = json.loads((out / "probes_meta.json").read_text())
    assert meta["checkpoint"]["g_step"] > 0 and meta["reextract"]["dither"] is False and meta["reextract"]["head"] == "refit"


@pytest.mark.slow
def test_probe_synthesis_is_the_predict_step_synthesis():
    """``probes.synthesize`` equals ``VocoderGANModule.predict_step`` on the same T1 batch (same fixed RNG, eval mode)."""
    import os

    from sparc.vocoders.eval.vocoder_loader import load_vocoder

    needed = ("HF_HUB_CACHE", "SPARC_REFIT_NPZ", "SPARC_VOC_CACHE", "SPARC_VOC_RUNS", "LIBRITTSR_RAW")
    if any(name not in os.environ for name in needed):
        pytest.skip("environment of the feature cache is not set")
    cache = Path(os.environ["SPARC_VOC_CACHE"])
    ckpts = list((Path(os.environ["SPARC_VOC_RUNS"]) / "main" / "ddsp" / "ckpt").glob("step*.ckpt"))  # DDSP draws noise: the RNG matters
    if not (cache / "packed" / "test.clean" / "index.parquet").exists() or not ckpts:
        pytest.skip("packed test.clean features or the main/ddsp checkpoint are missing")
    module, vcfg = load_vocoder("main", "ddsp", None, "cpu")
    index = pd.read_parquet(cache / "packed" / "test.clean" / "index.parquet", columns=["id", "T"])
    uid = str(index[(index["T"] > 40) & (index["T"] < 70)]["id"].iloc[0])
    (batch,) = probes.probe_batches(vcfg, [uid], "cpu", "test.clean")
    assert batch["features"].shape[:2] == (1, 15) and batch["condition"] == "T1" and batch["ref_id"] == uid
    ours = probes.synthesize(module, batch["features"], batch["spk_raw"], torch.device("cpu"))
    with torch.no_grad():
        theirs = module.predict_step({**batch, "id": [uid], "condition": ["T1"]}, 0)["wav"][0, 0].numpy()
    assert ours.shape == (batch["features"].shape[-1] * 480,)
    np.testing.assert_array_equal(ours, theirs)
    again = probes.synthesize(module, batch["features"], batch["spk_raw"], torch.device("cpu"))
    np.testing.assert_array_equal(ours, again)
