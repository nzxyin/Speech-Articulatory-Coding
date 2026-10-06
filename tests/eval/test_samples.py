"""Listening-sample selection and writing (CPU, seconds)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sparc.vocoders.eval import io, samples
from sparc.vocoders.eval.systems import load_systems
from toy_split import SEX, SPLIT, build_toy_cache, set_env, toy_config


@pytest.fixture
def toy(tmp_path, monkeypatch):
    build_toy_cache(tmp_path)
    set_env(monkeypatch, tmp_path)
    sex = io.read_speaker_sex(tmp_path / "eval" / "ref_data" / "LibriTTS_R" / "SPEAKERS.txt")
    table = io.build_utterance_table(tmp_path / "cache", SPLIT, sex=sex)
    return tmp_path, table


def big_table(n_speakers: int = 30, per_speaker: int = 6, seed: int = 0) -> pd.DataFrame:
    """A synthetic utterance table with many speakers of both sexes."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_speakers):
        for k in range(per_speaker):
            rows.append(
                {
                    "id": f"{1000 + s}_1_{k:06d}_000000",
                    "speaker": str(1000 + s),
                    "dur_s": float(rng.uniform(1.0, 14.0)),
                    "has_ref": bool(k > 0 or s % 7),
                    "sex": "F" if s % 2 else "M",
                }
            )
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def test_selection_is_deterministic_balanced_and_speaker_distinct():
    table = big_table()
    a = samples.select_samples(table, None, 10, 10, (3.0, 12.0), seed=0)
    assert a == samples.select_samples(table.sample(frac=1, random_state=3), None, 10, 10, (3.0, 12.0), seed=0)  # row order is irrelevant
    assert len(a) == 20 and len(set(a)) == 20
    rows = table.set_index("id").loc[a]
    assert rows["speaker"].is_unique
    assert (rows["sex"].tolist() == ["F"] * 10 + ["M"] * 10) and rows["dur_s"].between(3.0, 12.0).all() and rows["has_ref"].all()
    assert samples.select_samples(table, None, 10, 10, (3.0, 12.0), seed=1) != a
    # restricting the id set (a smoke run) can only choose among those ids
    subset = table["id"].iloc[::2].tolist()
    chosen = samples.select_samples(table, subset, 3, 3, (3.0, 12.0), seed=0)
    assert set(chosen) <= set(subset)


def test_selection_fails_loudly_when_a_sex_has_too_few_speakers():
    table = big_table(n_speakers=8)
    with pytest.raises(ValueError, match="only 4 F speakers"):
        samples.select_samples(table, None, 10, 10, (3.0, 12.0))
    with pytest.raises(ValueError, match="M speakers"):
        samples.select_samples(table.assign(sex="F"), None, 2, 2, (3.0, 12.0))


def test_write_samples_converts_to_pcm16_at_each_systems_rate_and_reports_gaps(toy):
    root, table = toy
    cfg = toy_config(root, "eval.samples.n_female=2", "eval.samples.n_male=2")
    systems = load_systems(cfg)
    paths = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, None)
    ids = table["id"].tolist()
    expected = samples.select_samples(table, ids, 2, 2, (3.0, 12.0), seed=0)
    rng = np.random.default_rng(0)
    frames = table.set_index("id")["T"]
    have_everything = expected[0]
    for uid in expected:
        for spec in samples.sample_systems(systems):
            for condition in spec.conditions:
                if uid != have_everything and spec.name == "ddsp":
                    continue  # ddsp was not synthesized for the other utterances
                n = io.expected_samples(int(frames[uid]), spec.sr)
                io.write_wav(paths.audio_path(spec, condition, uid), 0.4 * rng.standard_normal(n).astype(np.float32), spec.sr)
    summary = samples.write_samples(cfg, paths, systems, table, ids)
    assert summary["ids"] == expected
    out = paths.samples_dir()
    names = sorted(p.name for p in (out / have_everything).iterdir())
    want = sorted(f"{s.name}_{c}.wav" for s in samples.sample_systems(systems) for c in s.conditions)
    assert names == want  # audio_from systems (gt_shipped) have no files of their own
    assert "gt_shipped_T1.wav" not in names and "gt_T1.wav" in names and "enplus16_T2.wav" in names and "hifigan_T3.wav" in names
    info = sf.info(out / have_everything / "enplus16_T1.wav")
    assert info.subtype == "PCM_16" and info.samplerate == 16000  # at the system's own sample rate
    assert sf.info(out / have_everything / "hifigan_T1.wav").samplerate == 24000
    assert set(summary["missing"]) == set(expected) - {have_everything}
    assert "ddsp_T1" in summary["missing"][expected[1]]
    index = pd.read_csv(out / "index.csv", dtype=str, keep_default_na=False)
    assert list(index.columns) == ["id", "speaker", "sex", "duration", "transcript", "ref_id", "missing"]
    assert index["id"].tolist() == expected
    row = index.iloc[0]
    assert row["sex"] == SEX[row["speaker"]] and row["ref_id"] == table.set_index("id").loc[row["id"], "ref_id"]
    assert len(row["transcript"].split()) == 5 and row["missing"] == ""
    assert "ddsp_T2" in index.iloc[1]["missing"]
    assert io.read_json(out / "meta.json")["ids"] == expected


def test_allow_fewer_returns_what_exists_for_smoke_runs():
    table = big_table(n_speakers=8)
    chosen = samples.select_samples(table, None, 10, 10, (3.0, 12.0), allow_fewer=True)
    assert len(chosen) == 8 and table.set_index("id").loc[chosen]["speaker"].is_unique
