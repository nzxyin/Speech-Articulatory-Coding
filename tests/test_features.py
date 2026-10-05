"""Tests for the feature cache: frame counts, cache files, loudness exactness, pooling, packing and statistics (CPU)."""

import json
import os
import signal
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

import sparc
from sparc.src_extractor import AmplitudeHistogram
from sparc.vocoders.constants import EXTRACTOR_HOP, SAMPLE_RATE, feature_length
from sparc.vocoders.data.gain import gain_factor, sample_gain_db
from sparc.vocoders.features import cache, cache_module
from sparc.vocoders.features.cache_module import CacheWriter, ManifestShardDataModule, ManifestShardDataset, shard_rows
from sparc.vocoders.features.extractor import (
    SparcFeatureExtractor,
    fork_commit,
    pool_speaker,
    raw_loudness,
    set_fp32_numerics,
)
from sparc.vocoders.features.manifest import evenly_spaced, predicted_frames, select_rows
from sparc.vocoders.features.pack import pack_split
from sparc.vocoders.features.stats import Moments, compute_stats

CONF_DIR = str(Path(sparc.__file__).parent / "conf")


def synthetic_arrays(frames: int, seed: int = 0, meta_hash: str = "h") -> dict:
    rng = np.random.default_rng(seed)
    return {
        "feats": rng.normal(size=(frames, 15)).astype(np.float32),
        "loud_raw": rng.uniform(0.0, 0.1, size=frames).astype(np.float32),
        "spk_l0": rng.normal(size=1024).astype(np.float32),
        "spk_l6": rng.normal(size=1024).astype(np.float32),
        "spk_enplus64": rng.normal(size=64).astype(np.float32),
        "spk_wsum": np.float64(3.5),
        "spk_fallback": np.bool_(False),
        "n24": np.int64(480 * (frames + 1)),
        "T": np.int64(frames),
        "peak24": np.float64(0.5),
        "seed": np.int64(seed),
        "meta_hash": meta_hash,
        "gpu": "cpu",
    }


def synthetic_speech(seconds: float, seed: int = 0) -> np.ndarray:
    """Harmonic signal with moving F0, syllable-like envelope and a little noise, 24 kHz float64."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = 120.0 + 25.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    wave = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t)) ** 0.5
    return 0.2 * envelope * wave + 0.002 * rng.normal(size=len(t))


# ---------------------------------------------------------------------------------------------------------------------
# frame counts


def test_feature_length_formula():
    n24 = np.arange(9000, 14000)
    expected = n24 // 480 - 1 + (n24 % 480 >= 119)
    assert (predicted_frames(n24) == expected).all()
    assert all(feature_length(int(n)) == int(t) for n, t in zip(n24, expected))
    filtered = np.array([480 * k + r for k in (30, 31, 500) for r in (0, 4)])
    assert (predicted_frames(filtered) == filtered // 480 - 1).all()


@pytest.mark.parametrize("n24", [9239, 9240, 70080, 158400, 123457])
def test_resampled_length_matches_formula(n24):
    raw16 = librosa.resample(np.zeros(n24), orig_sr=SAMPLE_RATE, target_sr=16000)
    assert len(raw16) == -(-2 * n24 // 3)
    assert (len(raw16) - 80) // EXTRACTOR_HOP == feature_length(n24)


def test_shortest_encodable_utterance_has_19_frames():
    assert feature_length(9239) == 19
    assert feature_length(9238) == 18


def test_shards_partition_rows():
    table = pd.DataFrame({"id": [str(i) for i in range(23)]})
    parts = [shard_rows(table, k, 5)["id"].tolist() for k in range(5)]
    assert sorted(sum(parts, [])) == sorted(table["id"])
    assert all(abs(len(p) - 23 / 5) < 1 for p in parts)
    with pytest.raises(ValueError):
        shard_rows(table, 5, 5)


def test_select_rows_subsample_per_split():
    table = pd.DataFrame({"split": ["a"] * 100 + ["b"] * 50, "id": [str(i) for i in range(150)]})
    cfg = OmegaConf.create({"select": {"splits": None, "subsample": 10}})
    chosen = select_rows(table, cfg)
    assert chosen["split"].value_counts().to_dict() == {"a": 10, "b": 10}
    assert select_rows(table, cfg, "b")["id"].tolist() == chosen[chosen["split"] == "b"]["id"].tolist()
    assert evenly_spaced(5, 10).tolist() == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------------------------------------------------
# cache files


def test_write_validate_round_trip(tmp_path):
    path = cache.utt_path(tmp_path, "dev.clean", "1_2_3")
    arrays = synthetic_arrays(40)
    cache.write_utt(path, arrays)
    assert path.is_file() and not list(path.parent.glob("*.tmp"))
    assert cache.is_valid(path, 40)
    assert not cache.is_valid(path, 41)
    loaded = cache.read_utt(path)
    assert set(loaded) == set(cache.UTT_KEYS)
    np.testing.assert_array_equal(loaded["feats"], arrays["feats"])
    assert str(loaded["meta_hash"]) == "h" and int(loaded["seed"]) == 0


def test_write_is_atomic(tmp_path, monkeypatch):
    path = cache.utt_path(tmp_path, "dev.clean", "u")
    cache.write_utt(path, synthetic_arrays(30, seed=1))
    before = path.read_bytes()

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", broken)
    with pytest.raises(OSError):
        cache.write_utt(path, synthetic_arrays(30, seed=2))
    assert path.read_bytes() == before
    assert not list(tmp_path.rglob("*.tmp"))
    fresh = cache.utt_path(tmp_path, "dev.clean", "v")
    with pytest.raises(OSError):
        cache.write_utt(fresh, synthetic_arrays(30))
    assert not fresh.exists() and not list(tmp_path.rglob("*.tmp"))


def test_validation_rejects_bad_files(tmp_path):
    good = synthetic_arrays(25)
    cases = {
        "nan": {
            **good,
            "feats": np.where(np.arange(25 * 15).reshape(25, 15) == 7, np.nan, good["feats"]).astype(np.float32),
        },
        "dtype": {**good, "feats": good["feats"].astype(np.float64)},
        "shape": {**good, "loud_raw": good["loud_raw"][:-1]},
        "missing": {k: v for k, v in good.items() if k != "spk_l6"},
        "wrong_T": {**good, "T": np.int64(26)},
    }
    for name, arrays in cases.items():
        path = tmp_path / f"{name}.npz"
        cache.write_utt(path, arrays)
        assert not cache.is_valid(path, 25), name
    assert not cache.is_valid(tmp_path / "absent.npz", 25)
    truncated = tmp_path / "truncated.npz"
    cache.write_utt(truncated, good)
    truncated.write_bytes(truncated.read_bytes()[:-200])
    assert not cache.is_valid(truncated, 25)
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"not a zip file")
    assert not cache.is_valid(garbage, 25)


def test_meta_created_once_and_checked(tmp_path):
    path = tmp_path / "meta.json"
    first = cache.build_meta({"versions": {"torch": "2.9.0"}}, "abc")
    stored, digest = cache.ensure_meta(path, first)
    assert stored["fork_commit"] == "abc"
    later = cache.build_meta({"versions": {"torch": "2.9.0"}}, "def")
    stored2, digest2 = cache.ensure_meta(path, later)
    assert stored2["fork_commit"] == "abc" and digest2 == digest == cache.read_meta_hash(path)
    with pytest.raises(RuntimeError):
        cache.ensure_meta(path, cache.build_meta({"versions": {"torch": "2.10.0"}}, "abc"))
    assert not list(tmp_path.glob("*.tmp"))


def test_meta_temporary_file_names_do_not_collide_between_processes(tmp_path, monkeypatch):
    """Array tasks on different nodes can share a pid, so the temporary file must not be named by pid alone."""
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    sources = []
    real_link = os.link

    def recording_link(src, dst, *args, **kwargs):
        sources.append(Path(src).name)
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", recording_link)
    for _ in range(2):
        path = tmp_path / "meta.json"
        path.unlink(missing_ok=True)
        cache.ensure_meta(path, cache.build_meta({"versions": {}}, "abc"))
    assert len(sources) == 2 and sources[0] != sources[1]
    assert not list(tmp_path.glob("*.tmp"))


def test_utterance_seed_is_stable():
    assert cache.utterance_seed("1_2_3") == cache.utterance_seed("1_2_3")
    assert 0 <= cache.utterance_seed("84_121123_000007_000001") < 2**31
    assert cache.utterance_seed("a") != cache.utterance_seed("b")


# ---------------------------------------------------------------------------------------------------------------------
# extraction driver pieces (no model)


def driver_config(tmp_path: Path, table: pd.DataFrame):
    manifest = tmp_path / "manifest.parquet"
    table.to_parquet(manifest)
    return OmegaConf.create(
        {
            "select": {"splits": None, "subsample": None},
            "cache": {
                "utt_dir": str(tmp_path / "utt"),
                "errors_dir": str(tmp_path / "errors"),
                "summaries_dir": str(tmp_path / "summaries"),
            },
            "manifest": {"path": str(manifest)},
            "extract": {"num_workers": 0, "prefetch_factor": 2, "max_consecutive_errors": 3, "log_every": 0},
        }
    )


def driver_table(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for k, n24 in enumerate([9600, 9600, 4800, 9600]):
        path = tmp_path / f"u{k}.wav"
        sf.write(path, 0.1 * np.ones(n24 if k != 3 else 9000), SAMPLE_RATE, subtype="PCM_16")
        frames = feature_length(n24)
        rows.append(
            {
                "id": f"u{k}",
                "split": "dev.clean",
                "speaker": "1",
                "chapter": "1",
                "wav_path": str(path),
                "n24": n24,
                "duration": np.float32(n24 / SAMPLE_RATE),
                "T": frames,
                "encodable": frames >= 19,
                "text": "",
            }
        )
    return pd.DataFrame(rows)


def test_datamodule_lists_not_encodable_and_skips_valid(tmp_path):
    table = driver_table(tmp_path)
    cfg = driver_config(tmp_path, table)
    cache.write_utt(cache.utt_path(cfg.cache.utt_dir, "dev.clean", "u1"), synthetic_arrays(int(table["T"][1])))
    data = ManifestShardDataModule(cfg, 0, 1, skip_valid=True)
    data.setup()
    assert data.not_encodable == ["u2"]
    items = {item["id"]: item for item in data.predict_dataloader()}
    assert set(items) == {"u0", "u1", "u3"}
    assert items["u1"]["skipped"] and "wav24" not in items["u1"]
    assert items["u0"]["wav24"].shape == (9600,) and items["u0"]["wav24"].dtype == np.float64
    assert "wrong" not in items["u3"].get("error", "") and "expected 9600 mono samples" in items["u3"]["error"]
    rewrite = ManifestShardDataModule(cfg, 0, 1, skip_valid=False)
    rewrite.setup()
    assert "skipped" not in next(item for item in rewrite.predict_dataloader() if item["id"] == "u1")


def test_dataset_reports_missing_audio(tmp_path):
    table = driver_table(tmp_path)
    table.loc[0, "wav_path"] = str(tmp_path / "absent.wav")
    item = ManifestShardDataset(table, tmp_path / "utt", skip_valid=False)[0]
    assert "error" in item and "wav24" not in item


def test_writer_writes_logs_errors_and_stops(tmp_path):
    table = driver_table(tmp_path)
    cfg = driver_config(tmp_path, table)
    writer = CacheWriter(cfg, "00000of00001")
    ok = {
        "id": "u0",
        "split": "dev.clean",
        "T": 19,
        "n24": 9600,
        "status": "ok",
        "seconds": 0.5,
        "arrays": synthetic_arrays(19),
    }
    call = lambda prediction: writer.write_on_batch_end(None, None, prediction, [0], None, 0, 0)  # noqa: E731
    call(ok)
    assert cache.is_valid(cache.utt_path(cfg.cache.utt_dir, "dev.clean", "u0"), 19)
    call({**ok, "id": "u1", "status": "skipped"})
    bad = {"id": "u2", "split": "dev.clean", "status": "error", "error": "Traceback\nValueError: boom"}
    call(bad)
    call(ok)
    assert writer.consecutive_errors == 0
    call(bad)
    call(bad)
    with pytest.raises(RuntimeError, match="consecutive errors"):
        call(bad)
    lines = (tmp_path / "errors" / "00000of00001.jsonl").read_text().splitlines()
    assert len(lines) == 4 and "boom" in lines[0]
    assert writer.counts == {"written": 2, "skipped": 1, "errors": 4}
    summary = writer.summary()
    assert summary["written"] == 2 and summary["audio_hours"] == pytest.approx(2 * 9600 / SAMPLE_RATE / 3600)
    writer.consecutive_errors = 0
    cache_module.STOP.set()
    try:
        with pytest.raises(cache_module.StopRequested):
            call(ok)
    finally:
        cache_module.STOP.clear()


class StubExtractor:
    """Stands in for SparcFeatureExtractor: random arrays of the right shapes; amplitude > 0.5 fails; > 0.4 signals."""

    device = torch.device("cpu")
    device_name = "cpu"
    signals_sent = 0

    def __init__(self, cfg, device):
        pass

    def describe(self):
        return {"versions": {"stub": "1"}}

    def extract(self, wav24, seed):
        if wav24.max() > 0.5:
            raise ValueError("stub failure")
        if wav24.max() > 0.4 and not StubExtractor.signals_sent:
            StubExtractor.signals_sent = 1
            os.kill(os.getpid(), signal.SIGUSR1)
        frames = feature_length(len(wav24))
        arrays = synthetic_arrays(frames, seed=seed % 1000)
        return {
            **{key: arrays[key] for key in ("feats", "loud_raw", "spk_l0", "spk_l6", "spk_enplus64")},
            "spk_wsum": 2.0,
            "spk_fallback": False,
            "T": frames,
            "n24": len(wav24),
            "peak24": float(np.abs(wav24).max()),
            "seed": seed,
        }


def stub_driver_config(tmp_path: Path, amplitudes: list[float], lengths: list[int]):
    rows = []
    for k, (amp, n24) in enumerate(zip(amplitudes, lengths)):
        path = tmp_path / f"s{k}.wav"
        sf.write(path, amp * np.ones(n24), SAMPLE_RATE, subtype="FLOAT")
        frames = feature_length(n24)
        rows.append(
            {
                "id": f"s{k}",
                "split": "dev.clean",
                "speaker": "1",
                "chapter": "1",
                "wav_path": str(path),
                "n24": n24,
                "duration": np.float32(n24 / SAMPLE_RATE),
                "T": frames,
                "encodable": frames >= 19,
                "text": "",
            }
        )
    cfg = driver_config(tmp_path, pd.DataFrame(rows))
    cfg.cache.meta_path = str(tmp_path / "meta.json")
    cfg.extract.update({"accelerator": "cpu", "skip_valid": True})
    cfg.update({"shard_index": 0, "num_shards": 1})
    return cfg


@pytest.fixture
def stub_extractor(monkeypatch):
    monkeypatch.setattr(cache_module, "SparcFeatureExtractor", StubExtractor)
    monkeypatch.setattr(cache_module, "fork_commit", lambda: "test")
    StubExtractor.signals_sent = 0
    saved = {sig: signal.getsignal(sig) for sig in cache_module.STOP_SIGNALS}
    cache_module.STOP.clear()
    yield
    cache_module.STOP.clear()
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_driver_resumes_reports_errors_and_stops_on_signal(tmp_path, stub_extractor):
    cfg = stub_driver_config(tmp_path, [0.1, 0.9, 0.1, 0.1, 0.1], [9600, 9600, 4800, 9600, 9840])
    utt = lambda k: cache.utt_path(cfg.cache.utt_dir, "dev.clean", f"s{k}")  # noqa: E731
    errors = tmp_path / "errors" / "00000of00001.jsonl"

    assert cache_module.run_extract(cfg) == 1
    assert [cache.is_valid(utt(k), feature_length(n)) for k, n in ((0, 9600), (3, 9600), (4, 9840))] == [True] * 3
    assert not utt(1).exists() and not utt(2).exists()
    assert "stub failure" in errors.read_text() and len(errors.read_text().splitlines()) == 1
    assert (tmp_path / "errors" / "not_encodable_00000of00001.txt").read_text() == "s2\n"
    assert not list((tmp_path / "utt").rglob("*.tmp"))
    first = utt(0).read_bytes()

    sf.write(tmp_path / "s1.wav", 0.1 * np.ones(9600), SAMPLE_RATE, subtype="FLOAT")
    assert cache_module.run_extract(cfg) == 0
    assert cache.is_valid(utt(1), feature_length(9600)) and utt(0).read_bytes() == first
    summaries = [json.loads(line) for line in (tmp_path / "summaries" / "extract_00000of00001.jsonl").read_text().splitlines()]
    assert [(r["written"], r["skipped"], r["errors"]) for r in summaries] == [(3, 0, 1), (1, 3, 0)]

    for k in range(5):
        utt(k).unlink(missing_ok=True)
    sf.write(tmp_path / "s3.wav", 0.45 * np.ones(9600), SAMPLE_RATE, subtype="FLOAT")
    assert cache_module.run_extract(cfg) == cache_module.EXIT_STOPPED
    stopped = [k for k in (0, 1, 3, 4) if utt(k).exists()]
    assert stopped == [0, 1, 3]
    cache_module.STOP.clear()
    assert cache_module.run_extract(cfg) == 0
    assert all(cache.is_valid(utt(k), feature_length(n)) for k, n in ((0, 9600), (1, 9600), (3, 9600), (4, 9840)))


# ---------------------------------------------------------------------------------------------------------------------
# provenance and numerics


def test_set_fp32_numerics_controls_tf32():
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    try:
        set_fp32_numerics(True)
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
        set_fp32_numerics(False)
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved
    assert OmegaConf.load(Path(CONF_DIR) / "cache_config.yaml").extractor.allow_tf32 is False


def test_fork_commit_marks_uncommitted_changes(tmp_path):
    git = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True
    )
    try:
        git("init", "-q")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert fork_commit(tmp_path) == head
    (tmp_path / "src" / "b.py").write_text("y = 2\n")
    assert fork_commit(tmp_path) == head + "+dirty"
    assert fork_commit(tmp_path / "missing") == "unknown"


# ---------------------------------------------------------------------------------------------------------------------
# loudness and gain


def test_raw_loudness_scales_exactly_with_gain():
    wav24 = synthetic_speech(1.5, seed=3)
    histogram = AmplitudeHistogram(EXTRACTOR_HOP).eval()
    g = gain_factor(float(np.abs(wav24).max()), -3.0)
    raw16 = librosa.resample(wav24, orig_sr=SAMPLE_RATE, target_sr=16000)
    gained16 = librosa.resample(wav24 * g, orig_sr=SAMPLE_RATE, target_sr=16000)
    loud = raw_loudness(raw16, histogram, "cpu")
    loud_gained = raw_loudness(gained16, histogram, "cpu")
    np.testing.assert_allclose(loud_gained, loud * np.float32(g), rtol=1e-5, atol=1e-10)

    z = lambda x: (x - x.mean()) / x.std()  # noqa: E731
    z_loud = raw_loudness(z(raw16), histogram, "cpu")
    z_loud_gained = raw_loudness(z(gained16), histogram, "cpu")
    np.testing.assert_allclose(z_loud_gained, z_loud, rtol=1e-4, atol=1e-6)
    assert not np.allclose(z_loud * np.float32(g), z_loud, rtol=1e-2)


def test_raw_loudness_definition():
    x = np.random.default_rng(0).normal(size=3300)
    got = raw_loudness(x, AmplitudeHistogram(EXTRACTOR_HOP).eval(), "cpu")
    padded = np.concatenate([np.zeros(160), np.abs(x), np.zeros(480)])
    expected = np.array([padded[320 * i : 320 * i + 320].mean() for i in range(len(x) // 320 + 1)])
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_gain_factor_reaches_target_peak():
    wav = synthetic_speech(0.5)
    rng = np.random.default_rng(0)
    db = sample_gain_db(rng, (-6.0, -1.0))
    gained = wav * gain_factor(float(np.abs(wav).max()), db)
    assert abs(20 * np.log10(np.abs(gained).max()) - db) < 1e-9


# ---------------------------------------------------------------------------------------------------------------------
# speaker pooling


def test_pool_speaker_weighted_mean():
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(30, 1024)).astype(np.float32)
    weights = rng.uniform(0.4, 1.0, size=31).astype(np.float32)
    weights[::3] = 0.0
    vector, wsum, fallback = pool_speaker(hidden, weights)
    expected = (hidden * weights[:30, None]).sum(0) / weights[:30].sum()
    np.testing.assert_allclose(vector, expected, rtol=1e-5, atol=1e-6)
    assert not fallback and wsum == pytest.approx(float(weights[:30].sum()), rel=1e-6)
    assert vector.dtype == np.float32 and vector.shape == (1024,)


def test_pool_speaker_truncates_to_shorter_input():
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(40, 1024)).astype(np.float32)
    weights = np.ones(25, np.float32)
    vector, wsum, _ = pool_speaker(hidden, weights)
    np.testing.assert_allclose(vector, hidden[:25].mean(0), rtol=1e-5, atol=1e-6)
    assert wsum == 25.0


def test_pool_speaker_falls_back_to_uniform_mean():
    rng = np.random.default_rng(2)
    hidden = rng.normal(size=(30, 1024)).astype(np.float32)
    vector, wsum, fallback = pool_speaker(hidden, np.zeros(31, np.float32))
    assert fallback and wsum == 0.0
    assert np.isfinite(vector).all()
    np.testing.assert_allclose(vector, hidden.mean(0), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------------------------------------------------
# packing and statistics


def packing_config(tmp_path: Path, splits: dict[str, int]):
    rows = []
    for split, count in splits.items():
        for k in range(count):
            frames = 25 + 7 * k
            speaker = str(100 + k % 3)
            rows.append(
                {
                    "id": f"{speaker}_{k}_{split}",
                    "split": split,
                    "speaker": speaker,
                    "chapter": str(k),
                    "wav_path": f"/nowhere/{split}/{k}.wav",
                    "n24": 480 * (frames + 1),
                    "duration": np.float32(480 * (frames + 1) / SAMPLE_RATE),
                    "T": frames,
                    "encodable": True,
                    "text": "x",
                }
            )
    table = pd.DataFrame(rows)
    manifest = tmp_path / "manifest.parquet"
    table.to_parquet(manifest)
    cfg = OmegaConf.create(
        {
            "select": {"splits": None, "subsample": None},
            "cache": {
                "utt_dir": str(tmp_path / "utt"),
                "meta_path": str(tmp_path / "meta.json"),
                "errors_dir": str(tmp_path / "errors"),
                "packed_dir": str(tmp_path / "packed"),
            },
            "manifest": {"path": str(manifest)},
            "pack": {"num_threads": 2, "chunk_utterances": 3, "allow_missing": False},
            "stats": {
                "splits": None,
                "gain_seed": 0,
                "spk_std_floor": 1e-6,
                "chunk_frames": 50,
                "chunk_utterances": 4,
                "output": str(tmp_path / "stats.json"),
            },
            "data": {"train_splits": list(splits), "gain_db_range": [-6.0, -1.0]},
        }
    )
    _, digest = cache.ensure_meta(cfg.cache.meta_path, cache.build_meta({"versions": {}}, "test"))
    for i, row in table.iterrows():
        arrays = synthetic_arrays(int(row["T"]), seed=i, meta_hash=digest)
        arrays["feats"][:, 12] = np.abs(arrays["feats"][:, 12]) * 100.0 + 80.0
        arrays["feats"][:, 14] = np.clip(arrays["feats"][:, 14], 0.0, 1.0)
        arrays["peak24"] = np.float64(0.2 + 0.1 * (i % 5))
        cache.write_utt(cache.utt_path(cfg.cache.utt_dir, row["split"], row["id"]), arrays)
    return cfg, table


def test_pack_split_layout(tmp_path):
    cfg, table = packing_config(tmp_path, {"dev.clean": 7})
    directory = pack_split(cfg, "dev.clean")
    index = pd.read_parquet(directory / "index.parquet")
    feats = np.load(directory / "feats.npy", mmap_mode="r")
    assert list(index["id"]) == list(table["id"])
    assert feats.shape == (int(table["T"].sum()), 15)
    assert (index["offset"].to_numpy() == np.concatenate([[0], np.cumsum(table["T"])[:-1]])).all()
    for i in (0, 3, 6):
        arrays = cache.read_utt(cache.utt_path(cfg.cache.utt_dir, "dev.clean", table["id"][i]))
        lo = int(index["offset"][i])
        np.testing.assert_array_equal(feats[lo : lo + int(index["T"][i])], arrays["feats"])
        np.testing.assert_array_equal(
            np.load(directory / "loud_raw.npy", mmap_mode="r")[lo : lo + int(index["T"][i])], arrays["loud_raw"]
        )
        np.testing.assert_array_equal(np.load(directory / "spk_l6.npy", mmap_mode="r")[i], arrays["spk_l6"])
        np.testing.assert_array_equal(np.load(directory / "spk_enplus64.npy", mmap_mode="r")[i], arrays["spk_enplus64"])
        assert index["peak24"][i] == float(arrays["peak24"])
    assert not list(directory.parent.glob("*.tmp"))


def test_pack_split_reports_missing_files(tmp_path):
    cfg, table = packing_config(tmp_path, {"dev.clean": 4})
    cache.utt_path(cfg.cache.utt_dir, "dev.clean", table["id"][2]).unlink()
    with pytest.raises(FileNotFoundError):
        pack_split(cfg, "dev.clean")
    cfg.pack.allow_missing = True
    index = pd.read_parquet(pack_split(cfg, "dev.clean") / "index.parquet")
    assert len(index) == 3 and table["id"][2] not in set(index["id"])


def test_compute_stats_matches_numpy(tmp_path):
    cfg, table = packing_config(tmp_path, {"train.a": 6, "train.b": 5})
    for split in ("train.a", "train.b"):
        pack_split(cfg, split)
    stats = compute_stats(cfg)
    frames, loud, utt_spk = [], [], {"l0": [], "l6": []}
    for offset, split in enumerate(("train.a", "train.b")):
        sub = table[table["split"] == split].reset_index(drop=True)
        arrays = [cache.read_utt(cache.utt_path(cfg.cache.utt_dir, split, i)) for i in sub["id"]]
        rng = np.random.default_rng([0, offset])
        for a in arrays:
            g = gain_factor(float(a["peak24"]), sample_gain_db(rng, (-6.0, -1.0)))
            frames.append(a["feats"].astype(np.float64))
            loud.append(np.log(a["loud_raw"].astype(np.float64) * g + 1e-4))
            for layer in utt_spk:
                utt_spk[layer].append(a[f"spk_{layer}"].astype(np.float64))
    feats, loud = np.concatenate(frames), np.concatenate(loud)
    np.testing.assert_allclose(stats["ema_mean"], feats[:, :12].mean(0), atol=1e-6)
    np.testing.assert_allclose(stats["ema_std"], feats[:, :12].std(0), rtol=1e-6)
    np.testing.assert_allclose(stats["logf0_mean"], np.log(np.maximum(feats[:, 12], 1.0)).mean(), atol=1e-6)
    np.testing.assert_allclose(stats["loud_log_mean"], loud.mean(), atol=1e-6)
    np.testing.assert_allclose(stats["loud_log_std"], loud.std(), rtol=1e-6)
    np.testing.assert_allclose(stats["per_std"], feats[:, 14].std(), rtol=1e-6)
    np.testing.assert_allclose(stats["spk_l6_mean"], np.mean(utt_spk["l6"], axis=0), atol=1e-6)
    np.testing.assert_allclose(stats["spk_l0_std"], np.std(utt_spk["l0"], axis=0), rtol=1e-6)
    assert stats["frames"] == len(feats) and stats["utterances"] == 11 and stats["splits"] == ["train.a", "train.b"]
    assert len(stats["spk_l0_mean"]) == 1024 and len(stats["ema_std"]) == 12


def test_moments_streaming_matches_numpy():
    x = np.random.default_rng(0).normal(loc=1e4, scale=3.0, size=(1000, 4))
    m = Moments(4)
    for lo in range(0, 1000, 130):
        m.update(x[lo : lo + 130])
    np.testing.assert_allclose(m.mean(), x.mean(0), rtol=1e-12)
    np.testing.assert_allclose(m.std(), x.std(0), rtol=1e-6)


# ---------------------------------------------------------------------------------------------------------------------
# equivalence with SPARC's own encode (needs the pinned checkpoints; device from SPARC_TEST_DEVICE, default cpu)


def equivalence_cfg():
    from hydra import compose, initialize_config_dir

    needed = ("HF_HUB_CACHE", "SPARC_REFIT_NPZ", "SPARC_VOC_CACHE", "SPARC_VOC_RUNS", "LIBRITTSR_RAW")
    if any(name not in os.environ for name in needed):
        pytest.skip("environment of the feature cache is not set")
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="cache_config", overrides=["stage=extract"])


@pytest.fixture(scope="module")
def extractor():
    cfg = equivalence_cfg()
    try:
        return SparcFeatureExtractor(cfg, os.environ.get("SPARC_TEST_DEVICE", "cpu"))
    except FileNotFoundError as error:
        pytest.skip(str(error))


def equivalence_waves() -> list[np.ndarray]:
    paths = [p for p in os.environ.get("SPARC_TEST_WAVS", "").split(":") if p]
    if paths:
        import soundfile as sf

        return [sf.read(p, dtype="float64")[0] for p in paths]
    return [synthetic_speech(2.5, seed=0), synthetic_speech(1.2, seed=1)]


@pytest.mark.parametrize("which", [0, 1])
def test_extractor_matches_sparc_encode(extractor, which):
    waves = equivalence_waves()
    if which >= len(waves):
        pytest.skip("fewer waveforms than parameters")
    wav24 = waves[which]
    seed = cache.utterance_seed(f"equivalence_{which}")
    out = extractor.extract(wav24, seed)
    raw16 = librosa.resample(wav24, orig_sr=SAMPLE_RATE, target_sr=16000)
    np.random.seed(seed)
    ref = extractor.coder.encode(raw16, concat=True)
    ref_feats = ref["features"]
    assert out["feats"].shape == ref_feats.shape == (feature_length(len(wav24)), 15)
    assert np.abs(out["feats"][:, :12] - ref_feats[:, :12]).max() < 1e-5
    np.testing.assert_array_equal(out["feats"][:, 12], ref_feats[:, 12])
    np.testing.assert_array_equal(out["feats"][:, 13], ref_feats[:, 13])
    np.testing.assert_array_equal(out["feats"][:, 14], ref_feats[:, 14])
    if not out["spk_fallback"]:
        assert np.abs(out["spk_enplus64"] - ref["spk_emb"]).max() < 1e-4
    assert out["spk_l6"].shape == out["spk_l0"].shape == (1024,)
    assert out["loud_raw"].shape == (out["T"],) and (out["loud_raw"] >= 0).all()


@pytest.mark.parametrize("which", [0, 1])
def test_extractor_is_deterministic_and_pools_layers_independently(extractor, which):
    waves = equivalence_waves()
    if which >= len(waves):
        pytest.skip("fewer waveforms than parameters")
    wav24, seed = waves[which], cache.utterance_seed(f"determinism_{which}")
    first, second = extractor.extract(wav24, seed), extractor.extract(wav24, seed)
    for key in ("feats", "loud_raw", "spk_l0", "spk_l6", "spk_enplus64"):
        np.testing.assert_array_equal(first[key], second[key])
    assert not torch.backends.cudnn.allow_tf32

    raw16 = librosa.resample(wav24, orig_sr=SAMPLE_RATE, target_sr=16000)
    wavs = extractor.coder.process_wavfiles(raw16)
    with torch.no_grad():
        states = extractor.wavlm(wavs.input_values, output_hidden_states=True).hidden_states
    np.random.seed(seed)
    weights = extractor.coder.encode(raw16)["periodicity"][:, 0].astype(np.float64)
    if first["spk_fallback"]:
        pytest.skip("no voiced frames in the test signal")
    for layer, key in ((0, "spk_l0"), (6, "spk_l6")):
        hidden = states[layer][0].double().cpu().numpy()
        n = min(len(hidden), len(weights))
        expected = (hidden[:n] * weights[:n, None]).sum(0) / weights[:n].sum()
        np.testing.assert_allclose(first[key], expected, rtol=1e-3, atol=1e-3)
    assert first["spk_wsum"] == pytest.approx(float(weights[:n].sum()), rel=1e-4)
