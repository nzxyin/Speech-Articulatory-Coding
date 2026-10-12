"""Evaluation I/O helpers: atomic writers, stop flag, chunking and resume, utterance table, layout (CPU, seconds)."""

import os
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from sparc.vocoders.constants import HOP
from sparc.vocoders.data.dataset import FullUtteranceDataset
from sparc.vocoders.data.datamodule import VocoderDataModule
from sparc.vocoders.eval import io
from sparc.vocoders.eval.systems import SystemSpec, load_systems
from toy_split import SEX, SPLIT, build_toy_cache, set_env, toy_config


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("toy")
    build_toy_cache(path)
    return path


@pytest.fixture(scope="module")
def table(root) -> pd.DataFrame:
    sex = io.read_speaker_sex(root / "eval" / "ref_data" / "LibriTTS_R" / "SPEAKERS.txt")
    return io.build_utterance_table(root / "cache", SPLIT, sex=sex)


# ------------------------------------------------------------------------------------------------- atomic writes


def test_round_trips(tmp_path):
    frame = pd.DataFrame({"id": ["a", "b"], "x": [1.5, np.nan]})
    io.write_parquet(tmp_path / "p" / "f.parquet", frame)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "p" / "f.parquet"), frame)
    io.write_npz(tmp_path / "f.npz", {"a": np.arange(3), "s": np.array(["x", "yy"])})
    part = io.read_npz_part(tmp_path / "f.npz")
    assert part["a"].tolist() == [0, 1, 2] and part["s"].tolist() == ["x", "yy"]
    io.write_json(tmp_path / "f.json", {"n": np.int64(3), "p": Path("/x"), "v": np.arange(2)})
    assert io.read_json(tmp_path / "f.json") == {"n": 3, "p": "/x", "v": [0, 1]}
    io.write_csv(tmp_path / "f.csv", frame)
    assert pd.read_csv(tmp_path / "f.csv").shape == (2, 2)
    wav = np.linspace(-1, 1, 480, dtype=np.float32)
    io.write_wav(tmp_path / "w.wav", wav, 24000)
    back, rate = io.read_wav(tmp_path / "w.wav")
    assert rate == 24000 and back.dtype == np.float32 and np.array_equal(back, wav)
    io.write_wav(tmp_path / "w16.wav", np.array([0.5, 2.0, -2.0], np.float32), 16000, subtype="PCM_16")
    assert sf.info(tmp_path / "w16.wav").subtype == "PCM_16"
    assert io.read_wav(tmp_path / "w16.wav")[0][1] == pytest.approx(1.0, abs=1e-3)  # clipped, not wrapped
    assert sorted(p.name for p in tmp_path.rglob("*.tmp*")) == []


def test_failed_write_leaves_neither_file_nor_temporary(tmp_path):
    target = tmp_path / "out.parquet"
    target.write_bytes(b"old")

    def broken(tmp: Path) -> None:
        tmp.write_bytes(b"partial")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        io.atomic_write(target, broken)
    assert target.read_bytes() == b"old"  # the previous complete file survives
    assert list(tmp_path.iterdir()) == [target]


def test_write_is_never_visible_half_written(tmp_path):
    seen = []

    def writer(tmp: Path) -> None:
        tmp.write_bytes(b"x" * 10)
        seen.append(sorted(p.name for p in tmp_path.iterdir()))

    io.atomic_write(tmp_path / "f.bin", writer)
    assert "f.bin" not in seen[0] and len(seen[0]) == 1  # only the temporary file existed while writing
    assert (tmp_path / "f.bin").read_bytes() == b"x" * 10


def test_predict_wav_writer_is_atomic(tmp_path, monkeypatch):
    """The WavWriter of ``sparc-predict`` goes through ``write_wav`` (temporary file, then ``os.replace``)."""
    from sparc.vocoders.training import callbacks

    replaced = []
    real = os.replace
    monkeypatch.setattr(callbacks.os, "replace", lambda a, b: (replaced.append((Path(a).name, Path(b).name)), real(a, b))[1])
    callbacks.write_wav(tmp_path / "x.wav", np.zeros(10, np.float32))
    assert replaced == [("x.wav.tmp", "x.wav")] and not (tmp_path / "x.wav.tmp").exists()


# ------------------------------------------------------------------------------------------------- stop flag


def test_stop_flag_from_signals_and_restore():
    flag = io.StopFlag()
    previous = signal.getsignal(signal.SIGUSR1)
    with flag.installed():
        assert not flag.is_set()
        os.kill(os.getpid(), signal.SIGUSR1)
        assert flag.is_set() and flag.signal_number == signal.SIGUSR1
        with pytest.raises(io.StopRequested):
            flag.raise_if_set()
    assert signal.getsignal(signal.SIGUSR1) == previous
    flag = io.StopFlag()
    with flag.installed():
        os.kill(os.getpid(), signal.SIGTERM)
        assert flag.is_set() and flag.signal_number == signal.SIGTERM
    flag.clear()
    assert not flag.is_set()


# ------------------------------------------------------------------------------------------------- chunks


def test_chunk_plan_and_condition_restriction():
    ids = [f"i{k}" for k in range(7)]
    chunks = io.chunk_plan(ids, 3)
    assert chunks == [["i0", "i1", "i2"], ["i3", "i4", "i5"], ["i6"]]
    restricted = io.chunk_for_condition(chunks, ["i1", "i6"])
    assert restricted == [["i1"], [], ["i6"]]  # numbering preserved, a chunk may become empty
    with pytest.raises(ValueError):
        io.chunk_plan(ids, 0)
    assert io.part_name(12, "npz") == "part-0012.npz"


def test_run_chunks_resumes_after_a_stop_without_recomputing(tmp_path):
    ids = [f"u{k}" for k in range(7)]
    chunks = io.chunk_plan(ids, 2)  # 4 chunks
    flag = io.StopFlag()
    calls: list[int] = []

    def is_done(k, _ids):
        return (tmp_path / io.part_name(k, "txt")).exists()

    def work(k, chunk_ids):
        calls.append(k)
        io.write_text(tmp_path / io.part_name(k, "txt"), ",".join(chunk_ids))
        if k == 1:
            flag.set()  # a signal arrives while chunk 1 is being computed

    report = io.ChunkReport()
    with pytest.raises(io.StopRequested):
        io.run_chunks(chunks, is_done, work, flag, report)
    assert calls == [0, 1] and report.computed == [0, 1]  # chunk 1 finished, chunk 2 not started
    flag.clear()
    calls.clear()
    report = io.run_chunks(chunks, is_done, work, flag)
    assert calls == [2, 3] and report.skipped == [0, 1] and report.computed == [2, 3]  # chunk 0 was not recomputed
    assert len(io.list_parts(tmp_path, "txt")) == 4
    assert io.run_chunks(chunks, is_done, work, flag).computed == []


def test_stop_during_the_last_chunk_completes_normally():
    flag = io.StopFlag()
    done: set[int] = set()

    def work(k, _ids):
        done.add(k)
        flag.set()

    with pytest.raises(io.StopRequested):  # stop after chunk 0 of two: chunk 1 never starts
        io.run_chunks([["a"], ["b"]], lambda k, _i: k in done, work, flag)
    assert done == {0}
    flag.clear()
    done.clear()
    report = io.run_chunks([["a"]], lambda k, _i: k in done, work, flag)  # the only chunk: nothing is left to stop
    assert report.computed == [0] and flag.is_set()


def test_list_and_read_parquet_parts(tmp_path):
    for k in (2, 0, 1):
        io.write_parquet(tmp_path / io.part_name(k, "parquet"), pd.DataFrame({"id": [f"i{k}"]}))
    (tmp_path / "part-x.parquet").write_bytes(b"")  # not a part
    assert [k for k, _ in io.list_parts(tmp_path, "parquet")] == [0, 1, 2]
    assert io.read_parquet_parts(tmp_path)["id"].tolist() == ["i0", "i1", "i2"]
    assert io.read_parquet_parts(tmp_path / "none").empty


def test_stage_meta_records_runs_and_refuses_a_changed_fingerprint(tmp_path):
    header = {"stage": "signal", "chunk_size": 4, "fingerprint": {"g_step": 10}, "config_digest": "a"}
    meta = io.StageMeta(tmp_path / "meta.json", header)
    meta.begin()
    meta.update(models={"m": 1})
    meta.end("done", io.ChunkReport(n_chunks=2, computed=[0], skipped=[1]))
    data = io.read_json(tmp_path / "meta.json")
    assert data["runs"][0]["status"] == "done" and data["runs"][0]["models"] == {"m": 1}
    assert data["runs"][0]["computed_chunks"] == [0] and data["stage"] == "signal" and "fork_commit" in data["runs"][0]
    second = io.StageMeta(tmp_path / "meta.json", {**header, "config_digest": "b"})
    second.begin()
    assert second.run["config_changed"] is True and len(io.read_json(tmp_path / "meta.json")["runs"]) == 2
    with pytest.raises(RuntimeError, match="fingerprint"):
        io.StageMeta(tmp_path / "meta.json", {**header, "fingerprint": {"g_step": 11}}).begin()
    io.StageMeta(tmp_path / "meta.json", {**header, "fingerprint": {"g_step": 11}}, strict=False).begin()
    with pytest.raises(RuntimeError, match="chunk_size"):
        io.StageMeta(tmp_path / "meta.json", {**header, "chunk_size": 8}).begin()


# ------------------------------------------------------------------------------------------------- utterance table


def test_table_columns_and_t2_reference_equals_the_dataset(root, table):
    assert table["id"].is_monotonic_increasing and table["id"].is_unique
    dataset = FullUtteranceDataset(root / "cache" / "packed" / SPLIT, ids=None, condition="T2")
    expected = {dataset[i]["id"]: dataset[i]["ref_id"] for i in range(len(dataset))}
    got = table[table["has_ref"]].set_index("id")["ref_id"].to_dict()
    assert got == expected and len(expected) == len(table) - 1
    lonely = table[~table["has_ref"]]
    assert lonely["speaker"].tolist() == ["305"] and lonely["ref_id"].tolist() == [""]
    assert (table["ref_id"][table["has_ref"]] != table["id"][table["has_ref"]]).all()
    assert table.set_index("speaker")["sex"].to_dict() == SEX
    assert np.allclose(table["dur_s"], table["n24"] / 24000)


@pytest.mark.slow
def test_t2_reference_equals_the_dataset_on_real_test_clean():
    cache = os.environ.get("SPARC_VOC_CACHE")
    if not cache or not (Path(cache) / "packed" / "test.clean" / "index.parquet").is_file():
        pytest.skip("real feature cache not available")
    real = io.build_utterance_table(cache, "test.clean")
    dataset = FullUtteranceDataset(Path(cache) / "packed" / "test.clean", ids=None, condition="T2")
    expected = {dataset[i]["id"]: dataset[i]["ref_id"] for i in range(len(dataset))}
    assert real[real["has_ref"]].set_index("id")["ref_id"].to_dict() == expected
    assert len(real) == 4687 and real["has_ref"].sum() == len(expected)


def test_condition_ids_and_base_columns(table):
    all_ids = table["id"].tolist()
    assert io.condition_ids(table, "T1") == all_ids
    t2 = io.condition_ids(table, "T2")
    assert len(t2) == len(all_ids) - 1 and "305_1_000000_000000" not in t2
    assert io.condition_ids(table, "T3", all_ids[:5]) == sorted(set(t2) & set(all_ids[:5]))
    ids = t2[:3]
    t1 = io.base_columns(table, ids, "T1")
    assert t1["ref_id"].tolist() == ids
    assert io.base_columns(table, ids, "T2")["ref_id"].tolist() == table.set_index("id").loc[ids, "ref_id"].tolist()
    assert set(io.base_columns(table, ids, "T3")["ref_id"]) == {io.SPEAKER_MEAN_REF}
    assert list(t1.columns) == ["id", "speaker", "chapter", "dur_s", "ref_id"]
    assert io.chapter_of("300_2_000001_000000") == "2"


def test_select_limit_ids_is_fixed_and_matches_the_predict_limit(root, table):
    ids = table["id"].tolist()
    a = io.select_limit_ids(ids, 7, seed=3)
    assert a == io.select_limit_ids(ids[::-1], 7, seed=3) and len(a) == 7 and a == sorted(a)
    assert io.select_limit_ids(ids, 7, seed=4) != a
    assert io.select_limit_ids(ids, None) == ids and io.select_limit_ids(ids, 1000) == ids
    cfg = OmegaConf.create(
        {
            "paths": {"cache_root": str(root / "cache")},
            "data": {"predict_split": SPLIT, "predict_limit": 7},
            "seed": 3,
        }
    )
    assert VocoderDataModule(cfg).predict_ids() == a  # eval.limit and data.predict_limit draw the same subset


def test_gt_audio_is_the_gained_prefix(root, table):
    from sparc.vocoders.data.dataset import PackedStore

    store = PackedStore([io.packed_dir(root / "cache", SPLIT)], "l6")
    uid = table["id"][0]
    utt = store.index_of(uid)
    wav = io.gt_audio(store, utt, -3.0)
    assert wav.dtype == np.float32 and len(wav) == HOP * int(store.T[utt])
    assert np.abs(wav).max() <= 10 ** (-3 / 20) + 1e-6 and np.abs(wav).max() > 0.5


def test_speaker_sex_parsing(tmp_path):
    path = tmp_path / "SPEAKERS.txt"
    path.write_text("; header\n14 | F | train-clean-360 | 25.03 | Kristin LeMoine\n16\tM\tx\n\n")
    assert io.read_speaker_sex(path) == {"14": "F", "16": "M"}


# ------------------------------------------------------------------------------------------------- layout


def test_layout_and_audio_problems(root, table, monkeypatch):
    set_env(monkeypatch, root)
    cfg = toy_config(root, "eval.limit=null")
    systems = load_systems(cfg)
    paths = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, None)
    gt, vocoder, enplus = systems["gt"], systems["hifigan"], systems["enplus16"]
    uid = table["id"][0]
    assert paths.audio_path(gt, "T1", uid) == root / "eval" / "audio" / "gt" / SPLIT / "T1" / f"{uid}.wav"
    assert paths.audio_path(systems["gt_shipped"], "T1", uid) == paths.audio_path(gt, "T1", uid)  # audio_from
    assert paths.audio_path(vocoder, "T2", uid) == root / "runs" / "main" / "hifigan" / "predictions" / SPLIT / "T2" / f"{uid}.wav"
    assert paths.results_dir("hifigan", "T1", "asr") == root / "eval" / "results" / SPLIT / "hifigan" / "T1" / "asr"
    assert paths.tables_dir() == root / "eval" / "tables" / SPLIT
    smoke = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, 5)
    assert smoke.root == root / "eval" / "smoke_5"
    assert smoke.audio_path(vocoder, "T1", uid) == root / "eval" / "smoke_5" / "audio" / "hifigan" / SPLIT / "T1" / f"{uid}.wav"
    T = int(table["T"][0])
    assert io.expected_samples(T, 24000) == 480 * T and io.expected_samples(T, 16000) == 320 * T
    io.write_wav(paths.audio_path(enplus, "T1", uid), np.zeros(320 * T, np.float32), 16000)
    assert io.audio_problems(paths, enplus, "T1", table, [uid]) == {}
    io.write_wav(paths.audio_path(enplus, "T1", table["id"][1]), np.zeros(10, np.float32), 16000)
    problems = io.audio_problems(paths, enplus, "T1", table, [uid, table["id"][1], table["id"][2]])
    assert set(problems) == {table["id"][1], table["id"][2]} and "unreadable" in problems[table["id"][2]]
    assert io.read_system_audio(paths, enplus, "T1", uid).shape == (320 * T,)
    with pytest.raises(ValueError, match="sample rate"):
        io.write_wav(paths.audio_path(enplus, "T2", uid), np.zeros(320 * T, np.float32), 24000)
        io.read_system_audio(paths, enplus, "T2", uid)


def test_to_16k():
    x = np.random.default_rng(0).standard_normal(2400).astype(np.float32)
    y = io.to_16k(x, 24000)
    assert y.dtype == np.float32 and len(y) == 1600
    assert io.to_16k(x[:100], 16000) is not None and len(io.to_16k(x[:100], 16000)) == 100


def test_system_spec_lookup_is_validated(root, monkeypatch):
    set_env(monkeypatch, root)
    cfg = toy_config(root)
    systems = load_systems(cfg)
    assert isinstance(systems["gt"], SystemSpec) and systems["enplus16"].sr == 16000
    assert not systems["gt_shipped"].applies_to("utmos") and systems["gt_shipped"].applies_to("reextract")
    assert systems["gt"].applies_to("gt") and not systems["gt"].applies_to("signal")
    bad = toy_config(root, "+eval.systems.x={kind:vocoder,sr:24000,conditions:[T1]}")
    with pytest.raises(ValueError, match="experiment"):
        load_systems(bad)
