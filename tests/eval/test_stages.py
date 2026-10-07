"""Stage drivers on a toy cache with fake models: chunked outputs, stop and resume, provenance, dispatch (CPU, seconds)."""

import os
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf

from sparc.cli import eval_vocoder
from sparc.cli import predict_vocoder
from sparc.vocoders.constants import HOP
from sparc.vocoders.data.gain import gain_factor
from sparc.vocoders.eval import io, stages
from sparc.vocoders.eval.systems import STAGES, CheckpointInfo, compose_config
from toy_split import SPLIT, build_toy_cache, set_env, toy_config

N_IDS = 20
N_T2 = 19
CHUNK = 8  # 20 ids -> chunks of 8, 8 and 4


# ------------------------------------------------------------------------------------------------- fake models


class FakeUTMOS:
    def __init__(self):
        self.calls: list[int] = []
        self.after = None

    def score(self, wavs16, errors=None):
        self.calls.append(len(wavs16))
        out = [float(np.mean(np.abs(w))) for w in wavs16]
        if errors is not None:
            errors.extend([""] * len(out))
        if self.after:
            self.after()
        return out

    def provenance(self):
        return {"fake": "utmos"}


class FakeASR:
    def __init__(self):
        self.calls = 0

    def transcribe(self, wavs16, errors=None):
        self.calls += 1
        if errors is not None:
            errors.extend(["" if len(w) > 1600 else "ValueError: too short" for w in wavs16])
        return ["the quick brown fox jumps"] * len(wavs16)

    def provenance(self):
        return {"fake": "asr"}


class FakeSpeaker:
    dim = 6

    def __init__(self, name):
        self.name = name

    def embed(self, wavs16, errors=None):
        rows = []
        for w in wavs16:
            g = np.random.default_rng(int(abs(float(w[:5].sum())) * 1e6) % (2**31)).standard_normal(self.dim)
            rows.append(g / np.linalg.norm(g))
        if errors is not None:
            errors.extend([""] * len(rows))
        return np.array(rows, dtype=np.float32)

    def provenance(self):
        return {"name": self.name}


class FakeVocosMel:
    def synth(self, wav24):
        return (0.5 * wav24).astype(np.float32)

    def describe(self):
        return {"fake": "vocos_mel"}


class FakeEnPlus:
    """encode/decode stand-ins: the decoded audio is constant and equals the speaker tag it was given."""

    def __init__(self):
        self.encodes = 0

    def encode(self, wav24):
        self.encodes += 1
        return {"tag": float(np.sum(wav24[:10]))}

    def spk_emb(self, code):
        return np.full(4, code["tag"], dtype=np.float32)

    def decode(self, code, spk_emb, n16):
        return np.full(n16, spk_emb[0], dtype=np.float32)

    def describe(self):
        return {"fake": "enplus16"}


class FakeReExtractor:
    """Returns ``T - 1`` frames of a fixed pattern per head; an utterance of the marked length fails."""

    def __init__(self, head):
        self.head = head
        self.calls = 0
        self.fail_length = None

    def extract(self, wav, sr):
        self.calls += 1
        if self.fail_length is not None and len(wav) == self.fail_length:
            raise RuntimeError("boom")
        T = len(wav) // (HOP if sr == 24000 else 320) - 1
        feats = np.zeros((T, 15), dtype=np.float32)
        feats[:, 12] = 120.0
        feats[:, 14] = 0.7
        feats[:, 0] = {"refit": 1.0, "shipped": 2.0}[self.head]
        return {"feats": feats, "loud_raw": np.full(T, 0.05, np.float32), "T": T}

    def describe(self):
        return {"head": self.head}


class Models:
    def __init__(self):
        self.utmos, self.asr = FakeUTMOS(), FakeASR()
        self.vocos_mel, self.enplus = FakeVocosMel(), FakeEnPlus()
        self.reextract: dict[str, FakeReExtractor] = {}
        self.built: list[str] = []

    def make(self, key: str):
        self.built.append(key)
        if key == "utmos":
            return self.utmos
        if key == "asr":
            return self.asr
        if key == "vocos_mel":
            return self.vocos_mel
        if key == "enplus16":
            return self.enplus
        if key.startswith("spk:"):
            return FakeSpeaker(key[4:])
        if key.startswith("reextract:"):
            head = key.split(":")[1]
            return self.reextract.setdefault(head, FakeReExtractor(head))
        raise KeyError(key)


@pytest.fixture
def env(tmp_path, monkeypatch):
    build_toy_cache(tmp_path)
    set_env(monkeypatch, tmp_path)
    models = Models()
    monkeypatch.setattr(stages, "make_model", lambda ctx, key: models.make(key))

    def make_ctx(*overrides, flag=None):
        cfg = toy_config(tmp_path, f"eval.chunk_size={CHUNK}", "eval.device=cpu", *overrides)
        return stages.EvalContext(cfg, flag or io.StopFlag())

    return SimpleNamespace(root=tmp_path, make_ctx=make_ctx, models=models, monkeypatch=monkeypatch)


def runs_of(path: Path) -> list[dict]:
    return io.read_json(path / "meta.json")["runs"]


def prepare_audio(ctx, *names):
    """gt plus the requested reference systems, produced by the real stages with the fake models."""
    stages.run_stage(ctx, "gt", "gt", "T1")
    for name in names:
        stages.run_stage(ctx, "refs", name, "all")


# ------------------------------------------------------------------------------------------------- context


def test_context_ids_chunks_and_limit(env):
    ctx = env.make_ctx()
    assert len(ctx.ids) == N_IDS and ctx.ids == sorted(ctx.ids)
    assert [len(c) for c in ctx.chunks("T1")] == [8, 8, 4]
    t2 = ctx.chunks("T2")
    assert sum(len(c) for c in t2) == N_T2 and "305_1_000000_000000" not in sum(t2, [])
    assert ctx.chunks("T1")[1] == ctx.ids[8:16]
    small = env.make_ctx("eval.limit=6")
    assert len(small.ids) == 6 and small.paths.root == env.root / "eval" / "smoke_6"
    assert small.ids == io.select_limit_ids(ctx.ids, 6, 0)
    assert ctx.model("utmos") is ctx.model("utmos") and env.models.built == ["utmos"]  # built once


# ------------------------------------------------------------------------------------------------- gt and references


def test_gt_stage_writes_the_gained_prefix_and_skips_finished_chunks(env):
    ctx = env.make_ctx()
    spec = ctx.systems["gt"]
    report = stages.stage_gt(ctx, spec, "T1")
    assert report.computed == [0, 1, 2] and report.skipped == []
    out = ctx.paths.audio_dir(spec, "T1")
    for uid in ctx.ids:
        wav, rate = io.read_wav(out / f"{uid}.wav")
        row = ctx.table.set_index("id").loc[uid]
        assert rate == 24000 and wav.shape == (HOP * int(row["T"]),)
        assert np.abs(wav).max() == pytest.approx(10 ** (-3 / 20), rel=1e-3)
        assert np.array_equal(wav, ctx.gt_from_store(uid))
    meta = io.read_json(out / "meta.json")
    assert [r["status"] for r in meta["runs"]] == ["done"] and meta["runs"][0]["computed_chunks"] == [0, 1, 2]
    assert meta["chunk_size"] == CHUNK and meta["stage"] == "gt" and meta["split"] == SPLIT and "fork_commit" in meta["runs"][0]
    # nothing to do: no new run record, no rewrite
    stamp = (out / f"{ctx.ids[0]}.wav").stat().st_mtime_ns
    report = stages.stage_gt(ctx, spec, "T1")
    assert report.computed == [] and report.skipped == [0, 1, 2] and len(runs_of(out)) == 1
    assert (out / f"{ctx.ids[0]}.wav").stat().st_mtime_ns == stamp
    # a lost file recomputes its chunk only (and only that file)
    (out / f"{ctx.ids[9]}.wav").unlink()
    report = stages.stage_gt(ctx, spec, "T1")
    assert report.computed == [1] and report.skipped == [0, 2] and (out / f"{ctx.ids[9]}.wav").is_file()
    assert (out / f"{ctx.ids[0]}.wav").stat().st_mtime_ns == stamp


def test_vocos_mel_and_enplus16_use_the_gt_audio_and_the_t2_reference_embedding(env):
    ctx = env.make_ctx()
    prepare_audio(ctx, "vocos_mel", "enplus16")
    frames = ctx.table.set_index("id")["T"]
    for uid in ctx.ids[:4]:
        gt = ctx.gt_from_store(uid)
        mel, _ = io.read_wav(ctx.paths.audio_path(ctx.systems["vocos_mel"], "T1", uid))
        assert mel.shape == gt.shape and np.allclose(mel, 0.5 * gt)
    assert not ctx.paths.audio_dir(ctx.systems["vocos_mel"], "T2").exists()  # one condition only
    spec = ctx.systems["enplus16"]
    table = ctx.table.set_index("id")
    for uid in ctx.condition_ids("T2"):
        own, rate = io.read_wav(ctx.paths.audio_path(spec, "T1", uid))
        t2, rate_t2 = io.read_wav(ctx.paths.audio_path(spec, "T2", uid))
        assert rate == rate_t2 == 16000 and own.shape == t2.shape == (320 * int(frames[uid]),)
        own_tag = float(np.sum(ctx.gt_from_store(uid)[:10]))
        ref_tag = float(np.sum(ctx.gt_from_store(table.loc[uid, "ref_id"])[:10]))
        assert own[0] == pytest.approx(own_tag) and t2[0] == pytest.approx(ref_tag) and own_tag != ref_tag
    assert not ctx.paths.audio_path(spec, "T2", "305_1_000000_000000").exists()  # no reference
    assert ctx.paths.audio_path(spec, "T1", "305_1_000000_000000").exists()
    meta = io.read_json(ctx.paths.audio_dir(spec, "T2") / "meta.json")
    assert meta["runs"][0]["models"] == {"enplus16": {"fake": "enplus16"}}


def test_refs_stage_rejects_wrong_lengths(env):
    ctx = env.make_ctx()
    stages.run_stage(ctx, "gt", "gt", "T1")
    env.monkeypatch.setattr(FakeVocosMel, "synth", lambda self, wav24: wav24[:-1])
    with pytest.raises(ValueError, match="expected"):
        stages.run_stage(ctx, "refs", "vocos_mel", "T1")
    assert runs_of(ctx.paths.audio_dir(ctx.systems["vocos_mel"], "T1"))[-1]["status"] == "failed"
    assert not list((env.root / "eval" / "done").rglob("refs__*"))


def _fail_encode_for(env, ctx, uid: str, state: dict) -> None:
    """Makes ``FakeEnPlus.encode`` raise (like SPARC's filters on a too short input) for the gt audio of ``uid``."""
    bad = ctx.gt_from_store(uid)
    real = FakeEnPlus.encode

    def encode(self, wav24):
        if state["fail"] and len(wav24) == len(bad) and np.array_equal(wav24, bad):
            raise ValueError("The length of the input vector x must be greater than padlen, which is 18.")
        return real(self, wav24)

    env.monkeypatch.setattr(FakeEnPlus, "encode", encode)


def test_refs_stage_goes_on_past_an_utterance_the_model_cannot_process(env):
    ctx = env.make_ctx("eval.spk_models=[ecapa]", "eval.aggregate.n_boot=20")
    stages.run_stage(ctx, "gt", "gt", "T1")
    bad, state = ctx.ids[3], {"fail": True}
    _fail_encode_for(env, ctx, bad, state)
    spec = ctx.systems["enplus16"]
    stages.run_stage(ctx, "refs", "enplus16", "T1")  # does not raise
    failures = io.read_json(ctx.paths.audio_failures_path(spec, "T1"))
    assert list(failures) == [bad] and "padlen" in failures[bad]
    assert not ctx.paths.audio_path(spec, "T1", bad).exists()
    assert sum(ctx.paths.audio_path(spec, "T1", uid).is_file() for uid in ctx.ids) == N_IDS - 1
    assert (env.root / "eval" / "done" / SPLIT / "refs__enplus16__T1").is_file()
    # the metric stages are not blocked: the utterance is a NaN row with an error, the others are scored
    stages.run_stage(ctx, "utmos", "enplus16", "T1")
    part = pd.concat([pd.read_parquet(p) for p in sorted(ctx.paths.results_dir("enplus16", "T1", "utmos").glob("part-*.parquet"))])
    row = part.set_index("id").loc[bad]
    assert np.isnan(row["utmos"]) and row["err"] and part["utmos"].notna().sum() == N_IDS - 1
    stages.run_stage(ctx, "aggregate", "all", "all")  # the failure is reported, not just averaged away
    meta = io.read_json(ctx.paths.tables_dir() / "meta.json")
    assert list(meta["audio_failures"]["enplus16 T1"]) == [bad]
    assert "Audio that could not be made" in (ctx.paths.tables_dir() / "table.md").read_text()
    # a later run tries the failed utterance again and clears the record once it works
    state["fail"] = False
    stages.run_stage(ctx, "refs", "enplus16", "T1")
    assert io.read_json(ctx.paths.audio_failures_path(spec, "T1")) == {}
    assert ctx.paths.audio_path(spec, "T1", bad).is_file()


def test_refs_failures_beyond_the_budget_stop_the_stage(env):
    ctx = env.make_ctx("eval.max_audio_failures=0")
    stages.run_stage(ctx, "gt", "gt", "T1")
    _fail_encode_for(env, ctx, ctx.ids[3], {"fail": True})
    with pytest.raises(RuntimeError, match="max_audio_failures=0"):
        stages.run_stage(ctx, "refs", "enplus16", "T1")
    assert not list((env.root / "eval" / "done").rglob("refs__enplus16*"))


# ------------------------------------------------------------------------------------------------- metric stages


def test_utmos_stop_and_resume_do_not_recompute_finished_chunks(env):
    flag = io.StopFlag()
    ctx = env.make_ctx(flag=flag)
    prepare_audio(ctx)
    fake = env.models.utmos
    fake.after = lambda: flag.set() if len(fake.calls) == 2 else None  # the signal arrives during chunk 1
    with pytest.raises(io.StopRequested):
        stages.run_stage(ctx, "utmos", "gt", "T1")
    directory = ctx.paths.results_dir("gt", "T1", "utmos")
    assert [k for k, _ in io.list_parts(directory, "parquet")] == [0, 1] and fake.calls == [8, 8]
    assert runs_of(directory)[-1]["status"] == "stopped"
    assert not stages.done_path(ctx.paths.root, SPLIT, "utmos", "gt", "T1").exists()
    flag.clear()
    fake.after = None
    fake.calls.clear()
    stages.run_stage(ctx, "utmos", "gt", "T1")
    assert fake.calls == [4]  # chunk 2 only
    frame = io.read_parquet_parts(directory)
    assert frame["id"].tolist() == ctx.ids and list(frame.columns) == ["id", "speaker", "chapter", "dur_s", "ref_id", "utmos", "err"]
    assert (frame["ref_id"] == frame["id"]).all() and (frame["err"] == "").all()
    wav16 = io.to_16k(ctx.gt_from_store(ctx.ids[3]), 24000)
    assert frame["utmos"][3] == pytest.approx(float(np.mean(np.abs(wav16))), rel=1e-5)
    runs = runs_of(directory)
    assert [r["status"] for r in runs] == ["stopped", "done"] and runs[1]["computed_chunks"] == [2] and runs[1]["skipped_chunks"] == [0, 1]
    assert runs[1]["models"] == {"utmos": {"fake": "utmos"}}
    assert stages.done_path(ctx.paths.root, SPLIT, "utmos", "gt", "T1").is_file()
    fake.calls.clear()
    stages.run_stage(ctx, "utmos", "gt", "T1")
    assert fake.calls == [] and len(runs_of(directory)) == 2  # nothing left: no model call, no new record


def test_signal_stage_records_a_failing_utterance_as_nan_with_its_error(env):
    from sparc.vocoders.eval.metrics import signal as signal_module

    calls = []

    def fake_signal(ref24, deg, deg_sr, cfg):
        calls.append((len(ref24), len(deg), deg_sr))
        if len(calls) == 3:
            raise ValueError("boom")
        return {"pesq_wb": 3.0, "mcd": 1.5, "mrstft": 0.5, "mel_l1": 0.1}

    env.monkeypatch.setattr(signal_module, "signal_metrics", fake_signal)
    ctx = env.make_ctx()
    prepare_audio(ctx, "vocos_mel", "enplus16")
    stages.run_stage(ctx, "signal", "vocos_mel", "all")
    stages.run_stage(ctx, "signal", "enplus16", "all")
    frame = io.read_parquet_parts(ctx.paths.results_dir("vocos_mel", "T1", "signal"))
    assert len(frame) == N_IDS and frame["err"].tolist().count("") == N_IDS - 1
    bad = frame[frame["err"] != ""].iloc[0]
    assert "boom" in bad["err"] and np.isnan(bad["pesq_wb"]) and frame["pesq_wb"].notna().sum() == N_IDS - 1
    assert bad["id"] == ctx.ids[2]
    t2 = io.read_parquet_parts(ctx.paths.results_dir("enplus16", "T2", "signal"))
    assert len(t2) == N_T2 and t2["ref_id"].tolist() == ctx.table.set_index("id").loc[t2["id"], "ref_id"].tolist()
    assert any(c[2] == 16000 and c[1] < c[0] for c in calls)  # a 16 kHz system is passed at its own rate
    assert not io.list_parts(ctx.paths.results_dir("gt", "T1", "signal"), "parquet")  # gt is its own reference


def test_asr_stage_counts_edits_and_corpus_sums(env):
    from sparc.vocoders.eval.metrics import asr as asr_module

    env.monkeypatch.setattr(asr_module, "normalize_text", lambda text: " ".join(text.lower().split()))
    ctx = env.make_ctx()
    prepare_audio(ctx)
    stages.run_stage(ctx, "asr", "gt", "T1")
    frame = io.read_parquet_parts(ctx.paths.results_dir("gt", "T1", "asr"))
    assert list(frame.columns[5:]) == ["hyp", "ref_norm", "hyp_norm", "word_errors", "word_ref_len", "char_errors", "char_ref_len", "err"]
    assert (frame["word_ref_len"] == 5).all() and (frame["hyp"] == "the quick brown fox jumps").all()
    for _, row in frame.iterrows():
        want = asr_module.edit_counts(row["ref_norm"], row["hyp_norm"])
        assert row["word_errors"] == want["word_errors"] and row["char_ref_len"] == want["char_ref_len"]
    assert frame["word_errors"].sum() / frame["word_ref_len"].sum() == pytest.approx(frame["word_errors"].mean() / 5)
    reference = (Path(ctx.table.set_index("id").loc[ctx.ids[0], "wav_path"]).with_suffix(".normalized.txt")).read_text().strip()
    assert frame["ref_norm"][0] == reference.lower()


def test_asr_model_error_gives_nan_counts_and_the_error_text(env):
    from sparc.vocoders.eval.metrics import asr as asr_module

    env.monkeypatch.setattr(asr_module, "normalize_text", lambda text: text.lower())
    env.monkeypatch.setattr(FakeASR, "transcribe", lambda self, w, errors=None: (errors.extend(["RuntimeError: oom"] * len(w)), [""] * len(w))[1])
    ctx = env.make_ctx()
    prepare_audio(ctx)
    stages.run_stage(ctx, "asr", "gt", "T1")
    frame = io.read_parquet_parts(ctx.paths.results_dir("gt", "T1", "asr"))
    assert (frame["err"] == "RuntimeError: oom").all() and frame["word_errors"].isna().all() and frame["word_ref_len"].isna().all()
    assert (frame["ref_norm"] != "").all()  # the reference text is still recorded


def test_spk_stage_writes_normalized_embeddings_per_model(env):
    ctx = env.make_ctx("eval.spk_models=[ecapa,wavlm_sv]")
    prepare_audio(ctx, "vocos_mel")
    stages.run_stage(ctx, "spk", "vocos_mel", "T1")
    for model in ("ecapa", "wavlm_sv"):
        directory = ctx.paths.embeddings_dir("vocos_mel", "T1", model)
        assert [k for k, _ in io.list_parts(directory, "npz")] == [0, 1, 2]
        parts = [io.read_npz_part(p) for _, p in io.list_parts(directory, "npz")]
        ids = np.concatenate([p["ids"] for p in parts]).tolist()
        emb = np.concatenate([p["emb"] for p in parts])
        assert ids == ctx.ids and emb.shape == (N_IDS, 6) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
        assert runs_of(directory)[-1]["models"] == {model: {"name": model}}
    ctx2 = env.make_ctx("eval.spk_models=[ecapa]")  # a model that is not configured is not run
    assert stages.speaker_models(ctx2) == ["ecapa"]


# ------------------------------------------------------------------------------------------------- re-extraction and prosody


def test_reextract_part_layout_failures_and_resume(env):
    ctx = env.make_ctx()
    prepare_audio(ctx)
    victim = ctx.ids[10]
    env.models.make("reextract:refit").fail_length = HOP * int(ctx.table.set_index("id").loc[victim, "T"])
    stages.run_stage(ctx, "reextract", "gt", "T1")
    directory = ctx.paths.arrays_dir("gt", "T1")
    assert [k for k, _ in io.list_parts(directory, "npz")] == [0, 1, 2]
    part = stages.unpack_reextraction(io.read_npz_part(directory / "part-0001.npz"))
    assert list(part)[:3] == ctx.ids[8:11] and list(part) == ctx.ids[8:16]
    T = int(ctx.table.set_index("id").loc[ctx.ids[8], "T"])
    feats, loud, err = part[ctx.ids[8]]
    assert feats.shape == (T - 1, 15) and loud.shape == (T - 1,) and err == "" and feats[0, 0] == 1.0
    feats, loud, err = part[victim]
    assert len(feats) == 0 and len(loud) == 0 and "boom" in err
    calls = env.models.reextract["refit"].calls
    stages.run_stage(ctx, "reextract", "gt", "T1")
    assert env.models.reextract["refit"].calls == calls  # nothing recomputed
    # gt_shipped re-extracts the gt audio with the other head and has its own array directory
    stages.run_stage(ctx, "reextract", "gt_shipped", "T1")
    shipped = stages.unpack_reextraction(io.read_npz_part(ctx.paths.arrays_dir("gt_shipped", "T1") / "part-0000.npz"))
    assert shipped[ctx.ids[0]][0][0, 0] == 2.0 and env.models.built.count("reextract:shipped") == 1
    assert not (ctx.paths.root / "audio" / "gt_shipped").exists()  # it reuses the gt audio


def write_reextraction(ctx, system, condition, make):
    """Writes reextract parts for every chunk; ``make(uid, T)`` returns ``(feats (T-1, 15), loud_raw (T-1,))``."""
    frames = ctx.table.set_index("id")["T"]
    for k, ids in enumerate(ctx.chunks(condition)):
        if not ids:
            continue
        feats, loud = zip(*(make(uid, int(frames[uid])) for uid in ids))
        io.write_npz(
            ctx.paths.arrays_dir(system, condition) / io.part_name(k, "npz"),
            {
                "ids": np.array(ids),
                "lengths": np.array([len(f) for f in feats]),
                "feats": np.concatenate(feats),
                "loud_raw": np.concatenate(loud),
                "err": np.array([""] * len(ids)),
            },
        )


def test_prosody_references_gt_floor_vocoder_and_shipped_head(env):
    ctx = env.make_ctx()
    store, gain = ctx.store, ctx.gain_db

    def cached(uid, T):
        utt = store.index_of(uid)
        feats, loud = store.frames(utt, 0, T)
        return feats, loud * gain_factor(store.peak24[utt], gain)

    def gt_make(uid, T):  # extraction floor 0: the gt re-extraction reproduces the cached input exactly
        feats, loud = cached(uid, T)
        return feats[: T - 1].copy(), loud[: T - 1]

    def vocoder_make(uid, T):  # pitch 100 cents sharp relative to gt, loudness +6.02 dB
        feats, loud = gt_make(uid, T)
        feats[:, 12] *= 2 ** (1 / 12)
        return feats, loud * 2.0

    def shipped_make(uid, T):  # shipped-head gt: EMA offset by 0.5 of the unit std
        feats, loud = gt_make(uid, T)
        feats[:, :12] += 0.5 * np.asarray(ctx.stats["ema_std"])
        return feats, loud

    def enplus_make(uid, T):  # reproduces the shipped-head EMA exactly, pitch exact
        return shipped_make(uid, T)

    write_reextraction(ctx, "gt", "T1", gt_make)
    write_reextraction(ctx, "gt_shipped", "T1", shipped_make)
    write_reextraction(ctx, "enplus16", "T1", enplus_make)
    for cond in ("T1", "T2", "T3"):
        write_reextraction(ctx, "hifigan", cond, vocoder_make)
    spec = ctx.systems["hifigan"]
    fingerprint = {"ckpt": "/x/step4.ckpt", "g_step": 4}
    io.write_json(ctx.paths.synth_meta_path(spec), {"fingerprint": fingerprint, "checkpoint": {"path": "/x/step4.ckpt", "g_step": 4, "max_g_steps": 4, "final": True}})

    stages.run_stage(ctx, "prosody", "gt", "T1")
    gt = io.read_parquet_parts(ctx.paths.results_dir("gt", "T1", "prosody"))
    assert list(gt.columns[:6]) == ["id", "speaker", "chapter", "dur_s", "ref_id", "f0_rmse_cents"] and list(gt.columns)[-1] == "err"
    voiced = gt["n_voiced_both"] > 0
    assert voiced.sum() >= N_IDS - 2
    assert np.allclose(gt.loc[voiced, "f0_rmse_cents"], 0, atol=1e-6) and np.allclose(gt["ema_rmse_mean"], 0, atol=1e-6)
    assert np.allclose(gt["loud_db_rmse"], 0, atol=1e-4) and np.allclose(gt["vde"], 0) and (gt["n_frames_dropped"] == 1).all()

    stages.run_stage(ctx, "prosody", "hifigan", "T2")
    h = io.read_parquet_parts(ctx.paths.results_dir("hifigan", "T2", "prosody"))
    assert len(h) == N_T2 and np.allclose(h["f0_rmse_cents"].dropna(), 100.0, atol=1e-3)  # against the gt re-extraction, not the cache
    assert np.allclose(h["loud_db_bias"].dropna(), 20 * np.log10(2), atol=0.05) and np.allclose(h["ema_rmse_mean"], 0, atol=1e-6)
    assert io.read_json(ctx.paths.results_dir("hifigan", "T2", "prosody") / "meta.json")["fingerprint"] == fingerprint

    stages.run_stage(ctx, "prosody", "enplus16", "T1")
    e = io.read_parquet_parts(ctx.paths.results_dir("enplus16", "T1", "prosody"))
    assert np.allclose(e["ema_rmse_mean"], 0, atol=1e-6)  # compared with the shipped-head gt (ema_ref), which would be 0.5 off the cache
    assert np.allclose(e["f0_rmse_cents"].dropna(), 0, atol=1e-6)


def test_prosody_needs_the_reextraction_first(env):
    ctx = env.make_ctx()
    with pytest.raises(stages.MissingPrerequisite, match="stage=reextract"):
        stages.run_stage(ctx, "prosody", "gt", "T1")


# ------------------------------------------------------------------------------------------------- vocoder systems


def fake_checkpoint(root: Path, experiment: str, vocoder: str, step: int) -> Path:
    path = root / "runs" / experiment / vocoder / "ckpt" / f"step{step:09d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}, "vocoder_counters": {"g_step": step, "d_step": step}}, path)
    return path


class FakeVocoder(LightningModule):
    calls: list = []
    after = None

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        FakeVocoder.calls.append((batch["id"][0], batch["condition"][0]))
        if FakeVocoder.after:
            FakeVocoder.after(len(FakeVocoder.calls))
        n = batch["features"].shape[-1]
        return {"id": batch["id"], "condition": batch["condition"], "wav": torch.full((1, 1, HOP * n), 0.25)}


@pytest.fixture
def synth_env(env):
    FakeVocoder.calls, FakeVocoder.after = [], None
    env.monkeypatch.setattr(predict_vocoder, "VocoderGANModule", lambda cfg, stats=None: FakeVocoder())
    return env


def test_synth_requires_the_final_checkpoint_and_pins_it(synth_env):
    env = synth_env
    fake_checkpoint(env.root, "main", "hifigan", 3)
    ctx = env.make_ctx("eval.systems.hifigan.overrides=['train.max_g_steps=5']")
    with pytest.raises(RuntimeError, match="g_step 3 of 5"):
        stages.run_stage(ctx, "synth", "hifigan", "T1")
    assert FakeVocoder.calls == []
    fake_checkpoint(env.root, "main", "hifigan", 5)
    stages.run_stage(ctx, "synth", "hifigan", "all")
    spec = ctx.systems["hifigan"]
    assert len(FakeVocoder.calls) == N_IDS + 2 * N_T2
    meta = io.read_json(ctx.paths.synth_meta_path(spec))
    assert meta["checkpoint"]["g_step"] == 5 and meta["checkpoint"]["final"] is True
    assert meta["fingerprint"] == {"ckpt": str(env.root / "runs/main/hifigan/ckpt/step000000005.ckpt"), "g_step": 5}
    assert io.audio_problems(ctx.paths, spec, "T2", ctx.table, ctx.condition_ids("T2")) == {}
    # a newer checkpoint appears later: continuing into the same prediction directory is refused
    fake_checkpoint(env.root, "main", "hifigan", 6)
    ctx.paths.audio_path(spec, "T1", ctx.ids[0]).unlink()
    with pytest.raises(RuntimeError, match="fingerprint"):
        stages.stage_synth(ctx, spec, ["T1"])


def test_synth_stop_and_resume_without_recomputing(synth_env):
    env = synth_env
    fake_checkpoint(env.root, "main", "ddsp", 2)
    flag = io.StopFlag()
    ctx = env.make_ctx("eval.systems.ddsp.overrides=['train.max_g_steps=2']", flag=flag)
    FakeVocoder.after = lambda n: flag.set() if n == 5 else None  # the signal arrives while the fifth utterance is synthesized
    with pytest.raises(io.StopRequested):
        stages.run_stage(ctx, "synth", "ddsp", "T1")
    spec = ctx.systems["ddsp"]
    done = sorted(p.stem for p in ctx.paths.audio_dir(spec, "T1").glob("*.wav"))
    assert len(done) == 5 and io.read_json(ctx.paths.synth_meta_path(spec))["runs"][-1]["status"] == "stopped"
    flag.clear()
    FakeVocoder.after = None
    first = [c[0] for c in FakeVocoder.calls]
    FakeVocoder.calls = []
    stages.run_stage(ctx, "synth", "ddsp", "T1")
    assert not set(first) & {c[0] for c in FakeVocoder.calls} and len(FakeVocoder.calls) == N_IDS - 5
    assert stages.done_path(ctx.paths.root, SPLIT, "synth", "ddsp", "T1").is_file()
    assert [r["status"] for r in io.read_json(ctx.paths.synth_meta_path(spec))["runs"]] == ["stopped", "done"]


def test_results_of_a_vocoder_are_tied_to_its_checkpoint(synth_env):
    env = synth_env
    ctx = env.make_ctx()
    spec = ctx.systems["vocos"]
    with pytest.raises(stages.MissingPrerequisite, match="stage=synth"):
        stages.run_stage(ctx, "utmos", "vocos", "T1")
    frames = ctx.table.set_index("id")["T"]

    def fake_synth(step: int) -> None:
        io.write_json(ctx.paths.synth_meta_path(spec), {"fingerprint": {"ckpt": "c", "g_step": step}})
        for uid in ctx.ids:
            io.write_wav(ctx.paths.audio_path(spec, "T1", uid), np.zeros(HOP * int(frames[uid]), np.float32), 24000)

    fake_synth(1)
    stages.run_stage(ctx, "utmos", "vocos", "T1")
    fake_synth(2)
    # complete parts of the old checkpoint are not taken for results of the new one, also when nothing would run
    with pytest.raises(RuntimeError, match="fingerprint"):
        stages.stage_utmos(ctx, spec, "T1")
    (ctx.paths.results_dir("vocos", "T1", "utmos") / "part-0002.parquet").unlink()
    with pytest.raises(RuntimeError, match="fingerprint"):
        stages.stage_utmos(ctx, spec, "T1")
    relaxed = env.make_ctx("eval.allow_fingerprint_change=true")
    stages.stage_utmos(relaxed, relaxed.systems["vocos"], "T1")
    assert len(io.read_parquet_parts(ctx.paths.results_dir("vocos", "T1", "utmos"))) == N_IDS


# ------------------------------------------------------------------------------------------------- dispatch and CLI


def test_system_all_goes_on_past_a_missing_prerequisite_and_fails_at_the_end(env):
    ctx = env.make_ctx()
    prepare_audio(ctx, "vocos_mel", "enplus16")
    with pytest.raises(stages.MissingPrerequisite, match="9 item") as error:
        stages.run_stage(ctx, "utmos", "all", "all")  # the three vocoders were not synthesized
    assert "utmos hifigan T1" in str(error.value) and "utmos vocos T3" in str(error.value)
    for name, conds in (("gt", ["T1"]), ("vocos_mel", ["T1"]), ("enplus16", ["T1", "T2"])):
        for cond in conds:
            assert len(io.read_parquet_parts(ctx.paths.results_dir(name, cond, "utmos"))) > 0  # the others ran
    assert not stages.done_path(ctx.paths.root, SPLIT, "utmos", "all", "all").exists()
    with pytest.raises(stages.MissingPrerequisite):
        stages.run_stage(ctx, "utmos", "hifigan", "T1")  # an explicit system fails at once


def test_stage_and_system_selection_errors(env):
    ctx = env.make_ctx()
    with pytest.raises(ValueError, match="does not apply"):
        stages.run_stage(ctx, "signal", "gt", "T1")
    with pytest.raises(ValueError, match="unknown system"):
        stages.run_stage(ctx, "utmos", "nope", "T1")
    with pytest.raises(ValueError, match="no condition T3"):
        stages.run_stage(ctx, "utmos", "enplus16", "T3")
    with pytest.raises(ValueError, match="unknown stage"):
        stages.run_stage(ctx, "bogus", "all", "all")
    assert not (ctx.paths.root / "done").exists()


def test_probes_and_efficiency_delegate_and_honour_a_stop(env):
    from sparc.vocoders.eval import efficiency as efficiency_module
    from sparc.vocoders.eval import probes as probes_module

    flag = io.StopFlag()
    ctx = env.make_ctx(flag=flag)
    calls = []

    def fake_probes(cfg, system, device, out_dir, stop):
        calls.append(("probes", system, str(device), Path(out_dir).relative_to(env.root), stop()))
        if len(calls) == 1:
            flag.set()  # the stop arrives before the work is finished: no results file
            return
        io.write_parquet(Path(out_dir) / cfg.eval_features.probes.results_name, pd.DataFrame({"x": [1]}))

    def fake_efficiency(cfg, system, device, out_path):
        calls.append(("efficiency", system, device, Path(out_path).relative_to(env.root)))
        return {}

    env.monkeypatch.setattr(probes_module, "run_probes", fake_probes)
    env.monkeypatch.setattr(efficiency_module, "run_efficiency", fake_efficiency)
    env.monkeypatch.setattr(stages, "resolve_checkpoint", lambda vcfg: CheckpointInfo("/runs/x/ckpt/step000000010.ckpt", 10, 10))
    with pytest.raises(io.StopRequested):
        stages.run_stage(ctx, "probes", "hifigan", "all")
    assert calls == [("probes", "hifigan", "cpu", Path("eval/probes/hifigan"), False)]
    flag.clear()
    stages.run_stage(ctx, "probes", "hifigan", "all")
    assert len(calls) == 2 and (env.root / "eval/probes/hifigan/results.parquet").is_file()
    stages.run_stage(ctx, "probes", "hifigan", "all")
    assert len(calls) == 2  # results exist: nothing to do
    calls.clear()
    stages.run_stage(ctx, "efficiency", "all", "all")
    assert [c[1] for c in calls] == ["extractor", "vocos_mel", "enplus16", "hifigan", "ddsp", "vocos"]
    assert calls[0][3] == Path("eval/efficiency/extractor.json") and all(c[2] == "cpu" for c in calls)
    calls.clear()
    stages.run_stage(ctx, "efficiency", "extractor", "all")
    assert [c[1] for c in calls] == ["extractor"]
    with pytest.raises(ValueError, match="does not apply"):
        stages.run_stage(ctx, "probes", "enplus16", "all")


def test_probes_and_efficiency_pin_and_check_the_checkpoint(env):
    """Both stages evaluate the checkpoint ``synth`` would use: final only, pinned in the config, never silently changed."""
    import json

    from sparc.vocoders.eval import efficiency as efficiency_module
    from sparc.vocoders.eval import probes as probes_module

    seen = []
    info = {"value": CheckpointInfo("/runs/x/ckpt/step000000007.ckpt", 7, 10)}
    env.monkeypatch.setattr(stages, "resolve_checkpoint", lambda vcfg: info["value"])

    def fake_probes(cfg, system, device, out_dir, stop):
        seen.append(("probes", cfg.eval.systems[system].ckpt))
        io.write_parquet(Path(out_dir) / cfg.eval_features.probes.results_name, pd.DataFrame({"x": [1]}))
        io.write_json(Path(out_dir) / "probes_meta.json", {"checkpoint": {"ckpt": info["value"].path, "g_step": info["value"].g_step}})

    def fake_efficiency(cfg, system, device, out_path):
        if system == "extractor":
            seen.append(("efficiency", None))
            return
        seen.append(("efficiency", cfg.eval.systems[system].ckpt))
        io.write_json(out_path, {"meta": {"checkpoint": {"ckpt": info["value"].path, "g_step": info["value"].g_step}}})

    env.monkeypatch.setattr(probes_module, "run_probes", fake_probes)
    env.monkeypatch.setattr(efficiency_module, "run_efficiency", fake_efficiency)
    ctx = env.make_ctx()
    with pytest.raises(RuntimeError, match="final checkpoints only"):
        stages.run_stage(ctx, "probes", "hifigan", "all")
    with pytest.raises(RuntimeError, match="final checkpoints only"):
        stages.run_stage(ctx, "efficiency", "hifigan", "all")
    assert seen == []
    ctx = env.make_ctx("eval.require_final=false")
    stages.run_stage(ctx, "probes", "hifigan", "all")
    stages.run_stage(ctx, "efficiency", "hifigan", "all")
    assert seen == [("probes", info["value"].path), ("efficiency", info["value"].path)]
    # the original config is not modified by the pinning
    assert ctx.cfg.eval.systems.hifigan.ckpt is None
    # a newer checkpoint appeared: finished results of the old one are not reused
    info["value"] = CheckpointInfo("/runs/x/ckpt/step000000009.ckpt", 9, 10)
    with pytest.raises(RuntimeError, match="made with checkpoint"):
        stages.run_stage(ctx, "probes", "hifigan", "all")
    with pytest.raises(RuntimeError, match="made with checkpoint"):
        stages.run_stage(ctx, "efficiency", "hifigan", "all")
    allowed = env.make_ctx("eval.require_final=false", "eval.allow_fingerprint_change=true")
    stages.run_stage(allowed, "probes", "hifigan", "all")  # results exist: nothing recomputed, no error
    assert len(seen) == 2
    # reference systems and the extractor have no checkpoint
    stages.run_stage(ctx, "efficiency", "extractor", "all")
    assert seen[-1] == ("efficiency", None) and json.loads((env.root / "eval/efficiency/hifigan.json").read_text())["meta"]["checkpoint"]["g_step"] == 7


def test_a_different_id_subset_cannot_reuse_numbered_parts_of_a_smoke_directory(env):
    """``eval.limit`` with another ``eval.seed`` selects other ids; part numbers alone would make the old parts look finished."""
    first = env.make_ctx("eval.limit=6", "eval.chunk_size=3")
    prepare_audio(first)
    stages.run_stage(first, "utmos", "gt", "T1")
    other = env.make_ctx("eval.limit=6", "eval.chunk_size=3", "eval.seed=1")
    assert other.ids != first.ids and other.paths.root == first.paths.root
    stages.run_stage(other, "gt", "gt", "T1")  # audio is keyed by id: a different subset only adds files
    with pytest.raises(RuntimeError, match="ids_digest"):
        stages.run_stage(other, "utmos", "gt", "T1")
    assert [r["status"] for r in runs_of(first.paths.results_dir("gt", "T1", "utmos"))] == ["done"]
    stages.run_stage(first, "utmos", "gt", "T1")  # the original subset is still fine (nothing to do)


def test_cli_config_composes_for_every_stage(env):
    for stage in STAGES:
        cfg = compose_config("eval_config", [f"stage={stage}", "system=hifigan", "condition=T2", "eval.split=test.other", "eval.limit=8"])
        assert cfg.stage == stage and cfg.system == "hifigan" and cfg.condition == "T2"
        assert cfg.eval.split == "test.other" and cfg.eval.limit == 8 and cfg.eval.chunk_size == 256
        assert cfg.paths.eval_root.endswith("/eval") and cfg.eval.systems.vocos.vocoder == "vocos"
        assert cfg.eval_metrics.asr.model_id == "openai/whisper-large-v3" and cfg.eval_features.probes.n_utterances == 50
    assert OmegaConf.is_missing(compose_config("eval_config", []), "stage")  # stage is mandatory


def test_cli_run_exit_codes_and_done_marker(env):
    cfg = toy_config(env.root, "stage=gt", "system=gt", "condition=T1", f"eval.chunk_size={CHUNK}", "eval.device=cpu")
    assert eval_vocoder.run(cfg) == 0
    marker = env.root / "eval" / "done" / SPLIT / "gt__gt__T1"
    assert marker.is_file() and len(list((env.root / "eval/audio/gt" / SPLIT / "T1").glob("*.wav"))) == N_IDS
    before = runs_of(env.root / "eval/audio/gt" / SPLIT / "T1")
    assert eval_vocoder.run(cfg) == 0 and runs_of(env.root / "eval/audio/gt" / SPLIT / "T1") == before  # marker short-circuits
    flag = io.StopFlag()
    flag.set()
    stop_cfg = toy_config(env.root, "stage=utmos", "system=gt", "condition=T1", f"eval.chunk_size={CHUNK}", "eval.device=cpu")
    assert eval_vocoder.run(stop_cfg, flag) == 75
    assert not (env.root / "eval/done" / SPLIT / "utmos__gt__T1").exists()
    bad = toy_config(env.root, "stage=bogus")
    with pytest.raises(ValueError, match="stage must be one of"):
        eval_vocoder.run(bad)


def test_smoke_runs_never_mix_with_full_results(env):
    full = env.make_ctx()
    smoke = env.make_ctx("eval.limit=5")
    stages.run_stage(smoke, "gt", "gt", "T1")
    assert len(list((env.root / "eval/smoke_5/audio/gt" / SPLIT / "T1").glob("*.wav"))) == 5
    assert not (env.root / "eval/audio").exists() and smoke.paths.root != full.paths.root
    assert (env.root / "eval/smoke_5/done" / SPLIT / "gt__gt__T1").is_file()


def test_a_metric_stage_without_its_audio_fails_before_writing_anything(env):
    ctx = env.make_ctx()
    with pytest.raises(stages.MissingPrerequisite, match="run stage=gt first"):
        stages.run_stage(ctx, "utmos", "gt", "T1")
    with pytest.raises(stages.MissingPrerequisite, match="run stage=refs first"):
        stages.run_stage(ctx, "spk", "vocos_mel", "T1")
    prepare_audio(ctx, "vocos_mel")
    (ctx.paths.audio_path(ctx.systems["vocos_mel"], "T1", ctx.ids[7])).write_bytes(b"not a wav")
    with pytest.raises(stages.MissingPrerequisite, match="1 of 20"):
        stages.run_stage(ctx, "utmos", "vocos_mel", "T1")
    for group in ("utmos", "spk"):
        assert not ctx.paths.results_dir("vocos_mel", "T1", group).exists()
    assert not list((env.root / "eval" / "done").rglob("utmos__*")) and not ctx.paths.embeddings_dir("vocos_mel", "T1", "ecapa").exists()
    # the gt audio is a prerequisite of the signal stage as well
    ctx.paths.audio_path(ctx.systems["vocos_mel"], "T1", ctx.ids[7]).unlink()
    stages.run_stage(ctx, "refs", "vocos_mel", "T1")  # restores the corrupted file
    ctx.paths.audio_path(ctx.systems["gt"], "T1", ctx.ids[3]).unlink()
    with pytest.raises(stages.MissingPrerequisite, match="gt/T1"):
        stages.run_stage(ctx, "signal", "vocos_mel", "T1")


def test_aggregate_and_samples_are_rerun_every_time(env):
    ctx = env.make_ctx("eval.samples.n_female=2", "eval.samples.n_male=2", "eval.spk_models=[ecapa]", "eval.aggregate.n_boot=20")
    prepare_audio(ctx, "vocos_mel")
    stages.run_stage(ctx, "aggregate", "all", "all")  # nothing computed yet: everything is reported as missing
    out = ctx.paths.tables_dir()
    first = io.read_json(out / "meta.json")
    assert len(first["missing"]) > 0 and all(m["status"] == "missing" for m in first["missing"])
    stages.run_stage(ctx, "utmos", "gt", "T1")
    stages.run_stage(ctx, "aggregate", "all", "all")
    second = io.read_json(out / "meta.json")
    assert len(second["missing"]) == len(first["missing"]) - 1  # the new gt utmos group is picked up
    stages.run_stage(ctx, "samples", "all", "all")
    assert (ctx.paths.samples_dir() / "index.csv").is_file()
    assert not list((env.root / "eval" / "done").rglob("aggregate__*")) and not list((env.root / "eval" / "done").rglob("samples__*"))
    cfg = toy_config(env.root, "stage=aggregate", f"eval.chunk_size={CHUNK}", "eval.device=cpu", "eval.spk_models=[ecapa]", "eval.aggregate.n_boot=20")
    assert eval_vocoder.run(cfg) == 0 and eval_vocoder.run(cfg) == 0


@pytest.mark.parametrize("signum", [signal.SIGUSR1, signal.SIGTERM])
def test_a_real_signal_during_synthesis_stops_after_the_current_utterance(synth_env, signum):
    """Lightning's trainer must not replace the handler: the signal sets the flag and the stage exits cleanly."""
    env = synth_env
    fake_checkpoint(env.root, "main", "hifigan", 2)
    flag = io.StopFlag()
    ctx = env.make_ctx("eval.systems.hifigan.overrides=['train.max_g_steps=2']", flag=flag)
    FakeVocoder.after = lambda n: os.kill(os.getpid(), signum) if n == 4 else None
    before = signal.getsignal(signum)
    with flag.installed():
        with pytest.raises(io.StopRequested):
            stages.run_stage(ctx, "synth", "hifigan", "T1")
        assert flag.signal_number == signum
    spec = ctx.systems["hifigan"]
    assert len(list(ctx.paths.audio_dir(spec, "T1").glob("*.wav"))) == 4  # the utterance being synthesized was written
    assert signal.getsignal(signum) == before  # the previous handler is back once the context is left


def test_hydra_cleared_allows_composing_inside_a_running_app():
    """The probes and efficiency stages call B's loader, which composes its own config while the CLI's app is running."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from sparc.vocoders.eval.systems import hydra_cleared
    from sparc.vocoders.eval.vocoder_loader import CONF_DIR, compose_vocoder_config

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):  # stands for the running @hydra.main app
        outer = GlobalHydra.instance().hydra
        with pytest.raises(ValueError, match="already initialized"):
            compose_vocoder_config(None, "hifigan")
        with hydra_cleared():
            assert compose_vocoder_config(None, "hifigan").vocoder is not None
        assert GlobalHydra.instance().hydra is outer
        assert compose(config_name="vocoder_config", overrides=["vocoder=hifigan"]).vocoder is not None
    GlobalHydra.instance().clear()


def test_efficiency_markers_are_per_device(tmp_path):
    """GPU and CPU efficiency runs of the same item must not share a DONE marker (either would skip the other)."""
    from sparc.vocoders.eval.stages import done_path

    gpu = done_path(tmp_path, "test.clean", "efficiency", "vocos", "all", "cuda")
    cpu = done_path(tmp_path, "test.clean", "efficiency", "vocos", "all", "cpu")
    assert gpu != cpu and gpu.name.endswith("__cuda") and cpu.name.endswith("__cpu")
    assert done_path(tmp_path, "test.clean", "asr", "vocos", "all", "cuda").name == "asr__vocos__all"
