"""VCTK training set for Phase 5: listing, trimmed 24 kHz audio and the ECAPA leak check against the evaluation pools.

Phase 5 (``docs/speaker_embedding_plan.md``, "VCTK training set" and Amendment 1) trains on LibriTTS-R, GLOBE v3 and the
VCTK speakers that are not in the Phase 4 VCTK evaluation pool. This module builds that VCTK list and audio; feature
extraction and packing wait for the quality thresholds (``extract`` / ``pack`` are hooks, see below).

* ``list`` takes every ``wav48/p*`` speaker of the corpus (``mic2`` files skipped, as in ``clone_pools.list_vctk``) minus
  the speakers of ``phase4/pools/vctk/table.parquet`` (asserted disjoint), reads the native audio, finds the trim bounds
  (:func:`trim_bounds`) and keeps utterances of 1-15 s after trimming. Writes ``list.parquet`` (``id, speaker, gender,
  accent, wav48, text, native_sr, n_native, trim_start, trim_end, dur_trimmed, n24``; trim bounds are native-rate sample
  indices, so a scorer can measure bandwidth on ``wav48[trim_start:trim_end]``) and ``list_report.json`` (per-speaker
  counts and drop reasons). A speaker without transcripts is kept (``text`` is empty).
* ``prepare`` writes ``wav24/<speaker>/<id>.wav``: the trimmed native segment resampled to 24 kHz with soxr_hq (the
  resampler of ``clone_pools.write_wav24``), mono float. Sharded (``--shard i --nshards n``: speakers ``i::n``),
  resumable (existing files are kept; writes are atomic), safe to rerun.
* ``leak`` embeds the prepared audio and the utterances of all three evaluation pools (LibriTTS-R test.clean, VCTK, GLOBE)
  with the evaluation's ECAPA (``SpeakerEmbedder``, 16 kHz, soxr_hq), and drops every training utterance whose cosine to
  any pool utterance is >= ``--threshold`` (0.70, as ``globe_train``). Writes ``leak/dropped_ids.txt``, ``leak/max_cos.parquet``
  and ``leak/leak_report.json``; embeddings are cached under ``leak/emb`` (resumable).
* ``extract`` / ``pack`` are hooks: they need the Phase 5 thresholds and are refused until ``--enable`` is passed.
  :func:`kept_table` is the list without the dropped ids; the later extractor reuses
  ``sparc.vocoders.features.globe_train``'s cache procedure over it and ``features.pack.write_packed``.

Trimming. ``clone_pools`` resamples the VCTK pool to 24 kHz but does not trim it (no trimming code exists there), so
the trimming here is new: ``librosa.effects.trim`` on the native audio (``top_db`` 30 under the peak, 2048/512 frames)
plus 0.1 s of padding each side, clipped to the file. Both parameters are recorded in ``list_report.json``.

Usage (CPU)::

    python -m sparc.vocoders.features.vctk_train list
    python -m sparc.vocoders.features.vctk_train prepare --shard 0 --nshards 8
    python -m sparc.vocoders.features.vctk_train leak
"""

import argparse
import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.vocoders.constants import SAMPLE_RATE
from sparc.vocoders.features.globe_train import EXIT_STOPPED, THRESHOLD, Stop, max_cosine, write_json_atomic

VCTK_ROOT = "/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus"
EVAL_TABLE = "/data/user_data/xoy/spkemb/phase4/pools/vctk/table.parquet"
OUT_ROOT = "/data/user_data/xoy/spkemb/phase5/vctk_train"
POOLS = ("libritts", "vctk", "globe")
MIN_S, MAX_S = 1.0, 15.0
TOP_DB = 30.0
PAD_S = 0.1
LIST_COLUMNS = (
    "id", "speaker", "gender", "accent", "wav48", "text", "native_sr", "n_native", "trim_start", "trim_end",
    "dur_trimmed", "n24",
)  # fmt: skip
DROP_REASONS = ("unreadable", "too_short", "too_long")


def out_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "list": root / "list.parquet",
        "report": root / "list_report.json",
        "wav": root / "wav24",
        "leak": root / "leak",
        "emb": root / "leak" / "emb",
        "dropped": root / "leak" / "dropped_ids.txt",
        "max_cos": root / "leak" / "max_cos.parquet",
        "leak_report": root / "leak" / "leak_report.json",
    }


# ------------------------------------------------------------------------------------------------ pure helpers


def trim_bounds(wav: np.ndarray, sample_rate: int, top_db: float = TOP_DB, pad_s: float = PAD_S) -> tuple[int, int]:
    """``(start, end)`` sample indices of the non-silent part of ``wav`` plus ``pad_s`` each side, clipped to the file.

    An all-silent file (nothing above ``top_db``) returns ``(0, len(wav))`` unchanged."""
    import librosa

    _, (lo, hi) = librosa.effects.trim(np.asarray(wav, dtype=np.float32), top_db=top_db, frame_length=2048, hop_length=512)
    if hi <= lo:
        return 0, len(wav)
    pad = int(round(pad_s * sample_rate))
    return max(0, int(lo) - pad), min(len(wav), int(hi) + pad)


def drop_reason(duration_s: float) -> str | None:
    """Why an utterance of ``duration_s`` seconds after trimming is dropped (``None`` if kept)."""
    if duration_s < MIN_S:
        return "too_short"
    if duration_s > MAX_S:
        return "too_long"
    return None


def trim_resample(wav: np.ndarray, sample_rate: int, start: int, end: int) -> np.ndarray:
    """Native segment ``wav[start:end]`` at 24 kHz (soxr_hq, as ``clone_pools.write_wav24``)."""
    import librosa

    seg = np.asarray(wav[start:end], dtype=np.float32)
    if sample_rate == SAMPLE_RATE:
        return seg
    return librosa.resample(seg, orig_sr=sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")


def speaker_report(rows: pd.DataFrame) -> dict:
    """Per-speaker ``{files, kept, dropped: {reason: n}}`` of the candidate table (column ``reason`` is ``None`` if kept)."""
    out = {}
    for spk, g in rows.groupby("speaker", sort=True):
        dropped = g["reason"].dropna().value_counts()
        out[str(spk)] = {
            "files": int(len(g)),
            "kept": int(g["reason"].isna().sum()),
            "dropped": {r: int(dropped.get(r, 0)) for r in DROP_REASONS},
        }
    return out


def kept_table(root: str | Path = OUT_ROOT) -> pd.DataFrame:
    """``list.parquet`` without the ids of ``leak/dropped_ids.txt`` (error if the leak check has not run)."""
    paths = out_paths(root)
    if not paths["dropped"].is_file():
        raise FileNotFoundError(f"{paths['dropped']}: run the leak check first")
    table = pd.read_parquet(paths["list"])
    dropped = set(paths["dropped"].read_text().split())
    return table[~table["id"].isin(dropped)].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ list


def _probe(item: tuple[str, str, str]) -> dict:
    """Candidate row of one file: native header, trim bounds and drop reason (worker of the list pool)."""
    import soundfile as sf

    utt_id, speaker, path = item
    row = {"id": utt_id, "speaker": speaker, "wav48": path, "native_sr": 0, "n_native": 0, "trim_start": 0, "trim_end": 0,
           "dur_trimmed": 0.0, "n24": 0, "reason": None}  # fmt: skip
    try:
        wav, rate = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim != 1:
            wav = wav.mean(axis=1)
        lo, hi = trim_bounds(wav, rate)
    except Exception:
        row["reason"] = "unreadable"
        return row
    dur = (hi - lo) / rate
    row.update(
        native_sr=int(rate), n_native=int(len(wav)), trim_start=lo, trim_end=hi, dur_trimmed=float(dur),
        n24=int(round((hi - lo) * SAMPLE_RATE / rate)), reason=drop_reason(dur),
    )  # fmt: skip
    return row


def build_list(
    vctk_root: str | Path = VCTK_ROOT, eval_table: str | Path = EVAL_TABLE, workers: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Kept table (``LIST_COLUMNS``) and the report."""
    from sparc.vocoders.eval.clone_pools import parse_vctk_speaker_info

    root = Path(vctk_root)
    info = parse_vctk_speaker_info((root / "speaker-info.txt").read_text())
    eval_speakers = set(pd.read_parquet(eval_table, columns=["speaker"])["speaker"].astype(str))
    files = sorted(p for p in (root / "wav48").glob("p*/*.wav") if "mic2" not in p.name)
    all_speakers = {p.parent.name for p in files}
    train_speakers = all_speakers - eval_speakers
    assert not (train_speakers & eval_speakers), "training and evaluation speakers overlap"
    assert eval_speakers <= all_speakers, f"evaluation speakers missing from the corpus: {sorted(eval_speakers - all_speakers)}"
    items = [(p.stem, p.parent.name, str(p)) for p in files if p.parent.name in train_speakers]
    workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    with ProcessPoolExecutor(workers) as pool:
        rows = pd.DataFrame(list(pool.map(_probe, items, chunksize=64)))
    if not rows["id"].is_unique:
        raise ValueError("VCTK utterance ids are not unique")
    rows["gender"] = rows["speaker"].map(lambda s: info.get(s, ("", ""))[0])
    rows["accent"] = rows["speaker"].map(lambda s: info.get(s, ("", ""))[1])
    texts = []
    for utt_id, spk in zip(rows["id"], rows["speaker"]):
        t = root / "txt" / spk / f"{utt_id}.txt"
        texts.append(t.read_text().strip() if t.is_file() else "")
    rows["text"] = texts
    kept = rows[rows["reason"].isna()]
    kept = kept[list(LIST_COLUMNS)].sort_values("id").reset_index(drop=True)
    drops = rows["reason"].dropna().value_counts()
    report = {
        "trim": {"top_db": TOP_DB, "pad_s": PAD_S, "min_s": MIN_S, "max_s": MAX_S},
        "corpus_speakers": len(all_speakers),
        "eval_speakers_excluded": sorted(eval_speakers),
        "train_speakers": len(train_speakers),
        "speakers_kept": int(kept["speaker"].nunique()),
        "speakers_without_text": sorted(set(rows["speaker"]) - set(rows.loc[rows["text"] != "", "speaker"])),
        "candidates": int(len(rows)),
        "kept": int(len(kept)),
        "kept_hours": float(kept["dur_trimmed"].sum() / 3600.0),
        "drop_reasons": {r: int(drops.get(r, 0)) for r in DROP_REASONS},
        "per_speaker": speaker_report(rows),
        "native_sr_counts": {str(k): int(v) for k, v in kept["native_sr"].value_counts().items()},
    }
    return kept, report


def run_list(root: str | Path, vctk_root: str | Path = VCTK_ROOT, eval_table: str | Path = EVAL_TABLE) -> pd.DataFrame:
    paths = out_paths(root)
    table, report = build_list(vctk_root, eval_table)
    paths["list"].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths["list"].with_name("list.parquet.tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(paths["list"])
    write_json_atomic(paths["report"], report)
    print(
        f"kept {report['kept']} of {report['candidates']} utterances, {report['speakers_kept']} speakers, "
        f"{report['kept_hours']:.2f} h, drops {report['drop_reasons']} -> {paths['list']}"
    )
    return table


# ------------------------------------------------------------------------------------------------ prepare


def wav24_path(root: str | Path, speaker: str, utt_id: str) -> Path:
    return out_paths(root)["wav"] / speaker / f"{utt_id}.wav"


def run_prepare(root: str | Path, shard_index: int, num_shards: int, max_minutes: float | None = None, limit: int | None = None) -> int:
    """Writes the 24 kHz trimmed wavs of speakers ``sorted[i::n]``; 0 when done, ``EXIT_STOPPED`` if stopped early."""
    import soundfile as sf

    from sparc.vocoders.eval.io import write_wav

    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard {shard_index} outside [0, {num_shards})")
    stop = Stop(max_minutes)
    table = pd.read_parquet(out_paths(root)["list"])
    mine = sorted(table["speaker"].unique())[shard_index::num_shards]
    table = table[table["speaker"].isin(mine)].sort_values("id")
    if limit is not None:
        table = table.head(limit)
    written = skipped = 0
    for r in table.itertuples():
        if stop:
            print(f"[{shard_index}/{num_shards}] stopped: written {written}, skipped {skipped}", flush=True)
            return EXIT_STOPPED
        path = wav24_path(root, r.speaker, r.id)
        if path.is_file():
            skipped += 1
            continue
        wav, rate = sf.read(r.wav48, dtype="float32", always_2d=False)
        if wav.ndim != 1:
            wav = wav.mean(axis=1)
        if rate != r.native_sr or len(wav) != r.n_native:
            raise ValueError(f"{r.wav48} differs from the list (rate {rate}, {len(wav)} samples)")
        out = trim_resample(wav, rate, int(r.trim_start), int(r.trim_end))
        if abs(len(out) - int(r.n24)) > 1:
            raise ValueError(f"{r.id}: {len(out)} samples at 24 kHz, list says {r.n24}")
        write_wav(path, out, SAMPLE_RATE)
        written += 1
        if written % 500 == 0:
            print(f"[{shard_index}/{num_shards}] written {written}, skipped {skipped}", flush=True)
    print(f"[{shard_index}/{num_shards}] done: {len(table)} utterances, written {written}, skipped {skipped}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ leak


def embed_files(embedder, paths: Sequence[str], cache: Path, chunk: int = 1000) -> np.ndarray:
    """L2-normalized ECAPA embeddings (NaN rows for failures) of wav files, cached in chunks of ``chunk`` utterances."""
    from sparc.vocoders.eval.io import read_wav, to_16k

    cache.mkdir(parents=True, exist_ok=True)
    blocks = []
    for k, lo in enumerate(range(0, len(paths), chunk)):
        f = cache / f"chunk_{k:05d}.npy"
        if f.is_file():
            blocks.append(np.load(f))
            continue
        wavs = []
        for p in paths[lo : lo + chunk]:
            wav, rate = read_wav(p)
            wavs.append(to_16k(wav, rate))
        emb = embedder.embed(wavs, errors=[])
        tmp = f.with_name(f.name + ".tmp.npy")
        np.save(tmp, emb)
        tmp.replace(f)
        blocks.append(emb)
        print(f"embedded {min(lo + chunk, len(paths))}/{len(paths)} -> {cache}", flush=True)
    return np.concatenate(blocks) if blocks else np.zeros((0, embedder.dim), np.float32)


def run_leak(root: str | Path, threshold: float = THRESHOLD, device: str = "cpu", pools: Sequence[str] = POOLS) -> dict:
    from sparc.vocoders.eval.clone_pools import Pool
    from sparc.vocoders.eval.interp import metrics_config
    from sparc.vocoders.eval.metrics.speaker import SpeakerEmbedder

    paths = out_paths(root)
    table = pd.read_parquet(paths["list"])
    embedder = SpeakerEmbedder("ecapa", metrics_config().speaker, device)
    train_files = [str(wav24_path(root, s, i)) for s, i in zip(table["speaker"], table["id"])]
    missing = [p for p in train_files if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prepared wavs are missing (run prepare), first: {missing[:3]}")
    train = embed_files(embedder, train_files, paths["emb"] / "train")
    train_ok = np.isfinite(train).all(axis=1)
    max_cos, pool_sizes = {}, {}
    for name in pools:
        pt = Pool(name).table
        pool_emb = embed_files(embedder, pt["wav_path"].tolist(), paths["emb"] / f"pool_{name}")
        ok = np.isfinite(pool_emb).all(axis=1)
        pool_sizes[name] = {"utterances": int(len(pt)), "embedded": int(ok.sum())}
        mc = np.zeros(len(train))
        mc[train_ok] = max_cosine(train[train_ok], pool_emb[ok])
        max_cos[name] = mc
    cos = pd.DataFrame({"id": table["id"], **{f"max_cos_{n}": v for n, v in max_cos.items()}})
    cos["max_cos"] = cos[[f"max_cos_{n}" for n in pools]].max(axis=1)
    cos["embed_failed"] = ~train_ok
    drop = ((cos["max_cos"] >= threshold) | cos["embed_failed"]).to_numpy()
    paths["leak"].mkdir(parents=True, exist_ok=True)
    cos.to_parquet(paths["max_cos"], index=False)
    paths["dropped"].write_text("".join(f"{i}\n" for i in table.loc[drop, "id"]))
    by_pool = {n: int((max_cos[n] >= threshold).sum()) for n in pools}
    per_speaker = table.assign(drop=drop).groupby("speaker")["drop"].agg(["size", "sum"])
    report = {
        "threshold": float(threshold),
        "total": int(len(table)),
        "dropped": int(drop.sum()),
        "kept": int((~drop).sum()),
        "embed_failed": int(cos["embed_failed"].sum()),
        "dropped_by_pool_alone": by_pool,
        "pools": pool_sizes,
        "kept_hours": float(table.loc[~drop, "dur_trimmed"].sum() / 3600.0),
        "speakers_with_drops": {s: {"total": int(r["size"]), "dropped": int(r["sum"])} for s, r in per_speaker.iterrows() if r["sum"]},
        "max_cos_quantiles": {str(q): float(np.quantile(cos["max_cos"], q)) for q in (0.5, 0.9, 0.99, 0.999)},
    }
    write_json_atomic(paths["leak_report"], report)
    print(f"leak: dropped {report['dropped']} of {report['total']} (by pool {by_pool}) -> {paths['dropped']}")
    return report


# ------------------------------------------------------------------------------------------------ hooks


def run_extract(root: str | Path, *args, **kwargs):
    """Hook: feature extraction over :func:`kept_table` with the LibriTTS-R cache procedure of ``globe_train.run_extract``
    (SparcFeatureExtractor, meta check, per-utterance seed); wired after the Phase 5 thresholds are fixed."""
    raise NotImplementedError("extract waits for the Phase 5 quality thresholds")


def run_pack(root: str | Path, *args, **kwargs):
    """Hook: ``features.pack.write_packed`` over the extracted cache (speaker = real VCTK id, chapter = speaker)."""
    raise NotImplementedError("pack waits for the Phase 5 quality thresholds")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=OUT_ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="trim bounds, duration filter, evaluation-speaker exclusion")
    s.add_argument("--vctk-root", default=VCTK_ROOT)
    s.add_argument("--eval-table", default=EVAL_TABLE)
    s = sub.add_parser("prepare", help="write trimmed 24 kHz wavs of one shard")
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--nshards", type=int, required=True)
    s.add_argument("--max-minutes", type=float, default=None)
    s.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("leak", help="ECAPA cosine check against the three evaluation pools")
    s.add_argument("--threshold", type=float, default=THRESHOLD)
    s.add_argument("--device", default="cpu")
    sub.add_parser("extract", help="hook (not enabled)")
    sub.add_parser("pack", help="hook (not enabled)")
    a = p.parse_args(argv)
    if a.cmd == "list":
        run_list(a.root, a.vctk_root, a.eval_table)
    elif a.cmd == "prepare":
        return run_prepare(a.root, a.shard, a.nshards, a.max_minutes, a.limit)
    elif a.cmd == "leak":
        run_leak(a.root, a.threshold, a.device)
    elif a.cmd == "extract":
        run_extract(a.root)
    else:
        run_pack(a.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
