"""GLOBE v3 training cache for Phase 4 arm G: listing, feature extraction and packing next to the LibriTTS-R cache.

Arm G (``docs/speaker_embedding_plan.md``, Phase 4 addendum) trains the B0 recipe on LibriTTS-R plus the filtered GLOBE v3
train list of the accent work (``train.tsv``: ``stem, source, speaker, accent``; ``stem`` = ``<parquet shard>-<row>``).
GLOBE has no speaker labels in train, so every utterance is its own speaker (``speaker = "globe_" + stem``,
``chapter = stem``) and its speaker reference is always itself.

* ``list`` joins ``train.tsv`` with the GLOBE manifest and drops every utterance whose ECAPA embedding has cosine >=
  ``--threshold`` (0.70) to any utterance of the evaluation GLOBE pool (``pools/globe/table.parquet``), to reduce speaker
  leakage into the evaluation. The embeddings are the ones of ``GLOBE_V3/ecapa/*.npz`` (speechbrain ECAPA, the model of
  the evaluation; L2-normalized here). Writes ``list.parquet`` and ``list_report.json``.
* ``extract`` writes, for the kept utterances of the parquet shards ``sorted_shards[i::n]``, a 24 kHz mono PCM_16 wav
  (``wav24/<parquet shard>/<stem>.wav``) and the per-utterance cache file (``utt/<parquet shard>/<stem>.npz``) with
  exactly the procedure of the LibriTTS-R cache: :class:`~sparc.vocoders.features.extractor.SparcFeatureExtractor` built
  from the ``extractor`` / ``inversion`` blocks of ``cache_config.yaml`` (CREPE dither at its default), the written wav
  re-read as float64, batch size 1, seed ``crc32(id) & 0x7FFFFFFF``. ``meta_hash`` is that of the existing LibriTTS-R
  ``meta.json`` (read only; extraction refuses to run if the settings differ from it). Resumable (valid files are
  skipped), atomic, stops cleanly after ``--max-minutes`` or on SIGUSR1 / SIGTERM (exit code 75 while work is left).
* ``pack`` packs the cache files into ``packed/globe.train/`` in the layout of the LibriTTS-R packed splits (same
  arrays, ``index.parquet`` columns and dtypes; :func:`~sparc.vocoders.features.pack.write_packed`). Utterances too short
  to encode (fewer than ``manifest.min_frames`` frames) are listed in ``errors/not_encodable_<shard>.json`` by ``extract``
  and skipped; any other missing cache file is an error unless ``--allow-missing``.

Usage (``extract`` needs a GPU)::

    python -m sparc.vocoders.features.globe_train list
    python -m sparc.vocoders.features.globe_train extract --shard 0 --nshards 20 --max-minutes 50
    python -m sparc.vocoders.features.globe_train pack

Training then lists the packed directory in ``data.train_splits`` (an absolute path is used as it is), see
``conf/experiment/p4_g.yaml``.
"""

import argparse
import csv
import io
import json
import signal
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.vocoders.constants import SAMPLE_RATE, feature_length

GLOBE_ROOT = "/data/user_data/xoy/GLOBE_V3"
GLOBE_RAW = "/data/group_data/UTD-NAS/Databases/GLOBE_V3/data"
TRAIN_LIST = "/data/user_data/xoy/globe_v3_accent_splits/v1/train.tsv"
POOL_TABLE = "/data/user_data/xoy/spkemb/phase4/pools/globe/table.parquet"
OUT_ROOT = "/data/user_data/xoy/spkemb/phase4/globe_train"
SPLIT = "globe.train"
THRESHOLD = 0.70
EXIT_STOPPED = 75
LIST_COLUMNS = ("stem", "shard", "row", "accent", "source", "speaker_orig", "parquet", "max_cos_pool")


def out_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "list": root / "list.parquet",
        "report": root / "list_report.json",
        "wav": root / "wav24",
        "utt": root / "utt",
        "errors": root / "errors",
        "summaries": root / "summaries",
        "packed": root / "packed" / SPLIT,
    }


def cache_config():
    """The ``extractor`` / ``inversion`` blocks and ``manifest.min_frames`` of ``cache_config.yaml`` (resolved)."""
    from omegaconf import OmegaConf

    from sparc.vocoders.eval.reextract import CONF_DIR

    cache = OmegaConf.load(CONF_DIR / "cache_config.yaml")
    root = OmegaConf.create(
        {
            "paths": OmegaConf.load(CONF_DIR / "paths" / "default.yaml"),
            "extractor": cache.extractor,
            "inversion": cache.inversion,
            "min_frames": cache.manifest.min_frames,
        }
    )
    return OmegaConf.create(OmegaConf.to_container(root, resolve=True))


# ------------------------------------------------------------------------------------------------ list


def max_cosine(train_emb: np.ndarray, pool_emb: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Largest cosine similarity of every row of ``train_emb`` to any row of ``pool_emb`` (both L2-normalized here)."""
    train = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    pool = (pool_emb / np.linalg.norm(pool_emb, axis=1, keepdims=True)).T
    return np.concatenate([(train[lo : lo + chunk] @ pool).max(axis=1) for lo in range(0, len(train), chunk)])


def leak_filter(table: pd.DataFrame, max_cos: np.ndarray, threshold: float) -> tuple[pd.DataFrame, dict]:
    """Rows of ``table`` (with an ``accent`` column) whose ``max_cos`` is below ``threshold``, and the drop report."""
    drop = np.asarray(max_cos) >= threshold
    per_accent = {}
    for accent, group in table.assign(drop=drop).groupby("accent", sort=True):
        per_accent[str(accent)] = {"total": int(len(group)), "dropped": int(group["drop"].sum())}
        per_accent[str(accent)]["kept"] = per_accent[str(accent)]["total"] - per_accent[str(accent)]["dropped"]
    report = {
        "threshold": float(threshold),
        "total": int(len(table)),
        "dropped": int(drop.sum()),
        "kept": int((~drop).sum()),
        "per_accent": per_accent,
    }
    return table[~drop].reset_index(drop=True), report


def load_ecapa(ecapa_dir: str | Path, stems: set[str] | None = None) -> pd.DataFrame:
    """All ``ecapa/*.npz`` (keys ``stems``, ``emb``) as a float32 matrix indexed by stem (only ``stems`` if given)."""
    names, blocks = [], []
    for path in sorted(Path(ecapa_dir).glob("*.npz")):
        with np.load(path) as z:
            s, e = z["stems"].astype(str), z["emb"]
        keep = np.ones(len(s), bool) if stems is None else np.isin(s, list(stems))
        names.append(s[keep])
        blocks.append(e[keep].astype(np.float32))
    return pd.DataFrame(np.concatenate(blocks), index=np.concatenate(names))


def build_list(
    train_tsv: str | Path = TRAIN_LIST,
    globe_root: str | Path = GLOBE_ROOT,
    raw_root: str | Path = GLOBE_RAW,
    pool_table: str | Path = POOL_TABLE,
    threshold: float = THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Filtered train table (``LIST_COLUMNS``) and the report of the leak filter."""
    train = pd.read_csv(train_tsv, sep="\t", dtype=str, keep_default_na=False)
    if not train["stem"].is_unique:
        raise ValueError(f"{train_tsv}: stems are not unique")
    manifest = pd.read_csv(
        Path(globe_root) / "manifest.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
        usecols=["stem", "split", "shard", "row", "accent"],
    )  # fmt: skip
    joined = train.merge(manifest, on="stem", how="left", suffixes=("", "_manifest"), validate="one_to_one")
    if joined["shard"].eq("").any() or joined["shard"].isna().any():
        raise ValueError(f"{int(joined['shard'].isna().sum())} train stems are not in the GLOBE manifest")
    if not (joined["split"] == "train").all():
        raise ValueError("train.tsv holds stems outside the GLOBE train split")
    if not (joined["accent"] == joined["accent_manifest"]).all():
        raise ValueError("accent of train.tsv differs from the manifest")
    joined["row"] = joined["row"].astype(int)
    joined["parquet"] = [str(Path(raw_root) / f"{s}.parquet") for s in joined["shard"]]

    pool_ids = pd.read_parquet(pool_table, columns=["id"])["id"].astype(str).tolist()
    ecapa = load_ecapa(Path(globe_root) / "ecapa", set(joined["stem"]) | set(pool_ids))
    missing_train = set(joined["stem"]) - set(ecapa.index)
    missing_pool = set(pool_ids) - set(ecapa.index)
    if missing_train or missing_pool:
        raise ValueError(f"no ECAPA embedding for {len(missing_train)} train and {len(missing_pool)} pool utterances")
    joined["max_cos_pool"] = max_cosine(ecapa.loc[joined["stem"]].to_numpy(), ecapa.loc[pool_ids].to_numpy())
    kept, report = leak_filter(joined, joined["max_cos_pool"].to_numpy(), threshold)
    report.update(
        pool_utterances=len(pool_ids),
        max_cos_quantiles={str(q): float(np.quantile(joined["max_cos_pool"], q)) for q in (0.5, 0.9, 0.99, 0.999)},
        shards_kept=int(kept["shard"].nunique()),
    )
    kept = kept.rename(columns={"speaker": "speaker_orig"})
    return kept[list(LIST_COLUMNS)].sort_values("stem").reset_index(drop=True), report


def run_list(root: str | Path, threshold: float, train_tsv: str | Path = TRAIN_LIST, pool_table: str | Path = POOL_TABLE):
    paths = out_paths(root)
    table, report = build_list(train_tsv=train_tsv, pool_table=pool_table, threshold=threshold)
    paths["list"].parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(paths["list"], index=False)
    paths["report"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"kept {report['kept']} of {report['total']} (dropped {report['dropped']}) -> {paths['list']}")
    return table, report


# ------------------------------------------------------------------------------------------------ extract


def shard_assignment(shards: Sequence[str], shard_index: int, num_shards: int) -> list[str]:
    """Parquet shards of task ``shard_index``: ``sorted(shards)[i::n]``."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} is outside [0, {num_shards})")
    return sorted(set(shards))[shard_index::num_shards]


def decode_audio(blob: bytes) -> np.ndarray:
    """Mono 24 kHz float32 samples of an embedded GLOBE audio file."""
    import soundfile as sf

    wav, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE or wav.ndim != 1:
        raise ValueError(f"expected mono {SAMPLE_RATE} Hz, got shape {wav.shape} at {rate} Hz")
    return wav


def read_not_encodable(path: Path) -> dict[str, str]:
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=0, sort_keys=True))
    tmp.replace(path)


class Stop:
    """Stop request: a signal, or the time budget of ``--max-minutes``."""

    def __init__(self, max_minutes: float | None):
        self.deadline = None if max_minutes is None else time.monotonic() + 60.0 * max_minutes
        self.signalled = False
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum: int, frame) -> None:
        self.signalled = True

    def __bool__(self) -> bool:
        return self.signalled or (self.deadline is not None and time.monotonic() > self.deadline)


def run_extract(
    root: str | Path,
    shard_index: int,
    num_shards: int,
    max_minutes: float | None = None,
    device: str = "auto",
    limit: int | None = None,
    meta_path: str | Path | None = None,
) -> int:
    """Extracts the shard's utterances; returns 0 when every one of them is done, ``EXIT_STOPPED`` if stopped early,
    1 if some failed."""
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch

    from sparc.vocoders.eval.clone_pools import cache_root_default
    from sparc.vocoders.eval.io import write_wav
    from sparc.vocoders.features.cache import (
        build_meta,
        check_arrays,
        file_sha256,
        is_valid,
        meta_compat,
        utt_path,
        utterance_seed,
        write_utt,
    )
    from sparc.vocoders.features.extractor import SparcFeatureExtractor, fork_commit

    stop = Stop(max_minutes)
    paths = out_paths(root)
    table = pd.read_parquet(paths["list"])
    mine = shard_assignment(table["shard"], shard_index, num_shards)
    table = table[table["shard"].isin(mine)]
    if limit is not None:
        table = table.sort_values("stem").head(limit)
    cfg = cache_config()
    meta_path = Path(meta_path or cache_root_default() / "meta.json")
    if device == "auto" and not torch.cuda.is_available():
        # the LibriTTS-R cache was written on GPUs; CPU features differ slightly, so CPU needs an explicit --device cpu
        raise RuntimeError("no CUDA device visible: refusing to extract on CPU (pass --device cpu to force it)")
    dev = torch.device("cuda") if device == "auto" else torch.device(device)
    extractor = SparcFeatureExtractor(cfg, dev)
    stored = json.loads(meta_path.read_text())
    if meta_compat(stored) != meta_compat(build_meta(extractor.describe(), fork_commit())):
        raise RuntimeError(f"{meta_path} (LibriTTS-R cache) was written with different settings than this extractor")
    meta_hash = file_sha256(meta_path)
    min_frames = int(cfg.min_frames)

    tag = f"{shard_index:05d}of{num_shards:05d}"
    counts = {"written": 0, "skipped": 0, "not_encodable": 0, "errors": 0}
    audio_seconds, extract_seconds, started = 0.0, 0.0, time.perf_counter()
    consecutive, stopped = 0, False

    def log_error(stem: str, text: str) -> None:
        paths["errors"].mkdir(parents=True, exist_ok=True)
        with open(paths["errors"] / f"extract_{tag}.jsonl", "a") as f:
            f.write(json.dumps({"id": stem, "error": text}) + "\n")
        print(f"[{tag}] error on {stem}: {text.strip().splitlines()[-1] if text.strip() else text}", flush=True)

    for shard in mine:
        rows = table[table["shard"] == shard].sort_values("row")
        if rows.empty or stopped:
            continue
        bad_path = paths["errors"] / f"not_encodable_{shard}.json"
        bad = read_not_encodable(bad_path)
        wav_dir = paths["wav"] / shard
        todo = []
        for stem, row in zip(rows["stem"], rows["row"]):
            wav_path = wav_dir / f"{stem}.wav"
            if stem in bad:
                counts["not_encodable"] += 1
            elif wav_path.is_file() and is_valid(
                utt_path(paths["utt"], shard, stem), feature_length(int(sf.info(str(wav_path)).frames))
            ):
                counts["skipped"] += 1
            else:
                todo.append((stem, int(row), wav_path))
        if not todo:
            continue
        pf = pq.ParquetFile(rows["parquet"].iloc[0])
        sizes = [pf.metadata.row_group(g).num_rows for g in range(pf.metadata.num_row_groups)]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        by_group: dict[int, list] = {}
        for item in todo:
            by_group.setdefault(int(np.searchsorted(starts, item[1], side="right") - 1), []).append(item)
        for g, items in sorted(by_group.items()):
            blobs = None
            for stem, row, wav_path in items:
                if stop:
                    stopped = True
                    break
                try:
                    if not wav_path.is_file():
                        if blobs is None:
                            blobs = pf.read_row_group(g, columns=["audio"]).column(0)
                        wav = decode_audio(blobs[row - int(starts[g])].as_py()["bytes"])
                        if feature_length(len(wav)) < min_frames:
                            bad[stem] = f"too short: {len(wav)} samples"
                            counts["not_encodable"] += 1
                            continue
                        write_wav(wav_path, wav, SAMPLE_RATE, subtype="PCM_16")
                    wav, rate = sf.read(str(wav_path), dtype="float64", always_2d=False)
                    if rate != SAMPLE_RATE or wav.ndim != 1:
                        raise ValueError(f"{wav_path}: expected mono {SAMPLE_RATE} Hz")
                    t0 = time.perf_counter()
                    out = extractor.extract(wav, utterance_seed(stem))
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    extract_seconds += time.perf_counter() - t0
                    arrays = {
                        **out,
                        "spk_wsum": np.float64(out["spk_wsum"]),
                        "spk_fallback": np.bool_(out["spk_fallback"]),
                        "meta_hash": meta_hash,
                        "gpu": extractor.device_name,
                    }
                    problem = check_arrays(arrays, int(out["T"]))
                    if problem is not None:  # e.g. silent clip or non-finite features: skipped by pack, not retried
                        bad[stem] = f"unusable: {problem}"
                        counts["not_encodable"] += 1
                        continue
                    write_utt(utt_path(paths["utt"], shard, stem), arrays)
                except Exception:
                    counts["errors"] += 1
                    consecutive += 1
                    log_error(stem, traceback.format_exc())
                    if consecutive >= 20:
                        write_json_atomic(bad_path, bad)
                        raise RuntimeError(f"{consecutive} consecutive errors") from None
                    continue
                consecutive = 0
                counts["written"] += 1
                audio_seconds += len(wav) / SAMPLE_RATE
                if counts["written"] % 100 == 0:
                    wall = time.perf_counter() - started
                    print(
                        f"[{tag}] {counts} audio {audio_seconds / 3600:.2f} h, extract "
                        f"{extract_seconds / max(audio_seconds / 3600, 1e-9):.0f} s/audio-h, wall "
                        f"{wall / max(audio_seconds / 3600, 1e-9):.0f} s/audio-h",
                        flush=True,
                    )
            if stopped:
                break
        write_json_atomic(bad_path, bad)

    hours = audio_seconds / 3600.0
    summary = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shard": tag,
        "parquet_shards": len(mine),
        **counts,
        "audio_hours": hours,
        "extract_seconds_per_audio_hour": extract_seconds / hours if hours else None,
        "wall_seconds_per_audio_hour": (time.perf_counter() - started) / hours if hours else None,
        "wall_seconds": time.perf_counter() - started,
        "stopped": stopped,
        "gpu": extractor.device_name,
    }
    paths["summaries"].mkdir(parents=True, exist_ok=True)
    with open(paths["summaries"] / f"extract_{tag}.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[{tag}] done: {json.dumps(summary)}", flush=True)
    if stopped:
        return EXIT_STOPPED
    return 1 if counts["errors"] else 0


# ------------------------------------------------------------------------------------------------ pack


def run_pack(
    root: str | Path,
    allow_missing: bool = False,
    num_threads: int = 16,
    chunk: int = 512,
    meta_path: str | Path | None = None,
) -> Path:
    """Packs the cache files of the kept utterances into ``packed/globe.train``."""
    from sparc.vocoders.eval.clone_pools import cache_root_default
    from sparc.vocoders.features.cache import read_meta_hash, utt_path
    from sparc.vocoders.features.pack import write_packed

    paths = out_paths(root)
    table = pd.read_parquet(paths["list"]).sort_values("stem").reset_index(drop=True)
    bad: dict[str, str] = {}
    for shard in table["shard"].unique():
        bad.update(read_not_encodable(paths["errors"] / f"not_encodable_{shard}.json"))
    files = [utt_path(paths["utt"], s, i) for s, i in zip(table["shard"], table["stem"])]
    present = np.array([p.is_file() for p in files])
    unencodable = table["stem"].isin(bad).to_numpy() & ~present
    missing = ~present & ~unencodable
    paths["errors"].mkdir(parents=True, exist_ok=True)
    (paths["errors"] / "pack_not_encodable.txt").write_text("".join(f"{s}\n" for s in table.loc[unencodable, "stem"]))
    if missing.any():
        absent = table.loc[missing, "stem"].tolist()
        if not allow_missing:
            raise FileNotFoundError(f"{len(absent)} cache files are missing, first: {absent[:5]}")
        (paths["errors"] / "pack_missing.txt").write_text("".join(f"{s}\n" for s in absent))
    keep = present.nonzero()[0]
    if len(keep) == 0:
        raise ValueError("nothing to pack")
    table, files = table.iloc[keep].reset_index(drop=True), [files[i] for i in keep]

    def sizes(path: Path) -> tuple[int, int]:
        with np.load(path, allow_pickle=False) as z:
            return int(z["n24"]), int(z["T"])

    with ThreadPoolExecutor(num_threads) as pool:
        n24, frames = zip(*pool.map(sizes, files))
    n24 = np.asarray(n24, dtype=np.int64)
    rows = pd.DataFrame(
        {
            "id": table["stem"].astype(str),
            "split": SPLIT,
            "speaker": "globe_" + table["stem"].astype(str),
            "chapter": table["stem"].astype(str),
            "wav_path": [str(paths["wav"] / s / f"{i}.wav") for s, i in zip(table["shard"], table["stem"])],
            "n24": n24,
            "T": np.asarray(frames, dtype=np.int32),
            "duration": (n24 / SAMPLE_RATE).astype(np.float32),
        }
    )
    bad_len = [i for i, n, t in zip(rows["id"], n24, rows["T"]) if feature_length(int(n)) != int(t)]
    if bad_len:
        raise ValueError(f"{len(bad_len)} cache files have a frame count that differs from n24, first: {bad_len[:3]}")
    meta_hash = read_meta_hash(meta_path or cache_root_default() / "meta.json")
    return write_packed(rows, files, paths["packed"], meta_hash, num_threads, chunk)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=OUT_ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="join train.tsv with the manifest and apply the ECAPA leak filter")
    s.add_argument("--threshold", type=float, default=THRESHOLD)
    s.add_argument("--train-tsv", default=TRAIN_LIST)
    s.add_argument("--pool-table", default=POOL_TABLE)
    s = sub.add_parser("extract", help="write wav24 and per-utterance cache files of one work shard")
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--nshards", type=int, required=True)
    s.add_argument("--max-minutes", type=float, default=None)
    s.add_argument("--device", default="auto")
    s.add_argument("--limit", type=int, default=None, help="smoke tests: only the first N utterances (by stem)")
    s = sub.add_parser("pack", help="pack the cache files into packed/globe.train")
    s.add_argument("--allow-missing", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "list":
        run_list(a.root, a.threshold, a.train_tsv, a.pool_table)
        return 0
    if a.cmd == "extract":
        return run_extract(a.root, a.shard, a.nshards, a.max_minutes, a.device, a.limit)
    run_pack(a.root, a.allow_missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
