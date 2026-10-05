"""Packing: per-utterance cache files of one split into a few large arrays that the datasets memory-map.

``<packed_dir>/<split>/`` holds ``feats.npy [frames, 15]``, ``loud_raw.npy [frames]``, ``spk_l0.npy``, ``spk_l6.npy``,
``spk_enplus64.npy`` (one row per utterance) and ``index.parquet`` with the frame offset of every utterance. The
directory is built next to its final place and renamed, so a half-packed split is never visible.
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from sparc.vocoders.constants import N_FEATURES
from sparc.vocoders.features.cache import ENPLUS_DIM, check_arrays, read_meta_hash, read_utt, utt_path
from sparc.vocoders.features.manifest import load_manifest, select_rows

INDEX_COLUMNS = (
    "id",
    "split",
    "speaker",
    "chapter",
    "wav_path",
    "n24",
    "T",
    "offset",
    "peak24",
    "seed",
    "duration",
    "spk_wsum",
    "spk_fallback",
)


def load_checked(path: Path, frames: int, meta_hash: str) -> dict:
    """Reads one cache file and raises if it is invalid or was written under different metadata."""
    data = read_utt(path)
    reason = check_arrays(data, frames)
    if reason is not None:
        raise ValueError(f"{path}: {reason}")
    if str(data["meta_hash"]) != meta_hash:
        raise ValueError(f"{path}: written under a different meta.json")
    return data


def pack_split(cfg: DictConfig, split: str) -> Path:
    """Packs the selected, encodable utterances of ``split``; returns the packed directory."""
    meta_hash = read_meta_hash(cfg.cache.meta_path)
    rows = select_rows(load_manifest(cfg), cfg, split)
    rows = rows[rows["encodable"]].reset_index(drop=True)
    paths = [utt_path(cfg.cache.utt_dir, split, utt_id) for utt_id in rows["id"]]
    present = np.array([p.is_file() for p in paths])
    if not present.all():
        absent = rows.loc[~present, "id"].tolist()
        if not cfg.pack.allow_missing:
            raise FileNotFoundError(f"{len(absent)} cache files of {split} are missing, first: {absent[:5]}")
        errors = Path(cfg.cache.errors_dir)
        errors.mkdir(parents=True, exist_ok=True)
        (errors / f"pack_missing_{split}.txt").write_text("".join(f"{i}\n" for i in absent))
        rows, paths = rows[present].reset_index(drop=True), [p for p, ok in zip(paths, present) if ok]
    if rows.empty:
        raise ValueError(f"nothing to pack for split {split}")

    frames = rows["T"].to_numpy(dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(frames)[:-1]])
    total, count = int(frames.sum()), len(rows)

    final = Path(cfg.cache.packed_dir) / split
    work = final.with_name(final.name + ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    open_array = np.lib.format.open_memmap
    feats = open_array(work / "feats.npy", mode="w+", dtype=np.float32, shape=(total, N_FEATURES))
    loud = open_array(work / "loud_raw.npy", mode="w+", dtype=np.float32, shape=(total,))
    spk = {
        name: open_array(work / f"{name}.npy", mode="w+", dtype=np.float32, shape=(count, dim))
        for name, dim in (("spk_l0", 1024), ("spk_l6", 1024), ("spk_enplus64", ENPLUS_DIM))
    }
    peak, seed, wsum, fallback = np.zeros(count), np.zeros(count, np.int64), np.zeros(count), np.zeros(count, bool)

    chunk = int(cfg.pack.chunk_utterances)
    with ThreadPoolExecutor(int(cfg.pack.num_threads)) as pool:
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            loaded = pool.map(lambda i: load_checked(paths[i], int(frames[i]), meta_hash), range(start, stop))
            for i, data in zip(range(start, stop), loaded):
                lo, hi = int(offsets[i]), int(offsets[i] + frames[i])
                feats[lo:hi] = data["feats"]
                loud[lo:hi] = data["loud_raw"]
                for name, array in spk.items():
                    array[i] = data[name]
                peak[i], seed[i] = float(data["peak24"]), int(data["seed"])
                wsum[i], fallback[i] = float(data["spk_wsum"]), bool(data["spk_fallback"])
    for array in (feats, loud, *spk.values()):
        array.flush()
    del feats, loud, spk

    index = rows[["id", "split", "speaker", "chapter", "wav_path", "n24", "T", "duration"]].copy()
    index["offset"] = offsets
    index["peak24"], index["seed"], index["spk_wsum"], index["spk_fallback"] = peak, seed, wsum, fallback
    index[list(INDEX_COLUMNS)].to_parquet(work / "index.parquet", index=False)

    backup = final.with_name(final.name + ".old")
    shutil.rmtree(backup, ignore_errors=True)
    if final.exists():
        final.rename(backup)
    work.rename(final)
    shutil.rmtree(backup, ignore_errors=True)
    print(f"packed {split}: {count} utterances, {total} frames -> {final}")
    return final


def run_pack(cfg: DictConfig) -> list[Path]:
    """Stage ``pack``: packs every selected split."""
    table = select_rows(load_manifest(cfg), cfg)
    return [pack_split(cfg, split) for split in table["split"].unique()]
