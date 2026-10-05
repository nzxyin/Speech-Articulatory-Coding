"""Manifest of the filtered LibriTTS-R utterances: one row per file, with the predicted SPARC feature length.

The source lists are the validated TSV files of Phase 1 (read with ``csv.QUOTE_NONE``, because the transcripts contain
double quotes). The result is written to ``manifest.parquet`` and read back by every later stage.
"""

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sparc.vocoders.constants import HOP, SAMPLE_RATE, feature_length

COLUMNS = ("id", "split", "speaker", "chapter", "wav_path", "n24", "duration", "T", "encodable", "text")
SOURCE_COLUMNS = ("id", "split", "speaker", "chapter", "wav_path", "num_samples_24k", "duration_s", "text_normalized")


def predicted_frames(n24: np.ndarray) -> np.ndarray:
    """Vectorized :func:`sparc.vocoders.constants.feature_length`."""
    l16 = -((-2 * np.asarray(n24, dtype=np.int64)) // 3)
    return ((l16 - 80) // 320).astype(np.int32)


def read_sources(cfg: DictConfig) -> pd.DataFrame:
    """Concatenates the source TSV files as strings."""
    frames = []
    for name in cfg.manifest.source_files:
        path = Path(cfg.manifest.sources_dir) / name
        frames.append(
            pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False, dtype=str, encoding="utf-8")
        )
    return pd.concat(frames, ignore_index=True)


def build_manifest(cfg: DictConfig) -> pd.DataFrame:
    """Builds the manifest table and checks it against the source lists and the feature-length formula."""
    source = read_sources(cfg)
    missing = [c for c in (*SOURCE_COLUMNS, "n_frames_sparc_cache") if c not in source.columns]
    if missing:
        raise ValueError(f"source lists lack the columns {missing}")
    split_dirs = dict(cfg.manifest.split_dirs)
    unknown = sorted(set(source["split"]) - set(split_dirs))
    if unknown:
        raise ValueError(f"unknown splits in the source lists: {unknown}")
    if not source["id"].is_unique:
        raise ValueError("utterance ids are not unique")

    n24 = source["num_samples_24k"].astype(np.int64).to_numpy()
    frames = predicted_frames(n24)
    if not (frames == n24 // HOP - 1).all():
        raise ValueError("T differs from n24 // 480 - 1 for some rows")
    if not (frames == source["n_frames_sparc_cache"].astype(np.int32).to_numpy()).all():
        raise ValueError("T differs from n_frames_sparc_cache of the source lists for some rows")
    if any(feature_length(int(n)) != int(t) for n, t in zip(n24, frames)):
        raise ValueError("constants.feature_length disagrees with the vectorized formula")

    root = Path(cfg.paths.libritts_r_raw)
    wav_path = [
        str(root / split_dirs[split] / speaker / chapter / f"{utt_id}.wav")
        for utt_id, split, speaker, chapter in zip(source["id"], source["split"], source["speaker"], source["chapter"])
    ]
    if wav_path != source["wav_path"].tolist():
        raise ValueError("wav_path of the source lists differs from paths.libritts_r_raw / <split dir> / ...")

    table = pd.DataFrame(
        {
            "id": source["id"].astype(str),
            "split": source["split"].astype(str),
            "speaker": source["speaker"].astype(str),
            "chapter": source["chapter"].astype(str),
            "wav_path": wav_path,
            "n24": n24,
            "duration": (n24 / SAMPLE_RATE).astype(np.float32),
            "T": frames,
            "encodable": frames >= int(cfg.manifest.min_frames),
            "text": source["text_normalized"].astype(str),
        }
    )
    expected_rows = cfg.manifest.expected_rows
    if expected_rows is not None and len(table) != int(expected_rows):
        raise ValueError(f"expected {expected_rows} rows, found {len(table)}")
    if cfg.manifest.check_files_exist:
        absent = [p for p in table["wav_path"] if not os.path.isfile(p)]
        if absent:
            raise FileNotFoundError(f"{len(absent)} audio files are missing, first: {absent[0]}")
    return table[list(COLUMNS)]


def write_manifest(table: pd.DataFrame, path: str | Path) -> None:
    """Writes the manifest as parquet through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        table.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_manifest(cfg: DictConfig) -> pd.DataFrame:
    """Reads ``manifest.parquet``."""
    return pd.read_parquet(cfg.manifest.path)


def evenly_spaced(count: int, keep: int) -> np.ndarray:
    """``keep`` distinct indices spread evenly over ``range(count)``."""
    if keep >= count:
        return np.arange(count)
    return np.unique(np.round(np.linspace(0, count - 1, keep)).astype(np.int64))


def select_rows(table: pd.DataFrame, cfg: DictConfig, split: str | None = None) -> pd.DataFrame:
    """Rows chosen by ``cfg.select``: the listed splits (all if ``null``), then ``subsample`` evenly spaced per split.

    With ``split`` given, only that split is returned (it must be selected), so packing one split sees exactly the rows
    that extraction processed.
    """
    chosen = cfg.select.splits
    splits = list(table["split"].unique()) if chosen is None else list(chosen)
    if split is not None:
        if split not in splits:
            raise ValueError(f"split {split!r} is not selected by select.splits")
        splits = [split]
    parts = []
    for name in splits:
        rows = table[table["split"] == name]
        if rows.empty:
            raise ValueError(f"split {name!r} has no rows in the manifest")
        if cfg.select.subsample is not None:
            rows = rows.iloc[evenly_spaced(len(rows), int(cfg.select.subsample))]
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def run_manifest(cfg: DictConfig) -> pd.DataFrame:
    """Stage ``manifest``: builds and writes the manifest, returns it."""
    table = build_manifest(cfg)
    write_manifest(table, cfg.manifest.path)
    hours = float(table["n24"].sum()) / SAMPLE_RATE / 3600.0
    skipped = int((~table["encodable"]).sum())
    print(f"manifest: {len(table)} rows, {hours:.1f} h, {skipped} not encodable -> {cfg.manifest.path}")
    return table
