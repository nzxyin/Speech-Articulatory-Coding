"""Feature statistics over the training splits, used by ``FeatureFrontend`` and the speaker-vector normalization.

All statistics are frame- or utterance-level moments of the packed arrays, accumulated in float64 in chunks. The
loudness statistic uses ``ln(loud_raw * g + eps)`` with a gain ``g`` drawn per utterance from the training gain
distribution with a fixed seed, so it matches what the frontend sees during training.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sparc.vocoders.constants import (
    F0_CHANNEL,
    LOUDNESS_EPS,
    N_EMA,
    PERIODICITY_CHANNEL,
    SPEAKER_RAW_DIM,
)
from sparc.vocoders.data.gain import gain_factor, sample_gain_db
from sparc.vocoders.features.cache import read_meta_hash


class Moments:
    """Streaming mean and standard deviation per column, in float64 with a shift for numerical stability."""

    def __init__(self, columns: int | None = None):
        self.shape = () if columns is None else (columns,)
        self.count = 0
        self.shift = None
        self.sum = np.zeros(self.shape)
        self.sumsq = np.zeros(self.shape)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] == 0:
            return
        if self.shift is None:
            self.shift = x.mean(axis=0)
        d = x - self.shift
        self.count += x.shape[0]
        self.sum += d.sum(axis=0)
        self.sumsq += (d * d).sum(axis=0)

    def mean(self) -> np.ndarray:
        return self.shift + self.sum / self.count

    def std(self) -> np.ndarray:
        variance = self.sumsq / self.count - (self.sum / self.count) ** 2
        return np.sqrt(np.maximum(variance, 0.0))


def split_gains(index: pd.DataFrame, seed: int, db_range: tuple[float, float], rng_offset: int) -> np.ndarray:
    """One gain factor per utterance from a single generator seeded with ``[seed, rng_offset]``."""
    rng = np.random.default_rng([seed, rng_offset])
    return np.array([gain_factor(peak, sample_gain_db(rng, db_range)) for peak in index["peak24"]])


def compute_stats(cfg: DictConfig) -> dict:
    """Computes the statistics over ``cfg.stats.splits`` (default: ``cfg.data.train_splits``)."""
    splits = list(cfg.stats.splits) if cfg.stats.splits is not None else list(cfg.data.train_splits)
    db_range = tuple(float(v) for v in cfg.data.gain_db_range)
    floor = float(cfg.stats.spk_std_floor)
    chunk = int(cfg.stats.chunk_frames)
    speaker_chunk = int(cfg.stats.chunk_utterances)

    ema, logf0, loud, per = Moments(N_EMA), Moments(), Moments(), Moments()
    spk = {"l0": Moments(SPEAKER_RAW_DIM), "l6": Moments(SPEAKER_RAW_DIM)}
    utterances = 0
    for rng_offset, split in enumerate(splits):
        directory = Path(cfg.cache.packed_dir) / split
        index = pd.read_parquet(directory / "index.parquet")
        feats = np.load(directory / "feats.npy", mmap_mode="r")
        loud_raw = np.load(directory / "loud_raw.npy", mmap_mode="r")
        gains = np.repeat(split_gains(index, int(cfg.stats.gain_seed), db_range, rng_offset), index["T"].to_numpy())
        if len(gains) != feats.shape[0]:
            raise ValueError(f"{directory}: index frames {len(gains)} differ from feats rows {feats.shape[0]}")
        for lo in range(0, feats.shape[0], chunk):
            block = np.asarray(feats[lo : lo + chunk])
            ema.update(block[:, :N_EMA])
            logf0.update(np.log(np.maximum(block[:, F0_CHANNEL].astype(np.float64), 1.0)))
            per.update(block[:, PERIODICITY_CHANNEL])
            raw = np.asarray(loud_raw[lo : lo + chunk], dtype=np.float64)
            loud.update(np.log(raw * gains[lo : lo + chunk] + LOUDNESS_EPS))
        for layer, moments in spk.items():
            vectors = np.load(directory / f"spk_{layer}.npy", mmap_mode="r")
            for lo in range(0, vectors.shape[0], speaker_chunk):
                moments.update(vectors[lo : lo + speaker_chunk])
        utterances += len(index)
    frames = ema.count
    if frames == 0:
        raise ValueError("no frames found for the statistics")

    stats = {
        "ema_mean": ema.mean().tolist(),
        "ema_std": ema.std().tolist(),
        "logf0_mean": float(logf0.mean()),
        "logf0_std": float(logf0.std()),
        "loud_log_mean": float(loud.mean()),
        "loud_log_std": float(loud.std()),
        "per_mean": float(per.mean()),
        "per_std": float(per.std()),
    }
    for layer, moments in spk.items():
        stats[f"spk_{layer}_mean"] = moments.mean().tolist()
        stats[f"spk_{layer}_std"] = np.maximum(moments.std(), floor).tolist()
    stats.update(
        frames=int(frames),
        utterances=int(utterances),
        splits=splits,
        gain_seed=int(cfg.stats.gain_seed),
        gain_db_range=list(db_range),
        meta_hash=read_meta_hash(cfg.cache.meta_path),
    )
    return stats


def save_stats(stats: dict, path: str | Path) -> None:
    """Writes the statistics as JSON through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(stats, indent=2))
    os.replace(tmp, path)


def run_stats(cfg: DictConfig) -> dict:
    """Stage ``stats``: computes and saves the statistics."""
    stats = compute_stats(cfg)
    save_stats(stats, cfg.stats.output)
    print(f"stats over {stats['utterances']} utterances, {stats['frames']} frames -> {cfg.stats.output}")
    return stats
