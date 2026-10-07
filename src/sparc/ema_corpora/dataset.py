"""Loader for preprocessed EMA corpora (outputs of preprocess.py).

    rows = load_manifest()                         # list of dicts, one per utterance
    rows = [r for r in rows if r["speaker"] == "usc_M1" and r["split"] == "train"]
    u = load_utterance(rows[0])                    # audio (16 kHz), ema (T, 12) normalized, valid (T, 6), ...

Normalization is the standard per-articulator one: each of the 12 channels is z-scored with the mean and std
of its normalization group (a speaker, or a recording session where a speaker has a content-independent frame
change: usc_F5_s1 / usc_F5_s2). No cross-speaker transform is applied. Statistics come from stats.json, which
is computed over all utterances of a group; recompute on a training subset with group_stats() if needed.
"""

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from .preprocess import CHANNELS, OUT_ROOT

NUMERIC = {"start", "end", "duration", "n_frames", "ema_sr", "lag_s", "rotation_deg", "valid_frac", "sentence_id",
           "repetition", "source_utt", "cut_spread_s", "frame_offset_mm", "frame_offset_x", "frame_offset_y",
           "frame_offset_lat"}


def load_manifest(root=OUT_ROOT):
    rows = []
    with open(Path(root) / "manifest.csv") as f:
        for r in csv.DictReader(f):
            for k in NUMERIC & r.keys():
                if r[k] != "":
                    r[k] = float(r[k])
            rows.append(r)
    return rows


@lru_cache(maxsize=4)
def load_stats(root=OUT_ROOT):
    return json.loads((Path(root) / "stats.json").read_text())


def group_stats(rows, root=OUT_ROOT):
    """Per-group channel mean/std over the given rows only (e.g. a training split)."""
    acc = {}
    for r in rows:
        e = np.load(Path(root) / r["ema"])["ema"].astype(np.float64)
        ok = ~np.isnan(e)
        a = acc.setdefault(r["norm_group"], [np.zeros(12), np.zeros(12), np.zeros(12)])
        a[0] += np.where(ok, e, 0).sum(0)
        a[1] += np.where(ok, e * e, 0).sum(0)
        a[2] += ok.sum(0)
    out = {}
    for g, (s, q, n) in acc.items():
        m = s / n
        out[g] = {"channels": list(CHANNELS), "mean": m.tolist(), "std": np.sqrt(q / n - m**2).tolist()}
    return out


def load_utterance(row, root=OUT_ROOT, normalize=True, stats=None):
    """dict(audio (n,) float32 16 kHz, ema (T, 12), lateral (T, 6), valid (T, 6), ema_mm (T, 12), row).

    `ema` is z-scored per channel with the row's normalization-group statistics (stats.json unless `stats` is
    given) when normalize=True, else in mm. Frames in long dropouts are NaN; `valid` marks them per sensor."""
    root = Path(root)
    z = np.load(root / row["ema"])
    audio, sr = sf.read(root / row["audio"], dtype="float32")
    assert sr == 16000
    ema_mm = z["ema"]
    ema = ema_mm
    if normalize:
        st = (stats or load_stats(str(root)))[row["norm_group"]]
        ema = (ema_mm - np.asarray(st["mean"], np.float32)) / np.asarray(st["std"], np.float32)
    return {"audio": audio, "ema": ema, "ema_mm": ema_mm, "lateral": z["lateral"], "valid": z["valid"], "row": row}
