"""Robustness evaluation of finished adaptation runs (train.py) under acoustic perturbations.

For each run directory, rebuilds the encoder and head from results.json + best_trainable.pt and scores the
test set clean and under:
    white noise at SNR 20/10/5/0 dB (fixed seed per utterance)
    speed perturbation x0.9 / x1.1 (resampled audio; EMA targets are time-stretched to match)
Writes <run>/robustness.json.

Usage: python -m sparc.compression.evaluate <run_dir> [<run_dir> ...]
"""

import argparse
import json
import zlib
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from . import mngu0
from .metrics import ema_metrics
from .train import build, load_trainable, predict

SNRS = (20, 10, 5, 0)
SPEEDS = (0.9, 1.1)


def add_noise(u, snr_db):
    rng = np.random.default_rng(zlib.crc32(u.stem.encode()) + snr_db)
    p = np.mean(u.wav**2)
    noise = rng.normal(size=u.wav.shape).astype(np.float32) * np.sqrt(p / 10 ** (snr_db / 10))
    return replace(u, wav=u.wav + noise)


def speed(u, factor):
    up, down = (10, 9) if factor < 1 else (10, 11)  # factor 0.9 -> longer audio
    wav = resample_poly(u.wav, up, down).astype(np.float32)
    stretch = up / down
    n = int(round(len(u.ema_mm) * stretch))
    t_old = np.arange(len(u.ema_mm))
    t_new = np.linspace(0, len(u.ema_mm) - 1, n)
    ema = np.stack([np.interp(t_new, t_old, u.ema_mm[:, c]) for c in range(12)], 1).astype(np.float32)
    phones = [(a * stretch, b * stretch, l) for a, b, l in u.phones]
    return replace(u, wav=wav, ema_mm=ema, sil_end=u.sil_end * stretch, phones=phones)


def evaluate_run(run_dir, device):
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "results.json").read_text())
    args = Namespace(**res["config"])
    model, head, _ = build(args, device)
    if args.head == "linear":
        mean, std = torch.load(run_dir / "input_stats.pt")
        head.set_input_stats(mean, std)
    load_trainable(model, head, torch.load(run_dir / "best_trainable.pt"))

    splits = mngu0.stems_by_split()
    train = [mngu0.load_utterance(s) for s in splits["train"]]
    allY = np.concatenate([u.ema_mm for u in train])
    ym, ys = allY.mean(0), allY.std(0)
    test = [mngu0.load_utterance(s) for s in splits["test"]]

    out = {}
    conds = {"clean": test}
    conds.update({f"white_{s}dB": [add_noise(u, s) for u in test] for s in SNRS})
    conds.update({f"speed_{f}": [speed(u, f) for u in test] for f in SPEEDS})
    for name, utts in conds.items():
        preds, trues, _ = predict(model, head, utts, args, ym, ys, device)
        m = ema_metrics(preds, trues, n_boot=0)
        out[name] = {k: m[k] for k in ("rmse", "pcc", "vel_rmse")}
        print(f"{run_dir.name} {name}: RMSE {m['rmse']:.4f} PCC {m['pcc']:.4f}", flush=True)
    (run_dir / "robustness.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for r in args.runs:
        if (Path(r) / "robustness.json").exists():
            continue
        evaluate_run(r, device)


if __name__ == "__main__":
    main()
