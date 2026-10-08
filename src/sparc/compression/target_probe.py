"""Probe screen of target reparameterizations (datasets.TARGETS) under leave-one-speaker-out folds.

For one model, a ridge probe on its cached ema_multi features (one layer, fixed alignment shift) is fitted on each
LOSO fold with each target space; ridge strength is chosen on the fold's validation split in the fitting space.
All metrics are computed after mapping predictions back to per-channel z, so targets are directly comparable.

Usage:
    python -m sparc.compression.target_probe run --model wavlm-large --layer 7 --shift 0
    python -m sparc.compression.target_probe summary
"""

import argparse
import json

import numpy as np
import torch
from scipy import stats

from . import datasets
from .probe import load_targets, probe_layer

OUT = datasets.XLSR_EMA / "analysis_targets"
METRICS = (("rmse_z", -1), ("pcc", 1), ("rmse", -1))


def run(model, layer, shift, targets=datasets.TARGETS, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = datasets.features_root("ema_multi") / model
    index = json.loads((d / "index.json").read_text())
    feats = np.load(d / f"layer_{layer:02d}.npy", mmap_mode="r").astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    out_f = OUT / f"probe_{model}_k{layer}.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    for spk in datasets.ALL_SPEAKERS:
        for t in targets:
            key = f"{spk}/{t}"
            if key in res:
                continue
            name = f"ema_loso_{spk}" + ("" if t == "z" else f"_{t}")
            ds = datasets.get(name)
            splits = {k: [s for s in v if s in index] for k, v in ds.splits().items()}
            utts = load_targets([s for v in splits.values() for s in v], ds)
            r, _, _ = probe_layer(feats, index, utts, splits, device, n_boot=0, ds=ds, shifts=[shift])
            res[key] = {"alpha": r["alpha"], **{sp: {m: r[sp][m] for m, _ in METRICS} for sp in ("test", "test_unseen")}}
            print(f"{model} k{layer} {key}: unseen rmse_z {r['test_unseen']['rmse_z']:.4f} pcc {r['test_unseen']['pcc']:.4f}"
                  f" | seen rmse_z {r['test']['rmse_z']:.4f} pcc {r['test']['pcc']:.4f}", flush=True)
            out_f.write_text(json.dumps({"model": model, "layer": layer, "shift": shift, "results": res}, indent=1))


def summary():
    L = ["# Target reparameterization probe screen (LOSO, ridge at a fixed layer)\n",
         "Held-out-speaker test, mean over 7 folds; paired difference vs target z (positive = better than z), "
         "95% t interval, folds better.\n"]
    for f in sorted(OUT.glob("probe_*.json")):
        d = json.loads(f.read_text())
        res = d["results"]
        L.append(f"\n## {d['model']} layer {d['layer']} (shift {d['shift']})\n")
        L.append("| target | unseen rmse_z | unseen PCC | vs z: rmse_z | vs z: PCC | seen rmse_z | seen PCC |")
        L.append("|---|---|---|---|---|---|---|")
        for t in datasets.TARGETS:
            folds = [s for s in datasets.ALL_SPEAKERS if f"{s}/{t}" in res and f"{s}/z" in res]
            if not folds:
                continue
            g = lambda sp, m, tt=t: np.array([res[f"{s}/{tt}"][sp][m] for s in folds])  # noqa: E731
            cells = []
            for m, better in METRICS[:2]:
                diff = better * (g("test_unseen", m) - g("test_unseen", m, "z"))
                h = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
                cells.append("-" if t == "z" else f"{diff.mean():+.4f} [{diff.mean() - h:+.4f}, {diff.mean() + h:+.4f}] "
                             f"{(diff > 0).sum()}/{len(diff)}")
            L.append(f"| {t} ({len(folds)} folds) | {g('test_unseen', 'rmse_z').mean():.4f} | {g('test_unseen', 'pcc').mean():.4f} | "
                     f"{cells[0]} | {cells[1]} | {g('test', 'rmse_z').mean():.4f} | {g('test', 'pcc').mean():.4f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary"))
    ap.add_argument("--model")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--shift", type=int)
    ap.add_argument("--targets", nargs="*", default=list(datasets.TARGETS))
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.model, args.layer, args.shift, args.targets)
    else:
        summary()


if __name__ == "__main__":
    main()
