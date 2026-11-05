"""Leave-one-speaker-out depth sweep with fold-internal (nested) selection: the multi-speaker FLOPs frontier.

For every cached layer k of an encoder (ema_multi feature cache, low-passed as in SPARC), a ridge probe is fitted on
each LOSO fold (datasets ema_loso_<h>: six training speakers, held-out speaker h). Alignment shift (-2..3) and ridge
strength are chosen on the training speakers' validation sentences of that fold only, so nothing about h is used for
selection. Each layer is scored on h's test sentences (rmse_z, PCC in h's z units) and on the training speakers'
test sentences ("seen").

Two summaries:
    frontier -- mean held-out rmse_z / PCC per depth against GFLOPs per second of audio (compute_prefix.json);
    nested   -- per fold, the depth with the lowest validation rmse_z (training speakers only) is scored on h,
                which removes the selection leak of the fixed depths used elsewhere (chosen on ema_multi, whose
                validation sentences include speakers held out in LOSO).

Usage:
    python -m sparc.compression.depth_sweep run --model wavlm-large
    python -m sparc.compression.depth_sweep summary
"""

import argparse
import json

import numpy as np
import torch

from . import datasets
from .probe import load_targets, probe_layer

OUT = datasets.XLSR_EMA / "analysis_depth"
SHIFTS = (-2, -1, 0, 1, 2, 3)
WIDE_SHIFTS = tuple(range(-4, 6))  # Whisper re-sweep when the first-pass grid is clipped (docs/whisper_probe_plan.md)
MODELS = ("wavlm-large", "xlsr-300m", "xlsr-1b", "w2v2-large", "whisper-large-v3", "whisper-medium", "whisper-medium-en")
FIXED = {"wavlm-large": 7, "xlsr-300m": 18, "xlsr-1b": 10, "w2v2-large": 18}  # depths used in LOSO option A (none for Whisper)


def gflops(model, key="gflops_per_audio_s"):
    prof = json.loads((datasets.XLSR_EMA / "features" / model / "compute_prefix.json").read_text())
    return {p["layers"][-1]: p[key] for p in prof["prefixes"]}


def load_all(model, variant=""):
    """Merged results of the main file and any part files (parallel layer-range jobs write separate files).
    variant "_wide": the wide-shift-grid results (<model>_wide.json, <model>_wide_part<tag>.json), kept apart."""
    res = {}
    base = f"{model}{variant}"
    for f in sorted(OUT.glob(f"{base}*.json")):
        if f.stem == base or f.stem.startswith(f"{base}_part"):
            res.update(json.loads(f.read_text()))
    return res


def run(model, layers=None, device=None, tag=None, shifts=None, variant=""):
    shifts = tuple(shifts) if shifts else SHIFTS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = datasets.features_root("ema_multi") / model
    index = json.loads((d / "index.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    layers = layers if layers is not None else range(1, meta["num_layers"] + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    base = f"{model}{variant}"
    out_f = OUT / (f"{base}_part{tag}.json" if tag else f"{base}.json")
    done = load_all(model, variant)
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    folds = {}
    for h in datasets.ALL_SPEAKERS:
        ds = datasets.EMAMulti(held_out=(h,), norm_by="norm_group")
        splits = {k: [s for s in v if s in index] for k, v in ds.splits().items()}
        folds[h] = (ds, splits)
    # targets do not depend on the fold (normalization is per group, from each group's own training split)
    ds0 = folds[datasets.ALL_SPEAKERS[0]][0]
    utts = load_targets(sorted({s for _, sp in folds.values() for v in sp.values() for s in v}), ds0)
    for k in layers:
        if all(f"{k}/{h}" in res or f"{k}/{h}" in done for h in datasets.ALL_SPEAKERS):
            continue
        feats = np.load(d / f"layer_{k:02d}.npy", mmap_mode="r").astype(np.float32)
        for h, (ds, splits) in folds.items():
            key = f"{k}/{h}"
            if key in res:
                continue
            sp = {"train": splits["train"], "valid": splits["valid"], "test": splits["test"], "test_unseen": splits["test_unseen"]}
            r, _, _ = probe_layer(feats, index, utts, sp, device, n_boot=0, ds=ds, shifts=shifts)
            res[key] = {"shift": r["shift"], "alpha": r["alpha"], "valid_rmse_z": r["valid"]["rmse_z"],
                        "valid_pcc": r["valid"]["pcc"],
                        "seen": {m: r["test"][m] for m in ("rmse_z", "pcc")},
                        "unseen": {m: r["test_unseen"][m] for m in ("rmse_z", "pcc")}}
        out_f.write_text(json.dumps(res, indent=1))
        un = [res[f"{k}/{h}"]["unseen"] for h in datasets.ALL_SPEAKERS]
        print(f"{model} k={k}: held-out rmse_z {np.mean([u['rmse_z'] for u in un]):.4f} "
              f"PCC {np.mean([u['pcc'] for u in un]):.4f}", flush=True)
        del feats


def summary():
    L = ["# LOSO depth sweep (ridge probe; shift, alpha and nested depth chosen on training speakers only)\n",
         "Held-out speaker test, rmse_z / PCC in its z units, mean over the 7 folds. GFLOPs per second of audio for the "
         "encoder prefix (FlopCounterMode, 4 s reference). Pareto = not beaten on both rmse_z and FLOPs by a "
         "shallower-or-equal-cost depth of any encoder.\n"]
    pts, pts_ideal = [], []
    for model in MODELS:
        res = load_all(model)
        if not res:
            continue
        try:
            fl = gflops(model)
        except FileNotFoundError:
            L.append(f"\n## {model}\n\nNo compute profile yet (features/{model}/compute_prefix.json missing); rerun summary once it exists.\n")
            continue
        try:  # Whisper only: attention on the real frames (cost lower bound the checkpoint cannot deliver)
            fl_ideal = gflops(model, "gflops_per_audio_s_ideal")
        except KeyError:
            fl_ideal = None
        ks = sorted({int(x.split("/")[0]) for x in res if all(f"{x.split('/')[0]}/{h}" in res for h in datasets.ALL_SPEAKERS)})
        L.append(f"\n## {model}\n")
        L.append("| k | GFLOPs/s | held-out rmse_z | held-out PCC | seen rmse_z | seen PCC | shifts chosen |")
        L.append("|---|---|---|---|---|---|---|")
        for k in ks:
            rs = [res[f"{k}/{h}"] for h in datasets.ALL_SPEAKERS]
            ur = np.mean([r["unseen"]["rmse_z"] for r in rs])
            up = np.mean([r["unseen"]["pcc"] for r in rs])
            pts.append((model, k, fl[k], ur, up))
            if fl_ideal:
                pts_ideal.append((model, k, fl_ideal[k], ur, up))
            mark = " (LOSO A)" if k == FIXED.get(model) else ""
            L.append(f"| {k}{mark} | {fl[k]:.1f} | {ur:.4f} | {up:.4f} | {np.mean([r['seen']['rmse_z'] for r in rs]):.4f} | "
                     f"{np.mean([r['seen']['pcc'] for r in rs]):.4f} | {sorted({r['shift'] for r in rs})} |")
        # nested depth selection per fold
        nested = []
        for h in datasets.ALL_SPEAKERS:
            kbest = min(ks, key=lambda k: res[f"{k}/{h}"]["valid_rmse_z"])
            nested.append((h, kbest, res[f"{kbest}/{h}"]["unseen"]))
        L.append(f"\nNested depth (per fold, lowest training-speaker validation rmse_z): "
                 + ", ".join(f"{h} k={k}" for h, k, _ in nested))
        L.append(f"Nested held-out rmse_z {np.mean([u['rmse_z'] for _, _, u in nested]):.4f}, "
                 f"PCC {np.mean([u['pcc'] for _, _, u in nested]):.4f}; mean GFLOPs/s "
                 f"{np.mean([fl[k] for _, k, _ in nested]):.1f}")
        # cross-fold depth selection: for held-out h, the depth with the best mean held-out rmse_z on the OTHER six
        # folds (h's own test sentences are never used; those folds' models were trained with h's training data)
        cross = []
        for h in datasets.ALL_SPEAKERS:
            others = [o for o in datasets.ALL_SPEAKERS if o != h]
            kbest = min(ks, key=lambda k: np.mean([res[f"{k}/{o}"]["unseen"]["rmse_z"] for o in others]))
            cross.append((h, kbest, res[f"{kbest}/{h}"]["unseen"]))
        L.append("Cross-fold depth (best mean held-out rmse_z on the other six folds): "
                 + ", ".join(f"{h} k={k}" for h, k, _ in cross))
        L.append(f"Cross-fold held-out rmse_z {np.mean([u['rmse_z'] for _, _, u in cross]):.4f}, "
                 f"PCC {np.mean([u['pcc'] for _, _, u in cross]):.4f}; mean GFLOPs/s "
                 f"{np.mean([fl[k] for _, k, _ in cross]):.1f}")
        kf = FIXED.get(model)
        if kf in ks:
            rs = [res[f"{kf}/{h}"]["unseen"] for h in datasets.ALL_SPEAKERS]
            L.append(f"Fixed LOSO-A depth k={kf}: held-out rmse_z {np.mean([u['rmse_z'] for u in rs]):.4f}, "
                     f"PCC {np.mean([u['pcc'] for u in rs]):.4f}")
    # Pareto frontier on held-out rmse_z vs FLOPs, all encoders
    L.append("\n## Frontier (held-out rmse_z vs GFLOPs/s, all encoders)\n")
    L.append("| encoder | k | GFLOPs/s | rmse_z | PCC |")
    L.append("|---|---|---|---|---|")
    best = np.inf
    for m, k, f, r, p in sorted(pts, key=lambda x: (x[2], x[3])):
        if r < best:
            best = r
            L.append(f"| {m} | {k} | {f:.1f} | {r:.4f} | {p:.4f} |")
    if pts_ideal:
        # The main frontier above uses the actual cost: Whisper's 30 s padded forward (fixed per utterance) divided by the
        # 4 s reference. This second table gives Whisper the idealized cost (attention over the real frames only, which
        # the stock checkpoint cannot deliver); the other encoders are unchanged.
        L.append("\n## Frontier, Whisper at idealized cost (real frames only; not attainable with the stock checkpoint)\n")
        L.append("| encoder | k | GFLOPs/s | rmse_z | PCC |")
        L.append("|---|---|---|---|---|")
        ideal_names = {m for m, *_ in pts_ideal}
        best = np.inf
        for m, k, f, r, p in sorted([x for x in pts if x[0] not in ideal_names] + pts_ideal, key=lambda x: (x[2], x[3])):
            if r < best:
                best = r
                L.append(f"| {m} | {k} | {f:.1f} | {r:.4f} | {p:.4f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary"))
    ap.add_argument("--model")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--tag", default=None, help="write to <model>_part<tag>.json (parallel layer ranges)")
    ap.add_argument("--shifts", type=int, nargs="*", default=None, help="shift grid (default -2..3), results to <model>_custom*.json")
    ap.add_argument("--wide", action="store_true", help="shift grid -4..5, results to <model>_wide*.json")
    args = ap.parse_args()
    if args.cmd == "run":
        if args.wide and args.shifts:
            ap.error("--wide and --shifts are exclusive")
        run(args.model, args.layers, tag=args.tag, shifts=WIDE_SHIFTS if args.wide else args.shifts,
            variant="_wide" if args.wide else ("_custom" if args.shifts else ""))
    else:
        summary()


if __name__ == "__main__":
    main()
