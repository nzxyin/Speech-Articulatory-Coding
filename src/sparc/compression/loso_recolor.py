"""Leave-one-speaker-out: z-trained models re-colored with the held-out speaker's covariance, vs zca12-trained models.

For each LOSO fold (held-out speaker h), the option-A checkpoints trained on per-channel z targets (LoRA r8 + causal
conv, seeds 0-2, and the frozen-encoder causal-conv control) are re-run on h's test sentences and scored in h's z
units (rmse_z, PCC) three ways:
    z            -- as trained (reproduces results.json test_unseen; checked);
    recolor_zca / recolor_zca12 -- predictions whitened with the training speakers' mean z-correlation and re-colored
                    with h's own (A_h^-1 A_bar; symmetric whitening, per articulator or over all 12 channels). Uses
                    h's covariance from its training split (enrollment data), as the zca12-trained models do.
The zca12-trained runs (dataset ema_loso_<h>_zca12; mapped back with h's own mm-geometry transform) are read from
their results.json. Train effect for a real model = zca12-trained - (z-trained + recolor_zca12); the two use
whitenings that differ by a per-group rotation (mm vs z geometry), noted as a caveat.

Usage:
    python -m sparc.compression.loso_recolor run --model wavlm-large --k 7
    python -m sparc.compression.loso_recolor summary
"""

import argparse
import json
from argparse import Namespace

import numpy as np
import torch
from scipy import stats as sstats

from . import datasets
from .metrics import ema_metrics
from .train import build, load_trainable, predict

OUT = datasets.XLSR_EMA / "analysis_loso_recolor"
RUN = "prefix{k}_independent_r8_q_proj+v_proj_conv_causal_s{s}"
FROZEN = "prefix{k}_none_conv_causal_s0"
ARMS = (("wavlm-large", 7), ("xlsr-300m", 18), ("xlsr-1b", 10), ("w2v2-large", 18))


def zmetrics(P, Y):
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    return {"rmse_z": float(np.sqrt(((Pc - Yc) ** 2).mean(0)).mean()), "pcc": ema_metrics(P, Y, n_boot=0)["pcc"]}


def zcorr(ds, g):
    sd = ds._stats[g][1]
    return ds._cov[g] / np.outer(sd, sd)


def run(model_name, k, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    out_f = OUT / f"{model_name}_k{k}.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    for h in datasets.ALL_SPEAKERS:
        name = f"ema_loso_{h}"
        ds = datasets.get(name)
        splits = ds.splits()
        Ytr = np.concatenate([ds.load_targets(s).ema_mm for s in splits["train"]])
        ym, ys = Ytr.mean(0), Ytr.std(0)
        train_groups = sorted({r["norm_group"] for r in ds.rows.values() if r["speaker"] not in ds.held_out})
        Cbar = np.mean([zcorr(ds, g) for g in train_groups], 0)
        test = [ds.load_utterance(s) for s in splits["test_unseen"]]
        maps = {}
        for kind in ("zca", "zca12"):
            Abar = datasets.target_matrices(kind, Cbar, np.ones(12))[0]
            maps[kind] = {g: (datasets.target_matrices(kind, zcorr(ds, g), np.ones(12))[1] @ Abar).T
                          for g in {u.norm_key for u in test}}
        for run_name in [RUN.format(k=k, s=s) for s in (0, 1, 2)] + [FROZEN.format(k=k)]:
            key = f"{h}/{run_name}"
            d = datasets.adapt_root(name) / model_name / run_name
            if key in res or not (d / "results.json").exists():
                continue
            r = json.loads((d / "results.json").read_text())
            args = Namespace(**r["config"])
            args.dataset, args.bf16 = name, device == "cuda"
            model, head, _ = build(args, device)
            load_trainable(model, head, torch.load(d / "best_trainable.pt"))
            preds, trues, _, kept = predict(model, head, test, args, ym, ys, device, return_utts=True)
            P = [p.astype(np.float64) for p in preds]
            Y = [t.astype(np.float64) for t in trues]
            e = {"z": zmetrics(P, Y), "stored_rmse_z": r["test_unseen"]["rmse_z"], "stored_pcc": r["test_unseen"]["pcc"]}
            for kind in ("zca", "zca12"):
                e[f"recolor_{kind}"] = zmetrics([p @ maps[kind][u.norm_key] for p, u in zip(P, kept)], Y)
            res[key] = e
            print(f"{model_name} {key}: z {e['z']['rmse_z']:.4f} (stored {e['stored_rmse_z']:.4f}) / {e['z']['pcc']:.4f}"
                  f" | recolor12 {e['recolor_zca12']['rmse_z']:.4f} / {e['recolor_zca12']['pcc']:.4f}", flush=True)
            out_f.write_text(json.dumps(res, indent=1))
            del model, head
            torch.cuda.empty_cache() if device == "cuda" else None


def _ci(d):
    h = sstats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.4f} [{d.mean() - h:+.4f}, {d.mean() + h:+.4f}] {(d > 0).sum()}/{len(d)}"


def summary():
    L = ["# LOSO: z-trained models re-colored with the held-out speaker's covariance vs zca12-trained models\n",
         "Held-out speaker test, rmse_z / PCC in its z units; per fold, seed mean (LoRA: 3 seeds; frozen: 1). Paired "
         "differences over the 7 folds: mean [95% t CI] folds better (+ = first arm better).\n"]
    for model_name, k in ARMS:
        f = OUT / f"{model_name}_k{k}.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        L.append(f"\n## {model_name} k={k}\n")
        maxdev = max(abs(v["z"]["rmse_z"] - v["stored_rmse_z"]) for v in res.values())
        L.append(f"Reproduction check: max |rmse_z - stored| = {maxdev:.2e}\n")
        for label, runs, zdir_runs in (
                ("LoRA + causal conv", [RUN.format(k=k, s=s) for s in (0, 1, 2)], [RUN.format(k=k, s=s) for s in (0, 1, 2)]),
                ("frozen + causal conv", [FROZEN.format(k=k)], [FROZEN.format(k=k)])):
            arms = {"z": {}, "z+recolor[zca]": {}, "z+recolor[zca12]": {}, "zca12-trained (own map)": {}}
            for h in datasets.ALL_SPEAKERS:
                es = [res[f"{h}/{r}"] for r in runs if f"{h}/{r}" in res]
                if es:
                    arms["z"][h] = [np.mean([e["z"][m] for e in es]) for m in ("rmse_z", "pcc")]
                    for kind in ("zca", "zca12"):
                        arms[f"z+recolor[{kind}]"][h] = [np.mean([e[f"recolor_{kind}"][m] for e in es]) for m in ("rmse_z", "pcc")]
                wd = datasets.adapt_root(f"ema_loso_{h}_zca12") / model_name
                ws = [json.loads((wd / r / "results.json").read_text())["test_unseen"] for r in zdir_runs
                      if (wd / r / "results.json").exists()]
                if ws:
                    arms["zca12-trained (own map)"][h] = [np.mean([w["rmse_z"] for w in ws]), np.mean([w["pcc"] for w in ws])]
            folds = [h for h in datasets.ALL_SPEAKERS if all(h in a for a in arms.values())]
            L.append(f"\n### {label} ({len(folds)} folds complete in every arm)\n")
            L.append("| arm | rmse_z | PCC |")
            L.append("|---|---|---|")
            for a, v in arms.items():
                fs = [h for h in folds if h in v] or [h for h in datasets.ALL_SPEAKERS if h in v]
                if fs:
                    L.append(f"| {a} | {np.mean([v[h][0] for h in fs]):.4f} | {np.mean([v[h][1] for h in fs]):.4f} |")
            if len(folds) >= 2:
                L.append("\n| contrast | rmse_z (+ = lower) | PCC (+ = higher) |")
                L.append("|---|---|---|")
                for name, a, b in (("recolor_zca12 effect: z+recolor[zca12] - z", "z+recolor[zca12]", "z"),
                                   ("train effect: zca12-trained - z+recolor[zca12]", "zca12-trained (own map)", "z+recolor[zca12]"),
                                   ("total: zca12-trained - z", "zca12-trained (own map)", "z")):
                    dr = np.array([arms[b][h][0] - arms[a][h][0] for h in folds])
                    dp = np.array([arms[a][h][1] - arms[b][h][1] for h in folds])
                    L.append(f"| {name} | {_ci(dr)} | {_ci(dp)} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary"))
    ap.add_argument("--model")
    ap.add_argument("--k", type=int)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.model, args.k)
    else:
        summary()


if __name__ == "__main__":
    main()
