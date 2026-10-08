"""Leave-one-speaker-out cross-validation summary (datasets ema_loso_<speaker>).

For each fold, a model is trained on the other six speakers (their train splits; early stopping on their
valid splits) and tested on the held-out speaker's test sentences (test_unseen; text-disjoint from training).
Per fold, metrics are averaged over seeds; then per model: mean +/- std across the seven folds, per-corpus
means, and paired comparisons between models across folds (folds won, mean paired difference +/- std, and an
exact two-sided sign test).

Usage: python -m sparc.compression.loso --runs wavlm-large=<run name> xlsr-300m=<run name> xlsr-1b=<run name>
"""

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from . import datasets


def sign_test(wins, n):
    """Exact two-sided sign test p-value for `wins` successes out of n (ties excluded beforehand)."""
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def collect(model, run):
    """{speaker: {metric: mean over seeds}} for the held-out speaker of each fold."""
    out = {}
    for spk in datasets.ALL_SPEAKERS:
        d = datasets.adapt_root(f"ema_loso_{spk}") / model
        seeds = sorted(d.glob(f"{run}_s*/results.json"))
        if not seeds:
            continue
        rs = [json.loads(f.read_text()) for f in seeds]
        out[spk] = {"n_seeds": len(rs),
                    "rmse": float(np.mean([r["test_unseen"]["rmse"] for r in rs])),
                    "pcc": float(np.mean([r["test_unseen"]["pcc"] for r in rs])),
                    "rmse_z": float(np.mean([r["test_unseen"]["rmse_z"] for r in rs])),
                    "rmse_seed_sd": float(np.std([r["test_unseen"]["rmse"] for r in rs], ddof=1)) if len(rs) > 1 else 0.0,
                    "seen_rmse": float(np.mean([r["test"]["rmse"] for r in rs])),
                    "seen_pcc": float(np.mean([r["test"]["pcc"] for r in rs]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="model=run_name (without the _s<seed> suffix)")
    ap.add_argument("--out", default=str(datasets.XLSR_EMA / "analysis_loso"))
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    res = {m: collect(m, run) for m, run in runs.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    folds = [s for s in datasets.ALL_SPEAKERS if all(s in res[m] for m in res)]
    lines = ["# Leave-one-speaker-out cross-validation\n", f"Folds complete for all models: {len(folds)} / 7\n",
             "| model | configuration | " + " | ".join(f"{s} RMSE / PCC" for s in folds) + " | mean RMSE | mean PCC |",
             "|---|---|" + "---|" * (len(folds) + 2)]
    for m, r in res.items():
        cells = [f"{r[s]['rmse']:.3f} / {r[s]['pcc']:.3f}" for s in folds]
        lines.append(f"| {m} | {runs[m]} | " + " | ".join(cells) +
                     f" | {np.mean([r[s]['rmse'] for s in folds]):.3f} +/- {np.std([r[s]['rmse'] for s in folds], ddof=1):.3f}"
                     f" | {np.mean([r[s]['pcc'] for s in folds]):.3f} +/- {np.std([r[s]['pcc'] for s in folds], ddof=1):.3f} |")
    lines += ["\n## Per corpus (mean over held-out speakers of the corpus)\n",
              "| model | USC-TIMIT RMSE / PCC | EMA_5EMO RMSE / PCC |", "|---|---|---|"]
    for m, r in res.items():
        cell = []
        for pre in ("usc", "5emo"):
            ss = [s for s in folds if s.startswith(pre)]
            cell.append(f"{np.mean([r[s]['rmse'] for s in ss]):.3f} / {np.mean([r[s]['pcc'] for s in ss]):.3f}" if ss else "-")
        lines.append(f"| {m} | " + " | ".join(cell) + " |")
    lines += ["\n## Paired comparisons across folds (A vs B: positive difference = A better)\n",
              "| A | B | metric | folds A better | mean difference +/- std | sign-test p |", "|---|---|---|---|---|---|"]
    for a, b in combinations(res, 2):
        for metric, better in (("rmse", -1), ("pcc", 1)):
            diff = np.array([better * (res[a][s][metric] - res[b][s][metric]) for s in folds])
            wins, n = int((diff > 0).sum()), int((diff != 0).sum())
            lines.append(f"| {a} | {b} | {metric} | {wins} / {len(folds)} | {diff.mean():+.4f} +/- {diff.std(ddof=1):.4f} | "
                         f"{sign_test(wins, n):.3f} |")
    lines += ["\nSeen-speaker test (the six training speakers' test sentences, mean over folds):\n",
              "| model | RMSE | PCC |", "|---|---|---|"]
    for m, r in res.items():
        lines.append(f"| {m} | {np.mean([r[s]['seen_rmse'] for s in folds]):.3f} | {np.mean([r[s]['seen_pcc'] for s in folds]):.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps({"runs": runs, "per_fold": res}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
