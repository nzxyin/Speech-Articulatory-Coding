"""Leave-one-speaker-out cross-validation summary (datasets ema_loso_<speaker>).

For each fold, a model is trained on the other six speakers (their train splits; early stopping on their
valid splits) and tested on the held-out speaker's test sentences (test_unseen; text-disjoint from training).
Targets are z-scored per normalization group (speaker, or recording session for usc_F5). SPARC resynthesizes from
this normalized space and the models predict it directly, so z-unit RMSE (rmse_z) and PCC are the primary
metrics; mm RMSE additionally uses the held-out speaker's own statistics to de-normalize and is secondary.

Per fold, metrics are averaged over seeds. Per arm: mean +/- std across the seven folds and per-corpus means.
Paired comparisons across folds: mean paired difference with a 95% t interval, folds won, and an exact two-sided
sign test with Holm correction over all comparisons reported. With 7 folds only 7/7 can reach p < 0.05 (p=0.0156)
and folds share training speakers, so treat the intervals as the main evidence.

Usage: python -m sparc.compression.loso --arms label=model/run_name ...   (run name without the _s<seed> suffix)
"""

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

from . import datasets

METRICS = (("rmse_z", -1), ("pcc", 1), ("rmse", -1))  # (name, +1 if higher is better)


def sign_test(wins, n):
    """Exact two-sided sign test p-value for `wins` successes out of n (ties excluded beforehand)."""
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def collect(model, run):
    """{speaker: {metric: mean over seeds}} for the held-out speaker of each fold."""
    out = {}
    for spk in datasets.ALL_SPEAKERS:
        d = datasets.adapt_root(f"ema_loso_{spk}") / model
        rs = [json.loads(f.read_text()) for f in sorted(d.glob(f"{run}_s*/results.json"))]
        if not rs:
            continue
        u = [r["test_unseen"] for r in rs]
        out[spk] = {"n_seeds": len(rs), **{m: float(np.mean([x[m] for x in u])) for m, _ in METRICS},
                    "rmse_z_seed_sd": float(np.std([x["rmse_z"] for x in u], ddof=1)) if len(rs) > 1 else 0.0,
                    "seen_rmse_z": float(np.mean([r["test"]["rmse_z"] for r in rs])),
                    "seen_pcc": float(np.mean([r["test"]["pcc"] for r in rs]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="label=model/run_name")
    ap.add_argument("--out", default=str(datasets.XLSR_EMA / "analysis_loso"))
    args = ap.parse_args()
    arms = {}
    for a in args.arms:
        label, spec = a.split("=", 1)
        model, run = spec.split("/", 1)
        arms[label] = (model, run)
    res = {lab: collect(m, run) for lab, (m, run) in arms.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    folds = [s for s in datasets.ALL_SPEAKERS if all(s in res[a] for a in res)]
    L = ["# Leave-one-speaker-out cross-validation\n", f"Folds complete for every arm: {len(folds)} / 7 ({', '.join(folds)})\n"]
    for metric, _ in METRICS:
        L += [f"\n## Held-out speaker {metric} (seed mean per fold)\n",
              "| arm | " + " | ".join(folds) + " | mean +/- sd over folds | USC-TIMIT mean | EMA_5EMO mean |",
              "|---|" + "---|" * (len(folds) + 3)]
        for a, r in res.items():
            v = [r[s][metric] for s in folds]
            usc = [r[s][metric] for s in folds if s.startswith("usc")]
            emo = [r[s][metric] for s in folds if s.startswith("5emo")]
            L.append(f"| {a} | " + " | ".join(f"{x:.3f}" for x in v) +
                     f" | {np.mean(v):.3f} +/- {np.std(v, ddof=1):.3f} | {np.mean(usc):.3f} | {np.mean(emo):.3f} |")
    rows, ps = [], []
    for a, b in combinations(res, 2):
        for metric, better in METRICS[:2]:
            diff = np.array([better * (res[a][s][metric] - res[b][s][metric]) for s in folds])
            wins, n = int((diff > 0).sum()), int((diff != 0).sum())
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
            p = sign_test(wins, n)
            rows.append((a, b, metric, wins, diff.mean(), half, p))
            ps.append(p)
    adj = holm(np.array(ps)) if ps else []
    L += ["\n## Paired comparisons across folds (positive = first arm better; primary metrics)\n",
          "| A | B | metric | folds A better | mean difference [95% t CI] | sign-test p | Holm p |", "|---|---|---|---|---|---|---|"]
    for (a, b, metric, wins, m, h, p), pa in zip(rows, adj):
        L.append(f"| {a} | {b} | {metric} | {wins} / {len(folds)} | {m:+.4f} [{m - h:+.4f}, {m + h:+.4f}] | {p:.3f} | {pa:.3f} |")
    L += ["\nSeen speakers (the six training speakers' test sentences, mean over folds):\n", "| arm | rmse_z | PCC |", "|---|---|---|"]
    for a, r in res.items():
        L.append(f"| {a} | {np.mean([r[s]['seen_rmse_z'] for s in folds]):.3f} | {np.mean([r[s]['seen_pcc'] for s in folds]):.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "results.json").write_text(json.dumps({"arms": {k: list(v) for k, v in arms.items()}, "per_fold": res}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
