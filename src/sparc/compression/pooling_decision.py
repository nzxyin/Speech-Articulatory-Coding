"""Does layer pooling help? Applies the pre-registered rule (CLAUDE.md, 2026-10-07) to 3-seed runs.

Per model, the unpooled arm (hidden 256 or 48) and the pooled arm (static per-articulator h48, attention, attention
per-articulator) are each chosen by their 3-seed mean validation error. Pooling helps that model if the chosen pooled
arm beats the chosen unpooled arm on seen-test error by more than 2x the SE of the seed-paired difference, and is not
worse on unseen speakers by more than 2x the SE of that paired difference. Pooling is kept only if it helps
>= 2 of the 3 ema_multi models. The error is rmse_z on ema_multi and mm RMSE on MNGU0 (single speaker, no unseen set).

Usage: python -m sparc.compression.pooling_decision
"""

import json

import numpy as np

from . import datasets

BASE = "independent_r8_q_proj+v_proj_conv_causal"
UNPOOLED = {"h256": "", "h48": "_h48"}
POOLED = {"static_art_h48": "_poolstatic_art_h48", "attn": "_poolattn", "attn_art": "_poolattn_art"}
CASES = [("ema_multi", "wavlm-large", 7), ("ema_multi", "xlsr-300m", 18), ("ema_multi", "xlsr-1b", 10),
         ("mngu0", "xlsr-1b", 16)]
SEEDS = (0, 1, 2)


def load(dataset, model, k, suffix):
    d = datasets.adapt_root(dataset) / model
    out = {}
    for s in SEEDS:
        f = d / f"prefix{k}_{BASE}{suffix}_s{s}" / "results.json"
        if f.exists():
            out[s] = json.loads(f.read_text())
    return out


def err(r, split, metric):
    return r[split][metric]


def paired(a, b, split, metric):
    """Seed-paired difference b - a (positive = a better when lower is better); mean, SE."""
    d = np.array([err(b[s], split, metric) - err(a[s], split, metric) for s in SEEDS])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main():
    L = ["# Pooling decision (pre-registered rule)\n"]
    helps = {}
    for dataset, model, k in CASES:
        metric = "rmse_z" if dataset != "mngu0" else "rmse"
        # MNGU0 is a confirmatory check of the one arm carried over from the single-seed study
        pooled = POOLED if dataset != "mngu0" else {"static_art_h48": POOLED["static_art_h48"]}
        arms = {n: load(dataset, model, k, s) for n, s in {**UNPOOLED, **pooled}.items()}
        missing = {n: [s for s in SEEDS if s not in a] for n, a in arms.items() if len(a) < len(SEEDS)}
        L.append(f"\n## {dataset} {model} k={k} (metric {metric})\n")
        L.append("| arm | valid | seen test | unseen test | seen PCC | unseen PCC |")
        L.append("|---|---|---|---|---|---|")
        for n, a in arms.items():
            if not a:
                continue
            m = lambda sp, me: np.mean([err(r, sp, me) for r in a.values()])  # noqa: E731
            un = (f"{m('test_unseen', metric):.4f}", f"{m('test_unseen', 'pcc'):.4f}") if dataset != "mngu0" else ("-", "-")
            L.append(f"| {n} ({len(a)} seeds) | {m('valid', metric):.4f} | {m('test', metric):.4f} | {un[0]} | "
                     f"{m('test', 'pcc'):.4f} | {un[1]} |")
        if missing:
            L.append(f"\nIncomplete arms: {missing}; no decision.")
            continue
        vmean = {n: np.mean([err(r, "valid", metric) for r in a.values()]) for n, a in arms.items()}
        u = min(UNPOOLED, key=vmean.get)
        p = min(pooled, key=vmean.get)
        gain, se = paired(arms[p], arms[u], "test", metric)
        ok = gain > 2 * se
        line = f"\nChosen on validation: unpooled {u}, pooled {p}. Seen gain {gain:+.4f} (SE {se:.4f}, need > {2 * se:.4f})"
        if dataset != "mngu0":
            ugain, use = paired(arms[p], arms[u], "test_unseen", metric)
            ok = ok and ugain > -2 * use
            line += f"; unseen gain {ugain:+.4f} (SE {use:.4f}, must be > {-2 * use:.4f})"
        L.append(line + f". Pooling helps: **{ok}**")
        helps[(dataset, model)] = ok
    multi = [v for (d, _), v in helps.items() if d == "ema_multi"]
    if len(multi) == 3:
        keep = sum(multi) >= 2
        L.append(f"\n## Decision\nPooling helps {sum(multi)} / 3 ema_multi models: **{'keep' if keep else 'discard'} pooling**.")
    out = datasets.XLSR_EMA / "analysis_multi" / "pooling_decision.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
