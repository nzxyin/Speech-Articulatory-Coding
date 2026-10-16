"""Nested depth and head selection (Item 4 of the pre-registered rigor plan, docs/rigor_prereg.md).

Question: how optimistic is a depth recommendation (k12 for WavLM, k18 for XLS-R 300M) that was chosen with the
held-out speakers' scores in the loop? For each outer held-out speaker h (7 LOSO folds), an inner leave-one-speaker-out
runs over its 6 training speakers i: train on the other 5 speakers' training splits, score speaker i's test sentences
(`EMAMulti(held_out=(h, i))`, `test_unseen` filtered to speaker i). Nothing about h is used. The depth chosen from the
inner scores is then read off the existing outer result for fold h:
    probe  -- every cached depth 1..24 of wavlm-large and xlsr-300m; shift = the outer fold's depth-sweep shift at
              that depth, ridge alpha on the inner training speakers' validation sentences; depth = lowest mean inner
              rmse_z (the highest-PCC choice is reported too); outer score = depth_sweep's held-out result.
    heads  -- pre-declared gruconv9 (BiGRU -> conv 9) and mamba (BiMamba), WavLM depths {7, 9, 12, 15, 18}; one seed
              (0) per inner fold, smoother off, one fixed learning rate per head (the most frequent choice over the
              head screen's folds at k7/9/12; ties go to the smaller rate); shift = the outer fold's depth-sweep
              shift; depth = highest mean inner PCC (the rmse_z choice is reported too); outer score = the head
              screen's 3-seed mean at the chosen depth.
Reported per fold: chosen depth, nested outer score vs the fixed k12 (WavLM) / k18 (300M) and vs the cross-fold choice
(the depth with the best mean held-out score over the other six folds), mean difference, paired 95% t CI, folds won and
the exact sign-flip p. Pre-registered contrast N1: gruconv9 at the nested-selected depth vs at k12, PCC, + = nested
better. rmse_z / PCC are in the scored speaker's z units (pooled frames per channel, mean over 12 channels).

Inner results are written incrementally (resumable) to analysis_rigor/nested/inner/{probe_<model>_<h>,
head_<head>_<h>}.json with keys "<k>/<i>"; `summary` writes probe_<model>.json, heads_<head>.json and summary.md.

Usage:
    python -m sparc.compression.nested probe --model wavlm-large [--folds usc_M1 ...] [--layers 12 ...]
    python -m sparc.compression.nested heads --head gruconv9 --folds usc_M1 [--layers 7 9 12 15 18]
    python -m sparc.compression.nested summary
"""

import argparse
import json
from collections import Counter

import numpy as np
from scipy import stats as sstats

from . import datasets

SPK = datasets.ALL_SPEAKERS
OUT = datasets.XLSR_EMA / "analysis_rigor" / "nested"
INNER = OUT / "inner"
PROBE_MODELS = ("wavlm-large", "xlsr-300m")
FIXED = {"wavlm-large": 12, "xlsr-300m": 18}
HEAD_DEPTHS = (7, 9, 12, 15, 18)
HEADS = ("gruconv9", "mamba")
HEAD_MODEL = "wavlm-large"
LR_LAYERS = (7, 9, 12)


# ---------------------------------------------------------------- pure helpers

def argsel(scores, mode):
    """Depth with the lowest ('min') or highest ('max') score; ties go to the shallower depth."""
    ks = sorted(scores)
    if mode == "min":
        return min(ks, key=lambda k: (scores[k], k))
    return max(ks, key=lambda k: (scores[k], -k))


def most_frequent_lr(lrs):
    """Most frequent learning rate; ties go to the smaller rate."""
    c = Counter(float(x) for x in lrs)
    top = max(c.values())
    return min(x for x, n in c.items() if n == top)


def paired_stats(nested, ref, sign):
    """Paired difference sign*(nested - ref) over folds: mean, 95% t CI, folds won, exact sign-flip p."""
    from .rigor import sign_flip_p

    d = sign * (np.asarray(nested, float) - np.asarray(ref, float))
    n = len(d)
    hw = float(sstats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(d.mean()), "lo": float(d.mean() - hw), "hi": float(d.mean() + hw),
            "won": int((d > 0).sum()), "n": n, "p": sign_flip_p(d) if n > 1 else float("nan")}


def fmt_stats(s):
    return f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] {s['won']}/{s['n']}, p={s['p']:.3f}"


def head_lr(head, root=None):
    root = root or datasets.XLSR_EMA / "analysis_heads" / HEAD_MODEL
    lrs = []
    for k in LR_LAYERS:
        f = root / f"k{k}_{head}.json"
        if f.exists():
            lrs += [r["lr"] for r in json.loads(f.read_text()).values() if "lr" in r]
    if not lrs:
        raise FileNotFoundError(f"no head-screen learning rates for {head}")
    return most_frequent_lr(lrs)


# ---------------------------------------------------------------- inner computation

def zm(P, Y):
    from .lora_readout import zm as _zm

    return _zm(P, Y)


class Inner:
    """Inner LOSO data for one outer fold h: datasets, targets, and aligned train/valid/test arrays per inner speaker i."""

    def __init__(self, model, h, index):
        self.h, self.index = h, index
        self.inner = [s for s in SPK if s != h]
        self.ds = {i: datasets.EMAMulti(held_out=(h, i), norm_by="norm_group") for i in self.inner}
        self.sp = {}
        for i, ds in self.ds.items():
            sp = {k: [s for s in v if s in index] for k, v in ds.splits().items()}
            sp["test_i"] = [s for s in sp["test_unseen"] if ds.rows[s]["speaker"] == i]
            self.sp[i] = sp
        from .probe import load_targets

        stems = sorted(set(index) & set(self.ds[self.inner[0]].rows))
        self.utts = load_targets(stems, self.ds[self.inner[0]])  # z targets do not depend on the held-out set

    def arrays(self, feats, i, shift, splits):
        from .probe import aligned

        out = {}
        for s in splits:
            X, Y, _ = aligned(feats, self.index, self.utts, self.sp[i][s], shift)
            out[s] = (X, Y)
        return out


def _load_json(f):
    return json.loads(f.read_text()) if f.exists() else {}


def run_probe(model, folds=None, layers=None, device=None):
    import torch

    from .depth_sweep import load_all
    from .lora_readout import fit_ridge
    from .probe import Ridge

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = datasets.features_root("ema_multi") / model
    index = json.loads((d / "index.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    layers = list(layers) if layers else list(range(1, meta["num_layers"] + 1))
    depth = load_all(model)
    INNER.mkdir(parents=True, exist_ok=True)
    for h in folds or SPK:
        f = INNER / f"probe_{model}_{h}.json"
        res = _load_json(f)
        todo = [k for k in layers if not all(f"{k}/{i}" in res for i in SPK if i != h)]
        if not todo:
            continue
        inn = Inner(model, h, index)
        for k in todo:
            feats = np.load(d / f"layer_{k:02d}.npy", mmap_mode="r").astype(np.float32)
            shift = depth[f"{k}/{h}"]["shift"]
            for i in inn.inner:
                key = f"{k}/{i}"
                if key in res:
                    continue
                a = inn.arrays(feats, i, shift, ("train", "valid", "test_i"))
                ridge, W, b = fit_ridge(a["train"][0], a["train"][1], a["valid"][0], a["valid"][1], device)
                P = ridge.predict(W, b, a["test_i"][0])
                res[key] = {"shift": shift, **zm(P, a["test_i"][1]), "n_utts": len(P)}
            f.write_text(json.dumps(res, indent=1))
            m = [res[f"{k}/{i}"] for i in inn.inner]
            print(f"{model} outer {h} k={k} shift {shift}: inner rmse_z {np.mean([x['rmse_z'] for x in m]):.4f} "
                  f"PCC {np.mean([x['pcc'] for x in m]):.4f}", flush=True)
            del feats


def run_heads(head, folds=None, layers=None, device=None):
    import torch

    from . import head_screen as hs
    from .depth_sweep import load_all

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lr = head_lr(head)
    d = datasets.features_root("ema_multi") / HEAD_MODEL
    index = json.loads((d / "index.json").read_text())
    depth = load_all(HEAD_MODEL)
    INNER.mkdir(parents=True, exist_ok=True)
    for h in folds or SPK:
        f = INNER / f"head_{head}_{h}.json"
        res = _load_json(f)
        todo = [k for k in (layers or HEAD_DEPTHS) if not all(f"{k}/{i}" in res for i in SPK if i != h)]
        if not todo:
            continue
        inn = Inner(HEAD_MODEL, h, index)
        for k in todo:
            feats = np.load(d / f"layer_{k:02d}.npy", mmap_mode="r")
            shift = depth[f"{k}/{h}"]["shift"]
            for i in inn.inner:
                key = f"{k}/{i}"
                if key in res:
                    continue
                a = inn.arrays(feats, i, shift, ("train", "valid", "test_i"))
                data = {s: ([np.asarray(x, dtype=np.float32) for x in X], [y.astype(np.float64) for y in Y])
                        for s, (X, Y) in a.items()}
                Yall = np.concatenate(data["train"][1])
                ym, ys = Yall.mean(0), Yall.std(0)
                head_m, v = hs.train_one(head, False, lr, data, ym, ys, device, seed=0)
                if head_m is None:
                    res[key] = {"failed": True, "shift": shift, "lr": lr}
                else:
                    P = hs.predict(head_m, data["test_i"][0], ym, ys, device)
                    res[key] = {"shift": shift, "lr": lr, "valid_rmse_z": v, **zm(P, data["test_i"][1]), "n_utts": len(P)}
                f.write_text(json.dumps(res, indent=1))
                print(f"{head} outer {h} k={k} inner {i}: {res[key]}", flush=True)
            del feats


# ---------------------------------------------------------------- summary

def inner_means(h, prefix, ks):
    """{k: {metric: mean over inner speakers}} for the depths with all six inner results; failures drop the depth."""
    res = _load_json(INNER / f"{prefix}_{h}.json")
    out = {}
    for k in ks:
        r = [res.get(f"{k}/{i}") for i in SPK if i != h]
        if all(x is not None and "failed" not in x for x in r):
            out[k] = {m: float(np.mean([x[m] for x in r])) for m in ("rmse_z", "pcc")}
    return out


def head_outer(head, k, h):
    f = datasets.XLSR_EMA / "analysis_heads" / HEAD_MODEL / f"k{k}_{head}.json"
    rec = _load_json(f).get(h)
    if not rec:
        return None
    ok = [x for x in rec["seeds"].values() if "failed" not in x]
    if not ok:
        return None
    return {m: float(np.mean([x["test_unseen"][m] for x in ok])) for m in ("rmse_z", "pcc")}


def select_table(name, outer, inner_of, ks, fixed, crit, sel_mode, L, main_metric):
    """Per fold nested choice vs fixed depth and cross-fold choice; returns a JSON-able result or None if incomplete."""
    rows = {}
    for h in SPK:
        inn = inner_of(h)
        if not inn:
            return None
        ksel = [k for k in ks if k in inn]
        if fixed not in ksel or any(outer(k, h) is None for k in ksel):
            return None
        kn = argsel({k: inn[k][crit] for k in ksel}, sel_mode)
        kn_other = argsel({k: inn[k]["pcc" if crit == "rmse_z" else "rmse_z"] for k in ksel},
                          "max" if crit == "rmse_z" else "min")
        rows[h] = {"k": kn, "k_other_metric": kn_other, "ks": ksel}
    for h in SPK:
        ksel = rows[h]["ks"]
        others = [o for o in SPK if o != h]
        # cross-fold: depth with the best mean held-out score on the other six folds, same metric as the nested rule
        cm = {k: np.mean([outer(k, o)[crit] for o in others]) for k in ksel}
        rows[h]["k_cross"] = argsel(cm, sel_mode)
    res = {"name": name, "fixed": fixed, "crit": crit, "folds": {}}
    L.append(f"\n### {name}  (nested rule: {'lowest' if sel_mode == 'min' else 'highest'} mean inner {crit})\n")
    L.append("| fold | nested k | (other-metric k) | cross-fold k | nested rmse_z / PCC | "
             f"k{fixed} rmse_z / PCC | cross-fold rmse_z / PCC | best-in-hindsight k (PCC) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for h in SPK:
        r = rows[h]
        o = {k: outer(k, h) for k in r["ks"]}
        hind = max(r["ks"], key=lambda k: o[k]["pcc"])
        res["folds"][h] = {"nested_k": r["k"], "other_metric_k": r["k_other_metric"], "cross_k": r["k_cross"],
                           "nested": o[r["k"]], "fixed": o[fixed], "cross": o[r["k_cross"]], "hindsight_k": hind}
        L.append(f"| {h} | {r['k']} | {r['k_other_metric']} | {r['k_cross']} | {o[r['k']]['rmse_z']:.4f} / "
                 f"{o[r['k']]['pcc']:.4f} | {o[fixed]['rmse_z']:.4f} / {o[fixed]['pcc']:.4f} | "
                 f"{o[r['k_cross']]['rmse_z']:.4f} / {o[r['k_cross']]['pcc']:.4f} | {hind} |")
    F = res["folds"]
    mean = lambda key, m: float(np.mean([F[h][key][m] for h in SPK]))
    L.append(f"| mean | | | | {mean('nested', 'rmse_z'):.4f} / {mean('nested', 'pcc'):.4f} | "
             f"{mean('fixed', 'rmse_z'):.4f} / {mean('fixed', 'pcc'):.4f} | "
             f"{mean('cross', 'rmse_z'):.4f} / {mean('cross', 'pcc'):.4f} | |")
    L.append("\nPaired over the 7 folds, nested minus reference (+ = nested better); mean [95% t CI] folds better, "
             "exact sign-flip p:\n")
    res["tests"] = {}
    for ref, label in (("fixed", f"k{fixed}"), ("cross", "cross-fold choice")):
        for m, sign in (("pcc", 1), ("rmse_z", -1)):
            s = paired_stats([F[h]["nested"][m] for h in SPK], [F[h][ref][m] for h in SPK], sign)
            res["tests"][f"{ref}_{m}"] = s
            tag = "  **N1 (pre-registered)**" if (main_metric and ref == "fixed" and m == "pcc") else ""
            L.append(f"- vs {label}, {m}: {fmt_stats(s)}{tag}")
    return res


def summary():
    from .depth_sweep import load_all

    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# Nested depth and head selection (Item 4; design in docs/rigor_prereg.md)\n",
         "Inner LOSO over each outer fold's 6 training speakers (train on 5, score the 6th's test sentences); depth chosen "
         "on inner scores only, scored with the existing outer result at that depth. rmse_z / PCC in the held-out "
         "speaker's z units. All outer configurations were already visible on these folds (see CLAUDE.md caveats), so "
         "'cross-fold' is itself mildly optimistic.\n", "## Ridge probe, depth 1..24\n"]
    for model in PROBE_MODELS:
        depth = load_all(model)
        ks = list(range(1, 25))
        outer = lambda k, h, depth=depth: depth[f"{k}/{h}"]["unseen"] if f"{k}/{h}" in depth else None
        res = select_table(f"{model} (ridge probe)", outer, lambda h, model=model: inner_means(h, f"probe_{model}", ks),
                           ks, FIXED[model], "rmse_z", "min", L, False)
        if res is None:
            L.append(f"\n### {model}: incomplete (inner probe results missing)\n")
            continue
        (OUT / f"probe_{model}.json").write_text(json.dumps(res, indent=1))
        # PCC-selected variant
        res2 = select_table(f"{model} (ridge probe), PCC-selected", outer,
                            lambda h, model=model: inner_means(h, f"probe_{model}", ks), ks, FIXED[model], "pcc",
                            "max", L, False)
        if res2:
            (OUT / f"probe_{model}_pccsel.json").write_text(json.dumps(res2, indent=1))
    summ = {}
    L.append("\n## Frozen heads on WavLM Large, depths 7/9/12/15/18 (3-seed outer scores)\n")
    for head in HEADS:
        outer = lambda k, h, head=head: head_outer(head, k, h)
        inner_of = lambda h, head=head: inner_means(h, f"head_{head}", HEAD_DEPTHS)
        res = select_table(f"{head} (lr {head_lr(head):g})", outer, inner_of, HEAD_DEPTHS, 12, "pcc", "max", L,
                           head == "gruconv9")
        if res is None:
            L.append(f"\n### {head}: incomplete (inner results or the k15/k18 head-screen results missing)\n")
            continue
        (OUT / f"heads_{head}.json").write_text(json.dumps(res, indent=1))
        if head == "gruconv9":
            t = res["tests"]["fixed_pcc"]
            summ["N1"] = {"a": "frozen BiGRU->conv at nested depth", "b": "frozen BiGRU->conv at k12", "metric": "pcc",
                          "n_folds": t["n"], "mean": t["mean"], "t_ci": [t["lo"], t["hi"]], "won": t["won"],
                          "sign_flip_p": t["p"]}
        res2 = select_table(f"{head}, rmse_z-selected", outer, inner_of, HEAD_DEPTHS, 12, "rmse_z", "min", L, False)
        if res2:
            (OUT / f"heads_{head}_rmsesel.json").write_text(json.dumps(res2, indent=1))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("probe", "heads", "summary"))
    ap.add_argument("--model", choices=PROBE_MODELS)
    ap.add_argument("--head", choices=HEADS)
    ap.add_argument("--folds", nargs="*", default=None, choices=SPK)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    args = ap.parse_args()
    if args.cmd == "probe":
        run_probe(args.model, args.folds, args.layers)
    elif args.cmd == "heads":
        run_heads(args.head, args.folds, args.layers)
    else:
        summary()


if __name__ == "__main__":
    main()
