"""Speaker scaling and target normalization: does whitening each speaker's EMA make more training speakers more useful?

The model is fixed (a ridge probe on one cached encoder layer, fixed alignment shift); only the training speakers and
the target normalization vary. A ridge fit depends only on its training-speaker subset S, so every non-empty subset of
the seven speakers with |S| <= 6 is fitted once and scored on every speaker not in S ("unseen") and on each speaker of
S on its own test sentences ("seen").

Why a 2 x 2 design. Ridge with per-dimension target standardization is exactly equivariant to a linear mixing of the
targets, so for a linear probe the target space itself does not matter; what matters is where speaker-specific
transforms enter. With A_g the whitening (datasets.target_matrices, kind zca or zca12) of normalization group g in
z units and A_bar the same whitening computed from the mean training-group correlation matrix of S:
    train factor  -- 'z': fit on per-channel z targets of all speakers in S;
                     'w': fit on A_g z, every training group whitened with its own covariance before pooling;
    map factor    -- 'shared': predictions mapped to the scored speaker h's z with the subset-level A_bar
                     (z model: identity; w model: A_bar^-1);
                     'own': with h's own covariance (z model: A_h^-1 A_bar, i.e. test-time recoloring;
                     w model: A_h^-1). 'own' uses h's full covariance (enrollment data), 'shared' only h's means/stds.
All whitenings here are symmetric (ZCA) whitenings of z-correlation matrices (per articulator for zca, joint for
zca12); this differs from the datasets' mm-geometry zca/zca12 targets only by a per-group rotation, and keeps the
group and shared maps on one construction.
The train effect (w - z) is what tells whether per-speaker whitening lets pooled speakers help a new speaker more.
With one training group it is exactly zero (equivariance), so its change with |S| is the quantity of interest.

Data conditions: 'all' uses every training utterance of each speaker in S; 'frames' uses the first utterances of a
fixed random order per speaker until F / |S| frames (F = the smallest per-speaker training frame count), so the total
amount of training data is constant across |S|.

Ridge strength is chosen once per (S, condition, train factor) from probe.ALPHAS on the validation sentences of S,
by mean per-channel RMSE in z units (training groups mapped with their own transforms). All primary metrics are in the
scored speaker's per-channel z units: rmse (mean over channels of the pooled RMSE), pcc (per-utterance, per-channel
mean) and r2 (mean over channels of 1 - MSE / variance of the speaker's test targets). For the 'w'/'own' arm the error
in the whitened space is also kept ("own_space"), for reference only: whitening gives a speaker's low-variance
directions unit weight, so it is not comparable with z-unit numbers.

Fits come from per-speaker sufficient statistics, exactly equivalent to probe.Ridge on concatenated frames.

Usage:
    python -m sparc.compression.speaker_scaling run --model wavlm-large --layer 7 --shift 0
    python -m sparc.compression.speaker_scaling summary
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from . import datasets
from .mngu0 import align
from .probe import ALPHAS

OUT = datasets.XLSR_EMA / "analysis_scaling"
KINDS = ("zca", "zca12")
SPK = datasets.ALL_SPEAKERS
CONDS = ("all", "frames")


class Stats:
    """Sufficient statistics of (x, y) frames for several target spaces sharing the same x."""

    def __init__(self, dim, spaces):
        self.n = 0
        self.sx = np.zeros(dim)
        self.sxx = np.zeros((dim, dim))
        self.sy = {t: np.zeros(12) for t in spaces}
        self.sxy = {t: np.zeros((dim, 12)) for t in spaces}
        self.syy = {t: np.zeros(12) for t in spaces}

    def add(self, x, ys):
        self.n += len(x)
        self.sx += x.sum(0)
        self.sxx += x.T @ x
        for t, y in ys.items():
            self.sy[t] += y.sum(0)
            self.sxy[t] += x.T @ y
            self.syy[t] += (y * y).sum(0)

    def copy(self):
        c = Stats.__new__(Stats)
        c.n, c.sx, c.sxx = self.n, self.sx.copy(), self.sxx.copy()
        c.sy = {t: v.copy() for t, v in self.sy.items()}
        c.sxy = {t: v.copy() for t, v in self.sxy.items()}
        c.syy = {t: v.copy() for t, v in self.syy.items()}
        return c

    @staticmethod
    def sum(items):
        out = items[0].copy()
        for s in items[1:]:
            out.n += s.n
            out.sx += s.sx
            out.sxx += s.sxx
            for t in out.sy:
                out.sy[t] += s.sy[t]
                out.sxy[t] += s.sxy[t]
                out.syy[t] += s.syy[t]
        return out


class StatsRidge:
    """probe.Ridge computed from sufficient statistics: standardized features and targets, one eigendecomposition."""

    def __init__(self, st):
        n = st.n
        self.xm = st.sx / n
        Sxx = st.sxx - n * np.outer(self.xm, self.xm)
        self.xs = np.sqrt(np.clip(np.diag(Sxx) / (n - 1), 0, None)).clip(1e-6)
        evals, self.V = np.linalg.eigh(Sxx / np.outer(self.xs, self.xs))
        self.evals = evals.clip(0)
        self.st = st

    def weights(self, space, alphas):
        st, n = self.st, self.st.n
        ym = st.sy[space] / n
        ys = np.sqrt((st.syy[space] - n * ym**2) / (n - 1))
        Sxy = st.sxy[space] - n * np.outer(self.xm, ym)
        VtXtY = self.V.T @ (Sxy / self.xs[:, None] / ys[None, :])
        out = []
        for a in alphas:
            W = (self.V @ (VtXtY / (self.evals + a)[:, None])) / self.xs[:, None] * ys[None, :]
            out.append((W, ym - self.xm @ W))
        return out


def sse_from_stats(st, space, W, b):
    """Per-dimension sum of squared errors of y_hat = x W + b over the frames summarized by st."""
    quad = np.einsum("dc,de,ec->c", W, st.sxx, W)
    return (st.syy[space] - 2 * (W * st.sxy[space]).sum(0) - 2 * b * st.sy[space] + quad
            + 2 * b * (st.sx @ W) + st.n * b**2)


def _pcc_cols(p, y):
    p, y = p - p.mean(0), y - y.mean(0)
    d = np.sqrt((p * p).sum(0) * (y * y).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, (p * y).sum(0) / d, np.nan)


def metrics(P, Y):
    """P, Y: lists of (T, 12) arrays in one space."""
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    mse = ((Pc - Yc) ** 2).mean(0)
    pcc = np.nanmean(np.stack([_pcc_cols(p, y) for p, y in zip(P, Y)]), axis=0)
    return {"rmse": float(np.sqrt(mse).mean()), "pcc": float(np.nanmean(pcc)), "r2": float((1 - mse / Yc.var(0)).mean())}


def shared_map(kind, groups, cov, std):
    """Whitening (z units -> w) from the mean z-correlation matrix of the given normalization groups."""
    C = np.mean([cov[g] / np.outer(std[g], std[g]) for g in groups], 0)
    return datasets.target_matrices(kind, C, np.ones(12))[0]


def run(model, layer, shift, out=None, max_k=6):
    d = datasets.features_root("ema_multi") / model
    index = json.loads((d / "index.json").read_text())
    feats = np.load(d / f"layer_{layer:02d}.npy", mmap_mode="r")
    dim = feats.shape[1]
    spaces = ("z",) + KINDS
    dz = datasets.EMAMulti(held_out=(), norm_by="norm_group")
    std = {g: s for g, (_, s) in dz._stats.items()}
    # (A, A^-1) per group, z -> w: symmetric whitening of the group's z-correlation matrix, the same construction as
    # the subset-level shared map, so a single training group gives exactly the z model (train effect 0 at k = 1)
    tf = {k: {g: datasets.target_matrices(k, dz._cov[g] / np.outer(std[g], std[g]), np.ones(12)) for g in dz._stats}
          for k in KINDS}
    by = {s: {"train": [], "valid": [], "test": []} for s in SPK}
    groups_of = {s: set() for s in SPK}
    for stem, r in sorted(dz.rows.items()):
        if stem in index:
            by[r["speaker"]][r["_split"]].append(stem)
            groups_of[r["speaker"]].add(r["norm_group"])

    def block(stem):
        a, n = index[stem]
        x = np.asarray(feats[a:a + n], dtype=np.float64)
        u = dz.load_targets(stem)
        x, z = align(x, u.ema_mm.astype(np.float64), u.base_offset + shift)
        g = u.norm_key
        return x, {"z": z, **{k: z @ tf[k][g][0].T for k in KINDS}}, g

    rng = np.random.default_rng(0)
    orders = {s: list(rng.permutation(by[s]["train"])) for s in SPK}
    full, items = {}, {}
    for s in SPK:
        st = Stats(dim, spaces)
        for stem in orders[s]:
            x, ys, _ = block(stem)
            if len(x) >= 2:
                st.add(x, ys)
        full[s] = st
        for split in ("valid", "test"):
            lst = []
            for stem in by[s][split]:
                x, ys, g = block(stem)
                if len(x) >= 2:
                    lst.append((x.astype(np.float32), ys["z"], g))
            items[(s, split)] = lst
    F = min(full[s].n for s in SPK)
    budget = {}
    for s in SPK:  # frame-matched snapshots: first utterances until F // k frames
        targets = {k: F // k for k in range(1, 7)}
        st, snaps = Stats(dim, spaces), {}
        for stem in orders[s]:
            x, ys, _ = block(stem)
            if len(x) >= 2:
                st.add(x, ys)
            for k, tgt in targets.items():
                if k not in snaps and st.n >= tgt:
                    snaps[k] = st.copy()
            if len(snaps) == 6:
                break
        budget[s] = snaps
        print(f"{s}: train {len(by[s]['train'])} utts / {full[s].n} frames; frame budget F={F}", flush=True)

    def predict(itms, W, b):
        return [x.astype(np.float64) @ W + b for x, _, _ in itms]

    def to_z(P, itms, train, kind, mapping, Abar):
        """Map predictions of a model trained in `train` ('z' or 'w' of `kind`) to each item's z units."""
        out = []
        for p, (_, _, g) in zip(P, itms):
            if train == "z":
                out.append(p if mapping == "shared" else p @ (tf[kind][g][1] @ Abar).T)
            else:
                out.append(p @ (np.linalg.inv(Abar) if mapping == "shared" else tf[kind][g][1]).T)
        return out

    out_f = (OUT if out is None else out) / f"scaling_{model}_k{layer}_s{shift}.jsonl"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_f.exists():
        done = {(r["cond"], tuple(r["subset"])) for r in map(json.loads, out_f.read_text().splitlines())}
    with out_f.open("a") as fh:
        for k in range(1, max_k + 1):
            for subset in combinations(SPK, k):
                groups = sorted(set().union(*(groups_of[s] for s in subset)))
                Abar = {kind: shared_map(kind, groups, dz._cov, std) for kind in KINDS}
                val = [it for s in subset for it in items[(s, "valid")]]
                for cond in CONDS:
                    if (cond, subset) in done:
                        continue
                    st = Stats.sum([full[s] if cond == "all" else budget[s][k] for s in subset])
                    ridge = StatsRidge(st)
                    rec = {"model": model, "layer": layer, "shift": shift, "cond": cond, "k": k, "subset": list(subset),
                           "n_frames": st.n, "F": F, "fits": {}}
                    for space in spaces:
                        cands = ridge.weights(space, ALPHAS)
                        train = "z" if space == "z" else "w"
                        kind = KINDS[0] if space == "z" else space
                        Yv = [y for _, y, _ in val]
                        vmap = "shared" if train == "z" else "own"  # z: identity; w: each training group's own map
                        vr = [metrics(to_z(predict(val, W, b), val, train, kind, vmap, Abar[kind]), Yv)["rmse"]
                              for W, b in cands]
                        j = int(np.argmin(vr))
                        W, b = cands[j]
                        arms = ([("shared", None)] + [("own", kk) for kk in KINDS]) if space == "z" else [("shared", space), ("own", space)]
                        fit = {"alpha": float(ALPHAS[j]), "alpha_idx": j, "valid_rmse_z": float(vr[j]), "arms": {}}
                        for mapping, kk in arms:
                            name = mapping if kk is None else f"{mapping}_{kk}"
                            res = {}
                            for h in SPK:
                                itms = items[(h, "test")]
                                Pz = to_z(predict(itms, W, b), itms, train, kk or KINDS[0], mapping,
                                          Abar[kk or KINDS[0]])
                                res[h] = {"scope": "seen" if h in subset else "unseen",
                                          **metrics(Pz, [y for _, y, _ in itms])}
                            fit["arms"][name] = res
                        if space != "z":  # whitened-space error of the w/own arm (reference only)
                            fit["own_space"] = {}
                            for h in SPK:
                                itms = items[(h, "test")]
                                P = predict(itms, W, b)
                                Yw = [y @ tf[space][g][0].T for _, y, g in itms]
                                fit["own_space"][h] = metrics(P, Yw)
                        rec["fits"][space] = fit
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
            print(f"{model}: k={k} done", flush=True)


# ---------------------------------------------------------------- summary

ARMS = {  # label -> (fit space, arm name)
    "z": ("z", "shared"),
    "z+recolor[{k}]": ("z", "own_{k}"),
    "w[{k}] shared map": ("{k}", "shared_{k}"),
    "w[{k}] own map": ("{k}", "own_{k}"),
}


def curves(recs, cond, space, arm, metric):
    """{h: {k: mean over subsets of size k excluding h}} for unseen speaker h; seen: per-speaker mean."""
    un = {h: {} for h in SPK}
    seen = {}
    for r in recs:
        if r["cond"] != cond:
            continue
        for h, v in r["fits"][space]["arms"][arm].items():
            if v["scope"] == "unseen":
                un[h].setdefault(r["k"], []).append(v[metric])
            else:
                seen.setdefault(r["k"], []).append(v[metric])
    return ({h: {k: float(np.mean(v)) for k, v in ks.items()} for h, ks in un.items()},
            {k: float(np.mean(v)) for k, v in seen.items()})


def slope(c):
    """Least-squares slope of the metric against log2(k) for one speaker's curve."""
    ks = sorted(c)
    return float(np.polyfit(np.log2(ks), [c[k] for k in ks], 1)[0])


def summary(out=None):
    root = OUT if out is None else out
    L = ["# Speaker scaling and per-speaker whitening (ridge probe, fixed layer)\n",
         "All metrics in the scored speaker's per-channel z units. Unseen: per held-out speaker, mean over all training "
         "subsets of size k that exclude it, then mean over the 7 speakers. 'z' = per-channel z targets; 'w[kind]' = each "
         "training group whitened with its own covariance before pooling; 'shared map' = predictions mapped back with the "
         "subset's average whitening (uses only the new speaker's means/stds), 'own map' / 'recolor' = with the new "
         "speaker's own covariance. Train effect = w - z at the same map; it is exactly 0 at k=1.\n"]
    for f in sorted(root.glob("scaling_*.jsonl")):
        recs = [json.loads(x) for x in f.read_text().splitlines()]
        r0 = recs[0]
        L.append(f"\n## {r0['model']} layer {r0['layer']} (shift {r0['shift']}; frame budget F = {r0['F']})\n")
        edge = sum(1 for r in recs for fit in r["fits"].values() if fit["alpha_idx"] in (0, len(ALPHAS) - 1))
        L.append(f"Ridge strength at a grid edge in {edge} of {sum(len(r['fits']) for r in recs)} fits.\n")
        for cond in CONDS:
            L.append(f"\n### condition: {cond}\n")
            for metric in ("r2", "pcc", "rmse"):
                sign = -1 if metric == "rmse" else 1
                L.append(f"\n{metric} (unseen speakers; slope per doubling of training speakers, + = improves)\n")
                L.append("| arm | " + " | ".join(f"k={k}" for k in range(1, 7)) + " | slope (mean) | speakers with + slope | seen k=6 |")
                L.append("|---|" + "---|" * 9)
                cache = {}
                for kind in KINDS:
                    for lab, (sp, arm) in ARMS.items():
                        lab, sp, arm = lab.format(k=kind), sp.format(k=kind), arm.format(k=kind)
                        if lab in cache:
                            continue
                        un, seen = curves(recs, cond, sp, arm, metric)
                        cache[lab] = un
                        sl = np.array([sign * slope(un[h]) for h in SPK])
                        means = [np.mean([un[h][k] for h in SPK if k in un[h]]) for k in range(1, 7)]
                        L.append(f"| {lab} | " + " | ".join(f"{v:.4f}" for v in means) +
                                 f" | {sl.mean():+.4f} | {(sl > 0).sum()}/7 | {seen.get(6, float('nan')):.4f} |")
                L.append("\nTrain effect (w - z, same map; + = whitening before pooling helps), per k, then per speaker at k=6:\n")
                L.append("| contrast | " + " | ".join(f"k={k}" for k in range(1, 7)) + " | speakers better at k=6 | per speaker at k=6 |")
                L.append("|---|" + "---|" * 8)
                for kind in KINDS:
                    for name, a, b in ((f"w[{kind}] - z (shared map)", f"w[{kind}] shared map", "z"),
                                       (f"w[{kind}] - z+recolor (own map)", f"w[{kind}] own map", f"z+recolor[{kind}]"),
                                       (f"z+recolor[{kind}] - z (test-time recoloring only)", f"z+recolor[{kind}]", "z")):
                        A, B = cache[a], cache[b]
                        dk = [np.mean([sign * (A[h][k] - B[h][k]) for h in SPK if k in A[h]]) for k in range(1, 7)]
                        d6 = {h: sign * (A[h][6] - B[h][6]) for h in SPK}
                        L.append(f"| {name} | " + " | ".join(f"{v:+.4f}" for v in dk) +
                                 f" | {sum(v > 0 for v in d6.values())}/7 | " +
                                 ", ".join(f"{h} {v:+.3f}" for h, v in d6.items()) + " |")
        # per corpus, k = 6, all data
        L.append("\n### Per corpus at k=6 (all data, unseen r2)\n")
        L.append("| arm | USC-TIMIT held out | EMA_5EMO held out |")
        L.append("|---|---|---|")
        for kind in KINDS:
            for lab, (sp, arm) in ARMS.items():
                lab, sp, arm = lab.format(k=kind), sp.format(k=kind), arm.format(k=kind)
                if kind == KINDS[1] and lab == "z":
                    continue
                un, _ = curves(recs, "all", sp, arm, "r2")
                usc = np.mean([un[h][6] for h in SPK if h.startswith("usc")])
                emo = np.mean([un[h][6] for h in SPK if h.startswith("5emo")])
                L.append(f"| {lab} | {usc:.4f} | {emo:.4f} |")
    (root / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary"))
    ap.add_argument("--model")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--shift", type=int)
    ap.add_argument("--max-k", type=int, default=6, help="largest training-subset size (smaller for a smoke test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.model, args.layer, args.shift, None if args.out is None else Path(args.out), args.max_k)
    else:
        summary()


if __name__ == "__main__":
    main()
