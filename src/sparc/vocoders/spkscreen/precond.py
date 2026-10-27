"""Precondition checks of the speaker-encoder screen: paired speaker-level inference from saved embeddings.

Usage: ``python -m sparc.vocoders.spkscreen.precond --out <dir>`` writes ``precond.json`` (all numbers) there.

Nothing is refitted differently from the harness (``metrics.evaluate``): per run the ridge probe is fitted once on the
non-test held-out rows and kept fixed; every statistic is then recomputed on speaker subsets. For a pair of variants
the per-run statistic (or its leave-one-speaker-out replicate) is differenced seed by seed (same seed paired),
averaged over the seeds, and the leave-one-speaker-out jackknife SE of that seed-averaged difference is reported
(pooled statistics: M1, 1 s / 2 s M1, EER, M3, F0 PCC). 5-shot identification is a mean over speakers, so its SE is
SD / sqrt(n). The number of speakers improved uses per-speaker M1 (mean over channels of the per-speaker RMSE), per-
speaker 5-shot accuracy and per-speaker EER (the speaker's utterances against all others).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

from sparc.vocoders.spkscreen import cache as C
from sparc.vocoders.spkscreen import metrics as M

SEEDS = (0, 1, 2)
# metric -> sign such that sign * (baseline - variant) > 0 means the variant is better
SIGN = {"m1": 1, "m1_1s": 1, "m1_2s": 1, "eer": 1, "id": -1, "m3": 1, "f0": -1}


def rmse_mean(E, keep):
    return float(np.sqrt((E[keep] ** 2).mean(0)).mean())


class Pairs:
    """Cosine scores of all utterance pairs once, sorted, for EER of speaker subsets without re-sorting."""

    def __init__(self, Xn, sp):
        self.si = np.unique(sp, return_inverse=True)[1]
        i, j = np.triu_indices(len(sp), 1)
        s = (Xn @ Xn.T)[i, j]
        o = np.argsort(-s, kind="stable")
        self.i, self.j = i[o], j[o]
        self.same = (self.si[i] == self.si[j])[o]

    def eer(self, drop=None, only=None):
        m = np.ones(len(self.i), bool)
        if drop is not None:
            m &= (self.si[self.i] != drop) & (self.si[self.j] != drop)
        if only is not None:
            m &= (self.si[self.i] == only) | (self.si[self.j] == only)
        lab = self.same[m]
        if lab.sum() == 0 or (~lab).sum() == 0:
            return np.nan
        tp = np.cumsum(lab) / lab.sum()
        fp = np.cumsum(~lab) / (~lab).sum()
        k = int(np.argmin(np.abs((1 - tp) - fp)))
        return float((1 - tp[k] + fp[k]) / 2)


def heldout_stats(X, X1, X2, H, rows=None):
    """Full value, leave-one-speaker-out replicates and per-speaker values of every metric for one run.

    ``H`` holds the Phase 1 arrays; ``rows`` optionally restricts the test rows to a corpus (the ridge probes stay
    fitted on all non-test rows, as in the harness).
    """
    Y, logf0, spk, is_te = H["targets"].astype(np.float64), H["logf0"], H["spk"], H["is_te"]
    te = np.where(is_te)[0]
    ysd = Y[~is_te].std(0) + 1e-9
    partner, order = M.partners(spk)
    f = M.ridge_fit(X[~is_te], Y[partner[~is_te]])
    tgt = Y[partner[te]]
    E = (f(X[te]) - tgt) / ysd
    E1 = (f(X1[te]) - tgt) / ysd
    E2 = (f(X2[te]) - tgt) / ysd
    Xn = X[te] / (np.linalg.norm(X[te], axis=1, keepdims=True) + 1e-9)
    sp = spk[te]
    sub = np.ones(len(te), bool) if rows is None else rows[te]
    full_domain = rows is None
    E, E1, E2, Xn, sp = E[sub], E1[sub], E2[sub], Xn[sub], sp[sub]
    names = sorted(set(sp))
    si = np.array([names.index(s) for s in sp])
    S = len(names)
    out = {"speakers": names}
    for tag, EE in (("m1", E), ("m1_1s", E1), ("m1_2s", E2)):
        out[tag] = (rmse_mean(EE, np.ones(len(EE), bool)), np.array([rmse_mean(EE, si != s) for s in range(S)]),
                    np.array([np.sqrt((EE[si == s] ** 2).mean(0)).mean() for s in range(S)]))
    pr = Pairs(Xn, sp)
    out["eer"] = (pr.eer(), np.array([pr.eer(drop=s) for s in range(S)]), np.array([pr.eer(only=s) for s in range(S)]))
    if full_domain:
        # 5-shot identification, as the harness (centroid of the first 5 utterances of every test speaker)
        cents = np.stack([Xn[np.where(si == s)[0][:5]].mean(0) for s in range(S)])
        acc = np.full(S, np.nan)
        for s in range(S):
            rest = np.where(si == s)[0][5:]
            if len(rest):
                acc[s] = np.mean(np.argmax(Xn[rest] @ cents.T, 1) == s)
        out["id"] = (float(np.nanmean(acc)), None, acc)
        # M3
        spk_mean_X = {s: X[ids].mean(0) for s, ids in order.items()}
        spk_mean_Y = {s: Y[ids].mean(0) for s, ids in order.items()}
        Xd = X - np.stack([spk_mean_X[s] for s in spk])
        Yd = Y - np.stack([spk_mean_Y[s] for s in spk])
        g = M.ridge_fit(Xd[~is_te], Yd[~is_te])
        Pd, Ydt = g(Xd[is_te]), Yd[is_te]
        ysd_d = Yd[~is_te].std(0) + 1e-9

        def m3(keep):
            return M.score(Pd[keep], Ydt[keep], ysd_d)[1]

        s_te = spk[is_te]
        mk = lambda s: s_te != names[s]  # noqa: E731
        out["m3"] = (m3(np.ones(len(Pd), bool)), np.array([m3(mk(s)) for s in range(S)]), None)
        # F0 PCC
        ok = np.isfinite(logf0)
        h = M.ridge_fit(X[~is_te & ok], logf0[~is_te & ok, None])
        ph, th, sf = h(X[is_te & ok])[:, 0], logf0[is_te & ok], spk[is_te & ok]
        out["f0"] = (float(np.corrcoef(ph, th)[0, 1]),
                     np.array([np.corrcoef(ph[sf != s], th[sf != s])[0, 1] for s in names]), None)
    return out


def globe_stats(X, X1, X2, H, G):
    """M1 (full, 1 s, 2 s) and EER on the GLOBE set with the ridge probe fitted on the held-out non-test rows."""
    Y, spk, is_te = H["targets"].astype(np.float64), H["spk"], H["is_te"]
    ysd = Y[~is_te].std(0) + 1e-9
    partner, _ = M.partners(spk)
    f = M.ridge_fit(X[~is_te], Y[partner[~is_te]])
    gp, _ = M.partners(G["spk"])
    tgt = G["targets"].astype(np.float64)[gp]
    names = sorted(set(G["spk"]))
    si = np.array([names.index(s) for s in G["spk"]])
    S = len(names)
    out = {"speakers": names}
    for tag, key in (("m1", "full"), ("m1_1s", "s1"), ("m1_2s", "s2")):
        EE = (f(G[key]) - tgt) / ysd
        out[tag] = (rmse_mean(EE, np.ones(len(EE), bool)), np.array([rmse_mean(EE, si != s) for s in range(S)]),
                    np.array([np.sqrt((EE[si == s] ** 2).mean(0)).mean() for s in range(S)]))
    Xn = G["full"] / (np.linalg.norm(G["full"], axis=1, keepdims=True) + 1e-9)
    pr = Pairs(Xn, G["spk"])
    out["eer"] = (pr.eer(), np.array([pr.eer(drop=s) for s in range(S)]), np.array([pr.eer(only=s) for s in range(S)]))
    return out


def paired(A, B, metric):
    """Seed-paired difference ``sign * (A - B)`` (positive = B better than A). ``A``, ``B``: lists over seeds of stats."""
    sg = SIGN[metric]
    full = np.mean([sg * (a[metric][0] - b[metric][0]) for a, b in zip(A, B)])
    res = {"d": float(full)}
    j = A[0][metric][1]
    if j is not None:
        J = np.mean([sg * (a[metric][1] - b[metric][1]) for a, b in zip(A, B)], 0)
        n = len(J)
        se = float(np.sqrt((n - 1) / n * ((J - J.mean()) ** 2).sum()))
        per = np.mean([sg * (a[metric][2] - b[metric][2]) for a, b in zip(A, B)], 0) if A[0][metric][2] is not None else None
    else:  # per-speaker mean (5-shot identification)
        per = np.mean([sg * (a[metric][2] - b[metric][2]) for a, b in zip(A, B)], 0)
        v = per[np.isfinite(per)]
        n = len(v)
        se = float(v.std(ddof=1) / np.sqrt(n))
    res["se"] = se
    t = sps.t.ppf(0.975, n - 1)
    res["lo"], res["hi"] = res["d"] - t * se, res["d"] + t * se
    res["p"] = float(2 * sps.t.sf(abs(res["d"] / se), n - 1)) if se > 0 else float("nan")
    if per is not None:
        v = per[np.isfinite(per)]
        res["improved"], res["n"] = int((v > 0).sum()), int(len(v))
    return res


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, k in enumerate(order):
        run = max(run, (len(ps) - r) * ps[k])
        adj[k] = min(1.0, run)
    return adj


def load_heldout(root: Path):
    import pandas as pd

    man = pd.read_parquet(root / "heldout" / "manifest.parquet")
    p1 = np.load(root / "heldout" / "phase1.npz")
    return {"targets": p1["targets"], "logf0": p1["logf0"], "spk": man["speaker"].values.astype(str),
            "is_te": man["is_test"].values.astype(bool)}


def load_globe(root: Path):
    import pandas as pd

    man = pd.read_parquet(root / "globe" / "manifest.parquet")
    d = C.chunk_dir
    n = C.n_chunks(len(man))
    tg = np.concatenate([np.load(d(root, "globe", c) / "targets.npy") for c in range(n)])
    return {"spk": man["speaker"].values.astype(str), "targets": tg}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=C.ROOT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-globe", action="store_true")
    a = p.parse_args()
    root = a.root
    H = load_heldout(root)
    corp = np.array([s.split(":")[0].replace("libri_dev", "libri").replace("libri_test", "libri") for s in H["spk"]])
    domains = {"all": None, "libri": corp == "libri", "vctk": corp == "vctk"}
    variants = ["baseline", "seg_vus3", "seg_vuv2", "ms_l1_l6", "pool_trimmed", "seg_mel_on_vuv2", "cap_rep3", "cap_rep2"]
    have_globe = (not a.no_globe) and (root / "runs" / "baseline" / "seed0" / "globe_embeddings.npz").exists()
    G = load_globe(root) if have_globe else None
    S = {}  # domain -> variant -> list of stats
    for v in variants:
        for s in SEEDS:
            z = np.load(root / "runs" / v / f"seed{s}" / "embeddings.npz")
            X, X1, X2 = (z[k].astype(np.float64) for k in ("full", "s1", "s2"))
            for dn, rows in domains.items():
                S.setdefault(dn, {}).setdefault(v, []).append(heldout_stats(X, X1, X2, H, rows))
            if have_globe:
                zg = np.load(root / "runs" / v / f"seed{s}" / "globe_embeddings.npz")
                S.setdefault("globe", {}).setdefault(v, []).append(
                    globe_stats(X, X1, X2, H, {**G, **{k: zg[k].astype(np.float64) for k in ("full", "s1", "s2")}}))
        print(v, "done", flush=True)
    result = {"abs": {}, "diff": {}}
    for dn, per_v in S.items():
        for v, runs in per_v.items():
            result["abs"].setdefault(dn, {})[v] = {
                m: [float(r[m][0]) for r in runs] for m in runs[0] if m != "speakers"}
        result["abs"][dn]["_n_speakers"] = len(per_v[variants[0]][0]["speakers"])
    comps = [("baseline", v) for v in variants[1:]] + [("cap_rep3", "seg_vus3"), ("cap_rep2", "ms_l1_l6"),
                                                         ("cap_rep2", "seg_vuv2"), ("seg_vuv2", "seg_vus3")]
    for dn, per_v in S.items():
        for a_, b_ in comps:
            result["diff"].setdefault(dn, {})[f"{a_}|{b_}"] = {
                m: paired(per_v[a_], per_v[b_], m) for m in SIGN if m in per_v[a_][0]}
    # Holm across the three shortlisted variants, per metric, for the full held-out set
    short = ["seg_vus3", "seg_vuv2", "ms_l1_l6"]
    for m in SIGN:
        ps = [result["diff"]["all"][f"baseline|{v}"][m]["p"] for v in short]
        for v, pa in zip(short, holm(ps)):
            result["diff"]["all"][f"baseline|{v}"][m]["p_holm"] = float(pa)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "precond.json").write_text(json.dumps(result, indent=1))
    print("written", a.out / "precond.json")


if __name__ == "__main__":
    main()
