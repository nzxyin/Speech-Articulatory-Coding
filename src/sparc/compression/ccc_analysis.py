"""Concordance correlation (CCC) as the main AAI metric: Q1-Q6 of docs/ccc_metric_plan.md from cached LOSO predictions.

Every arm is re-scored from the cached held-out predictions that `rigor.py analyze` / `tier2` use (no model runs). Outputs:
/data/user_data/xoy/xlsr_ema/analysis_ccc/{summary.md,results.json}. All numbers: 7 leave-one-speaker-out folds, test_unseen
sentences, z units of the held-out speaker's training statistics, mean over seeds within a fold; every configuration was
already seen on these folds ("selected on the test folds").

    python -m sparc.compression.ccc_analysis [--boot 1000] [--skip-q6]
"""

import argparse
import itertools
import json
import re
from collections import defaultdict

import numpy as np
from scipy import stats as sstats

from . import datasets, rigor
from .ccc import apply_gain, components, ccc_spk_centered, log_shares, metrics_all
from .rigor import SPK, SEEDS, USC

OUT = datasets.XLSR_EMA / "analysis_ccc"
SIGN = {"rmse_z": -1, "rmse_centered": -1, "pcc": 1, "pcc_concat": 1, "ccc_utt": 1, "ccc_utt_1s": 1, "ccc_utt_3s": 1, "ccc_spk": 1,
        "ccc_spk_centered": 1}
Q2M = ("rmse_z", "pcc", "ccc_utt", "ccc_spk")  # metrics with bootstrap intervals
EXTRA = ("rmse_centered", "ccc_utt_1s", "ccc_utt_3s", "ccc_spk_centered")  # paired fold tests only
GRID = np.round(np.arange(0.5, 1.5 + 1e-9, 0.02), 4)
Q1M = ("rmse_z", "rmse_centered", "pcc", "pcc_concat", "ccc_utt", "ccc_spk", "ccc_spk_centered")
NAMES = {"rmse_z": "rmse_z", "rmse_centered": "centered rmse_z", "pcc": "PCC (per utterance)",
         "pcc_concat": "PCC (concatenated)", "ccc_utt": "concordance corr. (per utterance)",
         "ccc_spk": "concordance corr. (per speaker)", "ccc_spk_centered": "concordance corr. (per speaker, centered)",
         "ccc_utt_1s": "concordance corr. (per utterance, >= 1 s)"}


# ---------------------------------------------------------------- arms as sufficient statistics

def prep(st):
    """Add the per-utterance CCC matrix (used by the bootstrap)."""
    st["ccc_mat"] = components(st["n"], st["psum"], st["psq"], st["ysum"], st["ysq"], st["pysum"])["ccc"]
    return st


def to_stats(arm):
    """{fold: [(stems, P, Y)]} -> {fold: [(stems, stats)]}"""
    return {h: [(list(s), prep(rigor.utt_stats(P, Y))) for s, P, Y in runs] for h, runs in arm.items()}


def build_arms():
    """The arms of rigor.analyze() (same loaders, same names) plus the matched-pipeline LoRA of tier2."""
    arms, notes = {}, []
    W, X3, X1, V2 = "wavlm-large", "xlsr-300m", "xlsr-1b", "w2v2-large"
    raw = {}
    raw["W12 probe"] = rigor.load_arm("probe", W, 12, None)
    for hn, lab in (("conv_causal9", "causal conv"), ("mamba", "BiMamba"), ("gruconv9", "BiGRU->conv"), ("mlp", "MLP")):
        raw[f"W12 frozen {lab}"] = rigor.load_arm("frozen", W, 12, hn)
    lora_specs = [("W12", W, 12, rigor.CC, "k12_cc", "causal conv"), ("W12", W, 12, rigor.MB, "k12_mamba", "BiMamba"),
                  ("W12", W, 12, rigor.GC, "k12_gruconv", "BiGRU->conv"), ("W7", W, 7, rigor.CC, None, "causal conv"),
                  ("X18", X3, 18, rigor.CC, None, "causal conv"), ("B10", X1, 10, rigor.CC, None, "causal conv"),
                  ("V18", V2, 18, rigor.CC, None, "causal conv")]
    for short, model, k, tmpl, tag, lab in lora_specs:
        lora = rigor.load_arm("lora", model, k, tmpl)
        if not lora:
            continue
        raw[f"{short} LoRA {lab}"] = lora
        b, _ = rigor.blend_arm(model, k, tmpl, tag)
        if b:
            raw[f"{short} blend {lab}"] = b
        xs, xc, orc, _ = rigor.shrink_arms(lora)
        raw[f"{short} LoRA {lab} x gain (cross-fold)"] = xs
        raw[f"{short} LoRA {lab} x per-channel gain (cross-fold)"] = xc
        raw[f"{short} LoRA {lab} x gain (ORACLE)"] = orc
        if short != "W12":
            raw.setdefault(f"{short} probe", rigor.load_arm("probe", model, k, None))
    raw["X18 frozen BiGRU->conv"] = rigor.load_arm("frozen", X3, 18, "gruconv9")
    raw["matched LoRA BiGRU->conv"] = rigor.load_arm("matched", W, 12, None)
    for a, v in raw.items():
        if v:
            arms[a] = to_stats(v)
    raw.clear()
    return arms, lora_specs


def is_derived(a):
    return "gain" in a or "ORACLE" in a


# ---------------------------------------------------------------- fold-level values and contrasts

def fold_metrics(arms):
    """{arm: {fold: [metrics_all dict per seed]}}"""
    return {a: {h: [metrics_all(st) for _, st in runs] for h, runs in arm.items()} for a, arm in arms.items()}


class Tab:
    def __init__(self, arms):
        self.arms = arms
        self.M = fold_metrics(arms)

    def fv(self, a, m):
        return {h: float(np.mean([r[m] for r in runs])) for h, runs in self.M[a].items()}

    def mean(self, a, m):
        v = self.fv(a, m)
        return float(np.mean(list(v.values()))) if v else np.nan

    def seeds(self, a, m):
        """per seed index: mean over folds"""
        out = []
        for si in range(max(len(r) for r in self.M[a].values())):
            v = [r[si][m] for r in self.M[a].values() if len(r) > si]
            if len(v) == len(self.M[a]):
                out.append(float(np.mean(v)))
        return out


def core4(st, idx):
    n = st["n"][idx]
    N = n.sum()
    rm = np.sqrt(st["sse"][idx].sum(0) / N).mean()
    c = components(N, st["psum"][idx].sum(0), st["psq"][idx].sum(0), st["ysum"][idx].sum(0), st["ysq"][idx].sum(0),
                   st["pysum"][idx].sum(0))
    return np.array([rm, np.nanmean(st["pcc"][idx]), np.nanmean(st["ccc_mat"][idx]), np.nanmean(c["ccc"])])


def boot_pair(A, B, nb=1000, seed=0):
    """Hierarchical bootstrap CI (speakers, seeds within speaker, utterances; shared utterance draw across arms and
    seeds) of the paired fold-mean difference a - b, sign applied so that + = a better, for rmse_z, PCC, CCC_utt, CCC_spk."""
    rng = np.random.default_rng(seed)
    folds = [h for h in SPK if h in A and h in B]
    pre = {}
    for h in folds:
        common = sorted(set.intersection(*[set(r[0]) for r in A[h] + B[h]]))
        pre[h] = (len(common), [np.array([{s: i for i, s in enumerate(r[0])}[c] for c in common]) for r in A[h]],
                  [np.array([{s: i for i, s in enumerate(r[0])}[c] for c in common]) for r in B[h]])
    sg = np.array([SIGN[m] for m in Q2M], dtype=float)
    out = []
    for _ in range(nb):
        hs = rng.integers(0, len(folds), len(folds))
        d = []
        for fi in hs:
            h = folds[fi]
            nc, ia, ib = pre[h]
            u = rng.integers(0, nc, nc)
            va = np.mean([core4(A[h][s][1], ia[s][u]) for s in rng.integers(0, len(ia), len(ia))], 0)
            vb = np.mean([core4(B[h][s][1], ib[s][u]) for s in rng.integers(0, len(ib), len(ib))], 0)
            d.append(va - vb)
        out.append(np.mean(d, 0) * sg)
    out = np.array(out)
    return {m: [float(x) for x in np.percentile(out[:, i], [2.5, 97.5])] for i, m in enumerate(Q2M)}


def contrast(T, a, b, m, boot=None):
    fa, fb = T.fv(a, m), T.fv(b, m)
    folds = [h for h in SPK if h in fa and h in fb]
    s = SIGN.get(m, 1)
    d = np.array([s * (fa[h] - fb[h]) for h in folds])
    n = len(d)
    sd = d.std(ddof=1)
    tq = sstats.t.ppf(0.975, n - 1)
    mde = (tq + sstats.t.ppf(0.8, n - 1)) * sd / np.sqrt(n)
    sa, sb = [T.seeds(a, m), T.seeds(b, m)]
    k = min(len(sa), len(sb))
    sdiff = [s * (x - y) for x, y in zip(sa[:k], sb[:k])]
    r = {"n_folds": n, "mean": float(d.mean()), "t_ci": [float(d.mean() - tq * sd / np.sqrt(n)), float(d.mean() + tq * sd / np.sqrt(n))],
         "won": int((d > 0).sum()), "sign_flip_p": rigor.sign_flip_p(d), "fold_sd": float(sd),
         "seed_sd": float(np.std(sdiff, ddof=1)) if k > 1 else np.nan, "mde80": float(mde),
         "std_effect": float(abs(d.mean()) / sd) if sd > 0 else np.nan, "folds": {h: float(x) for h, x in zip(folds, d)}}
    if boot and m in boot:
        r["boot_ci"] = boot[m]
    return r


def contrast_set(T, a, b, boot_B, cache):
    """All metrics for one pair; bootstrap for the four Q2 metrics."""
    key = (a, b)
    if key not in cache:
        bt = boot_pair(T.arms[a], T.arms[b], boot_B) if boot_B else None
        res = {m: contrast(T, a, b, m, bt) for m in Q2M + EXTRA}
        # location/scale/correlation factor differences (fold level; descriptive)
        for m in ("utt_corr", "utt_scale_w", "utt_location_w"):
            res[m] = contrast(T, a, b, m)
        cb = {}
        for arm, nm in ((a, "a"), (b, "b")):
            cb[nm] = {h: float(np.mean([r["ccc_utt"] / r["utt_corr"] for r in runs])) for h, runs in T.M[arm].items()}
        folds = [h for h in SPK if h in cb["a"] and h in cb["b"]]
        d = np.array([cb["a"][h] - cb["b"][h] for h in folds])
        tq = sstats.t.ppf(0.975, len(d) - 1)
        res["cb"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                     "t_ci": [float(d.mean() - tq * d.std(ddof=1) / np.sqrt(len(d))), float(d.mean() + tq * d.std(ddof=1) / np.sqrt(len(d)))]}
        cache[key] = res
    return cache[key]


def arm_decomp(T, a, kind):
    """Arm-level (A, B, C) with A B C = CCC exactly, from fold values (ratio of fold means)."""
    if kind == "utt":
        A = T.fv(a, "utt_corr"); AB = {h: T.fv(a, "utt_corr")[h] * T.fv(a, "utt_scale_w")[h] for h in A}; ABC = T.fv(a, "ccc_utt")
    else:
        A = T.fv(a, "spk_corr"); AB = {h: T.fv(a, "spk_corr")[h] * T.fv(a, "spk_scale_w")[h] for h in A}; ABC = T.fv(a, "ccc_spk")
    mA, mAB, mABC = [float(np.mean(list(x.values()))) for x in (A, AB, ABC)]
    return {"corr": mA, "scale_w": mAB / mA, "location_w": mABC / mAB, "ccc": mABC}


# ---------------------------------------------------------------- Q1

def kt(x, y):
    return float(sstats.kendalltau(x, y)[0])


def q1(T, names, label, nb=1000, seed=0):
    names = [a for a in names if len(T.M[a]) == len(SPK)]
    F = {m: np.array([[T.fv(a, m)[h] for h in SPK] for a in names]) for m in Q1M}  # arm x fold
    sg = {m: SIGN[m] for m in Q1M}
    rng = np.random.default_rng(seed)
    res = {"n_arms": len(names), "arms": names, "means": {m: F[m].mean(1).tolist() for m in Q1M}, "tau": {}}
    pairs = [("ccc_utt", "rmse_z"), ("ccc_utt", "rmse_centered"), ("ccc_utt", "pcc"), ("ccc_utt", "pcc_concat"), ("ccc_utt", "ccc_spk"),
             ("ccc_spk", "rmse_z"), ("ccc_spk", "rmse_centered"), ("ccc_spk", "pcc_concat"), ("ccc_utt", "ccc_utt_1s"),
             ("ccc_spk", "ccc_spk_centered"), ("pcc", "rmse_z"), ("rmse_z", "rmse_centered")]
    F["ccc_utt_1s"] = np.array([[T.fv(a, "ccc_utt_1s")[h] for h in SPK] for a in names])
    sg["ccc_utt_1s"] = 1
    groups = {"all": list(range(len(names))),
              "probes": [i for i, a in enumerate(names) if "probe" in a],
              "frozen heads": [i for i, a in enumerate(names) if "frozen" in a],
              "LoRA": [i for i, a in enumerate(names) if ("LoRA" in a and not is_derived(a) and "blend" not in a)],
              "WavLM k12 (W12)": [i for i, a in enumerate(names) if a.startswith("W12")]}
    for gname, idx in groups.items():
        if len(idx) < 3:
            continue
        for m1, m2 in pairs:
            x, y = sg[m1] * F[m1][idx], sg[m2] * F[m2][idx]
            t0 = kt(x.mean(1), y.mean(1))
            bs = []
            for _ in range(nb):
                fi = rng.integers(0, len(SPK), len(SPK))
                bs.append(kt(x[:, fi].mean(1), y[:, fi].mean(1)))
            res["tau"][f"{gname}|{m1}|{m2}"] = {"tau": t0, "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                                                "n": len(idx)}
    flips = {}
    for m2 in ("pcc", "rmse_z"):
        x, y = F["ccc_utt"].mean(1), sg[m2] * F[m2].mean(1)
        fl = []
        for i, j in itertools.combinations(range(len(names)), 2):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx * dy < 0:
                fl.append({"a": names[i], "b": names[j], "d_ccc_utt": float(dx), f"d_{m2}": float(dy)})
        flips[m2] = {"n_flips": len(fl), "n_pairs": len(names) * (len(names) - 1) // 2,
                     "list": sorted(fl, key=lambda r: -abs(r["d_ccc_utt"]))}
    res["flips"] = flips
    res["label"] = label
    res["arm_sd"] = {m: float(np.std(F[m].mean(1), ddof=1)) for m in Q1M}
    return res


def reading(t):
    return "same ranking" if t >= 0.8 else ("partly" if t >= 0.5 else "different ranking")


# ---------------------------------------------------------------- Q4 gain sweep

def curves(st):
    """metric values over the grid for one (fold, seed) statistics: dict metric -> array(G)."""
    out = {m: np.zeros(len(GRID)) for m in ("rmse_z", "pcc", "ccc_utt", "ccc_spk")}
    idx = slice(None)
    for i, g in enumerate(GRID):
        v = core4(prep(apply_gain(st, g)), idx)
        for m, x in zip(("rmse_z", "pcc", "ccc_utt", "ccc_spk"), v):
            out[m][i] = x
    return out


def gain_tables(arm):
    """Curves per (fold, seed) and the leave-fold-out and oracle gains of every metric."""
    cur = {}
    for h, runs in arm.items():
        for si, (_, st) in enumerate(runs):
            cur[(h, si)] = curves(st)
    lfo, orc = {}, {}
    for (h, si), c in cur.items():
        oth = [k for k in cur if k[0] != h and k[1] == si]
        pys = sum(arm[o][si][1]["pysum"].sum() for o, _ in oth)
        pps = sum(arm[o][si][1]["psq"].sum() for o, _ in oth)
        g = {"rmse_z": float(pys / pps)}
        for m in ("ccc_utt", "ccc_spk"):
            g[m] = float(GRID[int(np.argmax(np.mean([cur[k][m] for k in oth], 0)))])
        lfo[(h, si)] = g
        orc[(h, si)] = {"rmse_z": float(GRID[int(np.argmin(c["rmse_z"]))]), "ccc_utt": float(GRID[int(np.argmax(c["ccc_utt"]))]),
                        "ccc_spk": float(GRID[int(np.argmax(c["ccc_spk"]))])}
    return cur, lfo, orc


def gained(arm, gains, metric):
    """Arm with each (fold, seed) scaled by gains[(fold, seed)][metric] (or a constant)."""
    out = {}
    for h, runs in arm.items():
        out[h] = [(s, prep(apply_gain(st, gains[(h, si)][metric] if isinstance(gains, dict) else gains)))
                  for si, (s, st) in enumerate(runs)]
    return out


def q4(T, lora_specs, boot_B):
    names = [a for a in T.arms if (("LoRA" in a and not is_derived(a)) or a.startswith("matched") or
                                   a.startswith("W12 frozen") or a == "W12 probe")]
    res = {"arms": {}, "grid": GRID.tolist(), "blend": {}}
    G = {}
    for a in names:
        cur, lfo, orc = gain_tables(T.arms[a])
        G[a] = (cur, lfo, orc)
        mean_curve = {m: np.mean([c[m] for c in cur.values()], 0) for m in ("rmse_z", "pcc", "ccc_utt", "ccc_spk")}
        i1 = int(np.argmin(np.abs(GRID - 1.0)))
        pooled = {"rmse_z": float(np.mean([T.arms[a][h][si][1]["pysum"].sum() for h in T.arms[a] for si in range(len(T.arms[a][h]))]) /
                                  np.mean([T.arms[a][h][si][1]["psq"].sum() for h in T.arms[a] for si in range(len(T.arms[a][h]))]))}
        opt = {"rmse_z_closed_form_pooled": pooled["rmse_z"], "rmse_z_grid": float(GRID[int(np.argmin(mean_curve["rmse_z"]))]),
               "ccc_utt": float(GRID[int(np.argmax(mean_curve["ccc_utt"]))]), "ccc_spk": float(GRID[int(np.argmax(mean_curve["ccc_spk"]))]),
               "pcc_range": float(np.ptp(mean_curve["pcc"]))}
        g_r = opt["rmse_z_grid"]
        ir = int(np.argmin(np.abs(GRID - g_r)))
        iu, isp = int(np.argmin(np.abs(GRID - opt["ccc_utt"]))), int(np.argmin(np.abs(GRID - opt["ccc_spk"])))
        loss = {"ccc_utt_loss_at_rmse_gain": float(mean_curve["ccc_utt"][iu] - mean_curve["ccc_utt"][ir]),
                "ccc_spk_loss_at_rmse_gain": float(mean_curve["ccc_spk"][isp] - mean_curve["ccc_spk"][ir]),
                "rmse_loss_at_ccc_utt_gain": float(mean_curve["rmse_z"][iu] - mean_curve["rmse_z"][ir]),
                "rmse_loss_at_ccc_spk_gain": float(mean_curve["rmse_z"][isp] - mean_curve["rmse_z"][ir])}
        at1 = {m: float(mean_curve[m][i1]) for m in mean_curve}
        # leave-fold-out gains and what the held-out folds score at them
        lf = {"gains": {m: [float(np.mean([g[m] for g in lfo.values()])), float(np.min([g[m] for g in lfo.values()])),
                            float(np.max([g[m] for g in lfo.values()]))] for m in ("rmse_z", "ccc_utt", "ccc_spk")}}
        for gm in ("rmse_z", "ccc_utt", "ccc_spk"):
            ga = gained(T.arms[a], lfo, gm)
            vals = defaultdict(list)
            for h, runs in ga.items():
                for si, (_, st) in enumerate(runs):
                    for m, x in zip(("rmse_z", "pcc", "ccc_utt", "ccc_spk"), core4(st, slice(None))):
                        vals[m].append(x)
            lf[f"scored_at_lfo_gain_{gm}"] = {m: float(np.mean(v)) for m, v in vals.items()}
        orc_best = {m: float(np.mean([max(c[m]) if m != "rmse_z" else min(c[m]) for c in cur.values()])) for m in ("rmse_z", "ccc_utt", "ccc_spk")}
        res["arms"][a] = {"curve": {m: mean_curve[m].tolist() for m in mean_curve}, "optimal_gain_pooled": opt, "at_gain_1": at1,
                          "losses": loss, "lfo": lf, "oracle_best_value": orc_best,
                          "oracle_gain_mean": {m: float(np.mean([g[m] for g in orc.values()])) for m in ("rmse_z", "ccc_utt", "ccc_spk")}}
    # (b), (c): blend against the gain-calibrated LoRA, with each metric's own gain and with a common gain
    cache = {}
    for short, model, k, tmpl, tag, lab in lora_specs:
        L, Bn = f"{short} LoRA {lab}", f"{short} blend {lab}"
        if L not in G or Bn not in T.arms:
            continue
        _, lfo, orc = G[L]
        _, lfo_b, orc_b = (gain_tables(T.arms[Bn]))
        arms = {Bn: T.arms[Bn], L: T.arms[L]}
        for gm in ("rmse_z", "ccc_utt", "ccc_spk"):
            arms[f"LoRA x LFO gain ({gm})"] = gained(T.arms[L], lfo, gm)
            arms[f"LoRA x oracle gain ({gm})"] = gained(T.arms[L], orc, gm)
            arms[f"blend x oracle gain ({gm})"] = gained(T.arms[Bn], orc_b, gm)
        T2 = Tab(arms)
        out = {"own_gain": {}, "common_gain_rmse_z": {}, "oracle_single_gain": {}, "blend_at_own_oracle_gain": {}, "raw": {}}
        for gm in ("rmse_z", "ccc_utt", "ccc_spk"):
            out["own_gain"][gm] = contrast(T2, Bn, f"LoRA x LFO gain ({gm})", gm)
            out["oracle_single_gain"][gm] = contrast(T2, Bn, f"LoRA x oracle gain ({gm})", gm)
            out["blend_at_own_oracle_gain"][gm] = contrast(T2, f"blend x oracle gain ({gm})", f"LoRA x oracle gain ({gm})", gm)
        for m in ("rmse_z", "pcc", "ccc_utt", "ccc_spk"):
            out["common_gain_rmse_z"][m] = contrast(T2, Bn, "LoRA x LFO gain (rmse_z)", m)
            out["raw"][m] = contrast(T2, Bn, L, m)
        out["gains_lora_lfo"] = {gm: float(np.mean([g[gm] for g in lfo.values()])) for gm in ("rmse_z", "ccc_utt", "ccc_spk")}
        out["gains_blend_oracle"] = {gm: float(np.mean([g[gm] for g in orc_b.values()])) for gm in ("rmse_z", "ccc_utt", "ccc_spk")}
        res["blend"][f"{short} {lab}"] = out
    return res


# ---------------------------------------------------------------- Q2 / Q3 / Q5 contrasts

def contrast_list(T):
    A = T.arms
    c = []

    def add(cid, a, b, note=""):
        if a in A and b in A:
            c.append((cid, a, b, note))

    add("P1", "W12 blend BiGRU->conv", "W12 LoRA BiGRU->conv x gain (cross-fold)", "pre-registered; blend vs shrunk LoRA")
    add("P2", "W12 frozen BiGRU->conv", "W12 probe", "pre-registered; sequence head vs ridge probe")
    add("P3", "W12 frozen BiGRU->conv", "X18 frozen BiGRU->conv", "pre-registered; WavLM k12 vs XLS-R 300M k18")
    add("T1", "matched LoRA BiGRU->conv", "W12 frozen BiGRU->conv", "pre-registered; matched LoRA vs frozen")
    add("T2", "matched LoRA BiGRU->conv", "W12 LoRA BiGRU->conv", "pre-registered; matched vs original LoRA pipeline")
    for e, nm in (("X18", "XLS-R 300M k18"), ("B10", "XLS-R 1B k10"), ("V18", "wav2vec2-large k18")):
        add(f"enc-probe-{e}", "W12 probe", f"{e} probe", f"encoder, probe: WavLM k12 vs {nm}")
        add(f"enc-lora-{e}", "W12 LoRA causal conv", f"{e} LoRA causal conv", f"encoder, LoRA causal conv: WavLM k12 vs {nm}")
    for lab in ("causal conv", "BiMamba", "BiGRU->conv"):
        add(f"head-{lab}-vs-MLP", f"W12 frozen {lab}", "W12 frozen MLP", f"head: frozen {lab} vs MLP")
        add(f"head-{lab}-vs-probe", f"W12 frozen {lab}", "W12 probe", f"head: frozen {lab} vs ridge probe")
        add(f"lora-vs-frozen-{lab}", f"W12 LoRA {lab}", f"W12 frozen {lab}", f"LoRA vs frozen {lab} (original pipeline)")
    add("head-MLP-vs-probe", "W12 frozen MLP", "W12 probe", "head: MLP vs ridge probe")
    add("head-BiMamba-vs-BiGRUconv", "W12 frozen BiMamba", "W12 frozen BiGRU->conv", "head: BiMamba vs BiGRU->conv")
    add("head-BiMamba-vs-causalconv", "W12 frozen BiMamba", "W12 frozen causal conv", "head: BiMamba vs causal conv")
    add("head-BiGRUconv-vs-causalconv", "W12 frozen BiGRU->conv", "W12 frozen causal conv", "head: BiGRU->conv vs causal conv")
    for short, lab in (("W12", "causal conv"), ("W12", "BiMamba"), ("W12", "BiGRU->conv"), ("W7", "causal conv"), ("X18", "causal conv"),
                       ("B10", "causal conv"), ("V18", "causal conv")):
        add(f"blend-vs-LoRA-{short}-{lab}", f"{short} blend {lab}", f"{short} LoRA {lab}", f"blend vs LoRA, {short} {lab}")
        add(f"blend-vs-shrunk-{short}-{lab}", f"{short} blend {lab}", f"{short} LoRA {lab} x gain (cross-fold)",
            f"blend vs shrunk LoRA, {short} {lab}")
        add(f"shrunk-vs-LoRA-{short}-{lab}", f"{short} LoRA {lab} x gain (cross-fold)", f"{short} LoRA {lab}",
            f"shrunk LoRA vs LoRA, {short} {lab}")
    add("blend-vs-perchannel-W12-BiGRU", "W12 blend BiGRU->conv", "W12 LoRA BiGRU->conv x per-channel gain (cross-fold)",
        "blend vs per-channel-shrunk LoRA")
    return c


def q5_label(ref, new):
    """Re-reading of a contrast under metric `new` relative to the reference metric `ref` (both contrast dicts):
    significance = exact sign-flip p <= 0.05 over the 7 folds."""
    if not np.isfinite(ref["std_effect"]) or abs(ref["mean"]) < 1e-12:
        return "reference metric cannot see this difference (gain only)"
    rs, ns = ref["sign_flip_p"] <= 0.05, new["sign_flip_p"] <= 0.05
    opp = ref["mean"] * new["mean"] < 0
    if rs and ns:
        if opp:
            return "reversed"
        return "weakened (smaller standardized effect)" if new["std_effect"] < 0.9 * ref["std_effect"] else "unchanged"
    if rs and not ns:
        return "weakened (no longer significant" + (", sign flips)" if opp else ")")
    if ns:
        return "now significant, " + ("opposite sign to the reference" if opp else "same sign as the reference")
    return "unchanged (neither significant" + ("; sign differs)" if opp else ")")


PRE = {"P1": "rmse_z", "P2": "pcc", "P3": "pcc", "T1": "pcc", "T2": "rmse_z"}


def summarize(res):
    """Everything computed from the contrast table: summary counts, rule 1, Q5 labels, sensitivity cuts."""
    q2, spread = res["q2"], res["q1"]["all"]["arm_sd"]
    res["q2_summary"] = {}
    for m in Q2M + ("ccc_utt_1s", "ccc_utt_3s"):
        rows = [q2[c][m] for c in q2]
        sig = [r for r in rows if r["sign_flip_p"] < 0.05 and (("boot_ci" not in r) or r["boot_ci"][0] > 0 or r["boot_ci"][1] < 0)]
        res["q2_summary"][m] = {"n_contrasts": len(rows), "n_sig_p05": int(sum(r["sign_flip_p"] < 0.05 for r in rows)),
                                "n_sig_p05_and_ci": len(sig), "median_std_effect": float(np.nanmedian([r["std_effect"] for r in rows])),
                                "median_mde_in_arm_spread": float(np.median([r["mde80"] / spread[m if not m.startswith("ccc_utt_") else "ccc_utt"] for r in rows])),
                                "median_seed_sd": float(np.nanmedian([r["seed_sd"] for r in rows])),
                                "median_fold_sd": float(np.median([r["fold_sd"] for r in rows]))}
    # rule 1: CCC_utt vs PCC standardized effects (contrasts in which PCC cannot differ by construction are set aside)
    hi = lo = tie = skipped = 0
    for c, r in q2.items():
        x, y = r["ccc_utt"]["std_effect"], r["pcc"]["std_effect"]
        if not np.isfinite(y) or abs(r["pcc"]["mean"]) < 1e-12:
            skipped += 1
        elif abs(x - y) <= 0.1 * abs(y):
            tie += 1
        elif x > y:
            hi += 1
        else:
            lo += 1
    # same comparison restricted to the contrasts with a PCC-detectable difference (p < 0.05 under PCC or CCC utt)
    res["rule1"] = {"higher": hi, "lower": lo, "tie": tie, "skipped_pcc_cannot_differ": skipped,
                    "mde_spread_ccc_utt": res["q2_summary"]["ccc_utt"]["median_mde_in_arm_spread"],
                    "mde_spread_pcc": res["q2_summary"]["pcc"]["median_mde_in_arm_spread"],
                    "mde_spread_rmse_z": res["q2_summary"]["rmse_z"]["median_mde_in_arm_spread"]}
    q5 = {}
    for c, r in q2.items():
        pre = PRE.get(c)
        row = {"reference": pre or "rmse_z and PCC"}
        for new in ("ccc_utt", "ccc_spk"):
            if pre:
                row[new] = q5_label(r[pre], r[new])
            else:
                l1, l2 = q5_label(r["rmse_z"], r[new]), q5_label(r["pcc"], r[new])
                row[new] = l1 if l1 == l2 else f"{l1} [vs rmse_z]; {l2} [vs PCC]"
        row["ccc_utt_1s_same_conclusion"] = bool((np.sign(r["ccc_utt_1s"]["mean"]) == np.sign(r["ccc_utt"]["mean"])) and
                                                 ((r["ccc_utt_1s"]["sign_flip_p"] < 0.05) == (r["ccc_utt"]["sign_flip_p"] < 0.05)))
        row["ccc_utt_3s_same_conclusion_exploratory"] = bool((np.sign(r["ccc_utt_3s"]["mean"]) == np.sign(r["ccc_utt"]["mean"])) and
                                                             ((r["ccc_utt_3s"]["sign_flip_p"] < 0.05) == (r["ccc_utt"]["sign_flip_p"] < 0.05)))
        q5[c] = row
    res["q5"] = q5
    res["rule4_sensitivity_1s_all_same"] = bool(all(v["ccc_utt_1s_same_conclusion"] for v in q5.values()))
    res["rule4_sensitivity_1s_changed"] = [c for c, v in q5.items() if not v["ccc_utt_1s_same_conclusion"]]
    res["sensitivity_3s_exploratory_changed"] = [c for c, v in q5.items() if not v["ccc_utt_3s_same_conclusion_exploratory"]]


# ---------------------------------------------------------------- Q6

def q6(res):
    """Existing evidence only: multi-speaker heads in SPARC, vocoder controls."""
    from .sparc_multihead import OUT as MH

    out = {}
    ship = np.load("/home/xoy/.claude/jobs/62569f48/tmp/ccc/shipped_head.npz")
    Ws, bs = ship["W"], ship["b"]
    z = np.load(MH / "states.npz", allow_pickle=True)
    states = {s: x.astype(np.float64) for s, x in zip(z["stems"], z["states"])}
    heads = json.loads((MH / "heads.json").read_text())
    ev = json.loads((MH / "eval.json").read_text())
    hw = {n: np.load(MH / f"head_{n}.npz") for n in heads}
    # check: the cached per-channel maps reproduce the pickle's output scale on the 'all' head's training states
    S = np.concatenate(list(states.values())) @ Ws + bs
    Z = np.concatenate(list(states.values())) @ hw["all"]["Wsh"] + hw["all"]["bsh"]
    out["head_check_std_ratio"] = [float(x) for x in (Z.std(0) / S.std(0))]
    ds = datasets.EMAMulti(held_out=(), norm_by="norm_group")
    a_all = heads["all"]
    shipped = ("shipped", "loso", "loso_lin", "all", "all_lin", "true_ema", "ship_err", "ship_shrink")
    items = sorted({k.split("|", 1)[1] for k, r in ev.items() if r["src"] == "ema" and r["speaker"].startswith("usc") and r["system"] == "ship_shrink"})
    items = [s for s in items if s in states]
    mu, sd = np.concatenate([states[s] @ Ws + bs for s in items]).mean(0), np.concatenate([states[s] @ Ws + bs for s in items]).std(0)
    outputs = {s: [] for s in shipped}
    spk_of = {}
    for stem in items:
        spk = ev[f"shipped|{stem}"]["speaker"]
        spk_of[stem] = spk
        st = states[stem]
        e_s = st @ Ws + bs
        e_a = st @ hw["all"]["Wlin"] + hw["all"]["blin"]
        o = {"shipped": e_s}
        for nm, key, W_, b_ in (("loso", f"loso_{spk}", "Wsh", "bsh"), ("loso_lin", f"loso_{spk}", "Wlin", "blin"),
                                ("all", "all", "Wsh", "bsh"), ("all_lin", "all", "Wlin", "blin")):
            o[nm] = st @ hw[key][W_] + hw[key][b_]
        e = e_s.copy()
        u = ds.load_targets(stem)
        t0 = u.base_offset + a_all["shift"]
        zz = u.ema_mm.astype(np.float64)
        mapped = zz * np.array(a_all["scale"]) + np.array(a_all["offset"])
        j0 = max(0, -t0)
        t_from, t_to = max(0, t0), min(len(e), t0 + len(zz))
        if t_to > t_from:
            e[t_from:t_to] = mapped[j0:j0 + (t_to - t_from)]
        o["true_ema"] = e
        o["ship_err"] = e_s + np.roll(e_a - e_s, len(e_s) // 2, axis=0)
        r = e_a.std(0) / np.maximum(e_s.std(0), 1e-8)
        o["ship_shrink"] = e_s.mean(0) + (e_s - e_s.mean(0)) * r
        for k_, v in o.items():
            outputs[k_].append(v)
    Y = [(x - mu) / sd for x in outputs["shipped"]]
    agree = {}
    for s in shipped:
        P = [(x - mu) / sd for x in outputs[s]]
        agree[s] = {k: float(v) for k, v in metrics_all(rigor.utt_stats(P, Y)).items()
                    if k in ("rmse_z", "rmse_centered", "pcc", "ccc_utt", "ccc_spk", "ccc_utt_1s", "std_ratio", "utt_scale_w", "utt_location_w", "offset_sd")}
    out["agreement"] = agree
    out["n_items"] = len(items)

    def corpus_wer(s, src):
        rows = [r for k, r in ev.items() if r["system"] == s and r["src"] == src and r["wer"] is not None and
                (src == "libri" or r["speaker"].startswith("usc")) and (src == "libri" or k.split("|", 1)[1] in set(items))]
        nw = sum(r["n_words"] for r in rows)
        return 100 * sum(r["wer"] * r["n_words"] for r in rows) / nw if nw else np.nan

    def mean_of(s, src, key):
        v = [r[key] for k, r in ev.items() if r["system"] == s and r["src"] == src and r.get(key) is not None and
             (src == "libri" or (r["speaker"].startswith("usc") and k.split("|", 1)[1] in set(items)))]
        return float(np.mean(v)) if v else np.nan

    import csv

    def utmos(s, src):
        f = MH / "utmos" / f"{s}_{src}.csv"
        if not f.exists():
            return np.nan
        v = []
        for row in csv.DictReader(open(f)):
            spk, _, stem = row["utt_id"].partition("__")
            if src == "ema" and spk.startswith("usc") and stem in set(items):
                v.append(float(row["mos"]))
            elif src == "libri":
                v.append(float(row["mos"]))
        return float(np.mean(v)) if v else np.nan

    down = {s: {"wer_usc": corpus_wer(s, "ema"), "wer_libri": corpus_wer(s, "libri"), "sim_usc": mean_of(s, "ema", "sim"),
                "sim_libri": mean_of(s, "libri", "sim"), "utmos_usc": utmos(s, "ema"), "utmos_libri": utmos(s, "libri")} for s in shipped}
    out["downstream"] = down
    # Spearman between each metric (goodness) and each downstream measure (goodness)
    metr = {"rmse_z": -1, "rmse_centered": -1, "pcc": 1, "ccc_utt": 1, "ccc_spk": 1}
    meas = {"wer_usc": -1, "wer_libri": -1, "sim_usc": 1, "sim_libri": 1, "utmos_usc": 1, "utmos_libri": 1}
    sp = {}
    for mk, ms in metr.items():
        for dk, ds_ in meas.items():
            sy = [s for s in shipped if np.isfinite(down[s][dk])]
            if len(sy) < 4:
                continue
            x = [ms * agree[s][mk] for s in sy]
            y = [ds_ * down[s][dk] for s in sy]
            sp[f"{mk}|{dk}"] = {"rho": float(sstats.spearmanr(x, y)[0]), "n": len(sy)}
    out["spearman"] = sp
    # same without the two controls (heads only) and the amplitude-only control
    sp2 = {}
    sub = ("shipped", "loso", "loso_lin", "all", "all_lin", "true_ema")
    for mk, ms in metr.items():
        x = [ms * agree[s][mk] for s in sub]
        y = [-down[s]["wer_usc"] for s in sub]
        sp2[mk] = float(sstats.spearmanr(x, y)[0])
    out["spearman_heads_only_wer_usc"] = sp2
    return out


# ---------------------------------------------------------------- checks of what is cached

def availability():
    import glob

    out = {}
    ks = sorted({int(re.search(r"probe_k(\d+)_ds", f).group(1)) for f in glob.glob(str(rigor.OUT / "preds" / "*" / "*" / "probe_k*_ds.npz"))})
    out["probe_depths_with_predictions"] = ks
    out["N1_assessable"] = len(ks) >= 12
    out["gender_npz_files"] = len(glob.glob(str(datasets.XLSR_EMA / "analysis_gender" / "**" / "*.npz"), recursive=True))
    out["gender_assessable"] = out["gender_npz_files"] > 0
    out["extra_npz_files"] = len(glob.glob(str(rigor.OUT / "extra" / "**" / "*.npz"), recursive=True))
    return out


# ---------------------------------------------------------------- summary

def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def ci(c):
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--skip-q6", action="store_true")
    ap.add_argument("--only-q6", action="store_true")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.render_only:
        r = json.loads((OUT / "results.json").read_text())
        summarize(r)
        (OUT / "results.json").write_text(json.dumps(r, indent=1, default=float))
        write_summary(r)
        return
    if a.only_q6:
        r = json.loads((OUT / "results.json").read_text())
        r["q6"] = q6(r)
        (OUT / "results.json").write_text(json.dumps(r, indent=1, default=float))
        write_summary(r)
        return
    res = {"availability": availability()}
    print(res["availability"], flush=True)
    arms, lora_specs = build_arms()
    T = Tab(arms)
    prev = json.loads((rigor.OUT / "results.json").read_text())["rows"]
    chk = {n: max(abs(T.mean(n, "rmse_z") - prev[n]["rmse_z"]), abs(T.mean(n, "pcc") - prev[n]["pcc"])) for n in T.arms if n in prev}
    res["check_vs_results_json"] = {"max_abs_diff": max(chk.values()), "missing_in_results_json": [n for n in T.arms if n not in prev],
                                    "not_rebuilt": [n for n in prev if n not in T.arms]}
    print("max |ours - results.json| (rmse_z, PCC):", res["check_vs_results_json"], flush=True)
    names = list(T.arms)
    # arm table
    keys = ["rmse_z", "rmse_centered", "pcc", "pcc_concat", "ccc_utt", "ccc_utt_1s", "ccc_spk", "ccc_spk_centered", "std_ratio", "bias_share",
            "offset_sd", "utt_corr", "utt_scale_w", "utt_location_w", "utt_cs", "utt_cl", "spk_corr", "spk_scale_w", "spk_location_w", "spk_cs",
            "spk_cl", "n_utt", "n_utt_1s"]
    res["arms"] = {}
    for n in names:
        r = {k: T.mean(n, k) for k in keys}
        r["folds"], r["seeds"] = len(T.arms[n]), int(np.median([len(x) for x in T.arms[n].values()]))
        r["decomp_utt"], r["decomp_spk"] = arm_decomp(T, n, "utt"), arm_decomp(T, n, "spk")
        for m in ("rmse_z", "pcc", "ccc_utt", "ccc_spk"):
            fvs = T.fv(n, m)
            r[f"fold_sd_{m}"] = float(np.std(list(fvs.values()), ddof=1))
            sds = [np.var([x[m] for x in runs], ddof=1) for runs in T.M[n].values() if len(runs) > 1]
            r[f"seed_var_{m}"] = float(np.mean(sds)) if sds else np.nan
            r[f"fold_var_{m}"] = float(np.var(list(fvs.values()), ddof=1))
        r["fold_values"] = {m: T.fv(n, m) for m in ("rmse_z", "pcc", "ccc_utt", "ccc_spk")}
        res["arms"][n] = r
    # frame-count distribution
    nf = []
    for arm in (T.arms["W12 probe"],):
        for h, runs in arm.items():
            nf += list(runs[0][1]["n"])
    res["frames_per_utterance"] = {"n_utt": len(nf), "min": float(np.min(nf)), "p5": float(np.percentile(nf, 5)), "median": float(np.median(nf)),
                                   "p95": float(np.percentile(nf, 95)), "share_under_50_frames": float(np.mean(np.array(nf) < 50))}
    # Q1
    res["q1"] = {"all": q1(T, names, "all arms"),
                 "plain": q1(T, [n for n in names if not is_derived(n)], "arms without gain-derived arms (exploratory)")}
    print("Q1 done", flush=True)
    # Q2 / Q3 / Q5
    cl = contrast_list(T)
    cache = {}
    q2 = {}
    for cid, a_, b_, note in cl:
        r = contrast_set(T, a_, b_, a.boot, cache)
        q2[cid] = {"a": a_, "b": b_, "note": note, **r,
                   "shares_utt": log_shares(arm_decomp(T, a_, "utt"), arm_decomp(T, b_, "utt")),
                   "shares_spk": log_shares(arm_decomp(T, a_, "spk"), arm_decomp(T, b_, "spk"))}
        print("contrast", cid, flush=True)
    res["q2"] = q2
    # seed-vs-fold variance components
    res["variance_components"] = {m: {"median_seed_over_fold_var": float(np.nanmedian([res["arms"][n][f"seed_var_{m}"] / res["arms"][n][f"fold_var_{m}"] for n in names])),
                                      "median_seed_sd": float(np.nanmedian([np.sqrt(res["arms"][n][f"seed_var_{m}"]) for n in names])),
                                      "median_fold_sd": float(np.nanmedian([res["arms"][n][f"fold_sd_{m}"] for n in names]))} for m in Q2M}
    # Q3: arm-level explanation of CCC; C_b variation vs fold noise
    ccc_a = np.array([res["arms"][n]["ccc_utt"] for n in names])
    pcc_a = np.array([res["arms"][n]["utt_corr"] for n in names])
    sr_a = np.array([res["arms"][n]["std_ratio"] for n in names])
    off_a = np.array([res["arms"][n]["offset_sd"] for n in names])
    cb_a = np.array([res["arms"][n]["ccc_utt"] / res["arms"][n]["utt_corr"] for n in names])

    def r2(X, y):
        X = np.column_stack([np.ones(len(y))] + X)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    plain_idx = [i for i, n in enumerate(names) if not is_derived(n)]
    res["q3"] = {"r2_ccc_on_pcc": r2([pcc_a], ccc_a), "r2_ccc_on_pcc_stdratio": r2([pcc_a, sr_a], ccc_a),
                 "r2_ccc_on_pcc_stdratio_offset": r2([pcc_a, sr_a, off_a], ccc_a),
                 "r2_ccc_on_pcc_plain_arms": r2([pcc_a[plain_idx]], ccc_a[plain_idx]),
                 "r2_ccc_on_pcc_stdratio_plain_arms": r2([pcc_a[plain_idx], sr_a[plain_idx]], ccc_a[plain_idx]),
                 "cb_across_arm_sd": float(cb_a.std(ddof=1)), "cb_range": [float(cb_a.min()), float(cb_a.max())],
                 "cb_across_arm_sd_plain": float(cb_a[plain_idx].std(ddof=1)), "cb_range_plain": [float(cb_a[plain_idx].min()), float(cb_a[plain_idx].max())],
                 "cb_median_paired_se_q2": float(np.median([q2[c]["cb"]["se"] for c in q2])),
                 "cb_share_q2_ci_excludes_0": float(np.mean([q2[c]["cb"]["t_ci"][0] > 0 or q2[c]["cb"]["t_ci"][1] < 0 for c in q2])),
                 "share_of_ccc_diff_q2": {}}
    for c, r in q2.items():
        t = r["shares_utt"]
        res["q3"]["share_of_ccc_diff_q2"][c] = {k: (t[k] / t["total"] if abs(t["total"]) > 1e-9 else np.nan) for k in ("correlation", "scale", "location")} | {"total_log": t["total"]}
    summarize(res)
    print("Q2/Q3/Q5 done", flush=True)
    res["q4"] = q4(T, lora_specs, a.boot)
    print("Q4 done", flush=True)
    if not a.skip_q6:
        try:
            res["q6"] = q6(res)
        except Exception as e:  # keep everything else
            import traceback
            res["q6_error"] = traceback.format_exc()
            print(res["q6_error"], flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_summary(res)


def write_summary(res):
    import sys

    from .ccc_report import render

    text = render(res)
    (OUT / "summary.md").write_text(text)
    sys.stdout.write(text[:3000])


if __name__ == "__main__":
    main()
