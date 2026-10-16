"""Rigor extras (pre-registered in docs/rigor_prereg.md, addendum): alignment sensitivity and the data curve.

Item 9, alignment sensitivity
    align-test    every rigor arm re-scored with the held-out targets shifted by d in {-2..+2} frames against the cached
                  predictions (p[t] vs y[t+d], ends truncated); rmse_z / PCC per arm and Kendall tau of the arm ranking
                  at d vs d = 0 (CPU, cached predictions only)
    align-train   ridge probes (cached low-passed features, LOSO folds, each fold's depth-sweep shift and alpha) whose
                  TRAINING speakers' alignment is perturbed by a random offset in {-1, 0, +1} frames (one offset per
                  training speaker, 5 draws shared across encoders); the held-out speaker is scored unperturbed
Item 10, data curve (descriptive)
    curve-probe   MNGU0 WavLM k9 ridge probe on N in {100, 200, 400, all} training utterances x 3 random subsets
                  (the same subsets as train.py --train-limit/--train-subset-seed), z = full-training-split mean/std
    curve-lora    collects the LoRA + causal conv runs trained with --train-limit (adapt/wavlm-large/*_n<N>d<S>_s<S>)
                  and the full-data runs; rmse_z from per-channel mm RMSE / full-training-split std
    within        each multi-speaker speaker's own train -> own test ridge probe (WavLM k9, k12; shift from the depth
                  sweep, alpha on its own validation sentences), all data and N = 100, 200 subsets, in its z units
    summary       tables -> analysis_rigor/extra/summary.md

Outputs: /data/user_data/xoy/xlsr_ema/analysis_rigor/extra/{*.json,summary.md,results.json}.

Usage (jobs; /data is not mounted on the login node):
    python -m sparc.compression.rigor_extra align-test
    python -m sparc.compression.rigor_extra align-train
    python -m sparc.compression.rigor_extra curve-probe
    python -m sparc.compression.rigor_extra curve-lora
    python -m sparc.compression.rigor_extra within
    python -m sparc.compression.rigor_extra summary
"""

import argparse
import json
import random

import numpy as np
from scipy import stats as sstats

from . import datasets
from .rigor import (CC, GC, MB, OUT as RIGOR, SEEDS, SPK, blend_arm, load_arm, metrics_from, utt_stats)

OUT = RIGOR / "extra"
ALIGN_D = (-2, -1, 0, 1, 2)
MNGU0_MODEL, MNGU0_K = "wavlm-large", 9
N_SUBSETS = (100, 200, 400)
RUN_FULL = "prefix9_independent_r8_q_proj+v_proj_conv_causal_s{s}"
RUN_N = "prefix9_independent_r8_q_proj+v_proj_conv_causal_n{n}d{s}_s{s}"


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))


def _load_json(name):
    f = OUT / name
    return json.loads(f.read_text()) if f.exists() else None


def _ridge_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- item 9a: test-time shift

def shift_pair(p, y, d):
    """Compare p[t] with y[t + d] (truncating the ends)."""
    n = min(len(p), len(y))
    p, y = p[:n], y[:n]
    if d > 0:
        return p[:n - d], y[d:]
    if d < 0:
        return p[-d:], y[:n + d]
    return p, y


def build_arms():
    """The plain (non-gain) arms of rigor.analyze, from the cached held-out predictions."""
    W, X3, X1, V2 = "wavlm-large", "xlsr-300m", "xlsr-1b", "w2v2-large"
    arms = {"W12 probe": load_arm("probe", W, 12, None)}
    for hn, lab in (("conv_causal9", "causal conv"), ("mamba", "BiMamba"), ("gruconv9", "BiGRU->conv"), ("mlp", "MLP")):
        arms[f"W12 frozen {lab}"] = load_arm("frozen", W, 12, hn)
    specs = [("W12", W, 12, CC, "k12_cc", "causal conv"), ("W12", W, 12, MB, "k12_mamba", "BiMamba"),
             ("W12", W, 12, GC, "k12_gruconv", "BiGRU->conv"), ("W7", W, 7, CC, None, "causal conv"),
             ("X18", X3, 18, CC, None, "causal conv"), ("B10", X1, 10, CC, None, "causal conv"),
             ("V18", V2, 18, CC, None, "causal conv")]
    for short, model, k, tmpl, tag, lab in specs:
        lora = load_arm("lora", model, k, tmpl)
        if not lora:
            continue
        arms[f"{short} LoRA {lab}"] = lora
        b, _ = blend_arm(model, k, tmpl, tag)
        if b:
            arms[f"{short} blend {lab}"] = b
        if short != "W12":
            arms.setdefault(f"{short} probe", load_arm("probe", model, k, None))
    arms["X18 frozen BiGRU->conv"] = load_arm("frozen", X3, 18, "gruconv9")
    return {a: v for a, v in arms.items() if v}


def arm_scores(arm, d):
    """Mean over folds of the seed-mean rmse_z / PCC with targets shifted by d."""
    rz, pc = [], []
    for runs in arm.values():
        vz, vp = [], []
        for _, P, Y in runs:
            pairs = [shift_pair(p, y, d) for p, y in zip(P, Y)]
            m = metrics_from(utt_stats([a for a, _ in pairs], [b for _, b in pairs]))
            vz.append(m["rmse_z"]); vp.append(m["pcc"])
        rz.append(np.mean(vz)); pc.append(np.mean(vp))
    return float(np.mean(rz)), float(np.mean(pc))


def align_test():
    arms = build_arms()
    res = {"d": list(ALIGN_D), "arms": {}}
    for a, arm in arms.items():
        res["arms"][a] = {str(d): dict(zip(("rmse_z", "pcc"), arm_scores(arm, d))) for d in ALIGN_D}
        print(a, {d: tuple(round(x, 4) for x in (res["arms"][a][str(d)]["rmse_z"], res["arms"][a][str(d)]["pcc"]))
                  for d in ALIGN_D}, flush=True)
    names = list(arms)
    tau = {}
    for m in ("rmse_z", "pcc"):
        base = [res["arms"][a]["0"][m] for a in names]
        tau[m] = {str(d): float(sstats.kendalltau(base, [res["arms"][a][str(d)][m] for a in names])[0]) for d in ALIGN_D}
    res["kendall_tau_vs_d0"] = tau
    # best (lowest rmse_z / highest PCC) shift per arm
    res["best_d"] = {a: {"rmse_z": min(ALIGN_D, key=lambda d: res["arms"][a][str(d)]["rmse_z"]),
                         "pcc": max(ALIGN_D, key=lambda d: res["arms"][a][str(d)]["pcc"])} for a in names}
    _dump("align_test.json", res)


# ---------------------------------------------------------------- item 9b: training-time perturbation

ALIGN_TRAIN_MODELS = (("wavlm-large", 7), ("wavlm-large", 12), ("xlsr-300m", 18))
N_DRAWS = 5


def align_train(models=ALIGN_TRAIN_MODELS, draws=N_DRAWS):
    from .depth_sweep import load_all
    from .probe import Ridge, aligned, load_targets

    device = _ridge_device()
    out = _load_json("align_train.json") or {}
    # one offset per training speaker and draw, shared by all encoders and folds
    offsets = [{s: int(o) for s, o in zip(SPK, np.random.default_rng(100 + d).choice([-1, 0, 1], len(SPK)))}
               for d in range(draws)]
    out["offsets"] = offsets
    for model, k in models:
        key = f"{model}/k{k}"
        if key in out and len(out[key]) == len(SPK):
            continue
        d = datasets.features_root("ema_multi") / model
        index = json.loads((d / "index.json").read_text())
        feats = np.load(d / f"layer_{k:02d}.npy", mmap_mode="r").astype(np.float32)
        depth = load_all(model)
        res, utts = {}, None
        for h in SPK:
            ds = datasets.EMAMulti(held_out=(h,), norm_by="norm_group")
            sp = {s: [x for x in v if x in index] for s, v in ds.splits().items()}
            if utts is None:
                utts = load_targets(sorted(set(index) & set(ds.rows)), ds)
            rec = depth[f"{k}/{h}"]
            shift, alpha = rec["shift"], rec["alpha"]
            Xt, Yt, kt = aligned(feats, index, utts, sp["test_unseen"], shift)

            def fit_score(off):
                X, Y = [], []
                for spk in SPK:
                    if spk == h:
                        continue
                    st = [s for s in sp["train"] if ds.rows[s]["speaker"] == spk]
                    x, y, _ = aligned(feats, index, utts, st, shift + off.get(spk, 0))
                    X += x; Y += y
                r = Ridge(X, Y, device)
                W, b = r.weights(alpha)
                m = metrics_from(utt_stats([p.astype(np.float64) for p in r.predict(W, b, Xt)],
                                           [y.astype(np.float64) for y in Yt]))
                return {"rmse_z": m["rmse_z"], "pcc": m["pcc"]}

            res[h] = {"shift": shift, "alpha": alpha, "base": fit_score({}), "draws": [fit_score(o) for o in offsets]}
            print(key, h, res[h]["base"], [round(x["rmse_z"], 4) for x in res[h]["draws"]], flush=True)
        out[key] = res
        _dump("align_train.json", out)


# ---------------------------------------------------------------- item 10a: MNGU0 probe data curve

def _mngu0_std(ds):
    from . import mngu0
    Y = np.concatenate([mngu0.load_ema_mm(s) for s in ds.splits()["train"]]).astype(np.float64)
    return Y.std(0)


def _rz(per_channel_rmse_mm, std):
    return float(np.mean(np.asarray(per_channel_rmse_mm) / std))


def curve_probe():
    import torch  # noqa: F401
    from .probe import ALPHAS, Ridge, aligned, load_targets, rmse

    device = _ridge_device()
    ds = datasets.get("mngu0")
    d = datasets.features_root("mngu0") / MNGU0_MODEL
    index = json.loads((d / "index.json").read_text())
    probe = json.loads((d / "probe_results.json").read_text())["layers"][str(MNGU0_K)]
    shift = probe["shift"]
    feats = np.load(d / f"layer_{MNGU0_K:02d}.npy", mmap_mode="r").astype(np.float32)
    splits = {k: [s for s in v if s in index] for k, v in ds.splits().items()}
    utts = load_targets([s for v in splits.values() for s in v], ds)
    std = _mngu0_std(ds)
    Xva, Yva, _ = aligned(feats, index, utts, splits["valid"], shift)
    Xte, Yte, kte = aligned(feats, index, utts, splits["test"], shift)
    res = {"shift": shift, "n_train_full": len(splits["train"]), "n_test": len(kte), "runs": []}
    jobs = [(n, s) for n in N_SUBSETS for s in range(3)] + [(None, 0)]
    for n, s in jobs:
        tr = splits["train"] if n is None else sorted(random.Random(s).sample(splits["train"], n))
        Xtr, Ytr, _ = aligned(feats, index, utts, tr, shift)
        r = Ridge(Xtr, Ytr, device)
        best = min(((rmse(r.predict(*r.weights(a), Xva), Yva), a) for a in ALPHAS))
        W, b = r.weights(best[1])
        P = r.predict(W, b, Xte)
        m = ds.evaluate([utts[x] for x in kte], P, Yte, n_boot=0)
        rec = {"n": n or len(splits["train"]), "subset": s, "alpha": float(best[1]), "frames": int(sum(len(y) for y in Ytr)),
               "rmse_mm": m["rmse"], "pcc": m["pcc"],
               "rmse_z": _rz([m["per_channel"][c]["rmse"] for c in m["per_channel"]], std)}
        res["runs"].append(rec)
        print(rec, flush=True)
    _dump("curve_probe.json", res)


# ---------------------------------------------------------------- item 10b: LoRA runs

def curve_lora():
    ds = datasets.get("mngu0")
    std = _mngu0_std(ds)
    root = datasets.adapt_root("mngu0") / MNGU0_MODEL
    runs = []
    for n in (None,) + N_SUBSETS:
        for s in SEEDS:
            f = root / (RUN_FULL.format(s=s) if n is None else RUN_N.format(n=n, s=s)) / "results.json"
            if not f.exists():
                print("missing", f)
                continue
            r = json.loads(f.read_text())
            t = r["test"]
            runs.append({"n": n or len(ds.splits()["train"]), "seed": s, "rmse_mm": t["rmse"], "pcc": t["pcc"],
                         "rmse_z": _rz([t["per_channel"][c]["rmse"] for c in t["per_channel"]], std),
                         "train_limit": r["config"].get("train_limit"), "shift": r["config"]["shift"]})
            print(runs[-1], flush=True)
    _dump("curve_lora.json", {"runs": runs})


# ---------------------------------------------------------------- item 10c: within-speaker probes

WITHIN_K = (9, 12)
WITHIN_N = (100, 200)


def within():
    from .depth_sweep import load_all
    from .probe import ALPHAS, Ridge, aligned, load_targets, rmse

    device = _ridge_device()
    model = "wavlm-large"
    d = datasets.features_root("ema_multi") / model
    index = json.loads((d / "index.json").read_text())
    depth = load_all(model)
    ds = datasets.EMAMulti(held_out=(), norm_by="norm_group")  # every speaker's own train / valid / test
    sp = {s: [x for x in v if x in index] for s, v in ds.splits().items()}
    utts = load_targets(sorted(set(index) & set(ds.rows)), ds)
    res = {}
    for k in WITHIN_K:
        feats = np.load(d / f"layer_{k:02d}.npy", mmap_mode="r").astype(np.float32)
        for h in SPK:
            shift = depth[f"{k}/{h}"]["shift"]
            by = {s: [x for x in sp[s] if ds.rows[x]["speaker"] == h] for s in ("train", "valid", "test")}
            Xva, Yva, _ = aligned(feats, index, utts, by["valid"], shift)
            Xte, Yte, _ = aligned(feats, index, utts, by["test"], shift)
            jobs = [(None, 0)] + [(n, s) for n in WITHIN_N if n <= len(by["train"]) for s in range(3)]
            for n, s in jobs:
                tr = by["train"] if n is None else sorted(random.Random(s).sample(by["train"], n))
                Xtr, Ytr, _ = aligned(feats, index, utts, tr, shift)
                r = Ridge(Xtr, Ytr, device)
                best = min(((rmse(r.predict(*r.weights(a), Xva), Yva), a) for a in ALPHAS))
                W, b = r.weights(best[1])
                m = metrics_from(utt_stats([p.astype(np.float64) for p in r.predict(W, b, Xte)],
                                           [y.astype(np.float64) for y in Yte]))
                rec = {"k": k, "speaker": h, "n": n or len(by["train"]), "full": n is None, "subset": s, "shift": shift,
                       "alpha": float(best[1]), "n_test": len(Xte), "rmse_z": m["rmse_z"], "pcc": m["pcc"],
                       "rmse_centered": m["rmse_centered"]}
                res.setdefault(f"{k}/{h}", []).append(rec)
                print(rec, flush=True)
    res["_loso_probe_unseen"] = {f"{k}/{h}": depth[f"{k}/{h}"]["unseen"] for k in WITHIN_K for h in SPK}
    _dump("within.json", res)


# ---------------------------------------------------------------- summary

def _ms(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.4f}" + (f" +/- {v.std(ddof=1):.4f}" if len(v) > 1 else "")


def _gap(cp, wi):
    """Data size vs speaker shift, in rmse_z / PCC at WavLM k9 (probe), from the numbers above."""
    def mn(n):
        rr = [r for r in cp["runs"] if r["n"] == n]
        return np.mean([r["rmse_z"] for r in rr]), np.mean([r["pcc"] for r in rr])
    def within_mean(n):
        v = [r for h in SPK for r in wi.get(f"9/{h}", []) if (r["full"] if n is None else (not r["full"] and r["n"] == n))]
        return np.mean([r["rmse_z"] for r in v]), np.mean([r["pcc"] for r in v])
    lo = (np.mean([wi["_loso_probe_unseen"][f"9/{h}"]["rmse_z"] for h in SPK]),
          np.mean([wi["_loso_probe_unseen"][f"9/{h}"]["pcc"] for h in SPK]))
    m_all, m100 = mn(cp["n_train_full"]), mn(100)
    w_all, w100 = within_mean(None), within_mean(100)
    f = lambda t: f"{t[0]:.3f} / {t[1]:.3f}"
    return ["## Data size vs speaker shift (WavLM k9 ridge probe, rmse_z / PCC)\n",
            "| condition | rmse_z / PCC |", "|---|---|",
            f"| MNGU0, own speaker, all {cp['n_train_full']} utterances | {f(m_all)} |",
            f"| MNGU0, own speaker, N=100 | {f(m100)} |",
            f"| multi-speaker corpora, own speaker, N=100 (mean of 7) | {f(w100)} |",
            f"| multi-speaker corpora, own speaker, all (183-368 utterances; mean of 7) | {f(w_all)} |",
            f"| multi-speaker corpora, new speaker (LOSO, ~1470 training utterances from 6 speakers) | {f(lo)} |",
            "",
            f"Reading: cutting MNGU0 from all data to N=100 costs {m100[0] - m_all[0]:+.3f} rmse_z / {m100[1] - m_all[1]:+.3f} PCC. "
            f"At the same N=100 the multi-speaker speakers are still {w100[0] - m100[0]:+.3f} rmse_z / {w100[1] - m100[1]:+.3f} PCC "
            f"from MNGU0 (corpus and speaker difficulty: sensor placement, sessions, emotional speech, alignment; not data size). "
            f"Moving from a speaker's own data to a new speaker costs a further {lo[0] - w_all[0]:+.3f} rmse_z / {lo[1] - w_all[1]:+.3f} PCC "
            f"(speaker shift, with ~4-8x more training utterances than the within-speaker models). "
            f"Of the total MNGU0 -> new-speaker gap of {lo[0] - m_all[0]:+.3f} rmse_z, the data-size share (MNGU0 at N=100) is "
            f"{(m100[0] - m_all[0]) / (lo[0] - m_all[0]):.0%}, the corpus-difficulty share at matched N=100 "
            f"{(w100[0] - m100[0]) / (lo[0] - m_all[0]):.0%}, and speaker shift (all-data within-speaker -> LOSO) "
            f"{(lo[0] - w_all[0]) / (lo[0] - m_all[0]):.0%} (shares do not sum to 100% because the multi-speaker N=100 -> all step "
            f"is {w100[0] - w_all[0]:+.3f}). Descriptive: different speakers and corpora, one probe family.", ""]


def summary():
    L = ["# Rigor extras: alignment sensitivity and the data curve\n",
         "Pre-registered as descriptive items in docs/rigor_prereg.md. All metrics are rmse_z (each speaker's own z "
         "units) and PCC; mm RMSE is secondary. Every configuration was already looked at on these folds.\n"]
    res = {}
    at = _load_json("align_test.json")
    if at:
        res["align_test"] = at
        names = list(at["arms"])
        L.append("## 9a. Held-out targets shifted by d frames against the cached predictions (p[t] vs y[t+d])\n")
        L.append("Mean over 7 LOSO folds; each arm in the rigor re-analysis (probe, frozen heads, LoRA, blends).\n")
        L.append("| arm | " + " | ".join(f"d={d} rmse_z / PCC" for d in at["d"]) + " |")
        L.append("|---|" + "---|" * len(at["d"]))
        for a in names:
            L.append(f"| {a} | " + " | ".join(f"{at['arms'][a][str(d)]['rmse_z']:.4f} / {at['arms'][a][str(d)]['pcc']:.4f}"
                                              for d in at["d"]) + " |")
        L.append("\nKendall tau of the arm ranking at d vs d = 0 (" + f"{len(names)} arms):\n")
        L.append("| metric | " + " | ".join(f"d={d}" for d in at["d"]) + " |")
        L.append("|---|" + "---|" * len(at["d"]))
        for m in ("rmse_z", "pcc"):
            L.append(f"| {m} | " + " | ".join(f"{at['kendall_tau_vs_d0'][m][str(d)]:.2f}" for d in at["d"]) + " |")
        dm = {d: np.mean([at["arms"][a][str(d)]["rmse_z"] - at["arms"][a]["0"]["rmse_z"] for a in names]) for d in at["d"]}
        dp = {d: np.mean([at["arms"][a][str(d)]["pcc"] - at["arms"][a]["0"]["pcc"] for a in names]) for d in at["d"]}
        L.append("\nMean change over arms vs d = 0, rmse_z: " + ", ".join(f"d={d} {dm[d]:+.4f}" for d in at["d"] if d)
                 + "; PCC: " + ", ".join(f"d={d} {dp[d]:+.4f}" for d in at["d"] if d))
        bd = [at["best_d"][a] for a in names]
        L.append(f"\nBest d per arm (rmse_z / PCC): " + ", ".join(
            f"d={d}: {sum(b['rmse_z'] == d for b in bd)} / {sum(b['pcc'] == d for b in bd)} arms" for d in at["d"]))
        L.append("")
    atr = _load_json("align_train.json")
    if atr:
        res["align_train"] = atr
        L.append("## 9b. Random +-1 frame alignment offsets on the training speakers (ridge probe, 5 draws)\n")
        L.append("One offset in {-1, 0, +1} per training speaker per draw (same draws for every encoder); the held-out "
                 "speaker is scored unperturbed with the fold's depth-sweep shift and alpha. Mean over 7 folds.\n")
        L.append("| encoder | unperturbed rmse_z / PCC | perturbed rmse_z / PCC (mean over draws) | change rmse_z / PCC | folds worse (rmse_z) |")
        L.append("|---|---|---|---|---|")
        keys = [k for k in atr if k != "offsets"]
        base, pert = {}, {}
        for key in keys:
            r = atr[key]
            b = np.array([[r[h]["base"]["rmse_z"], r[h]["base"]["pcc"]] for h in SPK if h in r])
            p = np.array([[[x["rmse_z"], x["pcc"]] for x in r[h]["draws"]] for h in SPK if h in r])  # folds, draws, 2
            base[key], pert[key] = b, p
            L.append(f"| {key} | {b[:, 0].mean():.4f} / {b[:, 1].mean():.4f} | {p[:, :, 0].mean():.4f} / {p[:, :, 1].mean():.4f} | "
                     f"{p[:, :, 0].mean() - b[:, 0].mean():+.4f} / {p[:, :, 1].mean() - b[:, 1].mean():+.4f} | "
                     f"{int((p[:, :, 0].mean(1) > b[:, 0]).sum())}/{len(b)} |")
        # ranking of encoders per draw and metric
        if len(keys) > 1:
            nd = min(p.shape[1] for p in pert.values())
            for mi, m in ((0, "rmse_z"), (1, "pcc")):
                sign = 1 if mi == 0 else -1
                b_rank = tuple(np.argsort([sign * base[k][:, mi].mean() for k in keys]))
                ranks = [tuple(np.argsort([sign * pert[k][:, :nd][:, d, mi].mean() for k in keys])) for d in range(nd)]
                same = sum(r == b_rank for r in ranks)
                L.append(f"\nEncoder ranking by {m} (best first, unperturbed): " + " > ".join(keys[i] for i in b_rank)
                         + f"; same ranking in {same}/{nd} perturbed draws.")
        L.append("")
    cp, cl, wi = _load_json("curve_probe.json"), _load_json("curve_lora.json"), _load_json("within.json")
    if cp:
        res["curve_probe"] = cp
        L.append("## 10a. MNGU0 (single speaker) data curve, WavLM k9\n")
        L.append(f"Test set: {cp['n_test']} utterances; z = per-channel standardization with the full training split "
                 f"({cp['n_train_full']} utterances); 3 random training subsets (same as the LoRA runs); ridge alpha on the "
                 f"validation split, shift {cp['shift']}.\n")
        L.append("| N train | model | rmse_z | PCC | rmse mm |")
        L.append("|---|---|---|---|---|")
        for n in list(N_SUBSETS) + [cp["n_train_full"]]:
            rr = [r for r in cp["runs"] if r["n"] == n]
            if rr:
                L.append(f"| {n} | ridge probe | {_ms([r['rmse_z'] for r in rr])} | {_ms([r['pcc'] for r in rr])} | {_ms([r['rmse_mm'] for r in rr])} |")
            if cl:
                lr = [r for r in cl["runs"] if r["n"] == n]
                if lr:
                    L.append(f"| {n} | LoRA + causal conv ({len(lr)} seeds) | {_ms([r['rmse_z'] for r in lr])} | {_ms([r['pcc'] for r in lr])} | {_ms([r['rmse_mm'] for r in lr])} |")
        L.append("")
    if cl:
        res["curve_lora"] = cl
    if wi:
        res["within"] = wi
        L.append("## 10c. Within-speaker probes (each multi-speaker speaker, own train -> own test), WavLM\n")
        L.append("Ridge on the speaker's own training split, alpha on its own validation sentences, shift from the depth "
                 "sweep; own z units (usc_F5: two session groups). Subsets: 3 random draws. Last column: LOSO held-out "
                 "ridge probe (trained on the other six speakers, depth-sweep selection) for the same layer.\n")
        for k in WITHIN_K:
            L.append(f"\nWavLM k{k}, rmse_z / PCC:\n")
            L.append("| speaker | N=100 | N=200 | all (n train) | LOSO held-out |")
            L.append("|---|---|---|---|---|")
            allr = {n: [] for n in ("100", "200", "all")}
            for h in SPK:
                rs = wi.get(f"{k}/{h}", [])
                cells = []
                for n in (100, 200, None):
                    rr = [r for r in rs if r["full"]] if n is None else [r for r in rs if not r["full"] and r["n"] == n]
                    if rr:
                        z, p = np.mean([r["rmse_z"] for r in rr]), np.mean([r["pcc"] for r in rr])
                        cells.append(f"{z:.3f} / {p:.3f}" + (f" ({rr[0]['n']})" if n is None else ""))
                        allr[str(n) if n else "all"].append((z, p))
                    else:
                        cells.append("-")
                u = wi["_loso_probe_unseen"][f"{k}/{h}"]
                L.append(f"| {h} | " + " | ".join(cells) + f" | {u['rmse_z']:.3f} / {u['pcc']:.3f} |")
            mean = lambda v: f"{np.mean([x[0] for x in v]):.3f} / {np.mean([x[1] for x in v]):.3f} ({len(v)} spk)" if v else "-"
            lo = np.mean([wi["_loso_probe_unseen"][f"{k}/{h}"]["rmse_z"] for h in SPK]), np.mean([wi["_loso_probe_unseen"][f"{k}/{h}"]["pcc"] for h in SPK])
            L.append(f"| mean | {mean(allr['100'])} | {mean(allr['200'])} | {mean(allr['all'])} | {lo[0]:.3f} / {lo[1]:.3f} |")
        L.append("\nMean rows average only the speakers that have the cell (N=200 needs >= 200 training utterances: "
                 "5emo_jr and 5emo_kf have 184 / 185).\n")
        if cp:
            L += _gap(cp, wi)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("align-test", "align-train", "curve-probe", "curve-lora", "within", "summary"))
    a = ap.parse_args()
    if a.cmd == "align-test":
        align_test()
    elif a.cmd == "align-train":
        align_train()
    elif a.cmd == "curve-probe":
        curve_probe()
    elif a.cmd == "curve-lora":
        curve_lora()
    elif a.cmd == "within":
        within()
    else:
        summary()


if __name__ == "__main__":
    main()
