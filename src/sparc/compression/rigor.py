"""Rigor re-analysis of the LOSO results (pre-registered in docs/rigor_prereg.md).

Every arm is scored from cached per-utterance held-out predictions (test_unseen of each LOSO fold, in the held-out
speaker's z units), so all metrics, controls and tests run on CPU from one cache:

    probe-preds  ridge probe predictions (cached features; shift/alpha from depth_sweep, plus the probe at fixed
                 shifts with alpha chosen on validation, which is what lora_readout blends with)
    head-preds   frozen-encoder heads retrained at head_screen's recorded learning rate (seeds 0-2); the head screen
                 itself kept only metrics
    (LoRA predictions come from `loso_posthoc run [--run TEMPLATE]`.)
    analyze      metrics, shrinkage control, enrollment, post-hoc smoother, metric-definition check and the
                 pre-registered tests (sign-flip permutation, hierarchical bootstrap)

Outputs: /data/user_data/xoy/xlsr_ema/analysis_rigor/{preds/,summary.md,results.json}.

Usage:
    python -m sparc.compression.rigor probe-preds --model wavlm-large --k 12 --shifts 0,1
    python -m sparc.compression.rigor head-preds --model wavlm-large --layer 12 --head gruconv9 --fold usc_M1
    python -m sparc.compression.rigor analyze
"""

import argparse
import itertools
import json

import numpy as np
from scipy import stats as sstats

from . import datasets

OUT = datasets.XLSR_EMA / "analysis_rigor"
POSTHOC = datasets.XLSR_EMA / "analysis_posthoc" / "preds"
READOUT = datasets.XLSR_EMA / "analysis_readout"
SPK = datasets.ALL_SPEAKERS
USC = tuple(h for h in SPK if h.startswith("usc"))
SEEDS = (0, 1, 2)


def _save(f, stems, preds, trues, **extra):
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, stems=np.array(stems), preds=np.array([p.astype(np.float32) for p in preds], dtype=object),
             trues=np.array([y.astype(np.float32) for y in trues], dtype=object), **extra)


def _load(f, lora=False):
    z = np.load(f, allow_pickle=True)
    return list(z["stems"]), [p.astype(np.float64) for p in z["preds"]], [t.astype(np.float64) for t in z["trues"]]


# ---------------------------------------------------------------- prediction caches (GPU / CPU jobs)

def probe_preds(model, k, shifts=(), device=None):
    import torch

    from .depth_sweep import load_all
    from .lora_readout import fit_ridge
    from .probe import Ridge, aligned, load_targets

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = datasets.features_root("ema_multi") / model
    index = json.loads((d / "index.json").read_text())
    feats = np.load(d / f"layer_{k:02d}.npy", mmap_mode="r").astype(np.float32)
    depth = load_all(model)
    utts = None
    for h in SPK:
        ds = datasets.EMAMulti(held_out=(h,), norm_by="norm_group")
        sp = {s: [x for x in v if x in index] for s, v in ds.splits().items()}
        if utts is None:
            utts = load_targets(sorted(set(index) & set(ds.rows)), ds)
        rec = depth[f"{k}/{h}"]
        jobs = [("ds", rec["shift"], rec["alpha"])] + [(f"shift{s}", s, None) for s in shifts]
        for label, shift, alpha in jobs:
            f = OUT / "preds" / model / h / f"probe_k{k}_{label}.npz"
            if f.exists():
                continue
            A = {s: aligned(feats, index, utts, sp[s], shift) for s in ("train", "valid", "test_unseen")}
            if alpha is None:  # as lora_readout: alpha on the training speakers' validation sentences
                r, W, b = fit_ridge(A["train"][0], A["train"][1], A["valid"][0], A["valid"][1], device)
            else:
                r = Ridge(A["train"][0], A["train"][1], device)
                W, b = r.weights(alpha)
            X, Y, keep = A["test_unseen"]
            _save(f, keep, r.predict(W, b, X), [y.astype(np.float64) for y in Y], shift=shift,
                  alpha=-1.0 if alpha is None else alpha)
            print(f"{model} k{k} {h} {label}: saved {len(keep)} utterances", flush=True)


def head_preds(model, layer, head_name, fold, device=None):
    import torch

    from .head_screen import OUT as HS, train_one, predict, zm
    from .probe import aligned, load_targets

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hs = json.loads((HS / model / f"k{layer}_{head_name}.json").read_text())[fold]
    d = datasets.features_root("ema_multi") / model
    index = json.loads((d / "index.json").read_text())
    feats = np.load(d / f"layer_{layer:02d}.npy", mmap_mode="r")
    ds = datasets.EMAMulti(held_out=(fold,), norm_by="norm_group")
    sp = {k: [s for s in v if s in index] for k, v in ds.splits().items()}
    utts = load_targets(sorted(set(index) & set(ds.rows)), ds)
    data, keeps = {}, {}
    for split in ("train", "valid", "test_unseen"):
        X, Y, keep = aligned(feats, index, utts, sp[split], hs["shift"])
        data[split] = ([np.asarray(x, dtype=np.float32) for x in X], [y.astype(np.float64) for y in Y])
        keeps[split] = keep
    Yall = np.concatenate(data["train"][1])
    ym, ys = Yall.mean(0), Yall.std(0)
    for s in SEEDS:
        f = OUT / "preds" / model / fold / f"frozen_k{layer}_{head_name}_s{s}.npz"
        if f.exists():
            continue
        head, v = train_one(head_name, False, hs["lr"], data, ym, ys, device, seed=s)
        if head is None:
            print(f"{model} k{layer} {head_name} {fold} s{s}: diverged", flush=True)
            continue
        P = predict(head, data["test_unseen"][0], ym, ys, device)
        m = zm(P, data["test_unseen"][1])
        rec = hs["seeds"].get(str(s), {}).get("test_unseen", {})
        _save(f, keeps["test_unseen"], P, data["test_unseen"][1], valid_rmse_z=v,
              screen_rmse_z=rec.get("rmse_z", np.nan))
        print(f"{model} k{layer} {head_name} {fold} s{s}: held-out rmse_z {m['rmse_z']:.4f} / PCC {m['pcc']:.4f} "
              f"(head screen {rec.get('rmse_z', float('nan')):.4f} / {rec.get('pcc', float('nan')):.4f})", flush=True)


# ---------------------------------------------------------------- per-utterance statistics and metrics

def utt_stats(P, Y):
    """Per-utterance sufficient statistics: sse (U, 12), centered sse, frame counts, error means, PCC (U, 12)."""
    sse, csse, n, emean, pcc, psum, psq, ysum, ysq = [], [], [], [], [], [], [], [], []
    for p, y in zip(P, Y):
        e = p - y
        sse.append((e ** 2).sum(0))
        csse.append(((e - e.mean(0)) ** 2).sum(0))
        n.append(len(e))
        emean.append(e.mean(0))
        ps, ysd = p.std(0), y.std(0)
        c = ((p - p.mean(0)) * (y - y.mean(0))).mean(0) / np.where((ps > 1e-8) & (ysd > 1e-8), ps * ysd, np.nan)
        pcc.append(c)
        psum.append(p.sum(0)); psq.append((p ** 2).sum(0)); ysum.append(y.sum(0)); ysq.append((y ** 2).sum(0))
    A = lambda v: np.array(v, dtype=np.float64)
    return {"sse": A(sse), "csse": A(csse), "n": A(n), "emean": A(emean), "pcc": A(pcc),
            "psum": A(psum), "psq": A(psq), "ysum": A(ysum), "ysq": A(ysq)}


def metrics_from(st, idx=None):
    st = st if idx is None else {k: v[idx] for k, v in st.items()}
    N = st["n"].sum()
    mse = st["sse"].sum(0) / N
    cmse = st["csse"].sum(0) / N
    pstd = np.sqrt(np.maximum(st["psq"].sum(0) / N - (st["psum"].sum(0) / N) ** 2, 0))
    ystd = np.sqrt(np.maximum(st["ysq"].sum(0) / N - (st["ysum"].sum(0) / N) ** 2, 0))
    # speaker-centered: remove the speaker-level mean error per channel
    em = (st["emean"] * st["n"][:, None]).sum(0) / N
    smse = mse - em ** 2
    return {"rmse_z": float(np.sqrt(mse).mean()), "pcc": float(np.nanmean(st["pcc"])),
            "rmse_centered": float(np.sqrt(cmse).mean()), "bias_share": float(1 - cmse.sum() / mse.sum()),
            "rmse_spk_centered": float(np.sqrt(np.maximum(smse, 0)).mean()),
            "std_ratio": float(np.mean(pstd / ystd))}


def pcc_concat(P, Y):
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    return float(np.mean([np.corrcoef(Pc[:, c], Yc[:, c])[0, 1] for c in range(Yc.shape[1])]))


_KERNEL = None


def smooth(P):
    """The deployed +-1 s output smoother (heads.Smooth), applied post hoc per utterance (replicate edges)."""
    global _KERNEL
    if _KERNEL is None:
        from .heads import lowpass_kernel
        _KERNEL = np.asarray(lowpass_kernel().numpy(), dtype=np.float64)
    k, pad = _KERNEL, (len(_KERNEL) - 1) // 2
    out = []
    for p in P:
        x = np.pad(p, ((pad, pad), (0, 0)), mode="edge")
        out.append(np.stack([np.correlate(x[:, c], k, mode="valid") for c in range(p.shape[1])], 1))
    return out


# ---------------------------------------------------------------- arms

def load_arm(kind, model, k, extra):
    """Returns {fold: [(stems, P, Y), ...per seed]}."""
    out = {}
    for h in SPK:
        runs = []
        if kind == "probe":
            f = OUT / "preds" / model / h / f"probe_k{k}_ds.npz"
            if f.exists():
                runs.append(_load(f))
        elif kind == "frozen":
            for s in SEEDS:
                f = OUT / "preds" / model / h / f"frozen_k{k}_{extra}_s{s}.npz"
                if f.exists():
                    runs.append(_load(f))
        elif kind == "lora":
            for s in SEEDS:
                f = POSTHOC / model / h / (extra.format(k=k, s=s) + ".npz")
                if f.exists():
                    runs.append(_load(f))
        elif kind == "matched":  # matched-pipeline LoRA at each fold's validation-chosen learning rate
            c = OUT / "matched" / f"{h}.json"
            if c.exists():
                lr = json.loads(c.read_text())["lr"]
                for s in SEEDS:
                    f = POSTHOC / model / h / (MATCHED.format(k=k, lr=lr, s=s) + ".npz")
                    if f.exists():
                        runs.append(_load(f))
        if runs:
            out[h] = runs
    return out


def blend_arm(model, k, tmpl, tag):
    """0.75/0.25 (validation-chosen weight from lora_readout) of LoRA and the probe at the LoRA run's shift."""
    out, check = {}, []
    for h in SPK:
        f = READOUT / model / (f"{h}.json" if tag is None else f"{tag}/{h}.json")
        if not f.exists():
            continue
        rd = json.loads(f.read_text())
        runs = []
        for s in SEEDS:
            name = tmpl.format(k=k, s=s)
            lf = POSTHOC / model / h / f"{name}.npz"
            if name not in rd or not lf.exists():
                continue
            pf = OUT / "preds" / model / h / f"probe_k{k}_shift{rd[name]['shift']}.npz"
            if not pf.exists():
                continue
            w = rd[name]["blend_w"]
            ls, lp, _ = _load(lf)
            ps, pp, py = _load(pf)
            L = dict(zip(ls, lp))
            stems, P, Y = [], [], []
            for st, p, y in zip(ps, pp, py):
                q = L.get(st, p)
                n = min(len(p), len(q), len(y))
                stems.append(st); P.append(w * q[:n] + (1 - w) * p[:n]); Y.append(y[:n])
            runs.append((stems, P, Y))
            check.append(abs(metrics_from(utt_stats(P, Y))["rmse_z"] - rd[name]["test_unseen"]["blend"]["rmse_z"]))
        if runs:
            out[h] = runs
    return out, (max(check) if check else np.nan)


def shrink_arms(lora):
    """Gain on LoRA predictions: scalar / per-channel fitted on the other folds (same seed), and a scalar oracle."""
    xs, xc, orc, gains = {}, {}, {}, {"scalar": [], "oracle": []}
    for h, runs in lora.items():
        xs[h], xc[h], orc[h] = [], [], []
        for si, (stems, P, Y) in enumerate(runs):
            others = [lora[o][si] for o in lora if o != h and len(lora[o]) > si]
            Po = np.concatenate([p for _, PP, _ in others for p in PP])
            Yo = np.concatenate([y for _, _, YY in others for y in YY])
            g = float((Po * Yo).sum() / (Po * Po).sum())
            gc = (Po * Yo).sum(0) / (Po * Po).sum(0)
            Pc, Yc = np.concatenate(P), np.concatenate(Y)
            go = float((Pc * Yc).sum() / (Pc * Pc).sum())
            gains["scalar"].append(g); gains["oracle"].append(go)
            xs[h].append((stems, [g * p for p in P], Y))
            xc[h].append((stems, [gc * p for p in P], Y))
            orc[h].append((stems, [go * p for p in P], Y))
    return xs, xc, orc, gains


_FOLD = {}


def _fold_enroll_data(h):
    """Fold dataset, training speakers' average statistics, and the held-out speaker's training-split EMA frames."""
    if h not in _FOLD:
        ds = datasets.get(f"ema_loso_{h}")
        train_groups = sorted({r["norm_group"] for r in ds.rows.values() if r["speaker"] != h})
        pop = (np.mean([ds._stats[g][0] for g in train_groups], 0), np.mean([ds._stats[g][1] for g in train_groups], 0))
        frames = {}
        for r in ds.rows.values():
            if r["speaker"] == h and r["_split"] == "train":
                e = np.load(ds.root / r["ema"])["ema"][r["_first"]:r["_last"]].astype(np.float64)
                frames.setdefault(r["norm_group"], []).append(e)
        _FOLD[h] = (ds, pop, frames)
    return _FOLD[h]


def enroll_rmse(arm, ns=(0, 10, 20), draws=10, seed=0):
    """rmse_z when the speaker's means/stds come from N random training utterances (N = 0: training average)."""
    rng = np.random.default_rng(seed)
    res = {n: [] for n in ns}
    for h, runs in arm.items():
        ds, pop, frames = _fold_enroll_data(h)
        for n in ns:
            vals = []
            for _ in range(1 if n == 0 else draws):
                est = {}
                for g, U in frames.items():
                    if n == 0:
                        est[g] = pop
                    else:
                        X = np.concatenate([U[i] for i in rng.choice(len(U), min(n, len(U)), replace=False)])
                        est[g] = (X.mean(0), X.std(0) + 1e-6)
                for stems, P, Y in runs:
                    Pz = []
                    for st, p in zip(stems, P):
                        g = ds.rows[st]["norm_group"]
                        m_all, sd_all = ds._stats[g]
                        m, sd = est[g]
                        Pz.append((p * sd + m - m_all) / sd_all)
                    vals.append(metrics_from(utt_stats(Pz, Y))["rmse_z"])
            res[n].append(float(np.mean(vals)))
    return {n: (float(np.mean(v)) if v else np.nan) for n, v in res.items()}


# ---------------------------------------------------------------- tests

def fold_values(arm, m):
    return {h: float(np.mean([metrics_from(utt_stats(P, Y))[m] for _, P, Y in runs])) for h, runs in arm.items()}


def sign_flip_p(d):
    d = np.asarray(d)
    obs = abs(d.mean())
    null = [abs((d * np.array(s)).mean()) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([x >= obs - 1e-12 for x in null]))


def hier_bootstrap(a, b, m, sign, B=2000, seed=0):
    """Paired: speakers with replacement, then seeds within speaker, then utterances (shared index across arms)."""
    rng = np.random.default_rng(seed)
    folds = [h for h in SPK if h in a and h in b]
    pre = {}
    for h in folds:
        sa = [utt_stats(P, Y) for _, P, Y in a[h]]
        sb = [utt_stats(P, Y) for _, P, Y in b[h]]
        # align utterances by stem (both arms score the same sentences; keep those every seed of both arms has)
        common = sorted(set.intersection(*[set(r[0]) for r in a[h] + b[h]]))
        ia = [{st: i for i, st in enumerate(r[0])} for r in a[h]]
        ib = [{st: i for i, st in enumerate(r[0])} for r in b[h]]
        pre[h] = (sa, sb, common, ia, ib)
    out = []
    for _ in range(B):
        hs = rng.choice(folds, len(folds), replace=True)
        diffs = []
        for h in hs:
            sa, sb, common, ia, ib = pre[h]
            u = rng.integers(0, len(common), len(common))
            pick = [common[i] for i in u]
            va = [metrics_from(sa[s], np.array([ia[s][p] for p in pick]))[m] for s in rng.integers(0, len(sa), len(sa))]
            vb = [metrics_from(sb[s], np.array([ib[s][p] for p in pick]))[m] for s in rng.integers(0, len(sb), len(sb))]
            diffs.append(sign * (np.mean(va) - np.mean(vb)))
        out.append(np.mean(diffs))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def contrast(a, b, m, sign, boot=True):
    fa, fb = fold_values(a, m), fold_values(b, m)
    folds = [h for h in SPK if h in fa and h in fb]
    d = np.array([sign * (fa[h] - fb[h]) for h in folds])
    n = len(d)
    hw = sstats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    mde = (sstats.t.ppf(0.975, n - 1) + sstats.t.ppf(0.8, n - 1)) * d.std(ddof=1) / np.sqrt(n)
    usc = [sign * (fa[h] - fb[h]) for h in folds if h in USC]
    emo = [sign * (fa[h] - fb[h]) for h in folds if h not in USC]
    r = {"n_folds": n, "mean": float(d.mean()), "t_ci": [float(d.mean() - hw), float(d.mean() + hw)],
         "won": int((d > 0).sum()), "sign_flip_p": sign_flip_p(d), "mde80": float(mde),
         "usc": float(np.mean(usc)) if usc else np.nan, "emo": float(np.mean(emo)) if emo else np.nan}
    if boot:
        r["boot_ci"] = hier_bootstrap(a, b, m, sign)
    return r


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    rej, m = [False] * len(ps), len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def seed_sd(arm, m):
    sds = [np.std([metrics_from(utt_stats(P, Y))[m] for _, P, Y in runs], ddof=1) for runs in arm.values() if len(runs) > 1]
    return float(np.mean(sds)) if sds else np.nan


# ---------------------------------------------------------------- analysis

CC = "prefix{k}_independent_r8_q_proj+v_proj_conv_causal_s{s}"
MB = "prefix{k}_independent_r8_q_proj+v_proj_mamba_nosm_s{s}"
GC = "prefix{k}_independent_r8_q_proj+v_proj_gruconv_k9_nosm_s{s}"
MATCHED = "prefix{k}_independent_r8_q_proj+v_proj_gruconv_k9_nosm_m{lr:g}_s{s}"


def head_screen_folds(model, k, name):
    """{fold: {rmse_z, pcc}} seed means of a head_screen result file (no cached predictions needed)."""
    from .head_screen import OUT as HS

    f = HS / model / f"k{k}_{name}.json"
    if not f.exists():
        return {}
    out = {}
    for h, rec in json.loads(f.read_text()).items():
        ok = [x for x in rec.get("seeds", {}).values() if "failed" not in x]
        if ok:
            out[h] = {m: float(np.mean([x["test_unseen"][m] for x in ok])) for m in ("rmse_z", "pcc")}
            out[h]["shift"] = rec.get("shift")
    return out


def fold_contrast(fa, fb, m, sign):
    """Paired over folds from per-fold values: mean, t 95% CI, folds won, exact sign-flip p."""
    folds = [h for h in SPK if h in fa and h in fb]
    d = np.array([sign * (fa[h][m] - fb[h][m]) for h in folds])
    if len(d) < 2:
        return None
    hw = sstats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"n_folds": len(d), "mean": float(d.mean()), "t_ci": [float(d.mean() - hw), float(d.mean() + hw)],
            "won": int((d > 0).sum()), "sign_flip_p": sign_flip_p(d)}


def tier2():
    """Pre-registered T1 / T2 (+ N1 from nested.py) and the descriptive causal-pipeline table."""
    W = "wavlm-large"
    arms = {"matched LoRA BiGRU->conv": load_arm("matched", W, 12, None),
            "frozen BiGRU->conv": load_arm("frozen", W, 12, "gruconv9"),
            "LoRA BiGRU->conv (original pipeline)": load_arm("lora", W, 12, GC),
            "probe": load_arm("probe", W, 12, None)}
    arms = {a: v for a, v in arms.items() if v}
    res = {"rows": {}, "primary": {}, "secondary": {}, "causal": {}}
    for a, arm in arms.items():
        fv = {m: fold_values(arm, m) for m in ("rmse_z", "pcc", "rmse_centered", "bias_share", "std_ratio")}
        res["rows"][a] = {m: float(np.mean(list(v.values()))) for m, v in fv.items()} | {"folds": len(arm)}
        print(a, res["rows"][a], flush=True)
    M, F, L = "matched LoRA BiGRU->conv", "frozen BiGRU->conv", "LoRA BiGRU->conv (original pipeline)"
    prim = [("T1", M, F, "pcc", 1), ("T2", M, L, "rmse_z", -1)]
    for pid, a, b, m, s in prim:
        if a in arms and b in arms:
            res["primary"][pid] = {"a": a, "b": b, "metric": m, **contrast(arms[a], arms[b], m, s)}
    nf = OUT / "nested" / "summary.json"
    if nf.exists():
        n1 = json.loads(nf.read_text()).get("N1")
        if n1:
            res["primary"]["N1"] = n1
    ids = [i for i in ("T1", "T2", "N1") if i in res["primary"]]
    for pid, rej in zip(ids, holm([res["primary"][i]["sign_flip_p"] for i in ids])):
        res["primary"][pid]["holm_reject"] = bool(rej)
    for a, b in ((M, F), (M, L), (F, L)):
        if a in arms and b in arms:
            res["secondary"][f"{a} vs {b}"] = {m: contrast(arms[a], arms[b], m, s, boot=False)
                                               for m, s in (("rmse_z", -1), ("pcc", 1), ("rmse_centered", -1))}
    # causal pipeline (descriptive): head_screen results on causal / raw / zero-phase features, smoother off
    cz = {"conv_causal9 zero-phase": head_screen_folds(W, 12, "conv_causal9"),
          "gru_uni zero-phase": head_screen_folds(W, 12, "gru_uni"),
          "conv_causal9 causal": head_screen_folds(W, 12, "conv_causal9_causal"),
          "gru_uni causal": head_screen_folds(W, 12, "gru_uni_causal"),
          "conv_causal9 raw": head_screen_folds(W, 12, "conv_causal9_raw"),
          "gru_uni raw": head_screen_folds(W, 12, "gru_uni_raw"),
          "BiGRU zero-phase": head_screen_folds(W, 12, "gru"),
          "BiGRU->conv zero-phase": head_screen_folds(W, 12, "gruconv9")}
    cz = {a: v for a, v in cz.items() if v}
    for a, v in cz.items():
        res["causal"][a] = {"folds": len(v), **{m: float(np.mean([x[m] for x in v.values()])) for m in ("rmse_z", "pcc")},
                            "shifts": sorted({x["shift"] for x in v.values() if x["shift"] is not None})}
    for head in ("conv_causal9", "gru_uni"):
        for feat in ("causal", "raw"):
            a, b = f"{head} {feat}", f"{head} zero-phase"
            if a in cz and b in cz:
                res["causal"][f"{a} vs {b}"] = {m: fold_contrast(cz[a], cz[b], m, s) for m, s in (("rmse_z", -1), ("pcc", 1))}
    if "gru_uni causal" in cz and "BiGRU zero-phase" in cz:
        res["causal"]["gru_uni causal vs BiGRU zero-phase"] = {m: fold_contrast(cz["gru_uni causal"], cz["BiGRU zero-phase"], m, s)
                                                               for m, s in (("rmse_z", -1), ("pcc", 1))}
    (OUT / "tier2.json").write_text(json.dumps(res, indent=1, default=float))
    L_ = ["# Rigor tier 2: matched-pipeline LoRA and causal pipeline (WavLM k12, held-out speaker)\n",
          "## Pre-registered (Holm over T1, T2, N1; exact sign-flip p; + = first arm better)\n",
          "| id | first | second | metric | diff | bootstrap 95% | t 95% | won | p | Holm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for pid in ids:
        r = res["primary"][pid]
        bc = _ci(r, "boot_ci") if "boot_ci" in r else "-"
        L_.append(f"| {pid} | {r.get('a', '-')} | {r.get('b', '-')} | {r.get('metric', '-')} | {r['mean']:+.4f} | {bc} | "
                  f"{_ci(r)} | {r['won']}/{r['n_folds']} | {r['sign_flip_p']:.3f} | {'reject' if r['holm_reject'] else 'n.s.'} |")
    L_ += ["\n## Arms\n", "| arm | folds | rmse_z | PCC | centered rmse_z | bias share | std ratio |", "|---|---|---|---|---|---|---|"]
    for a, r in res["rows"].items():
        L_.append(f"| {a} | {r['folds']} | {r['rmse_z']:.4f} | {r['pcc']:.4f} | {r['rmse_centered']:.4f} | "
                  f"{r['bias_share']:.3f} | {r['std_ratio']:.3f} |")
    L_ += ["\n## Secondary (descriptive)\n", "| contrast | metric | diff | t 95% | won | p |", "|---|---|---|---|---|---|"]
    for n, d in res["secondary"].items():
        for m, r in d.items():
            L_.append(f"| {n} | {m} | {r['mean']:+.4f} | {_ci(r)} | {r['won']}/{r['n_folds']} | {r['sign_flip_p']:.3f} |")
    L_ += ["\n## Causal pipeline (descriptive; frozen heads, smoother off, 3 seeds)\n", "| arm | folds | rmse_z | PCC | shifts |",
           "|---|---|---|---|---|"]
    for a, r in res["causal"].items():
        if "folds" in r:
            L_.append(f"| {a} | {r['folds']} | {r['rmse_z']:.4f} | {r['pcc']:.4f} | {r['shifts']} |")
    for a, d in res["causal"].items():
        if "folds" not in d:
            for m, r in d.items():
                if r:
                    L_.append(f"- {a}, {m}: {r['mean']:+.4f} {_ci(r)} {r['won']}/{r['n_folds']}, p {r['sign_flip_p']:.3f}")
    (OUT / "tier2.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def analyze():
    arms, notes = {}, []
    W, X3, X1, V2 = "wavlm-large", "xlsr-300m", "xlsr-1b", "w2v2-large"
    arms["W12 probe"] = load_arm("probe", W, 12, None)
    for hn, lab in (("conv_causal9", "causal conv"), ("mamba", "BiMamba"), ("gruconv9", "BiGRU->conv"), ("mlp", "MLP")):
        arms[f"W12 frozen {lab}"] = load_arm("frozen", W, 12, hn)
    shrink_gains = {}
    lora_specs = [("W12", W, 12, CC, "k12_cc", "causal conv"), ("W12", W, 12, MB, "k12_mamba", "BiMamba"),
                  ("W12", W, 12, GC, "k12_gruconv", "BiGRU->conv"), ("W7", W, 7, CC, None, "causal conv"),
                  ("X18", X3, 18, CC, None, "causal conv"), ("B10", X1, 10, CC, None, "causal conv"),
                  ("V18", V2, 18, CC, None, "causal conv")]
    for short, model, k, tmpl, tag, lab in lora_specs:
        lora = load_arm("lora", model, k, tmpl)
        if not lora:
            continue
        arms[f"{short} LoRA {lab}"] = lora
        b, chk = blend_arm(model, k, tmpl, tag)
        if b:
            arms[f"{short} blend {lab}"] = b
            notes.append(f"{short} blend {lab}: max |recomputed - stored| blend rmse_z = {chk:.2e}")
        xs, xc, orc, gains = shrink_arms(lora)
        arms[f"{short} LoRA {lab} x gain (cross-fold)"] = xs
        arms[f"{short} LoRA {lab} x per-channel gain (cross-fold)"] = xc
        arms[f"{short} LoRA {lab} x gain (ORACLE)"] = orc
        shrink_gains[f"{short} {lab}"] = {k_: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for k_, v in gains.items()}
        if short != "W12":
            arms.setdefault(f"{short} probe", load_arm("probe", model, k, None))
    arms["X18 frozen BiGRU->conv"] = load_arm("frozen", X3, 18, "gruconv9")
    arms = {a: v for a, v in arms.items() if v}

    # per-arm table
    rows = {}
    for a, arm in arms.items():
        fv = {m: fold_values(arm, m) for m in ("rmse_z", "pcc", "rmse_centered", "bias_share", "rmse_spk_centered", "std_ratio")}
        r = {m: float(np.mean(list(v.values()))) for m, v in fv.items()}
        r["pcc_concat"] = float(np.mean([np.mean([pcc_concat(P, Y) for _, P, Y in runs]) for runs in arm.values()]))
        r["folds"] = len(arm)
        r["seeds"] = int(np.median([len(x) for x in arm.values()]))
        r["seed_sd_rmse_z"], r["seed_sd_pcc"] = seed_sd(arm, "rmse_z"), seed_sd(arm, "pcc")
        r["fold_sd_rmse_z"] = float(np.std(list(fv["rmse_z"].values()), ddof=1))
        r["fold_sd_pcc"] = float(np.std(list(fv["pcc"].values()), ddof=1))
        if "ORACLE" not in a and "gain" not in a:
            sm = {h: [(s, smooth(P), Y) for s, P, Y in runs] for h, runs in arm.items()}
            r["smooth_d_rmse_z"] = float(np.mean(list(fold_values(sm, "rmse_z").values())) - r["rmse_z"])
            r["smooth_d_pcc"] = float(np.mean(list(fold_values(sm, "pcc").values())) - r["pcc"])
            r["enroll"] = enroll_rmse(arm)
        rows[a] = r
        print(f"{a}: rmse_z {r['rmse_z']:.4f} PCC {r['pcc']:.4f}", flush=True)

    # pre-registered contrasts
    prim = [("P1", "W12 blend BiGRU->conv", "W12 LoRA BiGRU->conv x gain (cross-fold)", "rmse_z", -1),
            ("P2", "W12 frozen BiGRU->conv", "W12 probe", "pcc", 1),
            ("P3", "W12 frozen BiGRU->conv", "X18 frozen BiGRU->conv", "pcc", 1)]
    prim_res = {}
    for pid, a, b, m, sign in prim:
        if a in arms and b in arms:
            prim_res[pid] = {"a": a, "b": b, "metric": m, **contrast(arms[a], arms[b], m, sign)}
    ids = list(prim_res)
    for pid, rej in zip(ids, holm([prim_res[i]["sign_flip_p"] for i in ids])):
        prim_res[pid]["holm_reject"] = bool(rej)

    # secondary (descriptive) contrasts: blend vs shrunk LoRA for every encoder/head, frozen vs LoRA heads
    sec = {}
    for short, model, k, tmpl, tag, lab in lora_specs:
        for b_ in ("gain (cross-fold)", "per-channel gain (cross-fold)"):
            A_, B_ = f"{short} blend {lab}", f"{short} LoRA {lab} x {b_}"
            if A_ in arms and B_ in arms:
                sec[f"{A_} vs {B_}"] = {m: contrast(arms[A_], arms[B_], m, s, boot=False) for m, s in (("rmse_z", -1), ("pcc", 1))}
        A_, B_ = f"{short} LoRA {lab} x gain (cross-fold)", f"{short} LoRA {lab}"
        if A_ in arms and B_ in arms:
            sec[f"{A_} vs {B_}"] = {"rmse_z": contrast(arms[A_], arms[B_], "rmse_z", -1, boot=False)}
    for lab in ("causal conv", "BiMamba", "BiGRU->conv"):
        A_, B_ = f"W12 frozen {lab}", f"W12 LoRA {lab}"
        if A_ in arms and B_ in arms:
            sec[f"{A_} vs {B_}"] = {m: contrast(arms[A_], arms[B_], m, s, boot=False) for m, s in (("rmse_z", -1), ("pcc", 1), ("rmse_centered", -1))}
    for lab in ("causal conv", "BiMamba", "BiGRU->conv"):
        A_ = f"W12 frozen {lab}"
        if A_ in arms and "W12 frozen MLP" in arms:
            sec[f"{A_} vs W12 frozen MLP"] = {m: contrast(arms[A_], arms["W12 frozen MLP"], m, s, boot=False) for m, s in (("rmse_z", -1), ("pcc", 1))}
    pairs = [("W12 frozen BiMamba", "W12 frozen BiGRU->conv"), ("W12 frozen BiMamba", "W12 frozen causal conv")]
    for A_, B_ in pairs:
        if A_ in arms and B_ in arms:
            sec[f"{A_} vs {B_}"] = {m: contrast(arms[A_], arms[B_], m, s, boot=False) for m, s in (("rmse_z", -1), ("pcc", 1))}

    # metric-definition agreement: rank the non-oracle arms under each definition
    plain = [a for a in rows if "gain" not in a]
    ranks = {}
    for m, sgn in (("rmse_z", 1), ("rmse_centered", 1), ("rmse_spk_centered", 1), ("pcc", -1), ("pcc_concat", -1)):
        ranks[m] = {a: int(i + 1) for i, a in enumerate(sorted(plain, key=lambda a: sgn * rows[a][m]))}
    taus = {f"{m1} vs {m2}": float(sstats.kendalltau([ranks[m1][a] for a in plain], [ranks[m2][a] for a in plain])[0])
            for m1, m2 in (("rmse_z", "rmse_centered"), ("rmse_z", "rmse_spk_centered"), ("pcc", "pcc_concat"))}

    res = {"rows": rows, "primary": prim_res, "secondary": sec, "shrink_gains": shrink_gains, "ranks": ranks,
           "kendall_tau": taus, "notes": notes}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_summary(res)


def _ci(r, key="t_ci"):
    return f"[{r[key][0]:+.4f}, {r[key][1]:+.4f}]"


def write_summary(res):
    rows, L = res["rows"], []
    L.append("# Rigor re-analysis (pre-registered in docs/rigor_prereg.md)\n")
    L.append("All numbers: held-out speaker (test_unseen), its own z units with full training-split statistics "
             "(oracle calibration), mean over 7 LOSO folds of seed means. Every configuration was already looked at "
             "on these folds (selected on the test folds).\n")
    L.append("## Pre-registered contrasts (+ = first arm better; Holm over 3 at alpha 0.05 on the sign-flip p)\n")
    L.append("| id | first arm | second arm | metric | mean diff | hier. bootstrap 95% | t 95% | folds won | sign-flip p | Holm | MDE (80%) | USC / 5EMO |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pid, r in res["primary"].items():
        L.append(f"| {pid} | {r['a']} | {r['b']} | {r['metric']} | {r['mean']:+.4f} | {_ci(r, 'boot_ci')} | {_ci(r)} | "
                 f"{r['won']}/{r['n_folds']} | {r['sign_flip_p']:.3f} | {'reject' if r['holm_reject'] else 'n.s.'} | "
                 f"{r['mde80']:.4f} | {r['usc']:+.4f} / {r['emo']:+.4f} |")
    L.append("\n## Every arm (descriptive)\n")
    L.append("| arm | seeds | rmse_z | PCC | PCC concat | centered rmse_z | bias share | spk-centered rmse_z | std ratio | seed sd rmse_z / PCC | fold sd rmse_z / PCC | smoother d rmse_z / PCC | enroll N=0 / 10 / 20 rmse_z |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, r in rows.items():
        sm = f"{r['smooth_d_rmse_z']:+.4f} / {r['smooth_d_pcc']:+.4f}" if "smooth_d_rmse_z" in r else "-"
        en = r.get("enroll")
        en = f"{en[0]:.2f} / {en[10]:.4f} / {en[20]:.4f}" if en else "-"
        L.append(f"| {a} | {r['seeds']} | {r['rmse_z']:.4f} | {r['pcc']:.4f} | {r['pcc_concat']:.4f} | {r['rmse_centered']:.4f} | "
                 f"{r['bias_share']:.3f} | {r['rmse_spk_centered']:.4f} | {r['std_ratio']:.3f} | "
                 f"{r['seed_sd_rmse_z']:.4f} / {r['seed_sd_pcc']:.4f} | {r['fold_sd_rmse_z']:.4f} / {r['fold_sd_pcc']:.4f} | {sm} | {en} |")
    L.append("\nShrinkage gains (mean, min, max over folds x seeds): " +
             "; ".join(f"{k}: cross-fold {v['scalar'][0]:.3f} ({v['scalar'][1]:.3f}-{v['scalar'][2]:.3f}), "
                       f"oracle {v['oracle'][0]:.3f} ({v['oracle'][1]:.3f}-{v['oracle'][2]:.3f})"
                       for k, v in res["shrink_gains"].items()))
    L.append("\n## Secondary contrasts (descriptive; paired over folds, t 95%, folds won, sign-flip p)\n")
    L.append("| contrast | metric | mean diff | t 95% | won | p |")
    L.append("|---|---|---|---|---|---|")
    for name, d in res["secondary"].items():
        for m, r in d.items():
            L.append(f"| {name} | {m} | {r['mean']:+.4f} | {_ci(r)} | {r['won']}/{r['n_folds']} | {r['sign_flip_p']:.3f} |")
    L.append("\n## Metric definitions: rank agreement over the non-gain arms (Kendall tau)\n")
    L.append(", ".join(f"{k}: {v:.2f}" for k, v in res["kendall_tau"].items()))
    L.append("\n" + "\n".join(f"- {n}" for n in res["notes"]))
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("probe-preds", "head-preds", "analyze", "tier2"))
    ap.add_argument("--model")
    ap.add_argument("--k", type=int)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--head")
    ap.add_argument("--fold")
    ap.add_argument("--shifts", default="", help="extra fixed shifts for probe-preds, e.g. 0,1")
    a = ap.parse_args()
    if a.cmd == "probe-preds":
        probe_preds(a.model, a.k, [int(s) for s in a.shifts.split(",") if s])
    elif a.cmd == "head-preds":
        folds = SPK if a.fold in (None, "all") else [a.fold]
        for h in folds:
            head_preds(a.model, a.layer, a.head, h)
    elif a.cmd == "tier2":
        tier2()
    else:
        analyze()


if __name__ == "__main__":
    main()
