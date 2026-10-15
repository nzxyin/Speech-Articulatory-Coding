"""Post-hoc analyses of the LOSO option-A checkpoints from cached held-out predictions.

`run` re-runs each fold's z-trained checkpoints (LoRA r8 + causal conv seeds 0-2, frozen causal-conv control) on the
held-out speaker's test sentences and caches per-utterance predictions (z units of that speaker) under
analysis_posthoc/preds/. Two CPU analyses then use the cache:

emotion   -- EMA_5EMO held-out speakers (jn, jr, kf): rmse_z / PCC per emotion (manifest 'emotion'), as trained and
             re-colored with the speaker's covariance (zca12; see loso_recolor).
enroll    -- how much of the new speaker's EMA is needed. The speaker's statistics (per normalization group: means,
             stds, and for re-coloring the z-correlation) are estimated from N random training-split utterances
             (N = 0 uses the training speakers' average statistics; 'all' = the full training split, as in every
             other result), predictions are mapped to mm with them and scored in the speaker's true z units
             (full-training-split statistics). 10 random draws per N.

Usage:
    python -m sparc.compression.loso_posthoc run --model wavlm-large --k 7
    python -m sparc.compression.loso_posthoc emotion
    python -m sparc.compression.loso_posthoc enroll
"""

import argparse
import json
from argparse import Namespace

import numpy as np
import torch

from . import datasets
from .metrics import ema_metrics
from .train import build, load_trainable, predict

OUT = datasets.XLSR_EMA / "analysis_posthoc"
RUNS = {"lora": ["prefix{k}_independent_r8_q_proj+v_proj_conv_causal_s%d" % s for s in (0, 1, 2)],
        "frozen": ["prefix{k}_none_conv_causal_s0"]}
ARMS = (("wavlm-large", 7), ("xlsr-300m", 18), ("xlsr-1b", 10), ("w2v2-large", 18))
NS = (0, 1, 2, 5, 10, 20, 50, "all")


def zmetrics(P, Y):
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    return {"rmse_z": float(np.sqrt(((Pc - Yc) ** 2).mean(0)).mean()), "pcc": ema_metrics(P, Y, n_boot=0)["pcc"]}


def run(model_name, k, device=None, run_tmpl=None):
    """run_tmpl: optional LoRA run-name template with {k} and {s} (seeds 0-2) instead of the option-A runs."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    runs_by_arm = RUNS if run_tmpl is None else {"lora": [run_tmpl.replace("{s}", str(s)) for s in (0, 1, 2)]}
    for h in datasets.ALL_SPEAKERS:
        name = f"ema_loso_{h}"
        ds = datasets.get(name)
        splits = ds.splits()
        Ytr = np.concatenate([ds.load_targets(s).ema_mm for s in splits["train"]])
        ym, ys = Ytr.mean(0), Ytr.std(0)
        test = [ds.load_utterance(s) for s in splits["test_unseen"]]
        for arm, runs in runs_by_arm.items():
            for run_name in [r.format(k=k) for r in runs]:
                out = OUT / "preds" / model_name / h / f"{run_name}.npz"
                d = datasets.adapt_root(name) / model_name / run_name
                if out.exists() or not (d / "results.json").exists():
                    continue
                r = json.loads((d / "results.json").read_text())
                args = Namespace(**r["config"])
                args.dataset, args.bf16 = name, device == "cuda"
                model, head, _ = build(args, device)
                load_trainable(model, head, torch.load(d / "best_trainable.pt"))
                preds, trues, _, kept = predict(model, head, test, args, ym, ys, device, return_utts=True)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, stems=np.array([u.stem for u in kept]), groups=np.array([u.norm_key for u in kept]),
                         preds=np.array(preds, dtype=object), trues=np.array(trues, dtype=object),
                         stored_rmse_z=r["test_unseen"]["rmse_z"])
                z = zmetrics([p.astype(np.float64) for p in preds], [t.astype(np.float64) for t in trues])
                print(f"{model_name} {h} {run_name}: rmse_z {z['rmse_z']:.4f} (stored {r['test_unseen']['rmse_z']:.4f})", flush=True)
                del model, head
                torch.cuda.empty_cache() if device == "cuda" else None


def load_preds(model_name, k, h, arm):
    out = []
    for run_name in [r.format(k=k) for r in RUNS[arm]]:
        f = OUT / "preds" / model_name / h / f"{run_name}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            out.append((list(z["stems"]), list(z["groups"]), [p.astype(np.float64) for p in z["preds"]],
                        [t.astype(np.float64) for t in z["trues"]]))
    return out


def zcorr(cov, sd):
    return cov / np.outer(sd, sd)


def recolor_maps(ds, groups):
    """A_h^-1 A_bar (zca12) per group, A_bar from the training groups' mean z-correlation."""
    train_groups = sorted({r["norm_group"] for r in ds.rows.values() if r["speaker"] not in ds.held_out})
    Cbar = np.mean([zcorr(ds._cov[g], ds._stats[g][1]) for g in train_groups], 0)
    Abar = datasets.target_matrices("zca12", Cbar, np.ones(12))[0]
    return {g: (datasets.target_matrices("zca12", zcorr(ds._cov[g], ds._stats[g][1]), np.ones(12))[1] @ Abar).T
            for g in set(groups)}, Abar, train_groups


def emotion():
    L = ["# Held-out EMA_5EMO speakers by emotion (LOSO option A checkpoints)\n",
         "rmse_z / PCC in the held-out speaker's z units; LoRA = mean of 3 seeds, frozen = 1 seed; 'recolor' = predictions "
         "re-colored with the speaker's covariance (zca12). n = test utterances per emotion (summed over jn, jr, kf).\n"]
    for model_name, k in ARMS:
        L.append(f"\n## {model_name} k={k}\n")
        L.append("| emotion | n | LoRA | LoRA recolor | frozen | frozen recolor |")
        L.append("|---|---|---|---|---|---|")
        acc = {}
        for h in ("5emo_jn", "5emo_jr", "5emo_kf"):
            ds = datasets.get(f"ema_loso_{h}")
            for arm in ("lora", "frozen"):
                for stems, groups, P, Y in load_preds(model_name, k, h, arm):
                    maps, _, _ = recolor_maps(ds, groups)
                    for s, g, p, y in zip(stems, groups, P, Y):
                        e = ds.rows[s]["emotion"] or "?"
                        acc.setdefault((e, arm), []).append((p, p @ maps[g], y))
        emos = sorted({e for e, _ in acc})
        for e in emos:
            cells = []
            for arm in ("lora", "frozen"):
                it = acc.get((e, arm), [])
                if not it:
                    cells += ["-", "-"]
                    continue
                a = zmetrics([p for p, _, _ in it], [y for _, _, y in it])
                b = zmetrics([q for _, q, _ in it], [y for _, _, y in it])
                cells += [f"{a['rmse_z']:.3f} / {a['pcc']:.3f}", f"{b['rmse_z']:.3f} / {b['pcc']:.3f}"]
            n = len(acc.get((e, "frozen"), []))
            L.append(f"| {e} | {n} | " + " | ".join(cells) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "emotion.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def enroll(draws=10):
    L = ["# Enrollment curve: new-speaker statistics from N utterances (LOSO option A LoRA, 3 seeds)\n",
         "Held-out rmse_z / PCC in the speaker's true z units, mean over 7 folds (and draws). z-score = means/stds from N "
         "utterances; recolor = means/stds and zca12 covariance from N. N = 0: training speakers' average statistics.\n"]
    rng = np.random.default_rng(0)
    res = {}
    for model_name, k in ARMS:
        per_n = {n: {"z": [], "rc": []} for n in NS}
        for h in datasets.ALL_SPEAKERS:
            ds = datasets.get(f"ema_loso_{h}")
            runs = load_preds(model_name, k, h, "lora")
            if not runs:
                continue
            _, Abar, train_groups = recolor_maps(ds, runs[0][1])
            pop_mean = np.mean([ds._stats[g][0] for g in train_groups], 0)
            pop_sd = np.mean([ds._stats[g][1] for g in train_groups], 0)
            frames = {}
            for r in ds.rows.values():
                if r["speaker"] == h and r["_split"] == "train":
                    e = np.load(ds.root / r["ema"])["ema"][r["_first"]:r["_last"]].astype(np.float64)
                    frames.setdefault(r["norm_group"], []).append(e)
            for n in NS:
                zs, rcs = [], []
                for _ in range(1 if n in (0, "all") else draws):
                    est = {}
                    for g, utts in frames.items():
                        if n == 0:
                            est[g] = (pop_mean, pop_sd, None)
                            continue
                        pick = utts if n == "all" else [utts[i] for i in rng.choice(len(utts), min(n, len(utts)), replace=False)]
                        X = np.concatenate(pick)
                        m, sd = X.mean(0), X.std(0) + 1e-6
                        C = np.corrcoef(X.T) if len(X) > 12 else np.eye(12)
                        est[g] = (m, sd, C)
                    for stems, groups, P, Y in runs:
                        Pz, Pr = [], []
                        for g, p in zip(groups, P):
                            m_all, sd_all = ds._stats[g]
                            m, sd, C = est[g]
                            Pz.append((p * sd + m - m_all) / sd_all)
                            if C is not None:
                                Ainv = datasets.target_matrices("zca12", C + 1e-6 * np.eye(12), np.ones(12))[1]
                                q = p @ (Ainv @ Abar).T
                            else:
                                q = p
                            Pr.append((q * sd + m - m_all) / sd_all)
                        zs.append(zmetrics(Pz, Y))
                        rcs.append(zmetrics(Pr, Y))
                per_n[n]["z"].append({m: float(np.mean([x[m] for x in zs])) for m in ("rmse_z", "pcc")})
                per_n[n]["rc"].append({m: float(np.mean([x[m] for x in rcs])) for m in ("rmse_z", "pcc")})
        res[model_name] = per_n
        L.append(f"\n## {model_name} k={k}\n")
        L.append("| N utterances | z-score rmse_z / PCC | recolor rmse_z / PCC |")
        L.append("|---|---|---|")
        for n in NS:
            if not per_n[n]["z"]:
                continue
            z = {m: np.mean([x[m] for x in per_n[n]["z"]]) for m in ("rmse_z", "pcc")}
            rc = {m: np.mean([x[m] for x in per_n[n]["rc"]]) for m in ("rmse_z", "pcc")}
            L.append(f"| {n} | {z['rmse_z']:.4f} / {z['pcc']:.4f} | {rc['rmse_z']:.4f} / {rc['pcc']:.4f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "enroll.md").write_text("\n".join(L) + "\n")
    (OUT / "enroll.json").write_text(json.dumps(res, indent=1, default=str))
    print("\n".join(L))


def phones():
    """MNGU0 per-phone-class test RMSE (mm; the multi-speaker corpora have no phone alignments)."""
    mk = (("wavlm-large", 9), ("xlsr-300m", 18), ("xlsr-1b", 16), ("w2v2-large", 18))
    L = ["# MNGU0 test RMSE by phone class (mm; LoRA + causal conv, mean of 3 seeds; frozen causal conv 1 seed)\n",
         "Phone classes from the MNGU0 forced-alignment labels (place of articulation / vowel / silence). The multi-speaker "
         "corpora have no phone alignments, so this is single-speaker only.\n"]
    table = {}
    for m, k in mk:
        for arm, pattern, root in (("LoRA", f"prefix{k}_independent_r8_q_proj+v_proj_conv_causal_s{{s}}", "mngu0"),
                                   ("LoRA, whitened targets", f"prefix{k}_independent_r8_q_proj+v_proj_conv_causal_s{{s}}", "mngu0_zca12"),
                                   ("frozen", f"prefix{k}_none_conv_causal_s0", "mngu0")):
            rs = []
            for s in (0, 1, 2):
                f = datasets.adapt_root(root) / m / pattern.format(s=s) / "results.json"
                if f.exists():
                    rs.append(json.loads(f.read_text())["test"])
            if not rs:
                continue
            classes = sorted(rs[0]["per_phone_class"])
            table[(m, arm)] = {c: (np.mean([r["per_phone_class"][c]["rmse"] for r in rs]), rs[0]["per_phone_class"][c]["n_frames"])
                               for c in classes}
            table[(m, arm)]["all"] = (np.mean([r["rmse"] for r in rs]), rs[0]["n_frames"])
    classes = sorted({c for v in table.values() for c in v if c != "all"}) + ["all"]
    L.append("| model | arm | " + " | ".join(f"{c} (n={next(iter(table.values()))[c][1]})" for c in classes) + " |")
    L.append("|---|---|" + "---|" * len(classes))
    for (m, arm), v in table.items():
        L.append(f"| {m} | {arm} | " + " | ".join(f"{v[c][0]:.3f}" if c in v else "-" for c in classes) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phones.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "emotion", "enroll", "phones"))
    ap.add_argument("--model")
    ap.add_argument("--k", type=int)
    ap.add_argument("--run", default=None, help="LoRA run-name template with {k} and {s} (run only)")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.model, args.k, run_tmpl=args.run)
    elif args.cmd == "emotion":
        emotion()
    elif args.cmd == "phones":
        phones()
    else:
        enroll()


if __name__ == "__main__":
    main()
