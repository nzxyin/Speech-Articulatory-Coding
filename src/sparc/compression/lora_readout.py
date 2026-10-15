"""Can a LoRA-adapted encoder keep its PCC gain without losing the probe's held-out rmse_z?

Under leave-one-speaker-out, LoRA + causal conv has the best held-out PCC but a plain ridge probe on the frozen encoder
has the best held-out rmse_z. For each fold and LoRA seed (option-A checkpoints) this compares, on the held-out
speaker's test sentences (rmse_z / PCC in its z units):
    lora        -- the checkpoint as trained (LoRA encoder + causal conv head);
    probe       -- ridge on the frozen encoder's cached features (same layer and shift);
    lora_ridge  -- ridge refit on the LoRA-adapted encoder's (low-passed) last-layer features;
    blend       -- w * lora + (1 - w) * probe, w in {0, .25, .5, .75, 1} chosen on the training speakers' validation.
Ridge strengths and w are chosen on the fold's validation sentences (training speakers only).

Usage:
    python -m sparc.compression.lora_readout run --model wavlm-large --k 7 --fold usc_M1 [--run TEMPLATE --tag NAME]
    python -m sparc.compression.lora_readout summary
"""

import argparse
import json
from argparse import Namespace

import numpy as np
import torch
from scipy import stats as sstats

from . import datasets
from .encoders import normalize_wav
from .extract import lowpass
from .metrics import ema_metrics
from .probe import ALPHAS, Ridge, aligned
from .train import build, load_trainable, predict

OUT = datasets.XLSR_EMA / "analysis_readout"
RUN = "prefix{k}_independent_r8_q_proj+v_proj_conv_causal_s{s}"
ARMS = (("wavlm-large", 7), ("xlsr-300m", 18), ("xlsr-1b", 10), ("w2v2-large", 18))
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def zm(P, Y):
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    return {"rmse_z": float(np.sqrt(((Pc - Yc) ** 2).mean(0)).mean()), "pcc": ema_metrics(P, Y, n_boot=0)["pcc"]}


def fit_ridge(Xtr, Ytr, Xva, Yva, device):
    r = Ridge(Xtr, Ytr, device)
    best = None
    for a in ALPHAS:
        W, b = r.weights(a)
        e = zm(r.predict(W, b, Xva), Yva)["rmse_z"]
        if best is None or e < best[0]:
            best = (e, W, b)
    return r, best[1], best[2]


@torch.no_grad()
def encode(model, utts, device, bf16):
    out = {}
    for u in utts:
        x = torch.from_numpy(normalize_wav(u.wav)).unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            h = model(x).last_hidden_state[0]
        out[u.stem] = lowpass(h.float().cpu().numpy()).astype(np.float32)
    return out


def run(model_name, k, fold, device=None, run_tmpl=RUN, tag=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = f"ema_loso_{fold}"
    ds = datasets.get(name)
    splits = ds.splits()
    cache = datasets.features_root("ema_multi") / model_name
    index = json.loads((cache / "index.json").read_text())
    feats = np.load(cache / f"layer_{k:02d}.npy", mmap_mode="r").astype(np.float32)
    sp = {s: [x for x in splits[s] if x in index] for s in ("train", "valid", "test", "test_unseen")}
    utts = {s: ds.load_utterance(s) for v in sp.values() for s in v}
    Ytr_all = np.concatenate([utts[s].ema_mm for s in sp["train"]])
    ym, ys = Ytr_all.mean(0), Ytr_all.std(0)
    out_f = OUT / model_name / (f"{fold}.json" if tag is None else f"{tag}/{fold}.json")
    out_f.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    for s in (0, 1, 2):
        run_name = run_tmpl.format(k=k, s=s)
        d = datasets.adapt_root(name) / model_name / run_name
        if run_name in res or not (d / "results.json").exists():
            continue
        r = json.loads((d / "results.json").read_text())
        args = Namespace(**r["config"])
        args.dataset, args.bf16 = name, device == "cuda"
        shift = args.shift
        # probe on the frozen encoder's cached features (same layer, shift)
        P = {split: aligned(feats, index, utts, sp[split], shift) for split in sp}
        pr, pW, pb = fit_ridge(P["train"][0], P["train"][1], P["valid"][0], P["valid"][1], device)
        model, head, _ = build(args, device)
        load_trainable(model, head, torch.load(d / "best_trainable.pt"))
        # LoRA head predictions on valid / test splits
        lp = {}
        for split in ("valid", "test", "test_unseen"):
            preds, trues, _, kept = predict(model, head, [utts[x] for x in sp[split]], args, ym, ys, device, return_utts=True)
            lp[split] = {u.stem: p.astype(np.float64) for u, p in zip(kept, preds)}
        # ridge on the LoRA encoder's own features
        lf = encode(model, list(utts.values()), device, args.bf16)
        idx2, off = {}, 0
        stems = list(lf)
        for st in stems:
            idx2[st] = (off, len(lf[st]))
            off += len(lf[st])
        F = np.concatenate([lf[st] for st in stems])
        L = {split: aligned(F, idx2, utts, sp[split], shift) for split in sp}
        lr_, lW, lb = fit_ridge(L["train"][0], L["train"][1], L["valid"][0], L["valid"][1], device)
        entry = {}
        for split in ("test", "test_unseen"):
            X, Y, keep = P[split]
            probe_p = [p.astype(np.float64) for p in pr.predict(pW, pb, X)]
            Xl, Yl, keepl = L[split]
            lr_p = {st: p.astype(np.float64) for st, p in zip(keepl, lr_.predict(lW, lb, Xl))}
            ys_ = [y.astype(np.float64) for y in Y]
            lora_p, lrr, blend_inputs = [], [], []
            for st, pp, y in zip(keep, probe_p, ys_):
                q = lp[split].get(st)
                n = min(len(pp), len(y), len(q) if q is not None else len(pp), len(lr_p.get(st, pp)))
                lora_p.append((q if q is not None else pp)[:n])
                lrr.append(lr_p.get(st, pp)[:n])
                blend_inputs.append((pp[:n], y[:n]))
            yy = [y for _, y in blend_inputs]
            pp_ = [p for p, _ in blend_inputs]
            entry[split] = {"lora": zm(lora_p, yy), "probe": zm(pp_, yy), "lora_ridge": zm(lrr, yy),
                            "_blend_parts": (lora_p, pp_, yy)}
        # blend weight on validation
        Xv, Yv, keepv = P["valid"]
        pv = [p.astype(np.float64) for p in pr.predict(pW, pb, Xv)]
        lv, pv2, yv = [], [], []
        for st, pp, y in zip(keepv, pv, Yv):
            q = lp["valid"].get(st)
            if q is None:
                continue
            n = min(len(pp), len(q), len(y))
            lv.append(q[:n]); pv2.append(pp[:n]); yv.append(y[:n].astype(np.float64))
        wbest = min(WEIGHTS, key=lambda w: zm([w * a + (1 - w) * b for a, b in zip(lv, pv2)], yv)["rmse_z"])
        rec = {"shift": shift, "blend_w": wbest}
        for split in ("test", "test_unseen"):
            lora_p, pp_, yy = entry[split].pop("_blend_parts")
            entry[split]["blend"] = zm([wbest * a + (1 - wbest) * b for a, b in zip(lora_p, pp_)], yy)
            rec[split] = entry[split]
        rec["stored_lora_unseen_rmse_z"] = r["test_unseen"]["rmse_z"]
        res[run_name] = rec
        out_f.write_text(json.dumps(res, indent=1))
        u = rec["test_unseen"]
        print(f"{model_name} {fold} s{s}: held-out rmse_z/PCC lora {u['lora']['rmse_z']:.4f}/{u['lora']['pcc']:.4f} "
              f"(stored {r['test_unseen']['rmse_z']:.4f}) probe {u['probe']['rmse_z']:.4f}/{u['probe']['pcc']:.4f} "
              f"lora_ridge {u['lora_ridge']['rmse_z']:.4f}/{u['lora_ridge']['pcc']:.4f} blend(w={wbest}) "
              f"{u['blend']['rmse_z']:.4f}/{u['blend']['pcc']:.4f}", flush=True)
        del model, head
        torch.cuda.empty_cache() if device == "cuda" else None


def summary():
    L = ["# LoRA + readout variants under LOSO (held-out speaker; seed mean per fold, then mean over folds)\n",
         "Paired vs probe over folds: mean [95% t CI] folds better; + = better (lower rmse_z, higher PCC).\n"]
    for model_name, k in ARMS:
        per = {}
        for h in datasets.ALL_SPEAKERS:
            f = OUT / model_name / f"{h}.json"
            if not f.exists():
                continue
            res = json.loads(f.read_text())
            if not res:
                continue
            per[h] = {v: [np.mean([r["test_unseen"][v][m] for r in res.values()]) for m in ("rmse_z", "pcc")]
                      for v in ("lora", "probe", "lora_ridge", "blend")}
            per[h]["seen"] = {v: [np.mean([r["test"][v][m] for r in res.values()]) for m in ("rmse_z", "pcc")]
                              for v in ("lora", "probe", "lora_ridge", "blend")}
            per[h]["w"] = [r["blend_w"] for r in res.values()]
        if not per:
            continue
        hs = list(per)
        L.append(f"\n## {model_name} k={k} ({len(hs)} folds)\n")
        L.append("| variant | held-out rmse_z | held-out PCC | vs probe rmse_z | vs probe PCC | seen rmse_z / PCC |")
        L.append("|---|---|---|---|---|---|")
        for v in ("probe", "lora", "lora_ridge", "blend"):
            r_ = np.mean([per[h][v][0] for h in hs])
            p_ = np.mean([per[h][v][1] for h in hs])
            cells = []
            for i, sign in ((0, -1), (1, 1)):
                dd = np.array([sign * (per[h][v][i] - per[h]["probe"][i]) for h in hs])
                if v == "probe" or len(dd) < 2:
                    cells.append("-")
                else:
                    hw = sstats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
                    cells.append(f"{dd.mean():+.4f} [{dd.mean() - hw:+.4f}, {dd.mean() + hw:+.4f}] {(dd > 0).sum()}/{len(dd)}")
            seen = f"{np.mean([per[h]['seen'][v][0] for h in hs]):.4f} / {np.mean([per[h]['seen'][v][1] for h in hs]):.4f}"
            L.append(f"| {v} | {r_:.4f} | {p_:.4f} | {cells[0]} | {cells[1]} | {seen} |")
        L.append(f"\nBlend weights chosen (LoRA share): {sorted(w for h in hs for w in per[h]['w'])}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary"))
    ap.add_argument("--model")
    ap.add_argument("--k", type=int)
    ap.add_argument("--run", default=RUN, help="LoRA run-name template with {k} and {s}")
    ap.add_argument("--tag", default=None, help="write to <model>/<tag>/<fold>.json (non-default run templates)")
    ap.add_argument("--fold")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.model, args.k, args.fold, run_tmpl=args.run, tag=args.tag)
    else:
        summary()


if __name__ == "__main__":
    main()
