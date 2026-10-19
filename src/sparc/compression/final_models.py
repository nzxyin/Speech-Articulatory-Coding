"""Deployable WavLM Large k12 + BiGRU -> conv 9 inverters (the recommended configuration) with saved checkpoints.

frozen     -- frozen encoder + head trained on the cached, zero-phase 10 Hz low-passed layer-12 features, recipe as
              head_screen (alignment shift chosen by ridge on validation; head learning rate from LRS on validation
              with seed 0, then seeds 1-2; smoother off). Datasets:
                mngu0  single speaker (MNGU0 day 1; targets in mm, also scored in z units of the training split)
                all7   all seven multi-speaker speakers (USC-TIMIT + EMA_5EMO; per-group z targets), scored on their
                       test sentences overall and per speaker (seen speakers: no held-out speaker here)
              Checkpoints: <OUT>/<dataset>/frozen_k12_gruconv9_s<seed>.pt with the head's state_dict and everything
              needed to run it (layer, shift, target mean / std, low-pass setting).
lora-lr    -- MNGU0 LoRA + BiGRU -> conv in the matched (frozen-head) pipeline, seed 0 at one LoRA learning rate; head
              learning rate and shift from the frozen MNGU0 result.
lora-seeds -- picks the LoRA learning rate on validation and runs seeds 1-2 (train.py writes best_trainable.pt).
summary    -- table of everything above.

Usage:
    python -m sparc.compression.final_models frozen --dataset mngu0
    python -m sparc.compression.final_models lora-lr --lr 5e-4
    python -m sparc.compression.final_models lora-seeds
    python -m sparc.compression.final_models summary
"""

import argparse
import json

import numpy as np

from . import datasets

OUT = datasets.XLSR_EMA / "final_models"
MODEL, K, HEAD = "wavlm-large", 12, "gruconv9"
SEEDS = (0, 1, 2)
LORA_LRS = (2e-4, 5e-4, 1e-3)


def _data(dataset):
    from .probe import load_targets

    if dataset == "mngu0":
        ds = datasets.get("mngu0")
        root = datasets.features_root("mngu0") / MODEL
    else:
        ds = datasets.EMAMulti(held_out=(), norm_by="norm_group")
        root = datasets.features_root("ema_multi") / MODEL
    index = json.loads((root / "index.json").read_text())
    feats = np.load(root / f"layer_{K:02d}.npy", mmap_mode="r")
    sp = {s: [x for x in v if x in index] for s, v in ds.splits().items() if s in ("train", "valid", "test")}
    utts = load_targets(sorted({x for v in sp.values() for x in v}), ds)
    return ds, feats, index, sp, utts


def _scores(P, Y, ys_train):
    from .head_screen import zm

    r = zm(P, Y)
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    rmse_c = np.sqrt(((Pc - Yc) ** 2).mean(0))
    return {"rmse": r["rmse_z"], "rmse_z": float((rmse_c / ys_train).mean()), "pcc": r["pcc"]}


def frozen(dataset, device=None, lrs=None, tag=""):
    """lrs: learning-rate grid (default head_screen.LRS); tag: suffix for the result / checkpoint files."""
    import torch

    from .head_screen import LRS, choose_shift, predict, train_one
    from .probe import aligned

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds, feats, index, sp, utts = _data(dataset)
    out_d = OUT / dataset
    out_d.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    lrs = tuple(lrs) if lrs else LRS
    res_f = out_d / f"frozen_k{K}_{HEAD}{sfx}.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    if "shift" not in res:
        res["shift"] = int(choose_shift(feats, index, utts, sp, device, shifts=range(-2, 5)))
        res_f.write_text(json.dumps(res, indent=1))
    shift = res["shift"]
    data, keep = {}, {}
    for split in ("train", "valid", "test"):
        X, Y, kp = aligned(feats, index, utts, sp[split], shift)
        data[split] = ([np.asarray(x, dtype=np.float32) for x in X], [y.astype(np.float64) for y in Y])
        keep[split] = kp
    Yall = np.concatenate(data["train"][1])
    ym, ys = Yall.mean(0), Yall.std(0)

    def finish(head, v, s, lr):
        P = predict(head, data["test"][0], ym, ys, device)
        rec = {"valid_rmse": v, "test": _scores(P, data["test"][1], ys)}
        if dataset == "all7":
            per = {}
            for spk in sorted({ds.rows[st]["speaker"] for st in keep["test"]}):
                idx = [i for i, st in enumerate(keep["test"]) if ds.rows[st]["speaker"] == spk]
                per[spk] = _scores([P[i] for i in idx], [data["test"][1][i] for i in idx], ys)
            rec["test_per_speaker"] = per
        torch.save({"state_dict": {k: t.cpu() for k, t in head.state_dict().items()}, "head": HEAD, "dim": 1024,
                    "model": MODEL, "layer": K, "shift": shift, "lr": lr, "seed": s, "ym": ym, "ys": ys,
                    "features": "hidden_states[12], zero-phase 10 Hz low-pass over the whole utterance (extract.lowpass)",
                    "targets": "mm (MNGU0)" if dataset == "mngu0" else "per-normalization-group z (multi-speaker)",
                    "make_head": "sparc.compression.head_screen.make_head('gruconv9', 1024, smooth=False)"},
                   out_d / f"frozen_k{K}_{HEAD}{sfx}_s{s}.pt")
        return rec

    if "lr" not in res:
        trials, best = {}, None
        for lr in lrs:
            head, v = train_one(HEAD, False, lr, data, ym, ys, device, seed=0)
            trials[str(lr)] = v
            if head is not None and (best is None or v < best[0]):
                best = (v, lr, head)
        res.update({"lr": best[1], "lr_valid": trials, "seeds": {"0": finish(best[2], best[0], 0, best[1])}})
        res_f.write_text(json.dumps(res, indent=1))
    for s in SEEDS[1:]:
        if str(s) in res["seeds"]:
            continue
        head, v = train_one(HEAD, False, res["lr"], data, ym, ys, device, seed=s)
        res["seeds"][str(s)] = finish(head, v, s, res["lr"]) if head is not None else {"failed": True}
        res_f.write_text(json.dumps(res, indent=1))
    ok = [x["test"] for x in res["seeds"].values() if "failed" not in x]
    print(f"{dataset} frozen k{K} {HEAD}: shift {shift} lr {res['lr']} test " +
          " ".join(f"{m} {np.mean([x[m] for x in ok]):.4f}" for m in ("rmse", "rmse_z", "pcc")), flush=True)


def lora_run_name(lr, s):
    return f"prefix{K}_independent_r8_q_proj+v_proj_gruconv_k9_nosm_m{lr:g}_s{s}"


def lora_train(lr, seed):
    from .train import main as train_main

    fr = json.loads((OUT / "mngu0" / f"frozen_k{K}_{HEAD}.json").read_text())
    train_main(["--model", MODEL, "--dataset", "mngu0", "--layers", f"1-{K}", "--variant", "independent",
                "--rank", "8", "--targets", "q_proj,v_proj", "--head", "gruconv", "--kernel", "9", "--no-smooth",
                "--input-lowpass", "--no-enc-dropout", "--no-bf16", "--shift", str(fr["shift"]),
                "--head-lr", str(fr["lr"]), "--lr", str(lr), "--seed", str(seed), "--tag", f"m{lr:g}"])


def _lora_result(lr, s):
    f = datasets.adapt_root("mngu0") / MODEL / lora_run_name(lr, s) / "results.json"
    return json.loads(f.read_text()) if f.exists() else None


def lora_seeds():
    scores = {}
    for lr in LORA_LRS:
        r = _lora_result(lr, 0)
        if r is None:
            raise RuntimeError(f"seed-0 LoRA run missing for lr {lr:g}")
        scores[lr] = r["valid"].get("rmse", r["valid"].get("rmse_z"))
    lr = min(scores, key=scores.get)
    (OUT / "mngu0").mkdir(parents=True, exist_ok=True)
    (OUT / "mngu0" / "lora_choice.json").write_text(json.dumps({"lr": lr, "valid": {f"{k:g}": v for k, v in scores.items()}}))
    print(f"MNGU0 LoRA lr {lr:g} (valid {scores})", flush=True)
    for s in (1, 2):
        lora_train(lr, s)


def summary():
    L = ["# Deployable WavLM Large k12 + BiGRU -> conv 9 (3 seeds; test sets)\n"]
    for dataset in ("mngu0", "all7"):
        f = OUT / dataset / f"frozen_k{K}_{HEAD}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        ok = [x for x in r["seeds"].values() if "failed" not in x]
        m = {k: (np.mean([x["test"][k] for x in ok]), np.std([x["test"][k] for x in ok])) for k in ("rmse", "rmse_z", "pcc")}
        unit = "mm" if dataset == "mngu0" else "z"
        L.append(f"## {dataset}, frozen (shift {r['shift']}, head lr {r['lr']}, {len(ok)} seeds)\n")
        L.append(f"RMSE {m['rmse'][0]:.4f} +/- {m['rmse'][1]:.4f} {unit}; rmse_z {m['rmse_z'][0]:.4f} +/- {m['rmse_z'][1]:.4f}; "
                 f"PCC {m['pcc'][0]:.4f} +/- {m['pcc'][1]:.4f}\n")
        if dataset == "all7":
            L.append("| speaker | rmse_z | PCC |")
            L.append("|---|---|---|")
            for spk in ok[0]["test_per_speaker"]:
                L.append(f"| {spk} | {np.mean([x['test_per_speaker'][spk]['rmse_z'] for x in ok]):.4f} | "
                         f"{np.mean([x['test_per_speaker'][spk]['pcc'] for x in ok]):.4f} |")
            L.append("")
    ch = OUT / "mngu0" / "lora_choice.json"
    if ch.exists():
        lr = json.loads(ch.read_text())["lr"]
        rs = [r for r in (_lora_result(lr, s) for s in SEEDS) if r]
        if rs:
            t = [r["test"] for r in rs]
            L.append(f"## mngu0, LoRA matched pipeline (LoRA lr {lr:g}, {len(rs)} seeds)\n")
            L.append(f"RMSE {np.mean([x['rmse'] for x in t]):.4f} +/- {np.std([x['rmse'] for x in t]):.4f} mm; "
                     f"PCC {np.mean([x['pcc'] for x in t]):.4f} +/- {np.std([x['pcc'] for x in t]):.4f}\n")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("frozen", "lora-lr", "lora-seeds", "summary"))
    ap.add_argument("--dataset", choices=("mngu0", "all7"))
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lrs", default="", help="frozen: comma-separated learning-rate grid (default head_screen's)")
    ap.add_argument("--tag", default="", help="frozen: suffix for result and checkpoint files")
    a = ap.parse_args()
    if a.cmd == "frozen":
        frozen(a.dataset, lrs=[float(x) for x in a.lrs.split(",") if x], tag=a.tag)
    elif a.cmd == "lora-lr":
        lora_train(a.lr, 0)
    elif a.cmd == "lora-seeds":
        lora_seeds()
    else:
        summary()


if __name__ == "__main__":
    main()
