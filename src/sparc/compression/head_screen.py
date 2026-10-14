"""Frozen-encoder head screen under leave-one-speaker-out: does a head with a different inductive bias beat spending
the same FLOPs on more encoder layers?

Heads are trained on the cached, low-passed encoder features of one layer (features_ema_multi/<model>/layer_kk.npy)
for every LOSO fold (training speakers' train split; early stopping and learning-rate choice on their validation
split, never on the held-out speaker). Heads (heads.Head and seqheads):
    linear            frame-wise readout (input standardized with training statistics);
    conv_causal9      the causal 9-frame conv head used so far (bridge to the earlier runs);
    conv9, conv17     centered (non-causal) conv heads, learned FIR filters;
    gru               bidirectional GRU;
    ssm               bidirectional diagonal linear SSM (S4D);
    mamba             bidirectional selective SSM (reference scan).
Each with and without the ±1 s output smoother. Learning rate in {3e-4, 1e-3} (AdamW, cosine with one warm-up epoch,
<= 40 epochs, patience 8), chosen per fold on the training speakers' validation rmse_z. Alignment shift: the most
common fold-internal shift of the depth sweep for that layer. Head FLOPs per second of audio are measured with
FlopCounterMode on a 4 s input (matmul / conv only, like the encoder profiles).

Usage:
    python -m sparc.compression.head_screen run --model wavlm-large --layer 7 --head gru
    python -m sparc.compression.head_screen summary
"""

import argparse
import json
import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from . import datasets
from .heads import Head
from .metrics import ema_metrics
from .probe import aligned, load_targets
from .seqheads import GRUHead, MambaHead, SSMHead

OUT = datasets.XLSR_EMA / "analysis_heads"
HEADS = ("linear", "conv_causal9", "conv9", "conv17", "gru", "ssm", "mamba")
LRS = (3e-4, 1e-3)
FT_SR = 50


def make_head(name, dim, smooth):
    if name == "linear":
        return Head(dim, "linear", smooth=smooth)
    if name == "conv_causal9":
        return Head(dim, "conv", hidden=256, kernel=9, causal=True, smooth=smooth)
    if name in ("conv9", "conv17"):
        return Head(dim, "conv", hidden=256, kernel=int(name[4:]), causal=False, smooth=smooth)
    if name == "gru":
        return GRUHead(dim, smooth=smooth)
    if name == "ssm":
        return SSMHead(dim, smooth=smooth)
    if name == "mamba":
        return MambaHead(dim, smooth=smooth)
    raise ValueError(name)


def head_gflops(name, dim):
    h = make_head(name, dim, smooth=False).eval()
    T = 4 * FT_SR
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        h(torch.zeros(1, T, dim))
    return fc.get_total_flops() / 4 / 1e9


def batches(X, Y, order, bs):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        T = max(len(X[j]) for j in idx)
        xb = torch.zeros(len(idx), T, X[idx[0]].shape[1])
        yb = torch.zeros(len(idx), T, 12)
        mask = torch.ones(len(idx), T, dtype=torch.bool)
        for b, j in enumerate(idx):
            n = len(X[j])
            xb[b, :n] = torch.from_numpy(np.asarray(X[j], dtype=np.float32))
            yb[b, :n] = torch.from_numpy(np.asarray(Y[j], dtype=np.float32))
            mask[b, :n] = False
        yield xb, yb, mask


@torch.no_grad()
def predict(head, X, ym, ys, device, bs=16):
    head.eval()
    out = []
    for xb, _, mask in batches(X, [np.zeros((len(x), 12)) for x in X], list(range(len(X))), bs):
        y = head(xb.to(device), mask.to(device)).float().cpu().numpy()
        for b in range(len(y)):
            n = int((~mask[b]).sum())
            out.append(y[b, :n] * ys + ym)
    return out


def zm(P, Y):
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    return {"rmse_z": float(np.sqrt(((Pc - Yc) ** 2).mean(0)).mean()), "pcc": ema_metrics(P, Y, n_boot=0)["pcc"]}


def train_one(name, smooth, lr, data, ym, ys, device, seed=0, epochs=40, patience=8, bs=8):
    torch.manual_seed(seed)
    random.seed(seed)
    Xtr, Ytr = data["train"]
    dim = Xtr[0].shape[1]
    head = make_head(name, dim, smooth).to(device)
    if name == "linear":
        F_ = np.concatenate(Xtr)
        head.set_input_stats(torch.from_numpy(F_.mean(0)), torch.from_numpy(F_.std(0)))
    Ytr_s = [(y - ym) / ys for y in Ytr]
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    steps = math.ceil(len(Xtr) / bs)
    total = epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / steps) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        head.train()
        order = list(range(len(Xtr)))
        random.shuffle(order)
        for xb, yb, mask in batches(Xtr, Ytr_s, order, bs):
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            y = head(xb, mask)
            valid = (~mask).float()[..., None]
            loss = (((y - yb) ** 2) * valid).sum() / (valid.sum() * 12)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
        v = zm(predict(head, data["valid"][0], ym, ys, device), data["valid"][1])["rmse_z"]
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.detach().clone() for k, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return head, best


def shift_for(model, layer):
    from .depth_sweep import load_all
    res = load_all(model)
    sh = [res[f"{layer}/{h}"]["shift"] for h in datasets.ALL_SPEAKERS if f"{layer}/{h}" in res]
    return Counter(sh).most_common(1)[0][0] if sh else 1


def run(model, layer, head_name, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = datasets.features_root("ema_multi") / model
    index = json.loads((d / "index.json").read_text())
    feats = np.load(d / f"layer_{layer:02d}.npy", mmap_mode="r")
    shift = shift_for(model, layer)
    out_f = OUT / model / f"k{layer}_{head_name}.json"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out_f.read_text()) if out_f.exists() else {"shift": shift}
    utts = None
    for h in datasets.ALL_SPEAKERS:
        if all(f"{h}/smooth{int(s)}" in res for s in (True, False)):
            continue
        ds = datasets.EMAMulti(held_out=(h,), norm_by="norm_group")
        sp = {k: [s for s in v if s in index] for k, v in ds.splits().items()}
        if utts is None:
            utts = load_targets(sorted(set(index) & set(ds.rows)), ds)  # targets do not depend on the fold
        data = {}
        for split in ("train", "valid", "test", "test_unseen"):
            X, Y, _ = aligned(feats, index, utts, sp[split], shift)
            data[split] = ([np.asarray(x, dtype=np.float32) for x in X], [y.astype(np.float64) for y in Y])
        Yall = np.concatenate(data["train"][1])
        ym, ys = Yall.mean(0), Yall.std(0)
        for smooth in (True, False):
            key = f"{h}/smooth{int(smooth)}"
            if key in res:
                continue
            best = None
            for lr in LRS:
                head, v = train_one(head_name, smooth, lr, data, ym, ys, device)
                if best is None or v < best[0]:
                    best = (v, lr, head)
            v, lr, head = best
            rec = {"lr": lr, "valid_rmse_z": v}
            for split in ("test", "test_unseen"):
                rec[split] = zm(predict(head, data[split][0], ym, ys, device), data[split][1])
            res[key] = rec
            out_f.write_text(json.dumps(res, indent=1))
            print(f"{model} k{layer} {head_name} {key}: lr {lr} held-out {rec['test_unseen']['rmse_z']:.4f} / "
                  f"{rec['test_unseen']['pcc']:.4f} seen {rec['test']['rmse_z']:.4f} / {rec['test']['pcc']:.4f}", flush=True)


def summary():
    from .depth_sweep import gflops, load_all
    L = ["# Frozen-encoder head screen (LOSO, 7 folds; held-out and seen rmse_z / PCC)\n",
         "Total GFLOPs/s = encoder prefix + head (FlopCounterMode, matmul/conv only). 'probe' = the depth sweep's ridge "
         "readout at the same layer. Paired vs the linear head over folds: mean, folds better (+ = better).\n"]
    rows = []
    for mdir in sorted(OUT.glob("*")):
        if not mdir.is_dir():
            continue
        model = mdir.name
        enc = gflops(model)
        depth = load_all(model)
        layers = sorted({int(f.stem.split("_")[0][1:]) for f in mdir.glob("k*_*.json")})
        dim = None
        for layer in layers:
            L.append(f"\n## {model} k={layer} (encoder {enc[layer]:.1f} GFLOPs/s)\n")
            L.append("| head | smoother | total GFLOPs/s | held-out rmse_z | held-out PCC | vs linear rmse_z | vs linear PCC | seen rmse_z / PCC | lr chosen |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            pr = [depth.get(f"{layer}/{h}") for h in datasets.ALL_SPEAKERS]
            if all(pr):
                L.append(f"| probe (ridge) | - | {enc[layer]:.2f} | {np.mean([p['unseen']['rmse_z'] for p in pr]):.4f} | "
                         f"{np.mean([p['unseen']['pcc'] for p in pr]):.4f} | - | - | "
                         f"{np.mean([p['seen']['rmse_z'] for p in pr]):.4f} / {np.mean([p['seen']['pcc'] for p in pr]):.4f} | - |")
            base = {}
            fj = mdir / f"k{layer}_linear.json"
            if fj.exists():
                base = json.loads(fj.read_text())
            for name in HEADS:
                f = mdir / f"k{layer}_{name}.json"
                if not f.exists():
                    continue
                r = json.loads(f.read_text())
                if dim is None:
                    dim = int(np.load(datasets.features_root("ema_multi") / model / f"layer_{layer:02d}.npy", mmap_mode="r").shape[1])
                hf = head_gflops(name, dim)
                for sm in (1, 0):
                    folds = [h for h in datasets.ALL_SPEAKERS if f"{h}/smooth{sm}" in r]
                    if not folds:
                        continue
                    g = lambda sp, m: np.array([r[f"{h}/smooth{sm}"][sp][m] for h in folds])  # noqa: E731
                    cells = []
                    for m, sign in (("rmse_z", -1), ("pcc", 1)):
                        bf = [h for h in folds if f"{h}/smooth{sm}" in base]
                        if name == "linear" or not bf:
                            cells.append("-")
                            continue
                        dd = np.array([sign * (r[f"{h}/smooth{sm}"]["test_unseen"][m] - base[f"{h}/smooth{sm}"]["test_unseen"][m]) for h in bf])
                        cells.append(f"{dd.mean():+.4f} ({(dd > 0).sum()}/{len(dd)})")
                    lrs = Counter(r[f"{h}/smooth{sm}"]["lr"] for h in folds)
                    tot = enc[layer] + hf
                    rows.append((model, layer, name, sm, tot, g("test_unseen", "rmse_z").mean(), g("test_unseen", "pcc").mean()))
                    L.append(f"| {name} | {'on' if sm else 'off'} | {tot:.2f} | {g('test_unseen', 'rmse_z').mean():.4f} | "
                             f"{g('test_unseen', 'pcc').mean():.4f} | {cells[0]} | {cells[1]} | {g('test', 'rmse_z').mean():.4f} / "
                             f"{g('test', 'pcc').mean():.4f} | {dict(lrs)} ({len(folds)} folds) |")
    L.append("\n## Frontier over all encoders, depths and heads (held-out rmse_z vs total GFLOPs/s)\n")
    L.append("| encoder | k | head | smoother | GFLOPs/s | rmse_z | PCC |")
    L.append("|---|---|---|---|---|---|---|")
    best = np.inf
    for m, k, n, sm, f, r, p in sorted(rows, key=lambda x: (x[4], x[5])):
        if r < best:
            best = r
            L.append(f"| {m} | {k} | {n} | {'on' if sm else 'off'} | {f:.2f} | {r:.4f} | {p:.4f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary", "flops"))
    ap.add_argument("--model")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--head", choices=HEADS)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.model, args.layer, args.head)
    elif args.cmd == "flops":
        for n in HEADS:
            print(n, f"{head_gflops(n, 1024):.3f} GFLOPs/s")
    else:
        summary()


if __name__ == "__main__":
    main()
