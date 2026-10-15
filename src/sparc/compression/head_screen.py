"""Frozen-encoder head screen under leave-one-speaker-out: does a head with a different inductive bias beat spending
the same FLOPs on more encoder layers?

Heads are trained on the cached, 10 Hz low-passed encoder features of one layer (features_ema_multi/<model>/
layer_kk.npy) for every LOSO fold. Everything is chosen on the fold's training speakers only (never the held-out
speaker): alignment shift = the fold's own probe-selected shift from depth_sweep; learning rate on their validation
sentences; early stopping on the same.

Heads (heads.Head and seqheads):
    linear        frame-wise readout of standardized features (probe-equivalent model class)
    mlp           LayerNorm -> Linear D->256 -> GELU -> Linear: same front end and capacity as the sequence heads but
                  no temporal context -- the control that separates temporal modelling from capacity
    conv_causal9  the causal 9-frame conv head used so far (bridge to the earlier runs)
    conv3 ... conv65  centered (non-causal) conv heads (learned FIR filters) of 3-65 frames (60 ms - 1.3 s)
    convgru9/17   centered conv then BiGRU; gruconv9 BiGRU then conv; gru2 two-layer BiGRU (depth control)
    gru           bidirectional GRU
    ssm           bidirectional diagonal linear SSM (S4D)
    mamba         bidirectional selective SSM (reference scan)

Protocol per (head, fold): primary arm without the output smoother (the features are already low-passed, so the
smoother would give every head the same ±1 s context): learning rate from {1e-4, 3e-4, 1e-3, 3e-3} with seed 0 on
validation, then seeds 1 and 2 at that rate; results are seed means. Secondary arm 'smoother on' (deployed
post-filter): seed 0 at the chosen rate. AdamW, weight decay 0.01 except biases, norms and SSM dynamics (none),
cosine schedule with a one-epoch warm-up, <= 40 epochs, patience 8, gradient clipping 1.0.

Head FLOPs per second of audio from FlopCounterMode on a 4 s input (matmul / conv only, like the encoder profiles;
the uncounted scan / FFT / elementwise work is < 0.02 GFLOPs/s, the smoother ~0.012).

Usage:
    python -m sparc.compression.head_screen run --model wavlm-large --layer 7 --head gru
    python -m sparc.compression.head_screen summary
"""

import argparse
import json
import math
import random

import numpy as np
import torch
from scipy import stats as sstats
from torch.utils.flop_counter import FlopCounterMode

from . import datasets
from .heads import Head
from .metrics import ema_metrics
from .probe import aligned, load_targets
from .seqheads import CascadeHead, GRUHead, MambaHead, SSMHead

OUT = datasets.XLSR_EMA / "analysis_heads"
HEADS = ("linear", "mlp", "conv_causal9", "gru_uni", "conv3", "conv5", "conv9", "conv17", "conv33", "conv65", "gru", "gru2",
         "convgru9", "convgru17", "gruconv9", "ssm", "mamba")
LRS = (1e-4, 3e-4, 1e-3, 3e-3)
SEEDS = (0, 1, 2)
FT_SR = 50
NO_DECAY = ("bias", "norm", "log_dt", "log_A_real", "A_imag", "A_log", ".D", "C")


def make_head(name, dim, smooth):
    if name == "linear":
        return Head(dim, "linear", smooth=smooth)
    if name == "mlp":
        return Head(dim, "conv", hidden=256, kernel=1, causal=False, smooth=smooth)
    if name == "conv_causal9":
        return Head(dim, "conv", hidden=256, kernel=9, causal=True, smooth=smooth)
    if name.startswith("convgru"):
        return CascadeHead(dim, "conv_gru", kernel=int(name[7:]), smooth=smooth)
    if name.startswith("gruconv"):
        return CascadeHead(dim, "gru_conv", kernel=int(name[7:]), smooth=smooth)
    if name == "gru2":
        return CascadeHead(dim, "gru_gru", smooth=smooth)
    if name.startswith("conv") and name[4:].isdigit():
        return Head(dim, "conv", hidden=256, kernel=int(name[4:]), causal=False, smooth=smooth)
    if name == "gru":
        return GRUHead(dim, smooth=smooth)
    if name == "gru_uni":
        return GRUHead(dim, smooth=smooth, bidirectional=False)
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


def param_groups(head):
    decay, no = [], []
    for n, p in head.named_parameters():
        (no if any(k in n for k in NO_DECAY) else decay).append(p)
    return [{"params": decay, "weight_decay": 0.01}, {"params": no, "weight_decay": 0.0}]


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
    opt = torch.optim.AdamW(param_groups(head), lr=lr)
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
        if not math.isfinite(v):
            break
        if best_state is None or v < best - 1e-5:
            best, best_state, bad = v, {k: t.detach().clone() for k, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        return None, float("inf")
    head.load_state_dict(best_state)
    return head, best


def fold_shift(model, layer, h, depth):
    key = f"{layer}/{h}"
    if key not in depth:
        raise KeyError(f"no depth-sweep shift for {model} {key}; run depth_sweep first")
    return depth[key]["shift"]


def evaluate(head, data, ym, ys, device):
    return {split: zm(predict(head, data[split][0], ym, ys, device), data[split][1]) for split in ("test", "test_unseen")}


def load_features(model, layer, features):
    """(feats, index). 'zp': the cached zero-phase 10 Hz low-passed features; 'raw': unfiltered features
    (extract --raw); 'causal': raw features through a one-sided (causal) 5th-order 10 Hz Butterworth per utterance."""
    if features == "zp":
        d = datasets.features_root("ema_multi") / model
        return np.load(d / f"layer_{layer:02d}.npy", mmap_mode="r"), json.loads((d / "index.json").read_text())
    d = datasets.features_root("ema_multi").with_name(datasets.features_root("ema_multi").name + "_raw") / model
    index = json.loads((d / "index.json").read_text())
    feats = np.load(d / f"layer_{layer:02d}.npy", mmap_mode="r")
    if features == "raw":
        return feats, index
    from scipy.signal import butter, lfilter

    b, a = butter(5, 10 / (FT_SR / 2), btype="low")
    out = np.empty(feats.shape, dtype=np.float32)
    for s, (i, n) in index.items():
        out[i:i + n] = lfilter(b, a, np.asarray(feats[i:i + n], dtype=np.float32), axis=0)
    return out, index


def choose_shift(feats, index, utts, sp, device, shifts=range(-2, 9)):
    """Shift by ridge on the training speakers' validation sentences (for filtered features with a delay)."""
    from .lora_readout import fit_ridge, zm as rzm

    best = None
    for s in shifts:
        Xtr, Ytr, _ = aligned(feats, index, utts, sp["train"], s)
        Xva, Yva, _ = aligned(feats, index, utts, sp["valid"], s)
        r, W, b_ = fit_ridge([np.asarray(x, dtype=np.float32) for x in Xtr], Ytr,
                             [np.asarray(x, dtype=np.float32) for x in Xva], Yva, device)
        e = rzm(r.predict(W, b_, [np.asarray(x, dtype=np.float32) for x in Xva]), Yva)["rmse_z"]
        if best is None or e < best[0]:
            best = (e, s)
    return best[1]


def run(model, layer, head_name, device=None, features="zp"):
    from .depth_sweep import load_all

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feats, index = load_features(model, layer, features)
    depth = load_all(model)
    out_f = OUT / model / (f"k{layer}_{head_name}.json" if features == "zp" else f"k{layer}_{head_name}_{features}.json")
    out_f.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    utts = None
    for h in datasets.ALL_SPEAKERS:
        if h in res and "smooth_on" in res[h]:
            continue
        ds = datasets.EMAMulti(held_out=(h,), norm_by="norm_group")
        sp = {k: [s for s in v if s in index] for k, v in ds.splits().items()}
        if utts is None:
            utts = load_targets(sorted(set(index) & set(ds.rows)), ds)  # targets do not depend on the fold
        if features == "zp":
            shift = fold_shift(model, layer, h, depth)
        else:
            shift = res[h]["shift"] if h in res else choose_shift(feats, index, utts, sp, device)
        data = {}
        for split in ("train", "valid", "test", "test_unseen"):
            X, Y, _ = aligned(feats, index, utts, sp[split], shift)
            data[split] = ([np.asarray(x, dtype=np.float32) for x in X], [y.astype(np.float64) for y in Y])
        Yall = np.concatenate(data["train"][1])
        ym, ys = Yall.mean(0), Yall.std(0)
        rec = res.get(h, {"shift": shift})
        assert rec["shift"] == shift, (rec["shift"], shift)
        if "lr" not in rec:  # learning rate on validation (seed 0, smoother off)
            trials, best_head = {}, None
            for lr in LRS:
                head, v = train_one(head_name, False, lr, data, ym, ys, device, seed=0)
                trials[lr] = v
                if head is not None and v <= min(trials.values()):
                    best_head = (lr, head)
            lr = min(trials, key=trials.get)
            if best_head is None:
                raise RuntimeError(f"{model} k{layer} {head_name} {h}: every learning rate diverged")
            rec.update({"lr": lr, "lr_valid": {str(k): v for k, v in trials.items()},
                        "seeds": {"0": {"valid_rmse_z": trials[lr], **evaluate(best_head[1], data, ym, ys, device)}}})
            res[h] = rec
            out_f.write_text(json.dumps(res, indent=1))
        for s in SEEDS[1:]:
            if str(s) in rec["seeds"]:
                continue
            head, v = train_one(head_name, False, rec["lr"], data, ym, ys, device, seed=s)
            rec["seeds"][str(s)] = {"valid_rmse_z": v, **evaluate(head, data, ym, ys, device)} if head is not None else {"failed": True}
            res[h] = rec
            out_f.write_text(json.dumps(res, indent=1))
        if "smooth_on" not in rec:
            head, v = train_one(head_name, True, rec["lr"], data, ym, ys, device, seed=0)
            rec["smooth_on"] = {"valid_rmse_z": v, **evaluate(head, data, ym, ys, device)} if head is not None else {"failed": True}
            res[h] = rec
            out_f.write_text(json.dumps(res, indent=1))
        ok = [x for x in rec["seeds"].values() if "failed" not in x]
        print(f"{model} k{layer} {head_name} {h}: shift {shift} lr {rec['lr']} held-out "
              f"{np.mean([x['test_unseen']['rmse_z'] for x in ok]):.4f} / {np.mean([x['test_unseen']['pcc'] for x in ok]):.4f} "
              f"({len(ok)} seeds)", flush=True)


# ---------------------------------------------------------------- summary

def seed_mean(rec, split, m, arm="off"):
    if arm == "on":
        x = rec.get("smooth_on", {})
        return x[split][m] if split in x else np.nan
    ok = [x for x in rec["seeds"].values() if "failed" not in x]
    return float(np.mean([x[split][m] for x in ok])) if ok else np.nan


def paired(a, b, sign):
    d = sign * (np.asarray(a) - np.asarray(b))
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return "-"
    hw = sstats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.4f} [{d.mean() - hw:+.4f}, {d.mean() + hw:+.4f}] {(d > 0).sum()}/{len(d)}"


def summary():
    from .depth_sweep import gflops, load_all

    SPK = datasets.ALL_SPEAKERS
    L = ["# Frozen-encoder head screen (LOSO, 7 folds)\n",
         "Held-out speaker rmse_z / PCC (seed mean of 3 for the primary, smoother-off arm). Total GFLOPs/s = encoder prefix + "
         "head (FlopCounterMode, matmul/conv; uncounted scan/FFT/elementwise < 0.02). Paired differences over folds: mean "
         "[95% t CI, unadjusted for the many comparisons] folds better; + = better. 'probe' = ridge readout of the depth "
         "sweep at the same layer (shift and alpha per fold). All selection on the training speakers.\n"]
    rows = []
    for mdir in sorted(p for p in OUT.glob("*") if p.is_dir()):
        model = mdir.name
        enc = gflops(model)
        depth = load_all(model)
        dim = None
        for k in sorted({int(f.stem.split("_")[0][1:]) for f in mdir.glob("k*_*.json")}):
            res = {n: json.loads((mdir / f"k{k}_{n}.json").read_text()) for n in HEADS if (mdir / f"k{k}_{n}.json").exists()}
            probe = {h: depth.get(f"{k}/{h}") for h in SPK}
            if dim is None:
                dim = int(np.load(datasets.features_root("ema_multi") / model / f"layer_{k:02d}.npy", mmap_mode="r").shape[1])
            L.append(f"\n## {model} k={k} (encoder {enc[k]:.2f} GFLOPs/s)\n")
            L.append("| head | GFLOPs/s | held-out rmse_z | held-out PCC | vs probe rmse_z | vs probe PCC | vs mlp rmse_z | vs mlp PCC | seen rmse_z / PCC | smoother on: held-out |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            if all(probe.values()):
                pu = {m: [probe[h]["unseen"][m] for h in SPK] for m in ("rmse_z", "pcc")}
                L.append(f"| probe | {enc[k]:.2f} | {np.mean(pu['rmse_z']):.4f} | {np.mean(pu['pcc']):.4f} | - | - | - | - | "
                         f"{np.mean([probe[h]['seen']['rmse_z'] for h in SPK]):.4f} / {np.mean([probe[h]['seen']['pcc'] for h in SPK]):.4f} | - |")
            for n, r in res.items():
                folds = [h for h in SPK if h in r and "seeds" in r[h]]
                if len(folds) < len(SPK):
                    L.append(f"| {n} | incomplete ({len(folds)}/7 folds) |")
                    continue
                u = {m: [seed_mean(r[h], "test_unseen", m) for h in SPK] for m in ("rmse_z", "pcc")}
                s = {m: np.mean([seed_mean(r[h], "test", m) for h in SPK]) for m in ("rmse_z", "pcc")}
                on = {m: np.nanmean([seed_mean(r[h], "test_unseen", m, "on") for h in SPK]) for m in ("rmse_z", "pcc")}
                cells = []
                for ref in ("probe", "mlp"):
                    for m, sign in (("rmse_z", -1), ("pcc", 1)):
                        if ref == "probe" and all(probe.values()):
                            cells.append(paired(u[m], [probe[h]["unseen"][m] for h in SPK], sign))
                        elif ref == "mlp" and "mlp" in res and n != "mlp" and all(h in res["mlp"] for h in SPK):
                            cells.append(paired(u[m], [seed_mean(res["mlp"][h], "test_unseen", m) for h in SPK], sign))
                        else:
                            cells.append("-")
                tot = enc[k] + head_gflops(n, dim)
                rows.append((model, k, n, tot, np.mean(u["rmse_z"]), np.mean(u["pcc"]),
                             np.mean([r[h]["seeds"]["0"]["valid_rmse_z"] for h in SPK])))
                L.append(f"| {n} | {tot:.2f} | {np.mean(u['rmse_z']):.4f} | {np.mean(u['pcc']):.4f} | " + " | ".join(cells) +
                         f" | {s['rmse_z']:.4f} / {s['pcc']:.4f} | {on['rmse_z']:.4f} / {on['pcc']:.4f} |")
        # probe at every depth, as the 'spend FLOPs on more layers' alternative
        for kk in sorted({int(x.split('/')[0]) for x in depth}):
            pr = [depth.get(f"{kk}/{h}") for h in SPK]
            if all(pr):
                rows.append((model, kk, "probe", enc[kk], np.mean([p["unseen"]["rmse_z"] for p in pr]),
                             np.mean([p["unseen"]["pcc"] for p in pr]), np.nan))
    L.append("\n## Descriptive frontier: held-out rmse_z vs total GFLOPs/s (heads at the screened layers + ridge probe at "
             "every depth). The minimum over many configurations is optimistic; use the paired tables for claims.\n")
    L.append("| encoder | k | readout | GFLOPs/s | rmse_z | PCC |")
    L.append("|---|---|---|---|---|---|")
    best = np.inf
    for m, k, n, f, r, p, _ in sorted(rows, key=lambda x: (x[3], x[4])):
        if r < best:
            best = r
            L.append(f"| {m} | {k} | {n} | {f:.2f} | {r:.4f} | {p:.4f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary", "flops"))
    ap.add_argument("--model")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--head", choices=HEADS)
    ap.add_argument("--features", default="zp", choices=("zp", "causal", "raw"),
                    help="zp: cached zero-phase low-passed features; causal / raw: see load_features")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.model, args.layer, args.head, features=args.features)
    elif args.cmd == "flops":
        for n in HEADS:
            print(n, f"{head_gflops(n, 1024):.3f} GFLOPs/s")
    else:
        summary()


if __name__ == "__main__":
    main()
