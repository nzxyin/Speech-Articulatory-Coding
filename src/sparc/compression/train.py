"""Train an EMA head (and optionally LoRA) on a layer-subset SSL encoder over MNGU0.

Examples:
    # frozen truncated XLS-R 1B (layers 1..12) + linear head initialized from the ridge probe
    python -m sparc.compression.train --model xlsr-1b --layers 1-12 --variant none
    # standard / shared-A / shared+gated LoRA on the same subset
    python -m sparc.compression.train --model xlsr-1b --layers 1-12 --variant independent --rank 8
    python -m sparc.compression.train --model xlsr-1b --layers 1-12 --variant shared_gated --gate rank
    # temporal heads: centered or causal local conv, windowed attention, layer pooling
    python -m sparc.compression.train --model xlsr-1b --layers 1-12 --head conv --kernel 9 --causal
    python -m sparc.compression.train --model xlsr-1b --layers 1-12 --head attn --window 25 --pool static
    # frame-wise attention pooling over layers, separately per articulator
    python -m sparc.compression.train --model xlsr-1b --layers 1-16 --head conv --causal --pool attn --per-articulator
    # non-contiguous subset
    python -m sparc.compression.train --model xlsr-1b --layers 1,2,4,7,9,12 --variant independent

The alignment shift defaults to the one the ridge probe chose on validation for the last kept layer.
Model selection (early stopping) uses validation RMSE; test is evaluated once at the end, both with
LoRA as separate parameters and after merging it into the base weights (they must agree).
Restart-safe: state is saved every epoch and resumed from <out>/state.pt.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import mngu0
from .encoders import MODELS, frame_lengths, load_subset, normalize_wav
from . import datasets
from .extract import OUT_ROOT
from .heads import POOL_MODES, Head
from .lora import VARIANTS, apply_lora, lora_param_count, merge_lora
from .metrics import ema_metrics

ADAPT_ROOT = OUT_ROOT.parent / "adapt"


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def layer_tag(layers):
    if layers == list(range(1, layers[-1] + 1)):
        return f"prefix{layers[-1]}"
    return "sub" + "-".join(map(str, layers))


def probe_info(model, k, dataset="mngu0"):
    d = datasets.features_root(dataset) / model
    res = json.loads((d / "probe_results.json").read_text())["layers"].get(str(k))
    heads = np.load(d / "probe_heads.npz") if (d / "probe_heads.npz").exists() else {}
    W = heads[f"W_{k:02d}"] if f"W_{k:02d}" in heads else None
    b = heads[f"b_{k:02d}"] if f"b_{k:02d}" in heads else None
    return res, W, b


def build(args, device):
    model = load_subset(args.model, args.layers)
    bank = None
    if args.variant != "none":
        bank = apply_lora(model, args.variant, args.rank, args.lora_alpha, tuple(args.targets.split(",")), args.gate)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    if args.grad_ckpt and bank is not None:
        model.gradient_checkpointing_enable()
    pool = getattr(args, "pool", "static" if args.layer_pool else "none")  # older configs only had layer_pool
    head = Head(model.config.hidden_size, args.head, args.hidden, args.kernel, args.window, args.causal,
                smooth=not args.no_smooth, n_pool_layers=len(args.layers) if pool != "none" else 0,
                pool=pool if pool != "none" else "static", per_articulator=getattr(args, "per_articulator", False),
                pool_norm=getattr(args, "pool_norm", "layer") == "layer")
    return model.to(device), head.to(device), bank


def load_data(splits, ds=None):
    ds = ds or datasets.get("mngu0")
    return {split: [ds.load_utterance(s) for s in stems] for split, stems in splits.items()}


def forward(model, head, wavs, device, layer_pool, train):
    """wavs: list of 1-D float arrays (already normalized). Returns (B, T, 12) preds and valid frame counts."""
    lens = [len(w) for w in wavs]
    x = torch.zeros(len(wavs), max(lens))
    mask = torch.zeros(len(wavs), max(lens), dtype=torch.long)
    for i, w in enumerate(wavs):
        x[i, : len(w)] = torch.from_numpy(w)
        mask[i, : len(w)] = 1
    x, mask = x.to(device), mask.to(device)
    n_frames = frame_lengths(model, torch.tensor(lens)).tolist()
    enc_trainable = any(p.requires_grad for p in model.parameters())
    with torch.set_grad_enabled(train and enc_trainable):
        out = model(x, attention_mask=mask if len(wavs) > 1 else None, output_hidden_states=layer_pool)
        h = list(out.hidden_states[1:]) if layer_pool else out.last_hidden_state
    T = (h[0] if layer_pool else h).shape[1]
    pad = torch.arange(T, device=device)[None, :] >= torch.tensor(n_frames, device=device)[:, None]
    with torch.set_grad_enabled(train):
        y = head(h, pad_mask=pad)
    return y, n_frames


def _targets(utts, n_frames, offset, ym, ys, device):
    """Aligned slices: list of (b, start, n) and the concatenated standardized targets."""
    sl, tgt = [], []
    for b, u in enumerate(utts):
        off = u.base_offset + offset
        e = u.ema_mm if off >= 0 else u.ema_mm[-off:]
        s = max(off, 0)
        n = min(n_frames[b] - s, len(e))
        if n <= 0:
            continue
        sl.append((b, s, n))
        tgt.append((e[:n] - ym) / ys)
    return sl, torch.from_numpy(np.concatenate(tgt)).to(device) if tgt else None


@torch.no_grad()
def predict(model, head, utts, args, ym, ys, device, return_utts=False):
    model.eval()
    head.eval()
    preds, trues, phones, kept = [], [], [], []
    ds = datasets.get(getattr(args, "dataset", "mngu0"))
    for u in utts:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            y, nf = forward(model, head, [normalize_wav(u.wav)], device, args.layer_pool, train=False)
        sl, _ = _targets([u], nf, args.shift, ym, ys, device)
        if not sl:
            continue
        _, s, n = sl[0]
        p = y[0, s : s + n].float().cpu().numpy() * ys + ym
        off = u.base_offset + args.shift
        e = u.ema_mm if off >= 0 else u.ema_mm[-off:]
        preds.append(p)
        trues.append(e[:n])
        phones.append(ds.frame_phones(u, off, n))
        kept.append(u)
    if return_utts:
        return preds, trues, phones, kept
    return preds, trues, phones


@torch.no_grad()
def feature_stats(model, head, utts, args, device):
    """Per-channel mean/std of the head's input over the training frames (initial encoder, no LoRA effect)."""
    model.eval()
    s = s2 = None
    n = 0
    for u in utts:
        out = model(torch.from_numpy(normalize_wav(u.wav)).unsqueeze(0).to(device),
                    output_hidden_states=args.layer_pool)
        h = head.pooled(list(out.hidden_states[1:])) if args.layer_pool else out.last_hidden_state
        if h.dim() == 4:  # per-articulator pooling: all groups are identical at initialization
            h = h[:, 0]
        h = h[0].double()
        s = h.sum(0) if s is None else s + h.sum(0)
        s2 = (h**2).sum(0) if s2 is None else s2 + (h**2).sum(0)
        n += len(h)
    mean = s / n
    return mean.float(), (s2 / n - mean**2).clamp_min(0).sqrt().float()


@torch.no_grad()
def pooled_ridge_init(model, head, data, args, ym, ys, device):
    """Closed-form ridge initialization of a pooled linear head.

    Gradient training cannot reach the ridge solution from a random start within the epoch budget on
    these ill-conditioned features, so the single-layer baselines start from the ridge probe. For a
    fair comparison, a pooled linear head starts from the ridge solution on its own initial input:
    the uniformly pooled layers, low-pass smoothed (the head smooths its output, and smoothing commutes
    with the linear map) and standardized. Alpha is chosen on validation RMSE (standardized units).
    """
    from .probe import ALPHAS

    model.eval()
    head.eval()

    def feats(utts):
        for u in utts:
            out = model(torch.from_numpy(normalize_wav(u.wav)).unsqueeze(0).to(device), output_hidden_states=True)
            h = head.pooled(list(out.hidden_states[1:]))
            if h.dim() == 4:
                h = h[:, 0]  # all articulator groups are identical at initialization
            if head.smooth is not None:
                h = head.smooth(h.float())
            h = ((h - head.in_mean[0]) / head.in_std[0])[0].double()
            sl, tgt = _targets([u], [h.shape[0]], args.shift, ym, ys, device)
            if sl:
                _, s, n = sl[0]
                yield h[s : s + n], tgt.double()

    D = head.in_mean.shape[-1]
    G = torch.zeros(D, D, dtype=torch.float64, device=device)
    C = torch.zeros(D, 12, dtype=torch.float64, device=device)
    sx = torch.zeros(D, dtype=torch.float64, device=device)
    sy = torch.zeros(12, dtype=torch.float64, device=device)
    n = 0
    for x, y in feats(data["train"]):
        G += x.T @ x
        C += x.T @ y
        sx += x.sum(0)
        sy += y.sum(0)
        n += len(x)
    mx, my = sx / n, sy / n
    evals, V = torch.linalg.eigh(G - n * torch.outer(mx, mx))
    VtC = V.T @ (C - n * torch.outer(mx, my))
    val = list(feats(data["valid"]))
    best = None
    for a in ALPHAS:
        W = V @ (VtC / (evals.clamp_min(0) + a)[:, None])
        b = my - mx @ W
        err = torch.cat([x @ W + b - y for x, y in val])
        r = float(err.pow(2).mean().sqrt())
        if best is None or r < best[0]:
            best = (r, float(a), W, b)
    _, alpha, W, b = best
    if head.per_articulator:
        head.out.weight.copy_(W.T.reshape(head.groups, -1, D).transpose(1, 2).float())
        head.out.bias.copy_(b.reshape(head.groups, -1).float())
    else:
        head.out.weight.copy_(W.T.float())
        head.out.bias.copy_(b.float())
    return alpha


def rmse(preds, trues):
    P, Y = np.concatenate(preds), np.concatenate(trues)
    return float(np.sqrt(((P - Y) ** 2).mean(0)).mean())


def trainable_state(model, head):
    params = {f"model.{n}": p for n, p in model.named_parameters() if p.requires_grad}
    params.update({f"head.{n}": p for n, p in head.named_parameters()})
    # the input standardization moves with the linear weights when a pooled head re-standardizes
    params.update({f"head.{n}": b for n, b in head.named_buffers() if n in ("in_mean", "in_std")})
    return {k: v.detach().cpu().clone() for k, v in params.items()}


def load_trainable(model, head, state):
    mp = dict(model.named_parameters())
    hp = dict(head.named_parameters())
    hp.update({n: b for n, b in head.named_buffers() if n in ("in_mean", "in_std")})
    with torch.no_grad():
        for k, v in state.items():
            dst = mp[k[6:]] if k.startswith("model.") else hp[k[5:]]
            dst.copy_(v.reshape(dst.shape))  # older runs stored (D,) input statistics


def fit(model, head, bank, data, args, ym, ys, device, epochs, state_path):
    """Train the head (and LoRA bank, if any) with early stopping on validation RMSE.

    Restart-safe through `state_path`. Returns (best validation RMSE, best trainable state, history).
    """
    pool_params = list(head.pool.parameters()) if head.pool is not None else []
    pool_ids = {id(p) for p in pool_params}
    groups = [{"params": [p for p in head.parameters() if id(p) not in pool_ids], "lr": args.head_lr}]
    if pool_params:  # softmax logits need larger steps than Adam at head_lr gives them to leave uniform
        groups.append({"params": pool_params, "lr": args.pool_lr, "weight_decay": 0.0})
    if bank is not None:
        groups.append({"params": list(bank.parameters()), "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(data["train"]) / args.batch_size)
    total, warm = epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )

    start_epoch, best, best_state, bad, history = 0, float("inf"), None, 0, []
    if state_path.exists():
        st = torch.load(state_path, weights_only=False)
        load_trainable(model, head, st["current"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, best, best_state, bad, history = st["epoch"] + 1, st["best"], st["best_state"], st["bad"], st["history"]
        random.setstate(st["py_rng"])
        print(f"resumed at epoch {start_epoch}")
    else:
        val0 = rmse(*predict(model, head, data["valid"], args, ym, ys, device)[:2])
        history.append({"epoch": -1, "valid_rmse": val0})
        best, best_state = val0, trainable_state(model, head)
        print(f"initial valid RMSE {val0:.4f} mm", flush=True)

    for epoch in range(start_epoch, epochs):
        if bad >= args.patience:
            break
        model.train() if bank is not None else model.eval()
        head.train()
        order = list(range(len(data["train"])))
        random.shuffle(order)
        t0, losses = time.time(), []
        for i in range(0, len(order), args.batch_size):
            utts = [data["train"][j] for j in order[i : i + args.batch_size]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                y, nf = forward(model, head, [normalize_wav(u.wav) for u in utts], device, args.layer_pool, True)
            sl, tgt = _targets(utts, nf, args.shift, ym, ys, device)
            pred = torch.cat([y[b_, s : s + n] for b_, s, n in sl]).float()
            loss = torch.nn.functional.mse_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if head.track_input_stats:
            head.restandardize()  # function-preserving; keeps the pooled linear head well conditioned
        val = rmse(*predict(model, head, data["valid"], args, ym, ys, device)[:2])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_rmse": val,
                        "seconds": time.time() - t0})
        if val < best - 1e-4:
            best, best_state, bad = val, trainable_state(model, head), 0
        else:
            bad += 1
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} valid RMSE {val:.4f} mm (best {best:.4f}) "
              f"{time.time() - t0:.0f}s", flush=True)
        torch.save({"current": trainable_state(model, head), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "best_state": best_state, "bad": bad, "history": history,
                    "py_rng": random.getstate()}, str(state_path) + ".tmp")
        Path(str(state_path) + ".tmp").replace(state_path)

    return best, best_state, history


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--dataset", default="mngu0", choices=datasets.DATASET_NAMES)
    ap.add_argument("--layers", required=True, help="e.g. 1-12 or 1,2,4,7")
    ap.add_argument("--variant", default="none", choices=("none",) + VARIANTS)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16)
    ap.add_argument("--targets", default="q_proj,v_proj")
    ap.add_argument("--gate", default="scalar", choices=("scalar", "rank"))
    ap.add_argument("--head", default="linear", choices=("linear", "conv", "attn"))
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--kernel", type=int, default=9)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--pool", default="none", choices=POOL_MODES,
                    help="pool all retained layers: static weights or frame-wise attention weights")
    ap.add_argument("--pool-norm", default="layer", choices=("layer", "none"),
                    help="layer-normalize each layer before pooling (default) or pool the raw hidden states")
    ap.add_argument("--per-articulator", action="store_true",
                    help="separate layer pooling (and head projections) per articulator")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--no-probe-init", action="store_true")
    ap.add_argument("--shift", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--pool-lr", type=float, default=3e-3, help="learning rate of the layer-pooling parameters")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap utterances per split")
    args = ap.parse_args(argv)
    args.layers = parse_layers(args.layers)
    args.layer_pool = args.pool != "none"
    if args.per_articulator and not args.layer_pool:
        ap.error("--per-articulator needs --pool static or attn")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.bf16 = args.bf16 and device == "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    k = args.layers[-1]
    try:
        probe_res, W, b = probe_info(args.model, k, args.dataset)
    except FileNotFoundError:
        probe_res, W, b = None, None, None
    if args.shift is None:
        args.shift = probe_res["shift"] if probe_res else 1
    name = (f"{layer_tag(args.layers)}_{args.variant}"
            + (f"_r{args.rank}_{args.targets.replace(',', '+')}" if args.variant != "none" else "")
            + (f"_g{args.gate}" if args.variant == "shared_gated" else "")
            + f"_{args.head}" + ("_causal" if args.causal else "") + (f"_w{args.window}" if args.window else "")
            + (f"_pool{args.pool}" if args.layer_pool else "") + ("_raw" if args.layer_pool and args.pool_norm == "none" else "")
            + ("_art" if args.per_articulator else "") + (f"_h{args.hidden}" if args.head != "linear" and args.hidden != 256 else "")
            + ("_randinit" if args.no_probe_init and args.head == "linear" and not args.layer_pool else "")
            + f"_s{args.seed}")
    out = Path(args.out or datasets.adapt_root(args.dataset) / args.model / name)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        print(f"{out} already finished")
        return
    print(f"run {out}", flush=True)

    ds = datasets.get(args.dataset)
    splits = ds.splits()
    if args.limit:
        splits = {s: v[: args.limit] for s, v in splits.items()}
    data = load_data(splits, ds)
    allY = np.concatenate([u.ema_mm for u in data["train"]])
    ym, ys = allY.mean(0), allY.std(0)

    model, head, bank = build(args, device)
    if args.head == "linear":
        stats_path = out / "input_stats.pt"
        if stats_path.exists():
            mean, std = torch.load(stats_path)
        else:
            mean, std = feature_stats(model, head, data["train"], args, device)
            torch.save((mean, std), stats_path)
        head.set_input_stats(mean, std)
    if args.head == "linear" and not args.layer_pool and W is not None and not args.no_probe_init \
            and args.layers == list(range(1, k + 1)):
        head.init_linear(W / ys[None, :], (b - ym) / ys)
        print(f"head initialized from the ridge probe of layer {k}")
    if args.head == "linear" and args.layer_pool and not args.no_probe_init:
        ridge_path = out / "pooled_ridge_init.pt"
        if ridge_path.exists():
            head.out.load_state_dict(torch.load(ridge_path))
        else:
            alpha = pooled_ridge_init(model, head, data, args, ym, ys, device)
            torch.save(head.out.state_dict(), ridge_path)
            print(f"pooled linear head initialized by ridge on the uniform pool (alpha {alpha:.3g})")

    best, best_state, history = fit(model, head, bank, data, args, ym, ys, device, args.epochs, out / "state.pt")

    load_trainable(model, head, best_state)
    def evaluate_split(split, n_boot):
        preds, trues, phones, kept = predict(model, head, data[split], args, ym, ys, device, return_utts=True)
        return ds.evaluate(kept, preds, trues, phones=phones if any(phones) else None, n_boot=n_boot)

    test = evaluate_split("test", 1000)
    extra_eval = {s: evaluate_split(s, 1000 if s.startswith("test") else 0)
                  for s in ("valid_unseen", "test_unseen") if data.get(s)}
    if head.pool is not None:
        head.pool.reset_stats()
    valid = evaluate_split("valid", 0)
    # mean layer weights over validation frames, (groups x retained layers); groups = articulators if per-articulator
    pool_weights = head.pool.mean_weights() if head.pool is not None else None
    merged_rmse = None
    if bank is not None:
        merge_lora(model)
        merged_rmse = rmse(*predict(model, head, data["test"], args, ym, ys, device)[:2])
    torch.save(best_state, out / "best_trainable.pt")
    cfg = {k_: v for k_, v in vars(args).items()}
    result = {
        "config": cfg,
        "n_lora_params": lora_param_count(bank) if bank is not None else 0,
        "n_head_params": sum(p.numel() for p in head.parameters()),
        "best_valid_rmse": best,
        "valid": valid,
        "test": test,
        **extra_eval,
        "test_rmse_after_merge": merged_rmse,
        "history": history,
        "probe_test_rmse_same_layer": probe_res["test"]["rmse"] if probe_res else None,
        "pool_weights_valid": pool_weights,
    }
    (out / "results.json").write_text(json.dumps(result, indent=1))
    print(f"test RMSE {test['rmse']:.4f} mm, PCC {test['pcc']:.4f}"
          + (f", after merge {merged_rmse:.4f}" if merged_rmse is not None else ""))


if __name__ == "__main__":
    main()
