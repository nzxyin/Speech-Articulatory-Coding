"""Structured pruning inside transformer layers: attention heads and FFN neurons.

Starting point: a finished adaptation run of train.py (e.g. XLS-R 300M layers 1-18 with LoRA and a
causal conv head). Its LoRA update is merged into the encoder, then pruning runs in `steps` rounds:

  1. importance: first-order Taylor scores of every attention head and FFN neuron on the training
     loss, from mask variables multiplied into each head's output and each neuron's activation
     (I = sum over batches of |dL/dm|, Michel et al. 2019), normalized per layer (L2) and ranked
     globally across layers, separately for heads and neurons (`--importance random` is a control)
  2. pruning: keep a fraction keep**(s/steps) of all heads and of all FFN neurons after round s;
     removed heads/neurons are cut out of the weight matrices, so FLOPs drop for real. A block that
     loses every head (or neuron) is replaced by its constant output bias, which is exact
  3. recovery: a fresh LoRA bank plus the head are fine-tuned (`step_epochs`), then LoRA is merged

then a final fine-tune with early stopping, and test evaluation plus a GPU cost profile of the pruned
encoder. With keep = 0.5 on XLS-R 300M layers 1-18 the encoder costs about the same as WavLM Large
layers 1-9, the most compute-efficient encoder found so far.

Outputs in <out>: step_XX.pt (pruned encoder + head after each round; the run resumes from the last
one), results.json (spec of kept heads/neurons per layer, metrics, cost).

Usage: python -m sparc.compression.components <source run dir> --keep 0.5 [--steps 4] [--out DIR]
"""

import argparse
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import mngu0
from .compute import profile
from .encoders import frame_lengths, normalize_wav
from .lora import apply_lora, lora_param_count, merge_lora
from .metrics import ema_metrics
from .train import ADAPT_ROOT, _targets, build, fit, load_trainable, predict, rmse


class ConstantAttention(nn.Module):
    """Attention block with every head removed: out_proj of an empty input is its bias."""

    def __init__(self, bias, dim):
        super().__init__()
        self.register_buffer("bias", bias.detach().clone() if bias is not None else torch.zeros(dim))

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        return self.bias.to(hidden_states.dtype).expand_as(hidden_states), None, None


class ConstantFeedForward(nn.Module):
    def __init__(self, bias, dim):
        super().__init__()
        self.register_buffer("bias", bias.detach().clone() if bias is not None else torch.zeros(dim))

    def forward(self, hidden_states):
        return self.bias.to(hidden_states.dtype).expand_as(hidden_states)


def _slice_linear(lin, rows=None, cols=None):
    W = lin.weight.data
    b = lin.bias.data if lin.bias is not None else None
    if rows is not None:
        W = W[rows]
        b = b[rows] if b is not None else None
    if cols is not None:
        W = W[:, cols]
    new = nn.Linear(W.shape[1], W.shape[0], bias=b is not None).to(W.device, W.dtype)
    new.weight.data.copy_(W)
    if b is not None:
        new.bias.data.copy_(b)
    return new


def n_heads(layer):
    a = layer.attention
    return 0 if isinstance(a, ConstantAttention) else a.q_proj.out_features // a.head_dim


def n_neurons(layer):
    f = layer.feed_forward
    return 0 if isinstance(f, ConstantFeedForward) else f.intermediate_dense.out_features


def prune_layer(layer, keep_heads, keep_neurons):
    """Keep the given (current-index) heads and FFN neurons of one encoder layer, in place."""
    dim = layer.layer_norm.normalized_shape[0]
    a = layer.attention
    if not isinstance(a, ConstantAttention):
        if len(keep_heads) == 0:
            layer.attention = ConstantAttention(a.out_proj.bias, dim)
        elif len(keep_heads) < n_heads(layer):
            hd = a.head_dim
            idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in sorted(keep_heads)]).to(a.q_proj.weight.device)
            a.q_proj, a.k_proj, a.v_proj = (_slice_linear(getattr(a, n), rows=idx) for n in ("q_proj", "k_proj", "v_proj"))
            a.out_proj = _slice_linear(a.out_proj, cols=idx)
            a.num_heads = len(keep_heads)
    f = layer.feed_forward
    if not isinstance(f, ConstantFeedForward):
        if len(keep_neurons) == 0:
            layer.feed_forward = ConstantFeedForward(f.output_dense.bias, dim)
        elif len(keep_neurons) < n_neurons(layer):
            idx = torch.as_tensor(sorted(keep_neurons), device=f.intermediate_dense.weight.device)
            f.intermediate_dense = _slice_linear(f.intermediate_dense, rows=idx)
            f.output_dense = _slice_linear(f.output_dense, cols=idx)


def apply_spec(model, spec):
    """Prune a freshly loaded (unpruned) subset model to `spec` = per-layer original head/neuron indices."""
    for layer, s in zip(model.encoder.layers, spec):
        prune_layer(layer, s["heads"], s["neurons"])


def importance(model, head, utts, args, ym, ys, device):
    """Taylor importance of every remaining head and neuron: lists (per layer) of score tensors."""
    model.eval()
    head.eval()
    masks, hooks = [], []
    for layer in model.encoder.layers:
        mh = torch.ones(n_heads(layer), device=device, requires_grad=True) if n_heads(layer) else None
        mn = torch.ones(n_neurons(layer), device=device, requires_grad=True) if n_neurons(layer) else None
        if mh is not None:
            hd = layer.attention.head_dim

            def pre_attn(mod, inp, mh=mh, hd=hd):
                x = inp[0]
                return (x * mh.to(x.dtype).repeat_interleave(hd),)

            hooks.append(layer.attention.out_proj.register_forward_pre_hook(pre_attn))
        if mn is not None:
            def pre_ffn(mod, inp, mn=mn):
                return (inp[0] * mn.to(inp[0].dtype),)

            hooks.append(layer.feed_forward.output_dense.register_forward_pre_hook(pre_ffn))
        masks.append((mh, mn))
    scores = [[torch.zeros_like(m) if m is not None else None for m in pair] for pair in masks]
    try:
        for i in range(0, len(utts), args.batch_size):
            batch = utts[i : i + args.batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                y, nf = _forward_with_grad(model, head, batch, device, args)
            sl, tgt = _targets(batch, nf, args.shift, ym, ys, device)
            loss = torch.nn.functional.mse_loss(torch.cat([y[b, s : s + n] for b, s, n in sl]).float(), tgt)
            params = [m for pair in masks for m in pair if m is not None]
            grads = torch.autograd.grad(loss, params)
            gi = iter(grads)
            for pair, sc in zip(masks, scores):
                for j in range(2):
                    if pair[j] is not None:
                        sc[j] += next(gi).abs().float()
    finally:
        for h in hooks:
            h.remove()
    out = []
    for sh, sn in scores:  # per-layer L2 normalization (Michel et al. 2019)
        out.append((sh / (sh.norm() + 1e-12) if sh is not None else None,
                    sn / (sn.norm() + 1e-12) if sn is not None else None))
    return out


def _forward_with_grad(model, head, batch, device, args):
    """Like train.forward, but with autograd on through the frozen encoder: train.forward disables it
    when no encoder parameter is trainable, and the mask gradients need it."""
    with torch.enable_grad():
        lens = [len(u.wav) for u in batch]
        x = torch.zeros(len(batch), max(lens))
        mask = torch.zeros(len(batch), max(lens), dtype=torch.long)
        for i, u in enumerate(batch):
            w = normalize_wav(u.wav)
            x[i, : len(w)] = torch.from_numpy(w)
            mask[i, : len(w)] = 1
        x, mask = x.to(device), mask.to(device)
        n_frames = frame_lengths(model, torch.tensor(lens)).tolist()
        out = model(x, attention_mask=mask if len(batch) > 1 else None, output_hidden_states=args.layer_pool)
        h = list(out.hidden_states[1:]) if args.layer_pool else out.last_hidden_state
        T = (h[0] if args.layer_pool else h).shape[1]
        pad = torch.arange(T, device=device)[None, :] >= torch.tensor(n_frames, device=device)[:, None]
        return head(h, pad_mask=pad), n_frames


def select(scores, spec, keep_frac, total_heads, total_neurons, rng=None):
    """Globally keep the top round(keep_frac * total) heads and neurons; returns the new spec (original indices)."""
    new = [{"heads": list(s["heads"]), "neurons": list(s["neurons"])} for s in spec]
    for kind, j, total in (("heads", 0, total_heads), ("neurons", 1, total_neurons)):
        cand = []  # (score, layer, current index)
        for li, sc in enumerate(scores):
            if sc[j] is None:
                continue
            vals = sc[j].tolist() if rng is None else [rng.random() for _ in range(len(sc[j]))]
            cand += [(v, li, ci) for ci, v in enumerate(vals)]
        n_keep = int(round(keep_frac * total))
        cand.sort(key=lambda c: -c[0])
        kept = {(li, ci) for _, li, ci in cand[:n_keep]}
        for li in range(len(new)):
            new[li][kind] = [orig for ci, orig in enumerate(spec[li][kind]) if (li, ci) in kept]
    return new


def to_current_indices(old_spec, new_spec, kind, li):
    return [old_spec[li][kind].index(o) for o in new_spec[li][kind]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="finished train.py run directory (with results.json, best_trainable.pt)")
    ap.add_argument("--keep", type=float, required=True, help="final fraction of heads and FFN neurons kept")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--step-epochs", type=int, default=4)
    ap.add_argument("--final-epochs", type=int, default=40)
    ap.add_argument("--importance", default="taylor", choices=("taylor", "random"))
    ap.add_argument("--importance-utts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap utterances per split")
    cli = ap.parse_args()

    src = Path(cli.source)
    cfg = json.loads((src / "results.json").read_text())["config"]
    args = Namespace(**cfg)
    if getattr(args, "variant", "none") == "none":
        raise SystemExit("source run must use LoRA (its update is merged before pruning)")
    args.pool_lr = getattr(args, "pool_lr", 3e-3)
    args.epochs, args.seed = cli.final_epochs, cli.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.bf16 = args.bf16 and device == "cuda"
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    out = Path(cli.out or ADAPT_ROOT / args.model / "pruned"
               / f"{src.name}_keep{cli.keep:g}_{cli.importance}_steps{cli.steps}_s{cli.seed}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        print(f"{out} already finished")
        return
    print(f"run {out}", flush=True)

    splits = mngu0.stems_by_split()
    if cli.limit:
        splits = {k: v[: cli.limit] for k, v in splits.items()}
    data ={s: [mngu0.load_utterance(x) for x in v] for s, v in splits.items()}
    allY = np.concatenate([u.ema_mm for u in data["train"]])
    ym, ys = allY.mean(0), allY.std(0)
    rng = random.Random(cli.seed)
    imp_utts = rng.sample(data["train"], min(cli.importance_utts, len(data["train"])))

    # source model: LoRA weights merged into the encoder
    model, head, bank = build(args, device)
    load_trainable(model, head, torch.load(src / "best_trainable.pt"))
    merge_lora(model)
    for p in model.parameters():
        p.requires_grad_(False)
    total_heads = sum(n_heads(l) for l in model.encoder.layers)
    total_neurons = sum(n_neurons(l) for l in model.encoder.layers)
    spec = [{"heads": list(range(n_heads(l))), "neurons": list(range(n_neurons(l)))} for l in model.encoder.layers]
    source_valid = rmse(*predict(model, head, data["valid"], args, ym, ys, device)[:2])
    print(f"source: {total_heads} heads, {total_neurons} FFN neurons, valid RMSE {source_valid:.4f}", flush=True)

    history = []
    start = 1
    done = sorted(out.glob("step_*.pt"))
    if done:  # resume from the last completed round
        st = torch.load(done[-1], weights_only=False)
        spec, history, start = st["spec"], st["history"], st["step"] + 1
        apply_spec(model, spec)
        model.load_state_dict(st["encoder"])
        head.load_state_dict(st["head"])
        print(f"resumed after round {st['step']}")

    for s in range(start, cli.steps + 1):
        frac = cli.keep ** (s / cli.steps)
        spec_path = out / f"spec_{s:02d}.json"  # fixed before recovery, so a preempted round resumes identically
        if spec_path.exists():
            new_spec = json.loads(spec_path.read_text())
        else:
            scores = importance(model, head, imp_utts, args, ym, ys, device)
            new_spec = select(scores, spec, frac, total_heads, total_neurons,
                              rng=random.Random(cli.seed * 1000 + s) if cli.importance == "random" else None)
            spec_path.write_text(json.dumps(new_spec))
        for li, layer in enumerate(model.encoder.layers):
            prune_layer(layer, to_current_indices(spec, new_spec, "heads", li),
                        to_current_indices(spec, new_spec, "neurons", li))
        spec = new_spec
        pruned_valid = rmse(*predict(model, head, data["valid"], args, ym, ys, device)[:2])
        bank = apply_lora(model, args.variant, args.rank, args.lora_alpha, tuple(args.targets.split(",")), args.gate)
        model.to(device)
        best, best_state, hist = fit(model, head, bank, data, args, ym, ys, device, cli.step_epochs,
                                     out / f"fit_step_{s:02d}.pt")
        load_trainable(model, head, best_state)
        merge_lora(model)
        for p in model.parameters():
            p.requires_grad_(False)
        kept_h = sum(len(x["heads"]) for x in spec)
        kept_n = sum(len(x["neurons"]) for x in spec)
        history.append({"step": s, "keep_frac": frac, "heads": kept_h, "neurons": kept_n,
                        "valid_after_prune": pruned_valid, "valid_after_recovery": best})
        print(f"round {s}: keep {frac:.3f} -> {kept_h} heads, {kept_n} neurons; valid {pruned_valid:.4f} "
              f"after pruning, {best:.4f} after recovery", flush=True)
        torch.save({"step": s, "spec": spec, "history": history, "encoder": model.state_dict(),
                    "head": head.state_dict()}, str(out / f"step_{s:02d}.pt") + ".tmp")
        Path(str(out / f"step_{s:02d}.pt") + ".tmp").replace(out / f"step_{s:02d}.pt")

    # final recovery fine-tune with early stopping
    bank = apply_lora(model, args.variant, args.rank, args.lora_alpha, tuple(args.targets.split(",")), args.gate)
    model.to(device)
    best, best_state, final_hist = fit(model, head, bank, data, args, ym, ys, device, cli.final_epochs,
                                       out / "fit_final.pt")
    load_trainable(model, head, best_state)
    n_lora = lora_param_count(bank)
    preds, trues, phones = predict(model, head, data["test"], args, ym, ys, device)
    test = ema_metrics(preds, trues, phones=phones)
    valid = ema_metrics(*predict(model, head, data["valid"], args, ym, ys, device)[:2], n_boot=0)
    merge_lora(model)
    merged = rmse(*predict(model, head, data["test"], args, ym, ys, device)[:2])
    cost = profile(model.eval(), device) if device == "cuda" else None
    result = {
        "source": str(src), "keep": cli.keep, "importance": cli.importance, "steps": cli.steps,
        "config": {k: v for k, v in vars(args).items()}, "spec": spec,
        "heads_kept": sum(len(x["heads"]) for x in spec), "heads_total": total_heads,
        "neurons_kept": sum(len(x["neurons"]) for x in spec), "neurons_total": total_neurons,
        "source_valid_rmse": source_valid, "rounds": history, "final_history": final_hist,
        "n_lora_params": n_lora, "best_valid_rmse": best, "valid": valid, "test": test,
        "test_rmse_after_merge": merged, "cost": cost,
    }
    torch.save({"spec": spec, "encoder": model.state_dict(), "head": head.state_dict()}, out / "pruned_model.pt")
    (out / "results.json").write_text(json.dumps(result, indent=1))
    print(f"test RMSE {test['rmse']:.4f} mm, PCC {test['pcc']:.4f}, after merge {merged:.4f}"
          + (f", {cost['gflops_per_audio_s']:.1f} GFLOPs/s audio" if cost else ""))


if __name__ == "__main__":
    main()
