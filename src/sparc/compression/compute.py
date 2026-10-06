"""Inference cost of an encoder: parameters, FLOPs, activation memory, latency.

FLOPs are counted with torch.utils.flop_counter (matmuls, convolutions and attention; 1 MAC = 2 FLOPs)
for a single utterance of `seconds` of audio. Attention is quadratic in length, so GFLOPs per
second of audio depend on the reference duration; REF_SECONDS is used for all reported comparisons.
Activation memory is the peak CUDA allocation during a no-grad forward minus the weights.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from .encoders import MODELS, encode, load_full, num_layers

REF_SECONDS = 4.0  # close to the mean MNGU0 utterance length
SR = 16000


def param_counts(model):
    groups = {
        "cnn": model.feature_extractor,
        "projection": model.feature_projection,
        "pos_conv": model.encoder.pos_conv_embed,
        "layers": model.encoder.layers,
    }
    out = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
    out["total"] = sum(p.numel() for p in model.parameters())
    if hasattr(model, "masked_spec_embed") and model.masked_spec_embed is not None:
        out["total"] -= model.masked_spec_embed.numel()  # training-only parameter
    return out


def _input(seconds, device, dtype, batch=1):
    g = torch.Generator().manual_seed(0)
    return torch.randn(batch, int(seconds * SR), generator=g).to(device=device, dtype=dtype)


@torch.no_grad()
def count_flops(model, seconds=REF_SECONDS, device="cpu"):
    dtype = next(model.parameters()).dtype
    x = _input(seconds, device, dtype)
    with FlopCounterMode(display=False) as fc:
        encode(model, x)
    return int(fc.get_total_flops())


@torch.no_grad()
def activation_memory(model, seconds=REF_SECONDS, device="cuda"):
    dtype = next(model.parameters()).dtype
    x = _input(seconds, device, dtype)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    encode(model, x)
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device) - base)


@torch.no_grad()
def latency(model, seconds=REF_SECONDS, device="cuda", warmup=5, iters=20):
    dtype = next(model.parameters()).dtype
    x = _input(seconds, device, dtype)
    times = []
    for i in range(warmup + iters):
        if device != "cpu":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        encode(model, x)
        if device != "cpu":
            torch.cuda.synchronize(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    return {"latency_s": med, "rtf": med / seconds}


def profile(model, device="cuda", seconds=REF_SECONDS, with_latency=True):
    """Cost summary for a (subset) encoder already on `device`."""
    params = param_counts(model)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    flops = count_flops(model, seconds, device)
    out = {
        "params": params,
        "weight_bytes": weight_bytes,
        "ref_seconds": seconds,
        "flops": flops,
        "gflops_per_audio_s": flops / seconds / 1e9,
    }
    if str(device).startswith("cuda"):
        out["activation_bytes"] = activation_memory(model, seconds, device)
        if with_latency:
            out.update(latency(model, seconds, device))
    return out


def profile_subsets(name, subsets, device="cuda", seconds=REF_SECONDS, with_latency=True):
    """Profile several layer subsets of one model, swapping layers on a single loaded copy."""
    model = load_full(name).to(device)
    all_layers, final_norm = model.encoder.layers, model.encoder.layer_norm
    total = len(all_layers)
    out = []
    try:
        for layers in subsets:
            layers = list(layers)
            model.encoder.layers = nn.ModuleList([all_layers[i - 1] for i in layers])
            model.encoder.layer_norm = final_norm if layers[-1] == total else nn.Identity()
            res = profile(model, device, seconds, with_latency)
            res["layers"] = layers
            out.append(res)
            print(
                f"{name} {len(layers)} layers (last {layers[-1]}): {res['params']['total'] / 1e6:.1f}M params, "
                f"{res['gflops_per_audio_s']:.1f} GFLOPs/s audio, rtf={res.get('rtf', float('nan')):.4f}",
                flush=True,
            )
    finally:
        model.encoder.layers, model.encoder.layer_norm = all_layers, final_norm
    return out


def main():
    from .extract import OUT_ROOT

    ap = argparse.ArgumentParser(description="Cost of every prefix truncation 1..L of an SSL model")
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()
    L = num_layers(args.model)
    res = profile_subsets(args.model, [range(1, k + 1) for k in range(1, L + 1)], with_latency=not args.no_latency)
    out = Path(args.out or OUT_ROOT / args.model / "compute_prefix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    out.write_text(json.dumps({"model": MODELS[args.model], "gpu": gpu, "prefixes": res}, indent=1))


if __name__ == "__main__":
    main()
