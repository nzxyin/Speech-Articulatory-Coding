"""Inference cost of Whisper encoder prefixes (docs/whisper_probe_plan.md, section 5).

(A) actual, stock use: the 30 s padded forward, so FLOPs per utterance are fixed (`flops_per_window_padded`) and per second
    of audio they are F_pad / duration; `gflops_per_audio_s` is F_pad / 4 s so `depth_sweep.gflops()` reads the actual cost.
(B) idealized: convolutions and attention over the real frames only (`PrefixEncoder.forward_ideal`), a lower bound the
    checkpoint cannot deliver; reported as `gflops_per_audio_s_ideal`. Never used to produce probe features.
FLOPs: torch FlopCounterMode, 1 MAC = 2 FLOPs, conv front end and positional embedding included, STFT/mel excluded.
Latency: batch 1, warm-up 5, median of 20 (the `compute.latency` protocol), padded forward for every prefix and the
idealized forward for a subset. Activation memory: peak CUDA allocation minus weights.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .compute import REF_SECONDS
from .whisper_encoders import MEL_FRAMES, WHISPER, PrefixEncoder, load_encoder, n_real_frames

IDEAL_LATENCY_K = (6, 12, 18, 24)  # plus the last layer


def _mel(enc, frames, device, dtype):
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, enc.num_mel_bins, frames, generator=g).to(device=device, dtype=dtype)


def param_counts(pre):
    enc = pre.enc
    conv = sum(p.numel() for m in (enc.conv1, enc.conv2) for p in m.parameters())
    pos = enc.embed_positions.weight.numel()
    layers = sum(p.numel() for layer in enc.layers[: pre.k] for p in layer.parameters())
    ln = sum(p.numel() for p in enc.layer_norm.parameters()) if pre.k == pre.last else 0
    return {"conv": conv, "pos_embed": pos, "layers": layers, "final_norm": ln, "total": conv + pos + layers + ln}


def count_flops(pre, ideal=False, seconds=REF_SECONDS):
    dev = next(pre.enc.parameters()).device
    dtype = next(pre.enc.parameters()).dtype
    n_w = n_real_frames(round(seconds * 16000))
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        if ideal:
            pre.forward_ideal(_mel(pre.enc, 2 * n_w, dev, dtype), n_w)
        else:
            pre(_mel(pre.enc, MEL_FRAMES, dev, dtype))
    return int(fc.get_total_flops())


@torch.no_grad()
def _timed(fn, device, warmup=5, iters=20):
    times = []
    for i in range(warmup + iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


@torch.no_grad()
def _peak(fn, device):
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    fn()
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device) - base)


def profile_prefix(pre, device, seconds=REF_SECONDS, with_latency=True, ideal_latency=False):
    dtype = next(pre.enc.parameters()).dtype
    n_w = n_real_frames(round(seconds * 16000))
    f_pad, f_ideal = count_flops(pre, False, seconds), count_flops(pre, True, seconds)
    params = param_counts(pre)
    out = {
        "params": params,
        "weight_bytes": params["total"] * next(pre.enc.parameters()).element_size(),
        "ref_seconds": seconds,
        "window_seconds": 30,
        "flops_per_window_padded": f_pad,
        "flops": f_pad,
        "gflops_per_audio_s": f_pad / seconds / 1e9,  # actual (padded) cost per second at the 4 s reference
        "gflops_per_audio_s_30s": f_pad / 30 / 1e9,  # best case: a full 30 s window
        "flops_ideal": f_ideal,
        "gflops_per_audio_s_ideal": f_ideal / seconds / 1e9,
    }
    if str(device).startswith("cuda"):
        mel_pad = _mel(pre.enc, MEL_FRAMES, device, dtype)
        out["activation_bytes"] = _peak(lambda: pre(mel_pad), device)
        if with_latency:
            med = _timed(lambda: pre(mel_pad), device)
            out.update({"latency_s": med, "rtf": med / seconds})
        if ideal_latency:
            mel_r = _mel(pre.enc, 2 * n_w, device, dtype)
            out["activation_bytes_ideal"] = _peak(lambda: pre.forward_ideal(mel_r, n_w), device)
            med = _timed(lambda: pre.forward_ideal(mel_r, n_w), device)
            out.update({"latency_s_ideal": med, "rtf_ideal": med / seconds})
    return out


def main():
    from .extract import OUT_ROOT

    ap = argparse.ArgumentParser(description="Cost of every prefix 1..L of a Whisper encoder (padded and idealized)")
    ap.add_argument("model", choices=sorted(WHISPER))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="redo a finished profile")
    args = ap.parse_args()
    target = Path(args.out or OUT_ROOT / args.model / "compute_prefix.json")
    if target.exists() and not args.force:
        print(f"{target} exists; use --force to redo")
        return
    enc = load_encoder(args.model).to(args.device)
    L = len(enc.layers)
    ideal_ks = set(IDEAL_LATENCY_K) | {L}
    res = []
    for k in range(1, L + 1):
        pre = PrefixEncoder(enc, k)
        r = profile_prefix(pre, args.device, with_latency=not args.no_latency, ideal_latency=k in ideal_ks and not args.no_latency)
        r["layers"] = list(range(1, k + 1))
        res.append(r)
        print(f"{args.model} k={k}: {r['gflops_per_audio_s']:.1f} GFLOPs/s padded, {r['gflops_per_audio_s_ideal']:.1f} ideal, "
              f"latency {r.get('latency_s', float('nan')):.4f}s", flush=True)
    out = Path(args.out or OUT_ROOT / args.model / "compute_prefix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name() if str(args.device).startswith("cuda") else "cpu"
    out.write_text(json.dumps({"model": WHISPER[args.model], "gpu": gpu, "padded_window": True, "prefixes": res}, indent=1))


if __name__ == "__main__":
    main()
