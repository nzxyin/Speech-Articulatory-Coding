"""Wall-clock latency and memory of the frontier encoders and heads (rigor plan item 3).

FLOP counts (FlopCounterMode: matmul / conv only) undercount recurrent and scan heads, which run sequentially. This
measures batch-1 forward latency on 1 s and 4 s of audio: truncated encoders on GPU (fp32 and bf16 autocast) and CPU
(8 threads), heads on the encoder's output, and the two-pass blend (frozen + LoRA encoder, merged LoRA has the same
cost as the frozen encoder). Median of 30 GPU / 10 CPU repeats after warm-up.

Usage: python -m sparc.compression.latency
"""

import json
import time

import numpy as np
import torch

from . import datasets
from .encoders import load_subset
from .head_screen import make_head

OUT = datasets.XLSR_EMA / "analysis_rigor"
ENCODERS = (("wavlm-large", 7), ("wavlm-large", 9), ("wavlm-large", 12), ("xlsr-300m", 18), ("xlsr-1b", 19))
HEADS = ("linear", "mlp", "conv_causal9", "conv9", "gru", "gruconv9", "mamba")
SECONDS = (1, 4)


def timeit(fn, device, reps):
    for _ in range(3 if device == "cpu" else 5):
        fn()
    ts = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    res = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "encoders": {}, "heads": {}}
    for name, k in ENCODERS:
        m = load_subset(name, range(1, k + 1))
        params = sum(p.numel() for p in m.parameters()) / 1e6
        r = {"params_M": params}
        for dev in ("cuda", "cpu"):
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            m = m.to(dev)
            for sec in SECONDS:
                x = torch.randn(1, 16000 * sec, device=dev)
                r[f"{dev}_fp32_{sec}s_ms"] = timeit(lambda: m(x), dev, 30 if dev == "cuda" else 10)
                if dev == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        r[f"cuda_bf16_{sec}s_ms"] = timeit(lambda: m(x), dev, 30)
                    torch.cuda.reset_peak_memory_stats()
                    m(x)
                    r[f"cuda_peak_mem_{sec}s_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
        res["encoders"][f"{name} k{k}"] = r
        print(name, k, json.dumps(r), flush=True)
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    for hn in HEADS:
        h = make_head(hn, 1024, smooth=False).eval()
        if hn == "linear":
            h.set_input_stats(torch.zeros(1024), torch.ones(1024))
        r = {"params_M": sum(p.numel() for p in h.parameters()) / 1e6}
        for dev in ("cuda", "cpu"):
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            h = h.to(dev)
            for sec in SECONDS:
                x = torch.randn(1, 50 * sec, 1024, device=dev)
                r[f"{dev}_{sec}s_ms"] = timeit(lambda: h(x), dev, 30 if dev == "cuda" else 10)
        res["heads"][hn] = r
        print(hn, json.dumps(r), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(res, indent=1))
    L = [f"# Latency (batch 1; GPU {res['gpu']}, CPU 8 threads; median ms)\n",
         "| encoder | params M | GPU fp32 1 s / 4 s | GPU bf16 1 s / 4 s | CPU 1 s / 4 s | GPU peak MB (4 s) |", "|---|---|---|---|---|---|"]
    for e, r in res["encoders"].items():
        L.append(f"| {e} | {r['params_M']:.0f} | {r.get('cuda_fp32_1s_ms', np.nan):.1f} / {r.get('cuda_fp32_4s_ms', np.nan):.1f} | "
                 f"{r.get('cuda_bf16_1s_ms', np.nan):.1f} / {r.get('cuda_bf16_4s_ms', np.nan):.1f} | "
                 f"{r['cpu_fp32_1s_ms']:.0f} / {r['cpu_fp32_4s_ms']:.0f} | {r.get('cuda_peak_mem_4s_MB', np.nan):.0f} |")
    L += ["\n| head (on 1024-dim features) | params M | GPU 1 s / 4 s | CPU 1 s / 4 s |", "|---|---|---|---|"]
    for hn, r in res["heads"].items():
        L.append(f"| {hn} | {r['params_M']:.2f} | {r.get('cuda_1s_ms', np.nan):.2f} / {r.get('cuda_4s_ms', np.nan):.2f} | "
                 f"{r['cpu_1s_ms']:.1f} / {r['cpu_4s_ms']:.1f} |")
    L.append("\nThe two-pass blend costs two encoder passes plus both heads (frozen probe + LoRA encoder and head).")
    (OUT / "latency.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
