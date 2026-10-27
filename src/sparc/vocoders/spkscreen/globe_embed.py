"""Embeds the GLOBE set (``cache extract --set globe``) with the saved encoders of finished screen runs.

Usage: ``python -m sparc.vocoders.spkscreen.globe_embed --variants baseline seg_vus3 ...`` writes
``<run>/seed<s>/globe_embeddings.npz`` (``full``, ``s1``, ``s2``). Encoders are loaded from ``encoder.pt`` (the z-score
buffers are part of the state dict, so ``setup`` is not called).
"""

import argparse

import numpy as np
import torch

from sparc.vocoders.spkscreen import cache as C
from sparc.vocoders.spkscreen.data import load_set
from sparc.vocoders.spkscreen.pooling import build
from sparc.vocoders.spkscreen.train import embed_all


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for v in a.variants:
        enc = build(v)
        fs = load_set("globe", enc.layers, enc.needs_mel, dev, C.ROOT)
        enc.to(dev)
        for s in a.seeds:
            d = C.ROOT / "runs" / v / f"seed{s}"
            enc.load_state_dict(torch.load(d / "encoder.pt", map_location=dev))
            np.savez(d / "globe_embeddings.npz", full=embed_all(enc, fs), s1=embed_all(enc, fs, 50), s2=embed_all(enc, fs, 100))
            print(v, s, "done", flush=True)


if __name__ == "__main__":
    main()
