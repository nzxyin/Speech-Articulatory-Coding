"""Embeds the out-of-domain sets with saved screen encoders; also builds the corpus-centered baseline (variant 5).

Usage::

    python -m sparc.vocoders.spkscreen.ood_embed --variants baseline ood_gruconv ... --sets heldout globe globe_spk
    python -m sparc.vocoders.spkscreen.ood_embed --center --sets heldout globe globe_spk   # runs/ood_center from baseline

Writes ``<run>/seed<s>/<set>_embeddings.npz`` (``full``, ``s1``, ``s2``; ``heldout`` writes the trainer's own
``embeddings.npz`` name only for the centered runs). Centering (variant 5): the pooled layer-6 vector of every utterance
has its evaluation corpus's mean (over all utterances of that corpus in the cache set, no labels) subtracted and the
training-set mean added back before the baseline FFN; the corpora are LibriTTS-R (dev + test), VCTK, ``globe``,
``globe_spk``. The 1 s and 2 s embeddings use the same full-utterance corpus mean.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from sparc.vocoders.spkscreen import cache as C
from sparc.vocoders.spkscreen.data import load_set
from sparc.vocoders.spkscreen.pooling import build
from sparc.vocoders.spkscreen.train import embed_all


def corpus_of(speakers: np.ndarray) -> np.ndarray:
    return np.array([s.split(":")[0].replace("libri_dev", "libri").replace("libri_test", "libri") for s in speakers])


@torch.no_grad()
def pooled_all(enc, fs, max_frames=None, chunk=64) -> np.ndarray:
    enc.eval()
    order = np.argsort(fs.lens)
    out = np.zeros((fs.n, 1024), np.float32)
    for i in range(0, fs.n, chunk):
        idx = order[i : i + chunk]
        out[idx] = enc.pooled(fs.full(idx, enc.layers, False, max_frames)).float().cpu().numpy()
    return out


@torch.no_grad()
def ffn_all(enc, V: np.ndarray, dev) -> np.ndarray:
    enc.eval()
    return enc.ffn(torch.as_tensor(V, device=dev)).float().cpu().numpy()


def centered_embeddings(enc, fs, dev) -> dict:
    corp = corpus_of(fs.speakers)
    full = pooled_all(enc, fs)
    out = {}
    for key, mf in (("full", None), ("s1", 50), ("s2", 100)):
        V = full if mf is None else pooled_all(enc, fs, mf)
        V = V.copy()
        for c in sorted(set(corp)):
            rows = corp == c
            V[rows] = V[rows] - full[rows].mean(0) + enc.ffn.mean.cpu().numpy()
        out[key] = ffn_all(enc, V, dev)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variants", nargs="*", default=[])
    p.add_argument("--center", action="store_true", help="corpus-centered baseline, written to runs/ood_center")
    p.add_argument("--sets", nargs="+", default=["heldout", "globe", "globe_spk"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--root", type=Path, default=C.ROOT)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # "<run>:<architecture>" embeds the run directory <run> with another variant's architecture (ood_aug:baseline)
    jobs = [(v.partition(":")[2] or v.partition(":")[0], v.partition(":")[0], False) for v in a.variants]
    jobs += [("baseline", "ood_center", True)] if a.center else []
    for arch, dst, center in jobs:
        src = dst if not center else "baseline"
        enc = build(arch)
        enc.to(dev)
        for st in a.sets:
            if st == "heldout" and not center:
                continue  # written by the trainer
            fs = load_set(st, enc.layers, enc.needs_mel, dev, a.root)
            for s in a.seeds:
                enc.load_state_dict(torch.load(a.root / "runs" / src / f"seed{s}" / "encoder.pt", map_location=dev))
                d = a.root / "runs" / dst / f"seed{s}"
                d.mkdir(parents=True, exist_ok=True)
                if center:
                    emb = centered_embeddings(enc, fs, dev)
                else:
                    emb = {"full": embed_all(enc, fs), "s1": embed_all(enc, fs, 50), "s2": embed_all(enc, fs, 100)}
                name = "embeddings.npz" if (st == "heldout") else f"{st}_embeddings.npz"
                np.savez(d / name, **emb)
                print(dst, st, s, "done", flush=True)
            del fs
    (a.root / "ood").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
