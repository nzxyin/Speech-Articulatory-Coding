"""All-layer pooled WavLM Large speaker statistics for the packed splits (speaker-embedding Phase 3).

For every utterance of a packed split: the periodicity-weighted mean and standard deviation over frames of all 25
WavLM Large hidden states (``hidden_states[0]`` = CNN + positional-conv output, 1..24 = transformer layers), the
same pooling as :func:`sparc.vocoders.features.extractor.pool_speaker`. The weights are the cached periodicity
channel (``feats[:, PERIODICITY_CHANNEL]``), which is the thresholded CREPE periodicity the cache pooled with, so
no CREPE pass is needed. Audio is resampled 24 -> 16 kHz (soxr_hq) and z-scored as in ``SpeechWave``
preprocessing; WavLM runs in fp32 with TF32 off and batch size 1, so layers 0 and 6 reproduce the cached
``spk_l0`` / ``spk_l6`` (checked on every utterance).

Outputs (``--out``, default ``$SPKEMB_P3_CACHE`` or ``/data/user_data/xoy/spkemb/phase3/cache``):
``<split>/mean.npy`` and ``<split>/std.npy`` ``[N, 25, 1024]`` float16 in the row order of the split's
``index.parquet``, ``<split>/ids.txt``, ``<split>/check.json``; ``stats.npz`` (per-layer mean / std of both over the
training splits) for the encoder's input z-score.

Usage::

    python -m sparc.vocoders.features.spk_layers extract --split train.clean.100 --shard 0 --nshards 8
    python -m sparc.vocoders.features.spk_layers merge --split train.clean.100 --nshards 8
    python -m sparc.vocoders.features.spk_layers stats
"""

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from sparc.vocoders.constants import PERIODICITY_CHANNEL, SAMPLE_RATE

N_LAYERS = 25
DIM = 1024
SV_ROOT = Path(os.environ.get("SV_ROOT", "/data/user_data/xoy/sparc-vocoders"))
PACKED = SV_ROOT / "cache" / "packed"
OUT = Path(os.environ.get("SPKEMB_P3_CACHE", "/data/user_data/xoy/spkemb/phase3/cache"))
WAVLM_DIR = SV_ROOT / "hf_hub" / "models--microsoft--wavlm-large" / "snapshots" / "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
TRAIN_SPLITS = ("train.clean.100", "train.clean.360", "train.other.500")


def pooled_stats(hidden: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and std over frames of ``hidden [L, frames, D]`` with ``weights [frames']``.

    Both are truncated to the shorter length; zero total weight falls back to uniform weights (as ``pool_speaker``).
    Returns float32 ``[L, D]`` arrays.
    """
    frames = min(hidden.shape[1], weights.shape[0])
    if frames == 0:
        raise ValueError("cannot pool zero frames")
    h = hidden[:, :frames].astype(np.float64)
    w = weights[:frames].astype(np.float64)
    if not w.sum() > 0.0:
        w = np.ones(frames)
    w = w / w.sum()
    mean = np.einsum("t,ltd->ld", w, h)
    var = np.einsum("t,ltd->ld", w, (h - mean[:, None]) ** 2)
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def load_wavlm(device: torch.device):
    from transformers import WavLMModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = WavLMModel.from_pretrained(str(WAVLM_DIR)).eval().to(device)
    if len(model.encoder.layers) != N_LAYERS - 1:
        raise ValueError(f"expected {N_LAYERS - 1} transformer layers, got {len(model.encoder.layers)}")
    return model


@torch.no_grad()
def hidden_states(model, wav24: np.ndarray, device: torch.device) -> np.ndarray:
    """All 25 hidden states ``[25, frames, 1024]`` of one 24 kHz utterance, preprocessed as the feature cache."""
    raw16 = librosa.resample(wav24, orig_sr=SAMPLE_RATE, target_sr=16000, res_type="soxr_hq")
    wav = (raw16 - raw16.mean()) / raw16.std()
    x = torch.from_numpy(wav).float()[None].to(device)
    out = model(x, attention_mask=torch.ones_like(x), output_hidden_states=True)
    return torch.stack([s[0] for s in out.hidden_states]).float().cpu().numpy()


def shard_rows(n: int, shard: int, nshards: int) -> range:
    bounds = np.linspace(0, n, nshards + 1).astype(int)
    return range(bounds[shard], bounds[shard + 1])


def extract(split: str, shard: int, nshards: int, out: Path, limit: int | None = None) -> None:
    packed = PACKED / split
    index = pd.read_parquet(packed / "index.parquet", columns=["id", "wav_path", "T", "offset"])
    rows = shard_rows(len(index), shard, nshards)
    if limit is not None:
        rows = rows[:limit]
    dest = out / split / "shards"
    dest.mkdir(parents=True, exist_ok=True)
    tag = f"{shard:03d}of{nshards:03d}"
    if (dest / f"{tag}.done").exists():
        print(f"{split} shard {tag} already done")
        return
    feats = np.load(packed / "feats.npy", mmap_mode="r")
    ref = {k: np.load(packed / f"spk_{k}.npy", mmap_mode="r") for k in ("l0", "l6")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_wavlm(device)
    means = np.zeros((len(rows), N_LAYERS, DIM), np.float16)
    stds = np.zeros((len(rows), N_LAYERS, DIM), np.float16)
    worst = {"l0": 1.0, "l6": 1.0}
    for j, r in enumerate(rows):
        wav24, rate = sf.read(str(index.wav_path[r]), dtype="float32", always_2d=False)
        if rate != SAMPLE_RATE or wav24.ndim != 1:
            raise ValueError(f"{index.id[r]}: unexpected audio {wav24.shape} at {rate} Hz")
        begin, frames = int(index.offset[r]), int(index["T"][r])
        weights = np.asarray(feats[begin : begin + frames, PERIODICITY_CHANNEL], dtype=np.float32)
        mean, std = pooled_stats(hidden_states(model, wav24, device), weights)
        for key, layer in (("l0", 0), ("l6", 6)):
            a, b = mean[layer].astype(np.float64), np.asarray(ref[key][r], dtype=np.float64)
            worst[key] = min(worst[key], float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        means[j], stds[j] = mean, std
        if j % 500 == 0:
            print(f"{split} {tag}: {j}/{len(rows)} worst cosine vs cache {worst}", flush=True)
    if min(worst.values()) < 0.9999:
        raise RuntimeError(f"pooled layers 0/6 do not reproduce the cache: worst cosine {worst}")
    np.save(dest / f"{tag}_mean.npy", means)
    np.save(dest / f"{tag}_std.npy", stds)
    json.dump({"rows": [rows.start, rows.stop], "worst_cosine": worst}, open(dest / f"{tag}.json", "w"))
    (dest / f"{tag}.done").touch()
    print(f"{split} shard {tag}: {len(rows)} utterances, worst cosine vs cache {worst}")


def merge(split: str, nshards: int, out: Path) -> None:
    index = pd.read_parquet(PACKED / split / "index.parquet", columns=["id"])
    src = out / split / "shards"
    tags = [f"{s:03d}of{nshards:03d}" for s in range(nshards)]
    missing = [t for t in tags if not (src / f"{t}.done").exists()]
    if missing:
        raise RuntimeError(f"{split}: shards not done: {missing}")
    checks = [json.load(open(src / f"{t}.json")) for t in tags]
    if checks[0]["rows"][0] != 0 or checks[-1]["rows"][1] != len(index):
        raise RuntimeError(f"{split}: shards do not cover all {len(index)} rows")
    for a, b in zip(checks, checks[1:]):
        if a["rows"][1] != b["rows"][0]:
            raise RuntimeError(f"{split}: shard rows are not contiguous")
    for kind in ("mean", "std"):
        dst = np.lib.format.open_memmap(out / split / f"{kind}.npy", "w+", np.float16, (len(index), N_LAYERS, DIM))
        for t, c in zip(tags, checks):
            dst[c["rows"][0] : c["rows"][1]] = np.load(src / f"{t}_{kind}.npy", mmap_mode="r")
        dst.flush()
        del dst
    (out / split / "ids.txt").write_text("\n".join(index.id.astype(str)) + "\n")
    worst = {k: min(c["worst_cosine"][k] for c in checks) for k in ("l0", "l6")}
    json.dump({"utterances": len(index), "worst_cosine_vs_cache": worst}, open(out / split / "check.json", "w"))
    for t in tags:
        for kind in ("mean", "std"):
            (src / f"{t}_{kind}.npy").unlink()
    print(f"{split}: merged {len(index)} utterances, worst cosine {worst}")


def stats(out: Path, chunk: int = 4096) -> None:
    """Per-layer, per-dimension mean and std of the pooled mean and std vectors over the training splits."""
    acc = {k: [np.zeros((N_LAYERS, DIM)), np.zeros((N_LAYERS, DIM))] for k in ("mean", "std")}
    n = 0
    for split in TRAIN_SPLITS:
        for kind in ("mean", "std"):
            arr = np.load(out / split / f"{kind}.npy", mmap_mode="r")
            for i in range(0, len(arr), chunk):
                x = np.asarray(arr[i : i + chunk], dtype=np.float64)
                acc[kind][0] += x.sum(0)
                acc[kind][1] += (x**2).sum(0)
        n += len(np.load(out / split / "mean.npy", mmap_mode="r"))
    res = {}
    for kind, (s, s2) in acc.items():
        m = s / n
        res[f"{kind}_mean"] = m.astype(np.float32)
        res[f"{kind}_std"] = np.sqrt(np.maximum(s2 / n - m**2, 1e-12)).astype(np.float32)
    np.savez(out / "stats.npz", utterances=n, splits=np.array(TRAIN_SPLITS), **res)
    print(f"stats over {n} training utterances -> {out / 'stats.npz'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--split", required=True)
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--nshards", type=int, default=1)
    e.add_argument("--limit", type=int, default=None, help="first N rows of the shard only (smoke test)")
    m = sub.add_parser("merge")
    m.add_argument("--split", required=True)
    m.add_argument("--nshards", type=int, required=True)
    sub.add_parser("stats")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    if a.cmd == "extract":
        extract(a.split, a.shard, a.nshards, a.out, a.limit)
    elif a.cmd == "merge":
        merge(a.split, a.nshards, a.out)
    else:
        stats(a.out)


if __name__ == "__main__":
    main()
