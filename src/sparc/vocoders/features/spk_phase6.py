"""Per-utterance speaker-encoder inputs of Phase 6 (layer 1 mean, voiced / unvoiced / silence layer 6 means).

Phase 6 changes only the vector fed to the speaker FFN (``speaker.layer`` in ``l1l6``, ``vus3_l6``, ``l6_rep3``):

* ``l1l6``: periodicity-weighted mean of WavLM Large layer 1, then that of layer 6 (``spk_l6``), 2048-d;
* ``vus3_l6``: the screen's ``seg_vus3`` pooling of layer 6, voiced / unvoiced speech / silence means, 3072-d
  (thresholds and back-off of ``spkscreen/variants/segment.py``; means only, no standard deviations);
* ``l6_rep3``: ``spk_l6`` repeated three times (width control for ``vus3_l6``), 3072-d.

Only layer 1 and the three-way layer 6 means need new per-utterance vectors; ``spk_l6`` is read from the packed split.
They are written as side files under ``$SPKEMB_P6_CACHE`` (default ``/data/user_data/xoy/spkemb/phase6/vec``), one
directory per packed split (a *tag*) holding ``l1.npy [N, 1024]`` and ``vus3.npy [N, 3, 1024]`` (float32, row order of
the split's ``index.parquet``), ``ids.txt`` and ``check.json``. :class:`~sparc.vocoders.data.dataset.PackedStore` finds a
row by utterance id across all tags, so a subset packed split (Phase 5 filtered splits) needs no files of its own.

The pooling is the screen's own code (``spkscreen.variants.segment.segment_masks`` / ``class_means`` and
``spkscreen.pooling.pool_weights`` / ``weighted_mean``), so training vectors, cloning-evaluation references computed from
audio (``eval.clone_pools.FeatureEncoder``) and the screen agree. Frames: WavLM frames and the cached periodicity /
z-scored loudness channels are truncated to the shorter length, as ``features.spk_layers.pooled_stats`` does.

Usage (GPU for ``extract`` / ``bench``)::

    python -m sparc.vocoders.features.spk_phase6 tasks                      # list tasks
    python -m sparc.vocoders.features.spk_phase6 extract --worker 0 --nworkers 12   # resumable, shard tasks[w::W]
    python -m sparc.vocoders.features.spk_phase6 merge                      # complete tags only
    python -m sparc.vocoders.features.spk_phase6 stats                      # FFN z-score over LibriTTS-R train
    python -m sparc.vocoders.features.spk_phase6 bench --tag test.clean --limit 300   # timing and checks, writes nothing
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.vocoders.constants import LOUDNESS_CHANNEL, PERIODICITY_CHANNEL, SAMPLE_RATE, SPEAKER_RAW_DIM

P6_LAYERS = ("l1l6", "vus3_l6", "l6_rep3")
INPUT_DIMS = {"l6": SPEAKER_RAW_DIM, "l1l6": 2 * SPEAKER_RAW_DIM, "vus3_l6": 3 * SPEAKER_RAW_DIM, "l6_rep3": 3 * SPEAKER_RAW_DIM}
NEEDS_SIDE = {"l1l6": ("l1",), "vus3_l6": ("vus3",), "l6_rep3": ()}
P6_ROOT = Path(os.environ.get("SPKEMB_P6_CACHE", "/data/user_data/xoy/spkemb/phase6/vec"))
STATS_FILE = P6_ROOT.parent / "stats.npz"
SV_ROOT = Path(os.environ.get("SV_ROOT", "/data/user_data/xoy/sparc-vocoders"))
PACKED = SV_ROOT / "cache" / "packed"
P3_CACHE = Path("/data/user_data/xoy/spkemb/phase3/cache")
SHARD_ROWS = 4000
TRAIN_TAGS = ("train.clean.100", "train.clean.360", "train.other.500")
TAGS = {
    **{s: PACKED / s for s in (*TRAIN_TAGS, "dev.clean", "dev.other", "test.clean", "test.other")},
    "globe.train": Path("/data/user_data/xoy/spkemb/phase4/globe_train/packed/globe.train"),
    "vctk.train": Path("/data/user_data/xoy/spkemb/phase5/packed/vctk.train"),
    "pool_vctk": Path("/data/user_data/xoy/spkemb/phase4/pools/vctk/packed/vctk"),
    "pool_globe": Path("/data/user_data/xoy/spkemb/phase4/pools/globe/packed/globe"),
}


# ------------------------------------------------------------------------------------------------ composition


def compose_input(kind: str, l6, l1=None, vus3=None):
    """Speaker-FFN input of layout ``kind`` from ``l6 [..., 1024]``, ``l1 [..., 1024]`` and ``vus3 [..., 3, 1024]``
    (numpy or torch; the arrays a layout does not use may be None). ``l6`` itself is returned unchanged for ``l6``."""
    if kind == "l6":
        return l6
    lib = _lib(l6)
    if kind == "l1l6":
        return lib.concatenate([l1, l6], -1) if lib is np else lib.cat([l1, l6], -1)
    if kind == "vus3_l6":
        return vus3.reshape(*vus3.shape[:-2], 3 * SPEAKER_RAW_DIM)
    if kind == "l6_rep3":
        return lib.concatenate([l6, l6, l6], -1) if lib is np else lib.cat([l6, l6, l6], -1)
    raise ValueError(f"unknown speaker layout {kind!r}")


def _lib(x):
    if isinstance(x, np.ndarray):
        return np
    import torch

    return torch


def pool_inputs(h1, h6, periodicity, zloud) -> dict[str, np.ndarray]:
    """``{"l1", "l6", "vus3"}`` (float32 ``[1024]``, ``[1024]``, ``[3, 1024]``) of one utterance.

    ``h1`` / ``h6`` are WavLM Large hidden states 1 and 6 ``[frames, 1024]``; ``periodicity`` / ``zloud`` the cached
    feature channels 14 and 13 ``[T]`` (torch tensors or numpy arrays). All run through the screen's pooling code.
    """
    import torch

    from sparc.vocoders.spkscreen.data import FrameBatch
    from sparc.vocoders.spkscreen.pooling import pool_weights, weighted_mean
    from sparc.vocoders.spkscreen.variants.segment import class_means, segment_masks

    def tensor(x, device):
        return torch.as_tensor(np.array(x, np.float32) if isinstance(x, np.ndarray) else x).float().to(device)

    device = h6.device if isinstance(h6, torch.Tensor) else torch.device("cpu")
    h1, h6 = tensor(h1, device), tensor(h6, device)
    frames = min(h6.shape[0], len(periodicity), len(zloud))
    if frames == 0:
        raise ValueError("cannot pool zero frames")
    per = tensor(periodicity, device)[None, :frames]
    zl = tensor(zloud, device)[None, :frames]
    mask = torch.ones(1, frames, dtype=torch.bool, device=device)
    h6, h1 = h6[None, :frames], h1[None, :frames]
    w = pool_weights(per, mask)
    base = weighted_mean(h6, w)
    batch = FrameBatch(h6[:, None], per, zl, torch.zeros_like(per), torch.zeros_like(per), None, mask)
    vus3, _ = class_means(h6, segment_masks(batch), base)
    return {
        "l1": weighted_mean(h1, w)[0].cpu().numpy().astype(np.float32),
        "l6": base[0].cpu().numpy().astype(np.float32),
        "vus3": vus3[0].cpu().numpy().astype(np.float32),
    }


def input_stats(kind: str, stats: dict, p6_stats: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, std)`` of the FFN z-score for layout ``kind``: the comparison's ``spk_l6`` statistics, the Phase 6 file
    ``stats.npz`` (layer 1 and three-way statistics over the LibriTTS-R training splits) and tilings / concatenations."""
    m6, s6 = np.asarray(stats["spk_l6_mean"], np.float64), np.asarray(stats["spk_l6_std"], np.float64)
    if kind == "l6":
        return m6, s6
    if kind == "l6_rep3":
        return np.tile(m6, 3), np.tile(s6, 3)
    with np.load(p6_stats or STATS_FILE) as z:
        if kind == "l1l6":
            return np.concatenate([z["l1_mean"], m6]), np.concatenate([z["l1_std"], s6])
        if kind == "vus3_l6":
            return z["vus3_mean"].reshape(-1), z["vus3_std"].reshape(-1)
    raise ValueError(f"unknown speaker layout {kind!r}")


# ------------------------------------------------------------------------------------------------ side files


def side_dirs(root: str | Path = P6_ROOT) -> list[Path]:
    """Tag directories under ``root`` that are merged (have ``ids.txt``), sorted."""
    return sorted(p.parent for p in Path(root).glob("*/ids.txt"))


class SideIndex:
    """Utterance id -> (array position, row) over the merged tags of ``root`` (first tag wins on a duplicate id)."""

    def __init__(self, root: str | Path = P6_ROOT):
        self.dirs = side_dirs(root)
        self.where: dict[str, tuple[int, int]] = {}
        for d, directory in enumerate(self.dirs):
            for row, uid in enumerate((directory / "ids.txt").read_text().split()):
                self.where.setdefault(uid, (d, row))
        self._arrays: dict[str, list] = {}

    def locate(self, ids) -> tuple[np.ndarray, np.ndarray]:
        missing = [str(u) for u in ids if str(u) not in self.where]
        if missing:
            raise KeyError(f"{len(missing)} utterance ids have no Phase 6 vectors (tags: {[d.name for d in self.dirs]}), e.g. {missing[:3]}")
        pairs = [self.where[str(u)] for u in ids]
        return np.array([p[0] for p in pairs], np.int64), np.array([p[1] for p in pairs], np.int64)

    def array(self, kind: str, d: int):
        if kind not in self._arrays:
            self._arrays[kind] = [None] * len(self.dirs)
        if self._arrays[kind][d] is None:
            self._arrays[kind][d] = np.load(self.dirs[d] / f"{kind}.npy", mmap_mode="r")
        return self._arrays[kind][d]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state


# ------------------------------------------------------------------------------------------------ extraction


class _Stop(Exception):
    pass


def load_truncated_wavlm(device):
    """WavLM Large (fp32, TF32 off) and a function giving hidden states 1 and 6 of a 24 kHz waveform.

    The forward pass stops after transformer layer 6 (hooks on layers 1 and 6): ``hidden_states[k]`` of the full model
    is the output of transformer layer ``k`` for ``k < 24``, so nothing above layer 6 is needed.
    """
    import librosa
    import torch

    from sparc.vocoders.features.spk_layers import load_wavlm

    model = load_wavlm(device)
    got: dict[int, torch.Tensor] = {}

    def hook(k):
        def fn(_module, _inputs, output):
            got[k] = (output[0] if isinstance(output, tuple) else output)[0]
            if k == 6:
                raise _Stop

        return fn

    model.encoder.layers[0].register_forward_hook(hook(1))
    model.encoder.layers[5].register_forward_hook(hook(6))

    @torch.no_grad()
    def run(wav24: np.ndarray):
        raw16 = librosa.resample(wav24, orig_sr=SAMPLE_RATE, target_sr=16000, res_type="soxr_hq")
        wav = (raw16 - raw16.mean()) / raw16.std()
        x = torch.from_numpy(wav).float()[None].to(device)
        try:
            model(x, attention_mask=torch.ones_like(x))
        except _Stop:
            pass
        return got[1].float(), got[6].float()

    return model, run


def tasks(shard_rows: int = SHARD_ROWS) -> list[tuple[str, int, int, int]]:
    """``(tag, shard, row_start, row_stop)`` of every shard of every tag, tag by tag."""
    out = []
    for tag, directory in TAGS.items():
        n = len(pd.read_parquet(directory / "index.parquet", columns=["id"]))
        for s, lo in enumerate(range(0, n, shard_rows)):
            out.append((tag, s, lo, min(lo + shard_rows, n)))
    return out


def shard_path(out: Path, tag: str, shard: int) -> Path:
    return out / tag / "shards" / f"{shard:04d}"


def extract_rows(directory: Path, lo: int, hi: int, run, log_every: int = 500):
    """Vectors of rows ``lo .. hi`` of a packed split: ``l1 [n, 1024]``, ``vus3 [n, 3, 1024]`` and worst cosine of the
    recomputed layer-6 mean against the packed ``spk_l6``."""
    import soundfile as sf

    index = pd.read_parquet(directory / "index.parquet", columns=["id", "wav_path", "T", "offset"])
    feats = np.load(directory / "feats.npy", mmap_mode="r")
    ref = np.load(directory / "spk_l6.npy", mmap_mode="r")
    l1 = np.zeros((hi - lo, SPEAKER_RAW_DIM), np.float32)
    vus3 = np.zeros((hi - lo, 3, SPEAKER_RAW_DIM), np.float32)
    worst = 1.0
    for j, r in enumerate(range(lo, hi)):
        wav24, rate = sf.read(str(index.wav_path[r]), dtype="float32", always_2d=False)
        if rate != SAMPLE_RATE or wav24.ndim != 1:
            raise ValueError(f"{index.id[r]}: unexpected audio {wav24.shape} at {rate} Hz")
        begin, frames = int(index.offset[r]), int(index["T"][r])
        block = np.asarray(feats[begin : begin + frames])
        h1, h6 = run(wav24)
        out = pool_inputs(h1, h6, block[:, PERIODICITY_CHANNEL], block[:, LOUDNESS_CHANNEL])
        a, b = out["l6"].astype(np.float64), np.asarray(ref[r], np.float64)
        worst = min(worst, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        l1[j], vus3[j] = out["l1"], out["vus3"]
        if log_every and j % log_every == 0:
            print(f"  row {r} ({j}/{hi - lo}) worst cosine of layer 6 vs spk_l6 {worst:.9f}", flush=True)
    return l1, vus3, worst


def extract(worker: int, nworkers: int, out: Path, tags: set[str] | None = None) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, run = load_truncated_wavlm(device)
    mine = [t for t in tasks() if tags is None or t[0] in tags][worker::nworkers]
    print(f"worker {worker}/{nworkers}: {len(mine)} shards", flush=True)
    for tag, shard, lo, hi in mine:
        base = shard_path(out, tag, shard)
        if base.with_suffix(".done").exists():
            continue
        base.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        l1, vus3, worst = extract_rows(TAGS[tag], lo, hi, run)
        if worst < 1.0 - 1e-6:
            raise RuntimeError(f"{tag} shard {shard}: layer 6 does not reproduce spk_l6 (worst cosine {worst})")
        tmp = base.parent / f"{base.name}.tmp"
        np.save(f"{tmp}_l1.npy", l1)
        np.save(f"{tmp}_vus3.npy", vus3)
        os.replace(f"{tmp}_l1.npy", f"{base}_l1.npy")
        os.replace(f"{tmp}_vus3.npy", f"{base}_vus3.npy")
        json.dump({"rows": [lo, hi], "worst_cosine_l6": worst, "seconds": time.time() - t0}, open(f"{base}.json", "w"))
        base.with_suffix(".done").touch()
        print(f"{tag} shard {shard} rows {lo}-{hi}: {time.time() - t0:.0f} s, worst cosine {worst:.9f}", flush=True)
    finish(out)


def _locked(path: Path) -> bool:
    """True when this process created ``path`` (an exclusive lock file); False if another process holds it."""
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def finish(out: Path) -> None:
    """Merges every tag whose shards are all done and writes ``stats.npz`` once the training tags are merged. Safe to call
    from several workers: one lock file per step (delete ``<out>/<tag>/merge.lock`` / ``<out>/stats.lock`` after a crash)."""
    for tag in TAGS:
        if (out / tag / "ids.txt").exists() or not (out / tag / "shards").is_dir():
            continue
        n = len(pd.read_parquet(TAGS[tag] / "index.parquet", columns=["id"]))
        if all(shard_path(out, tag, s).with_suffix(".done").exists() for _, s, _, _ in tasks_of(tag, n)):
            lock = out / tag / "merge.lock"
            if _locked(lock):
                merge(out, {tag})
    done = all((out / t / "ids.txt").exists() for t in TRAIN_TAGS)
    if done and not (out.parent / "stats.npz").exists() and _locked(out.parent / "stats.lock"):
        stats(out)


def merge(out: Path, tags: set[str] | None = None) -> None:
    for tag, directory in TAGS.items():
        if tags is not None and tag not in tags:
            continue
        ids = pd.read_parquet(directory / "index.parquet", columns=["id"])["id"].astype(str)
        mine = [t for t in tasks_of(tag, len(ids))]
        if not all(shard_path(out, tag, s).with_suffix(".done").exists() for _, s, _, _ in mine):
            print(f"{tag}: not all shards done, skipped")
            continue
        n = len(ids)
        l1 = np.lib.format.open_memmap(out / tag / "l1.npy.tmp", "w+", np.float32, (n, SPEAKER_RAW_DIM))
        v3 = np.lib.format.open_memmap(out / tag / "vus3.npy.tmp", "w+", np.float32, (n, 3, SPEAKER_RAW_DIM))
        worst = 1.0
        for _, s, lo, hi in mine:
            base = shard_path(out, tag, s)
            l1[lo:hi] = np.load(f"{base}_l1.npy", mmap_mode="r")
            v3[lo:hi] = np.load(f"{base}_vus3.npy", mmap_mode="r")
            worst = min(worst, json.load(open(f"{base}.json"))["worst_cosine_l6"])
        l1.flush(), v3.flush()
        del l1, v3
        os.replace(out / tag / "l1.npy.tmp", out / tag / "l1.npy")
        os.replace(out / tag / "vus3.npy.tmp", out / tag / "vus3.npy")
        json.dump({"utterances": n, "worst_cosine_l6_vs_spk_l6": worst}, open(out / tag / "check.json", "w"))
        (out / tag / "ids.txt").write_text("\n".join(ids) + "\n")  # last: marks the tag as complete
        print(f"{tag}: merged {n} utterances, worst cosine {worst:.9f}")


def tasks_of(tag: str, n: int, shard_rows: int = SHARD_ROWS) -> list[tuple[str, int, int, int]]:
    return [(tag, s, lo, min(lo + shard_rows, n)) for s, lo in enumerate(range(0, n, shard_rows))]


def stats(out: Path, chunk: int = 4096) -> None:
    """Mean / std of ``l1`` ``[1024]`` and the flattened ``vus3`` ``[3072]`` over the LibriTTS-R training splits."""
    acc = {k: [0.0, 0.0] for k in ("l1", "vus3")}
    n = 0
    for tag in TRAIN_TAGS:
        for kind in acc:
            arr = np.load(out / tag / f"{kind}.npy", mmap_mode="r")
            for i in range(0, len(arr), chunk):
                x = np.asarray(arr[i : i + chunk], np.float64).reshape(min(chunk, len(arr) - i), -1)
                acc[kind][0] = acc[kind][0] + x.sum(0)
                acc[kind][1] = acc[kind][1] + (x**2).sum(0)
        n += len(arr)
    res = {}
    for kind, (s, s2) in acc.items():
        m = s / n
        res[f"{kind}_mean"] = m.astype(np.float32)
        res[f"{kind}_std"] = np.maximum(np.sqrt(np.maximum(s2 / n - m**2, 0.0)), 1e-6).astype(np.float32)
    np.savez(out.parent / "stats.npz", utterances=n, splits=np.array(TRAIN_TAGS), **res)
    print(f"stats over {n} training utterances -> {out.parent / 'stats.npz'}")


def bench(tag: str, limit: int) -> None:
    """Times extraction of ``limit`` rows, checks layer 6 against ``spk_l6`` and layer 1 against the Phase 3 cache and
    against the full 25-layer forward; writes nothing."""
    import torch

    from sparc.vocoders.features.spk_layers import hidden_states, load_wavlm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, run = load_truncated_wavlm(device)
    directory = TAGS[tag]
    lo = 0
    t0 = time.time()
    # contiguous block for timing
    l1, vus3, worst = extract_rows(directory, lo, lo + limit, run, log_every=0)
    dt = time.time() - t0
    index = pd.read_parquet(directory / "index.parquet", columns=["id", "T"])
    hours = float(index["T"].iloc[lo : lo + limit].sum()) / 50 / 3600
    print(f"{tag}: {limit} utterances, {hours:.3f} audio h in {dt:.1f} s ({dt / hours:.0f} s per audio hour), device {device}")
    print(f"worst cosine layer 6 vs spk_l6: {worst:.12f}")
    p3 = P3_CACHE / tag / "mean.npy"
    if p3.exists():
        ref = np.load(p3, mmap_mode="r")[lo : lo + limit, 1].astype(np.float64)
        cos = (l1.astype(np.float64) * ref).sum(1) / (np.linalg.norm(l1, axis=1) * np.linalg.norm(ref, axis=1))
        print(f"layer 1 vs Phase 3 all-layer cache (fp16): min cosine {cos.min():.9f}")
    model = load_wavlm(device)
    import soundfile as sf

    sub = pd.read_parquet(directory / "index.parquet", columns=["wav_path", "T", "offset"])
    feats = np.load(directory / "feats.npy", mmap_mode="r")
    worst_full = 1.0
    for r in range(lo, lo + min(limit, 20)):
        wav24, _ = sf.read(str(sub.wav_path[r]), dtype="float32", always_2d=False)
        hs = hidden_states(model, wav24, device)
        block = np.asarray(feats[int(sub.offset[r]) : int(sub.offset[r]) + int(sub["T"][r])])
        full = pool_inputs(hs[1], hs[6], block[:, PERIODICITY_CHANNEL], block[:, LOUDNESS_CHANNEL])
        a, b = full["l1"].astype(np.float64), l1[r - lo].astype(np.float64)
        worst_full = min(worst_full, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        if not np.allclose(full["vus3"], vus3[r - lo], atol=1e-4):
            print(f"row {r}: vus3 differs from the full forward by {np.abs(full['vus3'] - vus3[r - lo]).max():.2e}")
    print(f"truncated forward vs full forward, layer 1 worst cosine: {worst_full:.12f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    e = sub.add_parser("extract")
    e.add_argument("--worker", type=int, default=0)
    e.add_argument("--nworkers", type=int, default=1)
    e.add_argument("--tags", nargs="*", default=None)
    m = sub.add_parser("merge")
    m.add_argument("--tags", nargs="*", default=None)
    sub.add_parser("stats")
    b = sub.add_parser("bench")
    b.add_argument("--tag", default="test.clean")
    b.add_argument("--limit", type=int, default=300)
    p.add_argument("--out", type=Path, default=P6_ROOT)
    a = p.parse_args()
    tags = set(a.tags) if getattr(a, "tags", None) else None
    if a.cmd == "tasks":
        t = tasks()
        print(len(t), "shards of", SHARD_ROWS, "rows")
        for tag in TAGS:
            print(tag, sum(1 for x in t if x[0] == tag))
    elif a.cmd == "extract":
        extract(a.worker, a.nworkers, a.out, tags)
    elif a.cmd == "merge":
        merge(a.out, tags)
    elif a.cmd == "stats":
        stats(a.out)
    else:
        bench(a.tag, a.limit)


if __name__ == "__main__":
    main()
