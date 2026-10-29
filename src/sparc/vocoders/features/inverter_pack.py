"""Phase 9: packed splits whose 12 articulatory channels come from a WavLM k12 inverter.

``extract`` runs the inverter (``k12_inverter``) over the 24 kHz wavs listed in a source packed split and writes the
channels in chunks (``<out>.parts/part_<chunk>.npy``, ``[sum T, 12]`` float32, written atomically, skipped when present),
so a shard can be killed and restarted. Chunk ``c`` holds source rows ``256 c .. 256 c + 255``; shard ``k`` of ``n`` does
the chunks with ``c % n == k``. ``merge`` checks every chunk and writes ``<out>``: ``feats.npy`` with columns 0-11
replaced (columns 12-14, F0 / loudness / periodicity, are copied bit for bit), every other file copied unchanged
(``loud_raw.npy``, ``spk_*.npy``, ``index.parquet``). ``stats`` writes a statistics file for the new channels.

``stats`` copies every field of the existing training statistics (``train_stats.json``, LibriTTS-R train splits) and
replaces ``ema_mean`` / ``ema_std`` by the moments of the new channels over the given packed splits (use the
LibriTTS-R training splits only); the existing file is only read.

Usage (GPU for ``extract``)::

    python -m sparc.vocoders.features.inverter_pack extract --inverter mngu0 --src <packed split> --out <new split> \
        [--shard 0 --nshards 4] [--limit-chunks N]
    python -m sparc.vocoders.features.inverter_pack merge   --inverter mngu0 --src <packed split> --out <new split>
    python -m sparc.vocoders.features.inverter_pack stats   --splits <new split> ... --out <stats.json>
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.vocoders.constants import N_EMA, N_FEATURES
from sparc.vocoders.features.stats import Moments, save_stats

CHUNK = 256
EXISTING_STATS = Path("/data/user_data/xoy/sparc-vocoders/cache/stats/train_stats.json")
COPIED = ("loud_raw.npy", "spk_l0.npy", "spk_l6.npy", "spk_enplus64.npy", "index.parquet")


def parts_dir(out: Path) -> Path:
    return out.with_name(out.name + ".parts")


def chunk_ranges(frames: np.ndarray, chunk: int | None = None) -> list[tuple[int, int]]:
    """Row ranges ``[lo, hi)`` of the chunks of a split with ``len(frames)`` rows."""
    chunk = chunk or CHUNK
    return [(lo, min(lo + chunk, len(frames))) for lo in range(0, len(frames), chunk)]


def chunk_path(out: Path, c: int) -> Path:
    return parts_dir(out) / f"part_{c:06d}.npy"


def save_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def write_meta(out: Path, inv, src: Path) -> None:
    meta = {"src": str(src), "chunk": CHUNK, **inv.describe()}
    path = parts_dir(out) / "meta.json"
    if path.is_file():
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in ("src", "chunk", "sha256", "mode")} != {k: meta[k] for k in ("src", "chunk", "sha256", "mode")}:
            raise ValueError(f"{path} was written for a different source, inverter or chunk size")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1))


def run_extract(src: Path, out: Path, mode: str, shard: int, nshards: int, device: str, limit_chunks: int | None) -> int:
    import soundfile as sf

    from sparc.vocoders.features.k12_inverter import K12Inverter

    index = pd.read_parquet(src / "index.parquet")
    frames = index["T"].to_numpy(dtype=np.int64)
    todo = [(c, lo, hi) for c, (lo, hi) in enumerate(chunk_ranges(frames)) if c % nshards == shard and not chunk_path(out, c).is_file()]
    if limit_chunks:
        todo = todo[:limit_chunks]
    if not todo:
        return 0
    inv = K12Inverter(mode, device)
    write_meta(out, inv, src)
    start, done = time.time(), 0
    for c, lo, hi in todo:
        pieces = []
        for _, r in index.iloc[lo:hi].iterrows():
            wav, rate = sf.read(r["wav_path"], dtype="float32", always_2d=False)
            if rate != 24000 or wav.ndim != 1 or len(wav) != int(r["n24"]):
                raise ValueError(f"{r['id']}: expected {int(r['n24'])} mono samples at 24 kHz, got {wav.shape} at {rate}")
            ch = inv.channels24(wav)
            if len(ch) != int(r["T"]):
                raise ValueError(f"{r['id']}: {len(ch)} frames, packed T {int(r['T'])}")
            pieces.append(ch)
        save_atomic(chunk_path(out, c), np.concatenate(pieces).astype(np.float32))
        done += 1
        print(f"shard {shard}/{nshards}: chunk {c} done ({done}/{len(todo)}, {time.time() - start:.0f} s)", flush=True)
    return done


def run_merge(src: Path, out: Path) -> dict:
    index = pd.read_parquet(src / "index.parquet")
    frames = index["T"].to_numpy(dtype=np.int64)
    ranges = chunk_ranges(frames)
    missing = [c for c in range(len(ranges)) if not chunk_path(out, c).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(ranges)} chunks are missing, first {missing[:5]}")
    offsets = np.concatenate([[0], np.cumsum(frames)])
    if not np.array_equal(offsets[:-1], index["offset"].to_numpy(dtype=np.int64)):
        raise ValueError("the source index offsets are not the cumulative frame counts")
    total = int(offsets[-1])
    feats_in = np.load(src / "feats.npy", mmap_mode="r")
    if feats_in.shape != (total, N_FEATURES):
        raise ValueError(f"{src}/feats.npy has shape {feats_in.shape}, index says {(total, N_FEATURES)}")
    work = out.with_name(out.name + ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    feats = np.lib.format.open_memmap(work / "feats.npy", mode="w+", dtype=np.float32, shape=(total, N_FEATURES))
    for c, (lo, hi) in enumerate(ranges):
        a, b = int(offsets[lo]), int(offsets[hi])
        part = np.load(chunk_path(out, c))
        if part.shape != (b - a, N_EMA):
            raise ValueError(f"chunk {c} has shape {part.shape}, expected {(b - a, N_EMA)}")
        if not np.isfinite(part).all():
            raise ValueError(f"chunk {c} has non-finite values")
        feats[a:b, :N_EMA] = part
        feats[a:b, N_EMA:] = feats_in[a:b, N_EMA:]
    feats.flush()
    del feats
    for name in COPIED:
        shutil.copyfile(src / name, work / name)
    shutil.copyfile(parts_dir(out) / "meta.json", work / "inverter.json")
    backup = out.with_name(out.name + ".old")
    shutil.rmtree(backup, ignore_errors=True)
    if out.exists():
        out.rename(backup)
    work.rename(out)
    shutil.rmtree(backup, ignore_errors=True)
    print(f"merged {len(ranges)} chunks, {len(index)} utterances, {total} frames -> {out}")
    return {"utterances": int(len(index)), "frames": total, "output": str(out)}


def run_stats(splits: list[Path], out: Path, existing: Path = EXISTING_STATS, chunk_frames: int = 2_000_000) -> dict:
    stats = json.loads(Path(existing).read_text())
    ema, utterances = Moments(N_EMA), 0
    for directory in splits:
        feats = np.load(directory / "feats.npy", mmap_mode="r")
        for lo in range(0, feats.shape[0], chunk_frames):
            ema.update(np.asarray(feats[lo : lo + chunk_frames, :N_EMA]))
        utterances += len(pd.read_parquet(directory / "index.parquet", columns=["id"]))
    if ema.count == 0:
        raise ValueError("no frames in the given splits")
    stats["ema_mean"], stats["ema_std"] = ema.mean().tolist(), ema.std().tolist()
    stats["inverter_stats"] = {"base": str(existing), "splits": [str(s) for s in splits], "frames": int(ema.count), "utterances": utterances}
    stats["frames"], stats["utterances"], stats["splits"] = int(ema.count), utterances, [str(s) for s in splits]
    save_stats(stats, out)
    print(f"ema statistics over {utterances} utterances, {ema.count} frames -> {out}")
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "merge"):
        s = sub.add_parser(name)
        s.add_argument("--src", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--inverter", required=name == "extract", choices=("mngu0", "all7"))
    sub.choices["extract"].add_argument("--shard", type=int, default=0)
    sub.choices["extract"].add_argument("--nshards", type=int, default=1)
    sub.choices["extract"].add_argument("--device", default="auto")
    sub.choices["extract"].add_argument("--limit-chunks", type=int, default=None)
    s = sub.add_parser("stats")
    s.add_argument("--splits", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--existing", default=str(EXISTING_STATS))
    a = p.parse_args(argv)
    if a.cmd == "extract":
        run_extract(Path(a.src), Path(a.out), a.inverter, a.shard, a.nshards, a.device, a.limit_chunks)
    elif a.cmd == "merge":
        run_merge(Path(a.src), Path(a.out))
    else:
        if Path(a.out).resolve() == Path(a.existing).resolve():
            raise ValueError("refusing to overwrite the existing statistics file")
        run_stats([Path(x) for x in a.splits], Path(a.out), Path(a.existing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
