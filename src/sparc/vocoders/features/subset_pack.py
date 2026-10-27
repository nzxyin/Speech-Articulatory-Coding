"""Subsets of packed splits for the Phase 5 filtered training data.

``subset_split`` copies the rows of a packed split (``features.pack`` layout: ``feats.npy``, ``loud_raw.npy``,
``spk_l0.npy``, ``spk_l6.npy``, ``spk_enplus64.npy``, ``index.parquet``) whose ids are in a keep list into a new packed
directory with recomputed frame offsets. Rows keep their source order, all index columns (speaker, chapter, wav_path,
seed, ...) are unchanged, so the subset reads exactly as the source rows did; the source is only read. The directory is
built next to its final place and renamed.

Selection tables come from ``features.quality select`` (``select/<corpus>.parquet``, column ``keep``).

Usage (CPU)::

    python -m sparc.vocoders.features.subset_pack libritts      # filtered train.clean.100 / .360 / .500
    python -m sparc.vocoders.features.subset_pack globe         # filtered phase 4 globe.train
"""

import argparse
import json
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.vocoders.constants import N_FEATURES

SELECT_ROOT = Path("/data/user_data/xoy/spkemb/phase5/select")
OUT_ROOT = Path("/data/user_data/xoy/spkemb/phase5/packed")
LIBRITTS_PACKED = Path("/data/user_data/xoy/sparc-vocoders/cache/packed")
LIBRITTS_SPLITS = ("train.clean.100", "train.clean.360", "train.other.500")
GLOBE_PACKED = Path("/data/user_data/xoy/spkemb/phase4/globe_train/packed/globe.train")
SPK_ARRAYS = ("spk_l0", "spk_l6", "spk_enplus64")


def kept_ids(select_file: str | Path) -> set[str]:
    table = pd.read_parquet(select_file, columns=["id", "keep"])
    return set(table.loc[table["keep"], "id"].astype(str))


def runs(rows: np.ndarray) -> Iterable[tuple[int, int]]:
    """Maximal runs ``[lo, hi)`` of consecutive integers in the sorted array ``rows``."""
    if len(rows) == 0:
        return
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    for part in np.split(rows, breaks):
        yield int(part[0]), int(part[-1]) + 1


def subset_split(src: str | Path, keep: set[str], final: str | Path) -> dict:
    """Writes the rows of ``src`` whose id is in ``keep`` to ``final``; returns counts (``kept``, ``source_rows``,
    ``frames``, ``hours``, ``keep_ids_not_in_source``)."""
    src, final = Path(src), Path(final)
    index = pd.read_parquet(src / "index.parquet")
    ids = index["id"].astype(str)
    rows = np.flatnonzero(ids.isin(keep).to_numpy())
    if len(rows) == 0:
        raise ValueError(f"no row of {src} is in the keep list")
    out = index.iloc[rows].reset_index(drop=True)
    frames = out["T"].to_numpy(dtype=np.int64)
    new_offset = np.concatenate([[0], np.cumsum(frames)[:-1]])
    total = int(frames.sum())

    work = final.with_name(final.name + ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    open_array = np.lib.format.open_memmap
    feats_in, loud_in = np.load(src / "feats.npy", mmap_mode="r"), np.load(src / "loud_raw.npy", mmap_mode="r")
    feats = open_array(work / "feats.npy", mode="w+", dtype=np.float32, shape=(total, N_FEATURES))
    loud = open_array(work / "loud_raw.npy", mode="w+", dtype=np.float32, shape=(total,))
    spk_in = {n: np.load(src / f"{n}.npy", mmap_mode="r") for n in SPK_ARRAYS}
    spk = {n: open_array(work / f"{n}.npy", mode="w+", dtype=np.float32, shape=(len(rows), a.shape[1])) for n, a in spk_in.items()}
    old_offset = index["offset"].to_numpy(dtype=np.int64)
    old_frames = index["T"].to_numpy(dtype=np.int64)
    # copy run by run: consecutive kept rows are contiguous in the source arrays
    pos = 0  # position in the output row list
    for lo, hi in runs(rows):
        n = hi - lo
        a, b = int(old_offset[lo]), int(old_offset[hi - 1] + old_frames[hi - 1])
        c = int(new_offset[pos])
        feats[c : c + (b - a)] = feats_in[a:b]
        loud[c : c + (b - a)] = loud_in[a:b]
        for name in SPK_ARRAYS:
            spk[name][pos : pos + n] = spk_in[name][lo:hi]
        pos += n
    assert pos == len(rows)
    for array in (feats, loud, *spk.values()):
        array.flush()
    del feats, loud, spk
    out["offset"] = new_offset
    out.to_parquet(work / "index.parquet", index=False)

    backup = final.with_name(final.name + ".old")
    shutil.rmtree(backup, ignore_errors=True)
    if final.exists():
        final.rename(backup)
    work.rename(final)
    shutil.rmtree(backup, ignore_errors=True)
    return {
        "source": str(src), "output": str(final), "source_rows": int(len(index)), "kept": int(len(rows)), "frames": total,
        "hours": float(out["duration"].sum() / 3600), "source_hours": float(index["duration"].sum() / 3600),
        "keep_ids_not_in_source": int(len(keep - set(ids))),
    }  # fmt: skip


def run_libritts(select_root: Path, out_root: Path, packed: Path = LIBRITTS_PACKED) -> dict:
    keep = kept_ids(select_root / "libritts_train.parquet")
    scored = set(pd.read_parquet(select_root / "libritts_train.parquet", columns=["id"])["id"].astype(str))
    packed_ids: set[str] = set()
    report = {"splits": {}}
    for split in LIBRITTS_SPLITS:
        packed_ids |= set(pd.read_parquet(packed / split / "index.parquet", columns=["id"])["id"].astype(str))
        report["splits"][split] = subset_split(packed / split, keep, out_root / "libritts_f" / split)
    report["scored_ids"], report["packed_ids"] = len(scored), len(packed_ids)
    report["packed_without_score"] = sorted(packed_ids - scored)[:20], len(packed_ids - scored)
    report["scored_not_packed"] = len(scored - packed_ids)
    report["kept_total"] = int(sum(s["kept"] for s in report["splits"].values()))
    report["hours_total"] = float(sum(s["hours"] for s in report["splits"].values()))
    return report


def run_globe(select_root: Path, out_root: Path, packed: Path = GLOBE_PACKED) -> dict:
    keep = kept_ids(select_root / "globe_train.parquet")
    scored = set(pd.read_parquet(select_root / "globe_train.parquet", columns=["id"])["id"].astype(str))
    packed_ids = set(pd.read_parquet(packed / "index.parquet", columns=["id"])["id"].astype(str))
    report = {"splits": {"globe.train": subset_split(packed, keep, out_root / "globe_f" / "globe.train")}}
    report["scored_ids"], report["packed_ids"] = len(scored), len(packed_ids)
    report["packed_without_score"] = sorted(packed_ids - scored)[:20], len(packed_ids - scored)
    report["scored_not_packed"] = len(scored - packed_ids)
    report["kept_total"] = report["splits"]["globe.train"]["kept"]
    report["hours_total"] = report["splits"]["globe.train"]["hours"]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("corpus", choices=("libritts", "globe"))
    p.add_argument("--select-root", default=str(SELECT_ROOT))
    p.add_argument("--out-root", default=str(OUT_ROOT))
    a = p.parse_args(argv)
    run = run_libritts if a.corpus == "libritts" else run_globe
    report = run(Path(a.select_root), Path(a.out_root))
    path = Path(a.out_root) / f"subset_report_{a.corpus}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
