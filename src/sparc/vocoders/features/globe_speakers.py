"""Real speaker labels for GLOBE v3 from the Common Voice 21 transcript tables (speaker-embedding Phase 7).

GLOBE v3 is a re-release of Common Voice English; each manifest row carries ``common_voice_path``
(``common_voice_en_<id>.mp3``). The Common Voice 21.0 transcript tables (``train.tsv`` / ``dev.tsv`` / ``test.tsv`` of
``fsicoli/common_voice_21_0``, copied to ``/data/user_data/xoy/common_voice_21_en_meta/``, CC0) list the same clips with
a ``client_id`` (a salted hash of one contributor). The hash is used only as an opaque grouping key: it is never joined
to anything outside these tables, and the files written here hold an integer index (``client_idx``, rank of the hash
among all GLOBE contributors) next to it in ``speakers.parquet``; training lists and packed splits carry only the label
``gcv<client_idx:06d>``.

Commands (CPU)::

    python -m sparc.vocoders.features.globe_speakers report        # join statistics, speaker-disjointness of the splits
    python -m sparc.vocoders.features.globe_speakers ids           # speakers.parquet + list_spk.parquet of the training list
    python -m sparc.vocoders.features.globe_speakers pack          # packed split globe_f_spk/globe.train (real speaker labels)
    python -m sparc.vocoders.features.globe_speakers pool-checks   # leak checks of the real-speaker pool globe_spk

``pack`` links the feature arrays of the Phase 5 filtered GLOBE split (``phase5/packed/globe_f/globe.train``) and writes a
new ``index.parquet`` with the same rows in the same order; only ``speaker`` changes (``chapter`` stays the unique
utterance stem, so a reference is always another utterance of the speaker). Existing files are never modified.
"""

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

CV_META = "/data/user_data/xoy/common_voice_21_en_meta"
GLOBE_ROOT = "/data/user_data/xoy/GLOBE_V3"
TRAIN_LIST = "/data/user_data/xoy/spkemb/phase4/globe_train/list.parquet"
FILTERED_SPLIT = "/data/user_data/xoy/spkemb/phase5/packed/globe_f/globe.train"
OUT_ROOT = "/data/user_data/xoy/spkemb/phase7"
SPLITS = ("train", "dev", "test")
ARRAYS = ("feats", "loud_raw", "spk_l0", "spk_l6", "spk_enplus64")
MIN_REF_S = 3.0
SAMPLE_RATE = 24000


def label(idx: int) -> str:
    return f"gcv{int(idx):06d}"


def read_cv_clients(meta_dir: str | Path = CV_META) -> pd.DataFrame:
    """``path, client_id`` of every clip of the three Common Voice tables (a path appearing twice is an error)."""
    csv.field_size_limit(10**9)
    frames = [
        pd.read_csv(Path(meta_dir) / f"{s}.tsv", sep="\t", usecols=["client_id", "path"], dtype=str, quoting=csv.QUOTE_NONE)
        for s in SPLITS
    ]
    cv = pd.concat(frames, ignore_index=True)
    if not cv["path"].is_unique:
        raise ValueError("a Common Voice clip path occurs in more than one table row")
    return cv


def globe_manifest(globe_root: str | Path = GLOBE_ROOT) -> pd.DataFrame:
    return pd.read_csv(
        Path(globe_root) / "manifest.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
        usecols=["stem", "split", "accent", "common_voice_path"],
    )  # fmt: skip


def join_speakers(manifest: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    """Manifest rows with ``client_id``, ``client_idx`` (rank of the hash among the GLOBE contributors) and ``speaker``.

    Every manifest row must find its clip (the join is checked to be complete and one to one)."""
    joined = manifest.merge(cv.rename(columns={"path": "common_voice_path"}), on="common_voice_path", how="left", validate="many_to_one")
    if joined["client_id"].isna().any():
        raise ValueError(f"{int(joined['client_id'].isna().sum())} GLOBE rows have no Common Voice clip")
    ids = np.array(sorted(joined["client_id"].unique()))
    joined["client_idx"] = np.searchsorted(ids, joined["client_id"].to_numpy()).astype(np.int64)
    joined["speaker"] = [label(i) for i in joined["client_idx"]]
    return joined


def speaker_table(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per contributor: ``client_idx, client_id, speaker`` and the number of GLOBE utterances per split."""
    counts = joined.pivot_table(index=["client_idx", "client_id", "speaker"], columns="split", values="stem", aggfunc="size", fill_value=0)
    return counts.reset_index()


def disjointness(joined: pd.DataFrame) -> dict:
    sets = {s: set(joined.loc[joined["split"] == s, "client_idx"]) for s in sorted(joined["split"].unique())}
    return {f"{a}&{b}": len(sets[a] & sets[b]) for a in sets for b in sets if a < b}


def utterance_stats(index: pd.DataFrame) -> dict:
    """Per-speaker utterance counts and reference availability of a table with ``speaker`` (and ``n24`` when present)."""
    per = index.groupby("speaker").size()
    out = {
        "utterances": int(len(index)), "speakers": int(len(per)), "singleton_speakers": int((per == 1).sum()),
        "utterances_of_singletons": int(per[per == 1].sum()),
        "max_utterances_per_speaker": int(per.max()), "median_utterances_per_speaker": float(per.median()),
        "mean_utterances_per_speaker": float(per.mean()),
        "share_utterances_of_largest_speaker": float(per.max() / len(index)),
        "speakers_with_ge_10": int((per >= 10).sum()),
    }
    if "n24" in index:
        long_ = index["n24"] >= MIN_REF_S * SAMPLE_RATE
        long_per = index.assign(long=long_).groupby("speaker")["long"].transform("sum")
        other_long = long_per - long_.astype(int)  # other utterances (>= 3 s) of the same speaker
        out["utterances_with_other_long_reference"] = int((other_long >= 1).sum())
        out["share_with_other_long_reference"] = float((other_long >= 1).mean())
    return out


def report(meta_dir=CV_META, globe_root=GLOBE_ROOT, train_list=TRAIN_LIST, filtered=FILTERED_SPLIT) -> dict:
    joined = join_speakers(globe_manifest(globe_root), read_cv_clients(meta_dir))
    rep = {"manifest_rows": {s: int((joined["split"] == s).sum()) for s in SPLITS}, "joined": int(len(joined)),
           "speakers_total": int(joined["client_idx"].nunique()),
           "speakers_per_split": {s: int(joined.loc[joined["split"] == s, "client_idx"].nunique()) for s in SPLITS},
           "split_overlap_speakers": disjointness(joined)}  # fmt: skip
    lst = pd.read_parquet(train_list).merge(joined[["stem", "speaker"]], on="stem", how="left", validate="one_to_one")
    rep["training_list"] = utterance_stats(lst)
    f_idx = pd.read_parquet(Path(filtered) / "index.parquet", columns=["id", "n24"])
    f_idx = f_idx.merge(joined[["stem", "speaker"]], left_on="id", right_on="stem", how="left", validate="one_to_one")
    rep["filtered_split"] = utterance_stats(f_idx)
    return rep


def write_ids(out_root: str | Path = OUT_ROOT, **kw) -> dict:
    meta_dir, globe_root, train_list = kw.get("meta_dir", CV_META), kw.get("globe_root", GLOBE_ROOT), kw.get("train_list", TRAIN_LIST)
    joined = join_speakers(globe_manifest(globe_root), read_cv_clients(meta_dir))
    out = Path(out_root) / "globe_train"
    out.mkdir(parents=True, exist_ok=True)
    speaker_table(joined).to_parquet(out / "speakers.parquet", index=False)
    lst = pd.read_parquet(train_list)
    new = lst.merge(joined[["stem", "client_idx", "speaker"]], on="stem", how="left", validate="one_to_one")
    if new["speaker"].isna().any() or len(new) != len(lst) or not (new["stem"].to_numpy() == lst["stem"].to_numpy()).all():
        raise ValueError("training list join changed rows or lost speakers")
    new = new.rename(columns={"speaker": "speaker_cv"})
    new.to_parquet(out / "list_spk.parquet", index=False)
    return {"list_spk": str(out / "list_spk.parquet"), "rows": int(len(new)), "speakers": int(new["speaker_cv"].nunique())}


def pack_split(out_root: str | Path = OUT_ROOT, src: str | Path = FILTERED_SPLIT, **kw) -> dict:
    """``globe_f_spk/globe.train``: linked arrays of ``src`` and an index with the real speaker labels."""
    meta_dir, globe_root = kw.get("meta_dir", CV_META), kw.get("globe_root", GLOBE_ROOT)
    src = Path(src)
    final = Path(out_root) / "packed" / "globe_f_spk" / "globe.train"
    joined = join_speakers(globe_manifest(globe_root), read_cv_clients(meta_dir))
    index = pd.read_parquet(src / "index.parquet")
    spk = index[["id"]].merge(joined[["stem", "speaker"]], left_on="id", right_on="stem", how="left", validate="one_to_one")
    if spk["speaker"].isna().any():
        raise ValueError("a packed utterance has no Common Voice speaker")
    new = index.copy()
    new["speaker"] = spk["speaker"].to_numpy()
    final.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        link = final / f"{name}.npy"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((src / f"{name}.npy").resolve())
    new.to_parquet(final / "index.parquet", index=False)
    return verify_split(src, final)


def verify_split(src: str | Path, final: str | Path) -> dict:
    """Checks that ``final`` has the rows, order and features of ``src`` and only differs in ``speaker``."""
    src, final = Path(src), Path(final)
    a, b = pd.read_parquet(src / "index.parquet"), pd.read_parquet(final / "index.parquet")
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for col in a.columns:
        if col != "speaker":
            assert a[col].equals(b[col]), f"column {col} differs"
    for name in ARRAYS:
        assert Path(final / f"{name}.npy").resolve() == (src / f"{name}.npy").resolve(), name
    per = b.groupby("speaker").size()
    return {"rows": int(len(b)), "identical_columns_except_speaker": True, "features_linked": list(ARRAYS),
            "old_speakers": int(a["speaker"].nunique()), "new_speakers": int(len(per)),
            "new_speaker_max_utterances": int(per.max())}  # fmt: skip


def pool_checks(
    pool_table="/data/user_data/xoy/spkemb/phase4/pools/globe_spk/table.parquet",
    other_pool_tables=(),
    old_pool_table="/data/user_data/xoy/spkemb/phase4/pools/globe/table.parquet",
    train_list=TRAIN_LIST,
    filtered=FILTERED_SPLIT,
    threshold: float = 0.70,
    meta_dir=CV_META,
    globe_root=GLOBE_ROOT,
) -> dict:
    """Leak checks of the real-speaker pool ``globe_spk``: exact speaker overlap with the GLOBE training lists, the ECAPA
    check of Phase 4 (cosine >= ``threshold`` between a training utterance and any pool utterance; reported, not
    applied) and the overlap with the old pseudo-speaker pool's utterances."""
    from sparc.vocoders.features.globe_train import load_ecapa, max_cosine

    joined = join_speakers(globe_manifest(globe_root), read_cv_clients(meta_dir))
    pool = pd.read_parquet(pool_table)
    pool_spk = set(pool["speaker"])
    full = pd.read_parquet(train_list)[["stem"]].merge(joined[["stem", "speaker"]], on="stem")
    filt = pd.read_parquet(Path(filtered) / "index.parquet", columns=["id"]).merge(joined[["stem", "speaker"]], left_on="id", right_on="stem")
    rep = {
        "pool_speakers": len(pool_spk), "pool_utterances": int(len(pool)),
        "speaker_overlap_full_training_list": len(pool_spk & set(full["speaker"])),
        "speaker_overlap_filtered_split": len(pool_spk & set(filt["speaker"])),
        "utterance_overlap_full_training_list": int(pool["id"].isin(full["stem"]).sum()),
    }  # fmt: skip
    # speakers of every GLOBE training-split row (all of train), not only the listed ones
    all_train = set(joined.loc[joined["split"] == "train", "speaker"])
    rep["speaker_overlap_all_globe_train_rows"] = len(pool_spk & all_train)
    old = pd.read_parquet(old_pool_table)
    rep["utterances_also_in_old_pool"] = int(pool["id"].isin(old["id"]).sum())
    for t in other_pool_tables:  # e.g. the dev real-speaker pool when checking the test one: speakers must be disjoint
        other = pd.read_parquet(t)
        rep[f"speaker_overlap_{Path(t).parent.name}"] = len(pool_spk & set(other["speaker"]))
        rep[f"utterance_overlap_{Path(t).parent.name}"] = int(pool["id"].isin(other["id"]).sum())
    ecapa = load_ecapa(Path(globe_root) / "ecapa", set(pool["id"]) | set(full["stem"]))
    pool_emb = ecapa.loc[pool["id"]].to_numpy()
    for name, stems in (("full_training_list", full["stem"]), ("filtered_split", filt["stem"])):
        cos = max_cosine(ecapa.loc[stems].to_numpy(), pool_emb)
        rep[f"ecapa_ge_{threshold}_{name}"] = int((cos >= threshold).sum())
        rep[f"ecapa_ge_{threshold}_{name}_share"] = float((cos >= threshold).mean())
        rep[f"ecapa_max_{name}"] = float(cos.max())
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("report", "ids", "pack", "pool-checks"))
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--pool", default="globe_spk", help="pool-checks: real-speaker pool to check (globe_spk or globe_spk_test)")
    a = p.parse_args(argv)
    if a.cmd == "report":
        result = report()
        path = Path(a.out_root) / "globe_train" / "speaker_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1))
    elif a.cmd == "pool-checks":
        base = "/data/user_data/xoy/spkemb/phase4/pools"
        if a.pool == "globe_spk":
            result = pool_checks()
            path = Path(a.out_root) / "globe_train" / "pool_checks.json"
        else:
            result = pool_checks(pool_table=f"{base}/{a.pool}/table.parquet", other_pool_tables=(f"{base}/globe_spk/table.parquet",))
            path = Path(a.out_root) / "globe_train" / f"pool_checks_{a.pool}.json"
        path.write_text(json.dumps(result, indent=1))
    elif a.cmd == "ids":
        result = write_ids(a.out_root)
    else:
        result = pack_split(a.out_root)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
