# Builds train/val/test manifests for ESD's English subset (speakers
# 0011-0020), in the same TSV format used by
# build_vctk_globe_accent_splits.py (stem, source, speaker, category) so
# both can be consumed the same way downstream.
#
# Unlike the VCTK+GLOBE splits, this does NOT re-derive a split -- ESD
# already ships an official train/test/evaluation partition per speaker
# per emotion (300/30/20 utterances respectively), already perfectly
# balanced across all 5 emotions and all 10 English speakers. This script
# just reads that existing structure into the same manifest format,
# mapping ESD's "evaluation" -> "val" to match the naming convention used
# elsewhere.
#
# Important difference from the VCTK+GLOBE splits: ESD's official split is
# NOT speaker-disjoint -- all 10 speakers appear in train, val, AND test
# (just with different utterance IDs per split). This matches ESD's
# intended usage (e.g. voice conversion needs target speakers present in
# training) but is a different property than the VCTK+GLOBE splits, which
# deliberately held speakers out of test/val.

import json
from collections import defaultdict
from pathlib import Path

ESD_ROOT = Path("/data/group_data/UTD-NAS/Databases/ESD/ESD")
ENGLISH_SPEAKERS = [f"{i:04d}" for i in range(11, 21)]  # 0011-0020
EMOTIONS = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
SPLIT_MAP = {"train": "train", "test": "test", "evaluation": "val"}

OUT_DIR = Path("/data/user_data/xoy/esd_english_splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_split(name, rows_, out_dir):
    with open(out_dir / f"{name}.tsv", "w") as f:
        f.write("stem\tsource\tspeaker\temotion\n")
        for r in rows_:
            f.write(f"{r['stem']}\t{r['source']}\t{r['speaker']}\t{r['emotion']}\n")


def main():
    splits = defaultdict(list)
    stats = defaultdict(lambda: defaultdict(int))

    for spk in ENGLISH_SPEAKERS:
        for emotion in EMOTIONS:
            for esd_split, out_split in SPLIT_MAP.items():
                d = ESD_ROOT / spk / emotion / esd_split
                if not d.is_dir():
                    print(f"missing dir: {d}")
                    continue
                for wav in sorted(d.glob("*.wav")):
                    splits[out_split].append({
                        "stem": wav.stem,
                        "source": "esd",
                        "speaker": f"esd_{spk}",
                        "emotion": emotion,
                    })
                    stats[out_split][emotion] += 1

    for name, rows_ in splits.items():
        write_split(name, rows_, OUT_DIR)

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"train: {len(splits['train'])}  val: {len(splits['val'])}  test: {len(splits['test'])}")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}:", dict(stats[split_name]))


if __name__ == "__main__":
    main()
