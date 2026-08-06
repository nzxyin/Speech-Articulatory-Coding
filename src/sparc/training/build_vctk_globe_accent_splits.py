"""
Builds train/val/test manifests combining VCTK (all 11 native accent labels)
and the GLOBE_V2 "United States English" subset (merged into VCTK's
"American" category -- same real-world accent, different corpora's labeling
convention).

Design (stated explicitly since these are judgment calls):
- VCTK speaker p280 has no accent label in speaker-info.txt -- excluded.
- GLOBE_V2 rows are grouped by their own `speaker_id` field, analogous to
  VCTK speaker IDs, for speaker-level (no-leakage) splitting.
- TEST: exactly N_TEST_PER_ACCENT utterances per accent, drawn from
  speakers reserved entirely for test (not used in train/val), except the
  two singleton-speaker VCTK accents (Welsh: p253 only, NewZealand: p335
  only) where the single available speaker's utterances are split by
  utterance across train/val/test instead, since a full speaker-level
  holdout would erase that accent from train/val entirely -- flagged in
  the output stats.
- TRAIN/VAL: everything else, split ~90/10, accent representation kept
  proportional to natural frequency (no rebalancing) via speaker-level
  assignment (whole speakers into train or val) wherever more than one
  non-reserved speaker remains for that accent.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

random.seed(0)

N_TEST_PER_ACCENT = 40
VAL_FRACTION = 0.10

VCTK_WAV_ROOT = Path("/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/wav48")
VCTK_SPK_INFO = Path("/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/speaker-info.txt")
GLOBE_DATA_DIR = Path("/data/group_data/UTD-NAS/Databases/GLOBE_V2/data")
OUT_DIR = Path("/data/user_data/xoy/vctk_globe_us_english_splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SINGLETON_ACCENTS = {"Welsh", "NewZealand"}

# ---------------------------------------------------------------- VCTK ----
spk_accent = {}
with open(VCTK_SPK_INFO) as f:
    f.readline()
    for line in f:
        parts = line.split()
        if len(parts) < 4:
            continue
        spk_accent["p" + parts[0]] = parts[3]

# rows: list of dicts {stem, source, speaker, accent}
rows = []
for spk_dir in sorted(VCTK_WAV_ROOT.iterdir()):
    spk = spk_dir.name
    accent = spk_accent.get(spk)
    if accent is None:
        continue  # p280
    for wav in spk_dir.glob("*.wav"):
        rows.append({"stem": wav.stem, "source": "vctk", "speaker": spk, "accent": accent})

print(f"VCTK rows: {len(rows)}")

# ------------------------------------------------------------ GLOBE_V2 ----
globe_rows = []
for shard in sorted(GLOBE_DATA_DIR.glob("*.parquet")):
    tbl = pq.read_table(shard, columns=["accent", "speaker_id"])
    accents = tbl.column("accent").to_pylist()
    speakers = tbl.column("speaker_id").to_pylist()
    for row_idx, (accent, speaker) in enumerate(zip(accents, speakers)):
        if accent and "United States English" in accent:
            globe_rows.append({
                "stem": f"{shard.stem}-{row_idx:06d}",
                "source": "globe_v2",
                "speaker": f"globe_{speaker}",
                "accent": "American",  # merged into VCTK's American category
            })

print(f"GLOBE_V2 (US English) rows: {len(globe_rows)}")
rows.extend(globe_rows)
print(f"combined rows: {len(rows)}")

# ------------------------------------------------------------ group up ----
by_accent_speaker = defaultdict(lambda: defaultdict(list))
for r in rows:
    by_accent_speaker[r["accent"]][r["speaker"]].append(r)

train, val, test = [], [], []
stats = {}

for accent, spk_map in by_accent_speaker.items():
    speakers = list(spk_map.keys())
    random.shuffle(speakers)
    total_utt = sum(len(v) for v in spk_map.values())

    if accent in SINGLETON_ACCENTS:
        # only one speaker total -- split by utterance instead of by speaker
        all_utts = [u for v in spk_map.values() for u in v]
        random.shuffle(all_utts)
        test_part = all_utts[:N_TEST_PER_ACCENT]
        rest = all_utts[N_TEST_PER_ACCENT:]
        n_val = max(1, int(len(rest) * VAL_FRACTION))
        val_part = rest[:n_val]
        train_part = rest[n_val:]
        test.extend(test_part)
        val.extend(val_part)
        train.extend(train_part)
        stats[accent] = dict(total=total_utt, n_speakers=len(speakers),
                              test=len(test_part), val=len(val_part), train=len(train_part),
                              note="singleton-speaker accent: utterance-level split, speaker leakage across splits")
        continue

    # reserve speaker(s) for test until we have enough utterances, then
    # subsample down to exactly N_TEST_PER_ACCENT (unused reserved
    # utterances beyond that are dropped, not used anywhere)
    reserved, reserved_count, i = [], 0, 0
    while reserved_count < N_TEST_PER_ACCENT and i < len(speakers):
        spk = speakers[i]
        reserved.append(spk)
        reserved_count += len(spk_map[spk])
        i += 1
    reserved_pool = [u for spk in reserved for u in spk_map[spk]]
    random.shuffle(reserved_pool)
    test_part = reserved_pool[:N_TEST_PER_ACCENT]
    test.extend(test_part)

    remaining_speakers = speakers[i:]
    random.shuffle(remaining_speakers)
    n_val_speakers = max(1, round(len(remaining_speakers) * VAL_FRACTION)) if len(remaining_speakers) > 1 else 0
    val_speakers = set(remaining_speakers[:n_val_speakers])
    train_speakers = set(remaining_speakers[n_val_speakers:])

    val_part = [u for spk in val_speakers for u in spk_map[spk]]
    train_part = [u for spk in train_speakers for u in spk_map[spk]]
    val.extend(val_part)
    train.extend(train_part)

    stats[accent] = dict(total=total_utt, n_speakers=len(speakers),
                          test=len(test_part), val=len(val_part), train=len(train_part),
                          test_speakers_reserved=reserved, val_speakers=sorted(val_speakers),
                          note=f"{reserved_count - N_TEST_PER_ACCENT} reserved-speaker utterances unused (dropped)")

for name, split in [("train", train), ("val", val), ("test", test)]:
    with open(OUT_DIR / f"{name}.tsv", "w") as f:
        f.write("stem\tsource\tspeaker\taccent\n")
        for r in split:
            f.write(f"{r['stem']}\t{r['source']}\t{r['speaker']}\t{r['accent']}\n")

with open(OUT_DIR / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print()
print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")
print()
for accent, s in sorted(stats.items(), key=lambda x: -x[1]["total"]):
    print(f"{accent}: total={s['total']} train={s['train']} val={s['val']} test={s['test']}  {s['note']}")
