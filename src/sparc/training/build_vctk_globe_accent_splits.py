"""
Builds train/val/test manifests combining VCTK (10 native accent labels --
Welsh dropped, see below) and GLOBE_V2 (its own self-reported, often
multi-tag accent field), mapped onto the same accent taxonomy:

    VCTK label      <-> GLOBE_V2 exact tag(s)
    English         <-> "England English"
    American        <-> "United States English"
    Scottish        <-> "Scottish English"
    Irish           <-> "Irish English"
    Canadian        <-> "Canadian English"
    NorthernIrish   <-> "Northern Irish"
    SouthAfrican    <-> "Southern African (South Africa, Zimbabwe, Namibia)",
                        "South African English"
    Indian          <-> "India and South Asia (India, Pakistan, Sri Lanka)"
    Australian      <-> "Australian English"
    NewZealand      <-> "New Zealand English"

Welsh is dropped entirely (VCTK: 1 speaker/375 utterances; GLOBE: 316
utterances) -- it was the thinnest category by a wide margin and set the
ceiling on how large a balanced test set could be. Dropping it raises that
ceiling to Irish's 2,658 GLOBE utterances (the new smallest category),
which is what N_TEST_PER_ACCENT below is sized against.

GLOBE_V2's accent field is a comma-separated list of self-reported tags
(Common Voice convention), and some individual tags themselves contain
commas inside parentheses (e.g. "Southern African (South Africa, Zimbabwe,
Namibia)") -- a naive `.split(",")` would shred those, so tag-splitting
here is parenthesis-depth-aware. A GLOBE_V2 row is assigned to a VCTK
category only if EXACTLY ONE of the mapped tags appears among its
(parsed) tag list; rows matching zero or multiple mapped tags (e.g.
"England English,United States English", a self-reported mixed accent)
are excluded, to avoid contaminating one category's data with another's.

Two split strategies are built and both written out (see STRATEGY below):

  "holdout": test is a fixed N utterances per accent, held out at the
  speaker level from the *combined* VCTK+GLOBE pool per accent; train/val
  get everything else, proportional to natural accent frequency.

  "vctk_train_globe_test": all of VCTK goes into train/val untouched (no
  VCTK held out for test at all); the balanced per-accent test set is
  drawn exclusively from GLOBE_V2. Per instruction, speaker identity is
  *not* held exclusive to test here -- SPARC features are assumed
  speaker-invariant (accent/rate carry the signal, not voice identity),
  so a GLOBE speaker can appear in both test and train/val as long as no
  single utterance is used in more than one split. Val still gets a
  speaker-disjoint split from train (ordinary best practice, unrelated to
  the test-speaker-overlap relaxation above).
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

random.seed(0)

N_TEST_PER_ACCENT = 300  # see module docstring: bounded by Irish's 2,658 GLOBE utterances
VAL_FRACTION = 0.10

VCTK_WAV_ROOT = Path("/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/wav48")
VCTK_SPK_INFO = Path("/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/speaker-info.txt")
GLOBE_DATA_DIR = Path("/data/group_data/UTD-NAS/Databases/GLOBE_V2/data")
OUT_DIR = Path("/data/user_data/xoy/vctk_globe_accent_splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROPPED_ACCENTS = {"Welsh"}

GLOBE_TAG_TO_ACCENT = {
    "England English": "English",
    "United States English": "American",
    "Scottish English": "Scottish",
    "Irish English": "Irish",
    "Canadian English": "Canadian",
    "Northern Irish": "NorthernIrish",
    "Southern African (South Africa, Zimbabwe, Namibia)": "SouthAfrican",
    "South African English": "SouthAfrican",
    "India and South Asia (India, Pakistan, Sri Lanka)": "Indian",
    "Australian English": "Australian",
    "New Zealand English": "NewZealand",
}


def split_tags(accent_str):
    """Comma-split that doesn't break on commas inside parentheses."""
    tags, depth, current = [], 0, []
    for ch in accent_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            tags.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        tags.append("".join(current).strip())
    return tags


def globe_accent_category(accent_str):
    if not accent_str:
        return None
    matched = {GLOBE_TAG_TO_ACCENT[t] for t in split_tags(accent_str) if t in GLOBE_TAG_TO_ACCENT}
    if len(matched) == 1:
        return next(iter(matched))
    return None  # zero or ambiguous (multiple) matches


# ---------------------------------------------------------------- VCTK ----
spk_accent = {}
with open(VCTK_SPK_INFO) as f:
    f.readline()
    for line in f:
        parts = line.split()
        if len(parts) < 4:
            continue
        spk_accent["p" + parts[0]] = parts[3]

vctk_rows = []
for spk_dir in sorted(VCTK_WAV_ROOT.iterdir()):
    spk = spk_dir.name
    accent = spk_accent.get(spk)
    if accent is None or accent in DROPPED_ACCENTS:
        continue  # p280 (no accent info), or a dropped accent (Welsh)
    for wav in spk_dir.glob("*.wav"):
        vctk_rows.append({"stem": wav.stem, "source": "vctk", "speaker": spk, "accent": accent})
print(f"VCTK rows (accent known, not dropped): {len(vctk_rows)}")

# ------------------------------------------------------------ GLOBE_V2 ----
globe_rows = []
excluded_ambiguous = excluded_unmatched = 0
for shard in sorted(GLOBE_DATA_DIR.glob("*.parquet")):
    tbl = pq.read_table(shard, columns=["accent", "speaker_id"])
    accents = tbl.column("accent").to_pylist()
    speakers = tbl.column("speaker_id").to_pylist()
    for row_idx, (accent, speaker) in enumerate(zip(accents, speakers)):
        cat = globe_accent_category(accent)
        if cat is None:
            tags = split_tags(accent) if accent else []
            n_matched = sum(1 for t in tags if t in GLOBE_TAG_TO_ACCENT)
            if n_matched >= 2:
                excluded_ambiguous += 1
            else:
                excluded_unmatched += 1
            continue
        globe_rows.append({
            "stem": f"{shard.stem}-{row_idx:06d}",
            "source": "globe_v2",
            "speaker": f"globe_{speaker}",
            "accent": cat,
        })

print(f"GLOBE_V2 rows matched to a single category: {len(globe_rows)}")
print(f"GLOBE_V2 rows excluded (ambiguous multi-accent tags): {excluded_ambiguous}")
print(f"GLOBE_V2 rows excluded (no matching tag): {excluded_unmatched}")
print(f"GLOBE_V2 total scanned: {len(globe_rows) + excluded_ambiguous + excluded_unmatched}")

combined_rows = vctk_rows + globe_rows
print(f"combined rows: {len(combined_rows)}")
print()
print("=== per-accent totals (VCTK + GLOBE combined) ===")
totals = defaultdict(lambda: [0, 0])  # accent -> [vctk_count, globe_count]
for r in vctk_rows:
    totals[r["accent"]][0] += 1
for r in globe_rows:
    totals[r["accent"]][1] += 1
for accent, (v, g) in sorted(totals.items(), key=lambda x: -(x[1][0] + x[1][1])):
    print(f"{accent}: vctk={v} globe={g} total={v+g}")


def speaker_group(rows_):
    by_accent_speaker = defaultdict(lambda: defaultdict(list))
    for r in rows_:
        by_accent_speaker[r["accent"]][r["speaker"]].append(r)
    return by_accent_speaker


def write_split(name, rows_, out_dir):
    with open(out_dir / f"{name}.tsv", "w") as f:
        f.write("stem\tsource\tspeaker\taccent\n")
        for r in rows_:
            f.write(f"{r['stem']}\t{r['source']}\t{r['speaker']}\t{r['accent']}\n")


def select_val_speakers(remaining_speakers, spk_map, remaining_pool_size, val_fraction):
    """Greedily assigns whole speakers to val (random order) until val's
    accumulated *utterance* count reaches val_fraction of the remaining
    pool, rather than picking a fraction of the *speaker headcount*.

    The naive speaker-count version breaks down for accents with both few
    unique speakers and a power-law-skewed per-speaker utterance count
    (e.g. GLOBE_V2's NorthernIrish: only 17 speakers -- round(17*0.10)=2
    speakers picked for val was very likely to land on two low-volume
    speakers by chance, producing a val set with single-digit utterances
    while train got the rest. Targeting utterance count instead keeps val
    genuinely close to val_fraction of the data while still keeping every
    speaker's utterances entirely on one side of the train/val boundary
    (no speaker split across both, which would leak speaker-specific
    characteristics into val and inflate its usefulness as a held-out
    estimate).
    """
    random.shuffle(remaining_speakers)
    val_target = max(1, round(remaining_pool_size * val_fraction)) if remaining_pool_size > 0 else 0
    val_speakers, val_count = set(), 0
    for spk in remaining_speakers:
        if val_count >= val_target:
            break
        val_speakers.add(spk)
        val_count += len(spk_map[spk])
    train_speakers = set(remaining_speakers) - val_speakers
    return val_speakers, train_speakers


# =====================================================================
# Strategy A: "holdout" -- speaker-level test holdout from the combined
# VCTK+GLOBE pool per accent (same approach as before, just with the
# fuller accent mapping now).
# =====================================================================
def build_holdout_strategy():
    by_accent_speaker = speaker_group(combined_rows)
    train, val, test = [], [], []
    stats = {}

    for accent, spk_map in by_accent_speaker.items():
        speakers = list(spk_map.keys())
        random.shuffle(speakers)
        total_utt = sum(len(v) for v in spk_map.values())

        # Note: VCTK's Welsh/NewZealand singleton-speaker problem from the
        # earlier VCTK-only split doesn't recur here -- GLOBE_V2 adds many
        # more speakers to every accent's pool, so a clean multi-speaker
        # holdout is always possible now.
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
        remaining_pool_size = sum(len(spk_map[spk]) for spk in remaining_speakers)
        val_speakers, train_speakers = select_val_speakers(
            remaining_speakers, spk_map, remaining_pool_size, VAL_FRACTION
        )

        val_part = [u for spk in val_speakers for u in spk_map[spk]]
        train_part = [u for spk in train_speakers for u in spk_map[spk]]
        val.extend(val_part)
        train.extend(train_part)

        stats[accent] = dict(total=total_utt, n_speakers=len(speakers),
                              test=len(test_part), val=len(val_part), train=len(train_part))
    return train, val, test, stats


# =====================================================================
# Strategy B: "vctk_train_globe_test" -- all VCTK stays in train/val;
# balanced test drawn only from GLOBE_V2. Test utterances are sampled
# directly per accent (no whole-speaker reservation) since a GLOBE speaker
# is allowed to also appear in train/val -- only the literal sampled
# utterances are excluded from the train/val remainder. Val is still a
# speaker-disjoint split of what's left, same as before.
# =====================================================================
def build_vctk_train_globe_test_strategy():
    globe_by_accent = defaultdict(list)
    for r in globe_rows:
        globe_by_accent[r["accent"]].append(r)

    train, val, test = list(vctk_rows), [], []
    stats = {}

    all_accents = set(totals.keys())
    for accent in all_accents:
        pool = list(globe_by_accent.get(accent, []))
        random.shuffle(pool)
        globe_total = len(pool)

        n_test = min(N_TEST_PER_ACCENT, globe_total)
        test_part = pool[:n_test]
        remainder = pool[n_test:]
        test.extend(test_part)

        spk_map = defaultdict(list)
        for r in remainder:
            spk_map[r["speaker"]].append(r)
        remaining_speakers = list(spk_map.keys())
        val_speakers, train_speakers = select_val_speakers(
            remaining_speakers, spk_map, len(remainder), VAL_FRACTION
        )

        val_part = [u for spk in val_speakers for u in spk_map[spk]]
        train_part = [u for spk in train_speakers for u in spk_map[spk]]
        val.extend(val_part)
        train.extend(train_part)

        vctk_total = totals[accent][0]
        stats[accent] = dict(vctk_total=vctk_total, globe_total=globe_total,
                              globe_n_speakers=len({r["speaker"] for r in pool}),
                              test=len(test_part),
                              globe_val=len(val_part), globe_train=len(train_part),
                              short_of_target=max(0, N_TEST_PER_ACCENT - globe_total))
    return train, val, test, stats


for strategy_name, builder in [
    ("holdout", build_holdout_strategy),
    ("vctk_train_globe_test", build_vctk_train_globe_test_strategy),
]:
    train, val, test, stats = builder()
    out_dir = OUT_DIR / strategy_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_split("train", train, out_dir)
    write_split("val", val, out_dir)
    write_split("test", test, out_dir)
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print()
    print(f"=== strategy: {strategy_name} ===")
    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")
    for accent, s in sorted(stats.items()):
        print(f"  {accent}: {s}")
