# Phase 7: real GLOBE speaker labels (pre-registered 2026-10-28, before any Phase 7 training or evaluation number)

## Question

GLOBE has no speaker labels, so since Phase 4 every GLOBE training utterance has been its own speaker and its
reference is the utterance itself (`speaker = "globe_" + stem`). The Common Voice 21 transcript tables give a
`client_id` (an opaque hash of one contributor) for every GLOBE clip (join: 100 % of train, dev and test rows). Does
training the speaker network with references drawn from *other utterances of the same real speaker* improve cloning
of unseen voices over the same data with utterance-as-speaker labels?

## Data

- Common Voice 21.0 English transcript tables (`fsicoli/common_voice_21_0`, CC0), copied to
  `/data/user_data/xoy/common_voice_21_en_meta/`. `client_id` is kept as an opaque hash: it is used only to group GLOBE
  utterances, written next to an integer index in `phase7/globe_train/speakers.parquet`, and never joined to outside
  data. Training files carry only the label `gcv<index>`.
- GLOBE splits are speaker-disjoint (checked exactly: train / dev / test share 0 contributors). The Phase 5 filtered
  GLOBE training split (114,609 utterances) has 13,362 contributors: median 2 utterances, 5,722 singletons (5.0 % of the
  utterances; their reference stays the utterance itself), 94.6 % of utterances have another utterance of at least 3 s
  of the same speaker, 1,660 speakers have at least 10, the largest speaker has 7,965 utterances (6.9 %).
- New packed split `/data/user_data/xoy/spkemb/phase7/packed/globe_f_spk/globe.train`: the Phase 5 filtered split's rows,
  order and feature arrays (linked, not copied); only the `speaker` column differs. `chapter` stays the unique stem.
  No existing file is modified.

## Arms

- **Arm `p7_gspk`:** experiment `p5_fv` (filtered LibriTTS-R + filtered GLOBE + filtered VCTK, HiFi-GAN, up to 8 references,
  speaker-relative F0, 100k generator steps, seed 0) with only `experiment_name` and the GLOBE split path changed. The
  utterances and the sampling weights are identical; the only change is which utterances the reference sampler may
  draw for a GLOBE target. There is no per-speaker cap, because a cap would change the training data. The largest
  contributor (6.9 % of the GLOBE utterances) therefore supplies references to its own 7,965 utterances only.
- **Reference:** `p5_fv` (same data, utterance-as-speaker), already trained and evaluated
  (`phase5/eval/p5_fv`).
- One seed per arm. The seed-to-seed difference behind the margin was measured on the Phase 4 recipe, not here.

## Evaluation

- Primary (decision): the three existing pools (LibriTTS-R test.clean, VCTK, GLOBE pseudo-speakers), variant
  `k5_f0_loud`, ECAPA / WavLM-SV and the held-out CAM++, the Phase 4/5 rule unchanged: primary gain in swap similarity to
  the target at least 0.0266 on LibriTTS-R with the speaker-level CI excluding 0, same sign in VCTK, GLOBE and CAM++,
  and all guards (own-speaker T2/T3 drop at most 0.01, UTMOS drop at most 0.10 in every pool, source similarity rise
  at most 0.02, WER at most +0.5 points). The comparison is `p7_gspk` vs `p5_fv` through `clone compare`.
- Secondary (reported, no decision): a new pool `globe_spk` of real GLOBE dev + test speakers (below), with both
  `p7_gspk` and `p5_fv` evaluated on it, same variant, metrics and guards. The old GLOBE pool's ECAPA caveat (clustered
  pseudo-speakers make ECAPA optimistic) does not apply to it, so ECAPA counts in its sign check.
- Descriptive: effect on the three guards, per pool.

## Real-speaker evaluation pool `globe_spk`

Written next to the old pool (`phase4/pools/globe_spk`, plan `phase4/plans/globe_spk.json`); the old pool and plan are
not changed.
- Source: GLOBE dev + test, speakers = Common Voice contributors. Rules of `globe`: speakers with at least 3 utterances
  of which at least 2 last at least 3 s (`min_refs` 2), gender by majority of `predicted_gender` (ties dropped), 400
  speakers stratified by gender and accent (a speaker's accent is the most common label of its utterances), the first 10
  utterances by stem. Swap plan: the Phase 4 rules (400 swaps, 100 own-speaker items, 100 curve items, up to 10
  references, seed 0).
- Checks: exact overlap of contributors with every GLOBE training list (the full 228,391-utterance list, the filtered
  split) must be 0 (a training speaker in the pool would void it). The ECAPA leak check of Phase 4 (cosine >= 0.70
  between a training utterance and a pool utterance) is reported for the full list and the filtered split. It is
  reported, not used to remove training data, because both arms share that data and changing it would break the design.
  Speakers cannot overlap with LibriTTS-R or VCTK (different corpora). Overlap of the new pool with the old pool's
  utterances is reported.
- Limits: contributors have 3-10 utterances in the pool (references 2-9, as in the old pool); one contributor can
  record on several devices or accounts, which no label can reveal.

## Interpretation rules (fixed now)

- `p7_gspk` advances only by the primary rule. A gain on `globe_spk` alone is reported as a secondary finding, not as an
  advance.
- If the primary rule fails but the secondary pool shows a gain with a CI excluding 0 on all three SV models, the result
  is reported as "real labels help on real speakers" with the 100k single-seed caveat.
- No further arms are added in this phase.

Outputs: `/data/user_data/xoy/spkemb/phase7/{globe_train,packed,eval}`.

## Pool as built (2026-10-28, before any Phase 7 evaluation number)

`globe_spk`: 400 contributors, 1,200 utterances (exactly 3 each; 263 male, 137 female because the female quota ran out;
13 accent labels, 164 American, 87 South Asian, 69 English), 2 references per contributor in 380+ cases as in the old pool.
Plan: 400 swaps (3 s bins: 57 / 201 / 137 / 5), 100 own-speaker items. Checks: contributor overlap with the full training
list, the filtered split and all GLOBE train rows 0; ECAPA cosine >= 0.70 to a pool utterance for 188 of the 228,391
listed training utterances and 79 of the 114,609 filtered ones (0.07 %, max cosine 0.82; reported only); 755 of the 1,200
pool utterances are also in the old pseudo-speaker pool.
