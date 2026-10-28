# Validation and test protocol for speaker-embedding work (2026-10-28)

## Rule

Every choice among ablations, variants, arms, thresholds or hyperparameters is made on **validation speakers**. **Test
speakers** only report the selected configuration(s) and the baseline. A single pre-registered confirmatory comparison (one
arm against its reference, named before any number exists) may use the test pools; it is not a selection.
Selection includes: which inference variant, which arm advances, which encoder variant is promising, and which layer, pooling,
head, filter threshold or data mixture. Fixed protocol parts (margins, guards, rule structure) are set
before the validation numbers are looked at.

## Splits (`docs/spkemb_validation_splits.json`, built by `python -m sparc.vocoders.eval.splits build`, seed 0)

* **LibriTTS-R:** dev-clean = validation, test-clean = test.
* **GLOBE:** dev split = validation, test split = test (GLOBE contributors are disjoint across splits, checked).
* **VCTK:** seeded gender-balanced halves. The 40 cloning-evaluation speakers are split first (10 + 10 per gender); the other
  held-out VCTK speakers of the encoder screen are split by the same rule among themselves (per gender: sorted ids, permuted
  with `default_rng([0, 31])`, first `n // 2` validation, rest test). p280 has no gender in `speaker-info.txt` and is in
  neither split. All remaining VCTK speakers are training speakers (Phase 5), so nothing else is available.

| use | pool | validation | test |
|---|---|---|---|
| cloning pools | LibriTTS-R (`libritts_dev` / `libritts`) | 36 | 36 |
| | VCTK (`vctk`) | 20 | 20 |
| | GLOBE real speakers (`globe_spk` / `globe_spk_test`) | 400 | 400 |
| | GLOBE pseudo-speakers (`globe`, ECAPA-clustered) | 370 | 30 |
| encoder screen (held-out speakers) | LibriTTS-R dev / test-clean | 13 | 10 |
| | VCTK (the 40 cloning-evaluation speakers) | 20 | 20 |
| | GLOBE real / pseudo | 400 / 370 | 0 / 30 |

**VCTK in the encoder screen.** The first screen split used 19 + 19 VCTK speakers; 19 of those are vocoder-training speakers
(Phase 5 VCTK training data), so they are not unseen. The screen (OOD screen Amendment 2) now scores only the 40
cloning-evaluation VCTK speakers (20 validation / 20 test, the cloning split above); every other held-out VCTK speaker is a
probe-fit speaker, never scored. Any screen number computed with the 19 + 19 split is superseded.

Primary pool: `libritts_dev` for validation, `libritts` for test.

**GLOBE real-speaker test pool (`globe_spk_test`, built 2026-10-28).** No GLOBE test contributor has three utterances (maximum
2; 1,318 have exactly 2, 1,092 of them with both >= 3 s), so the Phase 7 pool rules (>= 3 utterances, 2 references) select only
dev contributors: `globe_spk` is a pure validation pool. The test pool uses a relaxed rule (user-approved): test-split
contributors with exactly 2 utterances, both >= 3 s (the old pool's reference minimum); one reference and one content utterance
per speaker, 400 speakers stratified by gender and accent as before (200 F / 200 M; 178 American, 67 English, 64 South Asian,
40 Australian, 21 other-accent groups below 16 each), 800 utterances. Swap plan only: 400 swaps (3-6 s: 253, 6-10 s: 144, > 10 s:
3; same-gender 201), **no own-speaker items** (own-speaker resynthesis needs two references, so the own-speaker guard cannot be
evaluated on this pool), no curve items. A swap's reference is one utterance, so test-pool numbers are not comparable to
`globe_spk` (2-9 references): compare arms within a pool, never across the two. Leak checks (`clone_pools` / `globe_speakers
pool-checks --pool globe_spk_test`, `phase7/globe_train/pool_checks_globe_spk_test.json`): contributor overlap with the full GLOBE
training list, the filtered split and every GLOBE train row 0; contributor and utterance overlap with `globe_spk` 0; 41 utterances
are also in the old pseudo-speaker pool; ECAPA cosine >= 0.70 between a training utterance and a pool utterance for 30 of the
228,391 listed training utterances (13 of 114,609 filtered ones, max 0.83; reported, not removed). The old pools and plans are
untouched. The pool is not part of the encoder screen (no screen features exist for it).
VCTK halves have 20 speakers each, so validation margins are noisy; the margin of 0.0266 was measured on LibriTTS test.clean.

## Which metrics may drive a selection

* **Valid for selection:** similarity to the target (`sim_target`, with ECAPA, WavLM-SV and the held-out CAM++) and its gains over a
  reference arm, computed on the validation speakers; UTMOS and WER guards; own-speaker guards on pools that have own items.
  These depend only on the output and the target's references.
* **Valid only with per-split calibration:** acceptance (EER threshold) and top-1 identification. A pool-wide threshold or a
  pool-wide ranking mixes validation and test speakers, so a validation number would depend on test speakers' references. The split
  summary therefore recomputes both from the split's own speakers (below). Read them only from `summary_val` / `summary_test`
  written by `summarize-split`, never from the full-pool `summary.json`.
* Ground-truth similarity (`gt_same`, `gt_diff`) is per item and valid for either split.

## Implementation

* `sparc/vocoders/eval/splits.py`: split builder (`build`), `restrict_items` (cloning tables), `screen_masks` (screen).
* Cloning: **rescoring from stored embeddings, no re-synthesis.** `summarize-split` / `compare --split` (`clone.split_table`,
  `clone.rescore_split`) rebuild the per-output table of a split from the stored chunk and ground-truth embeddings:
  rows are the swap pairs whose **target** is in the split (`--require-source` additionally requires the content speaker in the
  split; otherwise sources may come from either split, and the summary reports the number of such cross-split rows and sources);
  the EER threshold, acceptance and the ground-truth calibration come from the split speakers' own references only; top-1
  identification ranks only the split's speakers (`top1_source` is empty for a source outside the split). Similarity columns still
  use every speaker's reference mean. The summary also lists per-split stratum counts (items / speakers per variant and stratum).
  `python -m sparc.vocoders.eval.clone summarize-split --out <scored dir> --split val|test|all` writes `summary_<split>.json/.md`
  next to the original (`all` reproduces `summary.json`). `clone compare --a .. --b .. --split val` filters both outputs the same
  way (primary pool defaults to the split's).
  Validation needs outputs on `libritts_dev`, `vctk` (validation half) and `globe_spk`; test needs `libritts`, `vctk` (test half),
  `globe_spk_test` (plan: `clone plan --pools globe_spk_test --n-own 0 --n-curve 0 --n-refs 1`) and optionally the pseudo pool `globe`.
  Unit tests (`tests/test_spkemb_clone.py`) check subset CIs, thresholds and top-1 against a direct recomputation for both halves.
* Encoder screen: `precond.heldout_stats(..., rows, pooled=True)` restricts pooled statistics (5-shot ID, M3, F0) to the split;
  `python -m sparc.vocoders.spkscreen.ood_report --out <dir> --split val|test` writes `ood_val.*` / `ood_test.*`. A corpus with
  no speakers in the split is skipped (the out-of-domain statistic is then VCTK alone). The promising rule is applied on `val`.
* Tests: `tests/test_validation_splits.py`.

## Reproduction check (no result file was modified)

Scored p5_fv outputs (`/data/user_data/xoy/spkemb/phase5/eval/p5_fv`, linked read-only into
`/data/user_data/xoy/spkemb/validation_protocol/p5_fv`): `summarize-split --split all` reproduces the whole `summary.json` pools
section (every strata, metric, CI, WER; same bootstrap settings) with 0 mismatches at 1e-9. Val and test targets partition the
VCTK pool (20 + 20 of 40; 4,400 of 4,400 rows) and the GLOBE pseudo pool (316 + 27 contributors with outputs, disjoint).
For the screen, an all-True mask reproduces the unrestricted `heldout_stats` of baseline seed 0 for every metric (1e-12).

## History of this protocol and of earlier selections

* **OOD encoder screen.** Amendment 1 (2026-10-28, `docs/spkemb_ood_screen.md`) moved the selection rule to validation speakers
  before any score was computed. A review of its first scoring found three leaks (probe fitted on scored VCTK speakers, centering
  statistics averaged over both splits, 19 + 19 VCTK speakers of which 19 are vocoder-training speakers); Amendment 2 fixed them
  and superseded the Amendment 1 results table. Only the Amendment 2 scoring counts.
* **Phase 4 inference variant.** The variant used as the baseline everywhere after Phase 4 (5 references, F0 median and range matched,
  loudness matched: `k5_f0_loud`) was chosen on the LibriTTS-R dev pool `libritts_dev`, for the existing speaker network at 100k (seed
  0), by `clone select-baseline`: the candidates (`base`, `k5`, `k10` and the `k*_f0*` variants) were ranked by ECAPA swap similarity
  to the target computed on the target's references that gave neither the code nor the matching statistics (not by the primary
  `sim_target`, which favours large K). That is a validation selection (dev-clean speakers only), but it was made with one network,
  and the arms were then evaluated with the same variant; the VCTK and GLOBE pools had no part in it.

## How earlier phases selected (for the record; nothing is re-run)

* Phase 1-3 (layer screen, FFN refit, resynthesis encoder arms): arms and layers were compared on the 62 held-out screen
  speakers / LibriTTS-R test.clean, i.e. on test speakers by this protocol.
* Phase 4: the **inference variant** (K = 5 references, F0 median and range matching, loudness matching) was selected on
  `libritts_dev` (`select-baseline`): validation. The **arm decisions** (references, speaker-relative pitch, GLOBE data,
  swapped pairs, identity loss, the 400k combination) and the margin were taken on the test pools (LibriTTS test.clean, VCTK,
  GLOBE pseudo-speakers): selections on test.
* Phase 5-6: the data filter thresholds and GLOBE accent floor were set by the user from the quality report (anchored on clean
  reference audio, no evaluation pool). The three data arms were compared on the test pools and the Phase 6 base (filtered +
  VCTK, by the tie-break) was chosen there: a selection on test.
* Encoder screens (segment, pooling, multi-scale, out-of-domain): variants were ranked on the 62 held-out speakers, which are
  the test speakers of this protocol; the OOD screen's GLOBE real-speaker pool (all dev) is validation-only data.
These results stay valid as descriptions; any claim that a selected arm is better than the baseline on the test pools carries
selection optimism and needs a fresh confirmation on test pools after selection on validation.

## From now on

* New arms, variants and hyperparameters: select on `val` (`summarize-split --split val`, `compare --split val`, `ood_report
  --split val`); then report the chosen configuration and the baseline on `test`, with the validation numbers next to them.
* Phase 7 (`p7_gspk` vs `p5_fv`): the pre-registered single comparison stays on the existing three pools (it is one
  confirmatory comparison, no selection). It is additionally reported per split, `--split val` (LibriTTS dev-clean if
  synthesized, VCTK validation half, `globe_spk`) and `--split test` (LibriTTS test-clean, VCTK test half), without
  changing its decision.
