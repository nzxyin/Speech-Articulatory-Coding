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
| | GLOBE real speakers (`globe_spk`) | 400 | **0** |
| | GLOBE pseudo-speakers (`globe`, ECAPA-clustered) | 370 | 30 |
| encoder screen (held-out speakers) | LibriTTS-R dev / test-clean | 13 | 10 |
| | VCTK | 19 | 19 |
| | GLOBE real / pseudo | 400 / 370 | 0 / 30 |

Primary pool: `libritts_dev` for validation, `libritts` for test.

**GLOBE real-speaker test pool does not exist.** No GLOBE test contributor has three utterances (maximum 2; 1,318 have
exactly 2, 1,092 of them with both utterances >= 3 s), so the Phase 7 pool rules (>= 3 utterances, 2 references) select only
dev contributors (738 eligible; all 400 pool speakers are dev). `globe_spk` is therefore a pure validation pool. A test pool
would need a relaxed rule (one reference per speaker, swap items only, no own-speaker guard) and has not been built; until
then the only GLOBE test pool is the 30 pseudo-speakers (descriptive: ECAPA is optimistic on them and 30 speakers give wide
intervals). VCTK halves have 20 speakers each, so validation margins are noisy; the margin of 0.0266 was measured on LibriTTS
test.clean.

## Implementation

* `sparc/vocoders/eval/splits.py`: split builder (`build`), `restrict_items` (cloning tables), `screen_masks` (screen).
* Cloning: **speaker filter, no re-synthesis.** `restrict_items` keeps the rows of `items.parquet` whose target speaker is in
  the split (`--require-source` also requires the content speaker). Per-output columns are unchanged, so top-1
  identification still ranks over the full pool's speakers and acceptance uses the full pool's EER threshold; calibration in a
  restricted summary is carried over from the full pool.
  `python -m sparc.vocoders.eval.clone summarize-split --out <scored dir> --split val|test|all` writes `summary_<split>.json/.md`
  next to the original. `clone compare --a .. --b .. --split val` filters both outputs (primary pool defaults to the split's).
  Validation needs outputs on `libritts_dev` and `globe_spk`: synthesize with `--pools libritts_dev,vctk,globe_spk`;
  outputs of `libritts,vctk,globe` runs only give the VCTK half and the pseudo-GLOBE speakers.
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
