# Speech-Articulatory-Coding (SPARC fork) — project state

Unofficial fork of SPARC (Berkeley). Changes are listed in `CHANGELOG.md`; this file tracks findings and
direction. Cluster conventions live in `~/.claude/CLAUDE.md`.

## Active direction: compressing large SSL encoders for speech-to-EMA (since 2026-10-06)

Question: does large-scale SSL pretraining followed by structural compression (layer truncation,
non-contiguous layer removal, LoRA adaptation) give a better compact encoder for articulatory (EMA)
prediction than a natively compact SSL model at the same inference compute? Primary model XLS-R 1B;
baselines XLS-R 300M and WavLM Large; XLS-R 2B only if the 1B result calls for a scaling test.

Framing (user, 2026-10-07): the question is (1) model FLOPs at inference time and (2) generalization to multiple
speakers, languages and styles. The end goal is to improve SPARC, which resynthesizes speech from the normalized
articulatory space, so z-unit RMSE (rmse_z) and PCC are the primary metrics; mm RMSE is secondary. Smaller SSL models
are not added (they need not carry articulatory information). Streaming is not required, though conceptually
interesting. Model set for the multi-speaker work: WavLM Large, XLS-R 300M, XLS-R 1B, plus wav2vec2-large-lv60
(English, same architecture as XLS-R 300M) to separate pretraining language from pretraining objective; HuBERT
is not added (WavLM builds on it); XLS-R 2B is dropped.

Current plan (2026-10-07):
1. Does pooling help? Pre-registered rule: pooling helps a model if its validation-best pooled arm (3-seed mean)
   beats the validation-best unpooled arm (hidden 256 or 48) on seen-test rmse_z by more than 2x the SE of the
   seed-paired difference, without hurting unseen speakers. Keep pooling only if this holds for >= 2 of 3 models;
   otherwise discard it.
2. Leave-one-speaker-out CV (7 folds, F5 normalized per session group) with one identical recipe for all models
   (LoRA r8 + causal 9-frame conv, 3 seeds) plus a frozen-encoder control; paired fold statistics
   (`compression/loso.py`). Shift per model fixed from a -2..3 shift grid on ema_multi.

Code: `src/sparc/compression/` (see the module docstrings), jobs: `scripts/xlsr_ema_*_slurm.sh`.
Outputs: `/data/user_data/xoy/xlsr_ema/{features/<model>,adapt/<model>/<run>,analysis}`.
Feature caches are large (XLS-R 1B: ~29 GB fp16) and can be regenerated with `sparc.compression.extract`.

### Protocol decisions
- Dataset MNGU0 day-1 EMA (single speaker). Standard corpus file sets: 1064 train / 63 valid / 61 test
  usable utterances (2 standard test utterances have no label file, 2 utterances have < 0.2 s of EMA).
  Single speaker, so speaker generalization cannot be measured on MNGU0; it would need another EMA corpus.
- EMA in mm: ema_norm stores (x - mean) / (4 std) in cm units; de-normalized with norm_parms and x10.
- Alignment: ema_norm is silence-trimmed, the wavs are not. Offset = round(leading-silence end x 50 Hz)
  + a global shift in {0..3} chosen on validation. This matches the earlier cross-correlation offsets
  (`mngu0_peralign_refit.py`) to ~1 frame (r = 0.95) without depending on the shipped WavLM model, which
  would have biased alignment toward WavLM.
- Probe: zero-phase 10 Hz low-pass on features (as SPARC), ridge with alpha and shift chosen on
  validation, best layer chosen on validation RMSE, test reported with 95% bootstrap CIs.
- Compute: FLOPs from torch's FlopCounterMode on a 4 s utterance, reported per second of audio; latency
  on one L40S. Profiles in `features/<model>/compute_prefix.json`.

### Bottom line (2026-10-06)
On MNGU0, compressing XLS-R 1B or 2B does **not** beat the native compact models at matched inference
compute, by any method tried: truncation, non-contiguous layer selection, LoRA variants, temporal heads.
WavLM Large truncated to 7-9 layers is the most compute-efficient encoder in the 14-17 GFLOPs/s range (the only
budget where all models were compared at matched compute). The best absolute
results come from XLS-R 300M k=18 with LoRA and a causal conv head. Larger pretraining gives a slightly better
best single layer (2B layer 11: 0.876 mm vs ~0.89 for the others), but that layer sits so deep in a much wider
model that it is never competitive per FLOP. Full analysis: `/data/user_data/xoy/xlsr_ema/analysis/summary.md`
and `fig_perf_vs_flops.png` / `fig_perf_vs_ratio.png` (regenerate with `python -m sparc.compression.analyze`).

Best adapted encoders (independent LoRA r=8 on q/v + causal 9-frame conv head; 3 seeds, test RMSE mean +/- std):
| model | k | GFLOPs/s audio | test RMSE mm | test PCC |
|---|---|---|---|---|
| WavLM Large | 9 | 17.4 | 0.753 +/- 0.001 | 0.933 |
| XLS-R 300M | 18 | 29.1 | 0.739 +/- 0.005 | 0.935 |
| XLS-R 1B | 16 | 38.4 | 0.758 +/- 0.002 | 0.932 |
| XLS-R 2B | 7 | 39.3 | 0.784 (1 seed) | 0.924 |
| XLS-R 2B | 11 | 57.2 | 0.757 (1 seed) | 0.930 |
Lower depths, same recipe (1 seed): WavLM k=6 0.785 @ 13.6, 300M k=8 0.783 @ 16.1, 1B k=9 0.780 @ 24.3,
1B k=12 0.769 @ 30.4, 2B k=5 0.835 @ 30.3.

Other findings:
- Temporal context is the largest single gain. A causal 9-frame conv head on a frozen encoder takes every model
  from ~0.89 to ~0.79-0.81 mm. A centered (non-causal) conv head on frozen 1B k=16 reaches 0.763, as good as LoRA.
  A windowed attention head (±25 frames) is no better than the causal conv (0.816). Every head also passes through a
  non-causal ±1 s FIR output smoother, so all numbers are offline (non-streaming) results.
- Frozen XLS-R 2B (k=11) + causal conv head is the best frozen encoder (0.772 vs 0.793 for 300M/WavLM, 0.810 for 1B),
  (single seed). This suggests richer features from larger pretraining, but LoRA closes the gap for the smaller
  models, and 2B needs 2-3x their FLOPs to get there.
- Cross-layer LoRA: with the causal conv head, shared-A/B with per-layer rank gates (41k LoRA params) matched
  independent LoRA (655k) on MNGU0: 0.759 vs 0.758 on 1B k=16 (single seed; not replicated on multi-speaker data,
  where it was not better). Not carried forward. With a linear head the shared variants trail
  (shared-gated 0.800, shared-A 0.769, independent 0.760). Merged LoRA weights reproduce the unmerged test
  RMSE in every run, so LoRA adds no inference cost.
- Non-contiguous selection on 1B (prefix 24 -> 8 layers): greedy backward elimination matches the prefix from
  10-24 layers (it picks exactly 1..16 at 16 layers). It only helps at aggressive compression (8 layers: 0.926
  vs prefix 0.981, by skipping layers 3 and 6 and reaching layer 10), which needs ~1.4x the FLOPs of
  300M k=8 (22.3 vs 16.1 GFLOPs/s) and is still worse than WavLM k=9. Block-Influence (ShortGPT) selection is worse than the prefix at every size.
- Robustness (white noise, +-10% speed): all adapted models degrade similarly (~1.4-1.5 mm at 0 dB SNR).
  300M is the most robust at 20-5 dB SNR, 1B marginally at 0 dB.

### Layer pooling and component pruning (2026-10-07)
Layer pooling (`heads.py` LayerPool; ridge-initialized for linear heads, per-group function-preserving
re-standardization; reviewed adversarially before the runs). Test RMSE mm, unpooled -> best pooled:
| model | frozen + linear | LoRA + causal conv |
|---|---|---|
| WavLM Large k=9 | 0.889 -> 0.867 (attn, global) | 0.753 +/- 0.001 -> 0.745 |
| XLS-R 300M k=18 | 0.893 -> 0.871 (attn, per-articulator) | 0.739 +/- 0.005 -> 0.737 |
| XLS-R 1B k=16 | 0.893 -> 0.882 (attn, per-articulator) | 0.758 +/- 0.002 -> 0.740 |
- Frame-wise attention pooling helps frozen encoders (-0.011 mm for 1B, -0.022 mm for 300M and WavLM). The
  last retained layer always gets the most weight. Layer 1 is second for WavLM and 1B and third for 300M,
  where the second-to-last layer is second. Static pooling does not help frozen encoders. (On WavLM it never left its
  ridge initialization; with raw, non-normalized layers it reached 0.881.)
- With LoRA, the last retained layer is always the top-weighted layer, at 0.20-0.80 weight: 0.5-0.8 for
  WavLM, 0.2-0.4 for 300M and 1B static pools. Gains are seed-noise-sized.
  The best LoRA results for 300M and 1B use static per-articulator pooling with a small head (hidden 48).
  For WavLM that ties attention per-articulator pooling (0.7451 vs 0.7451). That looks like regularization,
  not per-articulator specialization: all six articulators learn nearly the same layer weights.
- Per-articulator pooling gives no systematic gain. This matches the per-channel probe check (picking each
  channel's best layer on validation changes test RMSE by <= 0.003 mm).
- 1B still doesn't beat 300M (0.740 vs 0.737 mm, 38 vs 29 GFLOPs/s), so component pruning was run on 300M.

Component pruning of XLS-R 300M k=18 (LoRA + causal conv source; `components.py`: Taylor importance on
head/FFN masks, 4 iterative rounds with LoRA recovery, physical removal, 40-epoch final fine-tune):
| keep | heads | FFN neurons | params | GFLOPs/s | test RMSE | PCC | nearest truncation / native |
|---|---|---|---|---|---|---|---|
| 0.75 | 216/288 | 55k/74k | 183M | 23.2 | 0.754 | 0.932 | 1B k=9 0.780 @ 24.3 |
| 0.5 | 144/288 | 37k/74k | 127M | 17.4 | 0.786 | 0.929 | WavLM k=9 0.753 @ 17.4; 300M k=8 0.783 @ 16.1 |
| 0.5 random | 144/288 | 37k/74k | 127M | 17.4 | 0.858 | 0.916 | (control) |
| 0.4 | 115/288 | 29k/74k | 104M | 15.1 | 0.812 | 0.926 | 300M k=8 0.783 @ 16.1 |
| 0.3 | 86/288 | 22k/74k | 81M | 12.8 | 0.835 | 0.919 | WavLM k=6 0.785 @ 13.6 |
- Taylor importance clearly beats random (0.786 vs 0.858 at keep 0.5), so the scores carry real information.
- Mild pruning (keep 0.75, single seed, no matched random control) is better than truncation at that cost, but the
  margin is within seed noise. From keep 0.5 down, pruning inside the layers is no
  better than dropping late layers, and clearly worse than WavLM at the same FLOPs.
- Overall: no compression of XLS-R (1B or 300M; truncation, non-contiguous layers, pooling, head/FFN pruning)
  beats WavLM Large k=9 + LoRA + causal conv at its 17.4 GFLOPs/s (0.753 mm unpooled). The best result at that
  budget is pooled WavLM k=9 itself (0.745 mm, attention per-articulator or static per-articulator h48; single
  seed, and the pooled variant was picked by test RMSE, so this is optimistic; pooling is being re-decided with
  3 seeds and validation selection, see below). (Claims in this section were checked against the raw results by an independent verification pass;
  six corrections were applied.)

### Probe results (linear probes, test set; 2026-10-06)
| model | best k (valid) | GFLOPs/s audio | params | test RMSE mm (95% CI) | PCC |
|---|---|---|---|---|---|
| WavLM Large | 9 / 24 | 17.4 | 127M | 0.888 (0.853-0.925) | 0.904 |
| XLS-R 300M | 18 / 24 | 29.1 | 240M | 0.893 (0.858-0.931) | 0.902 |
| XLS-R 1B | 16 / 48 | 38.4 | 333M | 0.893 (0.859-0.931) | 0.901 |
| XLS-R 2B | 11 / 48 | 57.2 | 522M | 0.876 (0.842-0.913) | 0.901 |
Full models (last layer): WavLM 0.968, 300M 1.000, 1B 1.028, 2B 0.946 mm at 36.8 / 36.8 / 102.6 / 222.9 GFLOPs/s.

- WavLM's best layer (9) matches the SPARC paper's choice, which supports the new alignment/split.
- XLS-R 300M and 1B both have two peaks at the same relative depths (300M: layers 8-9 and 18;
  1B: layers 10-17 and 36), with a dip between. 1B reaches its ~0.90 mm plateau at k=10 (26% of its FLOPs).
- With linear probes, truncated XLS-R 1B does not beat the native models at matched compute: at the
  XLS-R 300M budget all three tie within CI, and WavLM gets there with less than half the FLOPs. At
  ~24 GFLOPs/s, 1B k=9 (0.934) does no better than 300M k=8 (0.926). At the 300M budget, 2B keeps only 6 layers
  (0.967 mm).

### Multi-speaker EMA corpora: USC-TIMIT EMA + USC EMA_5EMO (2026-10-07)
Preprocessed into audio-aligned 50 Hz utterances by `src/sparc/ema_corpora/` (module docstrings document every
decision). Outputs: `/data/user_data/xoy/ema_corpora/` (`manifest.csv`, `stats.json`, `checks.json`,
`<corpus>/<speaker>/<utt>.{wav,npz}`); load with `sparc.ema_corpora.dataset`. Regenerate with
`python -m sparc.ema_corpora.preprocess`, verify with `python -m sparc.ema_corpora.check`.
- 2757 utterances, ~2.65 h: USC-TIMIT M1, F1, M3, F5 (460 sentences each; text-disjoint train/valid/test
  split by sentence id) and 5EMO jn, jr, kf (917: sentences x 5 emotions, neutral repetitions, passage
  phrases). Same 12 channels as MNGU0 (TD, TB, TT, LI, UL, LL x anterior/up), lateral coordinate and native
  3-D trajectories kept for later PCA reparameterization. Normalization: per-articulator z-score per
  speaker, except usc_F5 which has two session groups. No cross-speaker transform.
- Facts found in the raw data (and handled):
  - Both corpora are mview .mat files; audio and EMA start together (EMA at 97.5-100.08 Hz, its own rate per
    file). In 276 USC files the EMA stops 70-110 ms before the audio. The .wav files equal the embedded audio
    (5EMO: peak-normalized copies).
  - Latency: model-based sync (SPARC inversion vs measured EMA, signed vertical channels) gives EMA leading audio
    by ~22 ms for all USC speakers (an independent /p b m/ vs lip-aperture check agrees in sign), 26 ms for
    5emo_jn, and ~0 for jr/kf. Corrected per speaker; residual lag on the outputs is within -4.5 to +6.2 ms.
  - 5emo_jn's frame is rotated ~50 deg (README: F1 not aligned to the occlusal plane); rotated so all speakers
    share x = anterior, y = up (lip line -13..+9 deg, tongue line -17..+6 deg across speakers after correction).
  - USC M3's transcripts are from its MRI session (no timing relation to the EMA audio). M3 is segmented by
    DTW-transferring sentence cuts from M1/F1/F5 recordings of the same sentences (2-4% wrong cuts in a
    leave-one-speaker-out test; M3's duration consistency matches transcript-segmented speakers).
  - USC F5 sentences 001-065 sit ~12 mm (all sensors) away from 066-460: two normalization groups.
  - USC M1's jaw sensor is ~13 mm off midline (plane tilt 17 deg); placement, left to per-articulator norm.
  - 5EMO emotion blocks carry 2-6 mm common-mode frame offsets (e.g. jr neutral 5.7 mm posterior). Emotion and
    session are confounded, so they are NOT corrected; per-utterance estimates are in the manifest
    (frame_offset_*). They inflate 5EMO channel std (5-7 mm vs 3-4 mm in USC).
  - pssg_short phrases are mostly exact excerpts of the full passages (used instead of them); 10 of 270 are
    not and are kept, flagged located=False; jn neutral's phrase split differs from the brief (text left empty).

### Multi-speaker generalization and fitting (2026-10-07)
Code: `compression/datasets.py` (dataset `ema_multi`), `crossspeaker.py`, `analyze_multi.py`; every experiment
module takes `--dataset ema_multi`. Results: `/data/user_data/xoy/xlsr_ema/{features_ema_multi,adapt_ema_multi,
analysis_multi}`. Protocol: train on usc_M1/M3/F5 + 5emo_jn/jr (1473 utterances), test on their text-disjoint
test sentences ("seen", 283) and on held-out speakers usc_F1 + 5emo_kf ("unseen", 114 test sentences). Targets
are z-scored per speaker (training-split stats); RMSE is reported in mm after per-speaker de-normalization.
Multi-speaker numbers are not comparable to MNGU0 ones (different speakers, sensor placements, mm spreads).

Zero-shot (MNGU0-trained models on all seven new speakers; mean over speakers):
| model | zero-shot PCC | PCC after per-speaker linear calibration |
|---|---|---|
| SPARC shipped (WavLM k9 linear) | 0.656 | 0.720 |
| WavLM k9 LoRA + causal conv | 0.670 | 0.716 |
| XLS-R 300M k18 LoRA + causal conv | 0.672 | 0.725 |
| XLS-R 1B k16 LoRA + causal conv | 0.687 | 0.755 |
| XLS-R 2B k11 LoRA + causal conv | 0.683 | 0.751 |
| XLS-R 300M pruned keep 0.5 | 0.642 | 0.694 |
- Zero-shot over all 3 MNGU0 seeds (speaker-mean PCC, mean +/- sd over seeds): 1B k16 0.682 +/- 0.004, 300M k18
  0.673 +/- 0.003, WavLM k9 0.672 +/- 0.002; after per-speaker linear calibration 0.746 / 0.730 / 0.723. 1B beats
  300M on 7/7 speakers (seed-averaged) on both measures and every 1B seed beats every 300M seed, unlike MNGU0
  in-domain where they tied. Small (+0.009 PCC zero-shot) but consistent. Pruning hurts transfer.

Fitted on the multi-speaker data (LoRA r8 + causal conv; 3 seeds mean +/- std):
| model | k | GFLOPs/s | seen RMSE mm | seen PCC | unseen RMSE mm | unseen PCC |
|---|---|---|---|---|---|---|
| WavLM Large | 7 | 14.9 | 2.374 +/- 0.006 | 0.801 +/- 0.001 | 3.453 +/- 0.034 | 0.703 +/- 0.006 |
| XLS-R 1B | 10 | 26.4 | 2.331 +/- 0.020 | 0.804 +/- 0.003 | 3.618 +/- 0.109 | 0.697 +/- 0.004 |
| XLS-R 300M | 18 | 29.1 | 2.401 +/- 0.009 | 0.815 +/- 0.004 | 3.497 +/- 0.025 | 0.724 +/- 0.009 |
| XLS-R 1B (1 seed) | 16 | 38.4 | 2.386 | 0.814 | 3.417 | 0.707 |
| XLS-R 1B (1 seed) | 36 | 78.5 | 2.362 | 0.823 | 3.355 | 0.723 |
- Best probe layers: WavLM 7, 300M 18 (as on MNGU0), 1B 10 (its layer 36 has the best unseen-speaker probe,
  3.208 mm / PCC 0.699, the best of any probe).
- Seen speakers: truncated 1B (k10) has lower pooled RMSE than 300M at less compute (2.331 vs 2.401 mm), but 300M
  has higher PCC (0.815 vs 0.804); WavLM k7 is within 0.04 mm at half the FLOPs. Per corpus (below) the 1B RMSE
  edge comes only from EMA_5EMO; on USC-TIMIT 300M wins both metrics.
- Unseen speakers: pooled over both held-out speakers, LoRA raises PCC but worsens RMSE vs frozen encoders. Per
  corpus this is only 5emo_kf: on usc_F1 LoRA improves RMSE too (WavLM 1.914 -> 1.759, 300M 1.894 -> 1.779 mm),
  on 5emo_kf it worsens (3.750 -> 3.917, 3.717 -> 3.969 mm), plausibly emotional variability and the 5EMO block
  frame offsets. 1B k10 is the most seed-sensitive here.

Per corpus (speaker-averaged within corpus; RMSE mm / PCC; LoRA r8 + causal conv, 3 seeds unless noted).
EMA_5EMO is much harder than USC-TIMIT (seen RMSE ~2.6-2.8 vs ~1.5-1.6 mm), so pooled mm figures are dominated by it.
| model | k | USC seen | 5EMO seen | USC unseen (F1) | 5EMO unseen (kf) |
|---|---|---|---|---|---|
| WavLM Large | 7 | 1.575 / 0.862 | 2.675 / 0.747 | 1.759 / 0.794 | 3.917 / 0.641 |
| XLS-R 1B | 10 | 1.550 / 0.865 | 2.626 / 0.748 | 1.946 / 0.784 | 4.083 / 0.639 |
| XLS-R 300M | 18 | 1.514 / 0.874 | 2.730 / 0.763 | 1.779 / 0.829 | 3.969 / 0.653 |
| XLS-R 1B (1 seed) | 36 | 1.550 / 0.880 | 2.667 / 0.771 | 1.814 / 0.804 | 3.788 / 0.668 |
Zero-shot per corpus (PCC, USC / 5EMO): 300M LoRA+cc 0.701 / 0.633, 1B LoRA+cc 0.715 / 0.648, 2B 0.715 / 0.641;
after per-speaker linear calibration 0.791 / 0.638 vs 0.800 / 0.695 (1B): the larger-model transfer advantage
holds in both corpora and is largest on 5EMO. Each corpus has one held-out speaker, so per-corpus unseen numbers
are single-speaker results.
- Component pruning of 300M (multi): keep 0.75 2.447 mm / 0.806 at 23.2; keep 0.5 2.433 / 0.796 at 17.4 vs WavLM
  k7 2.374 / 0.801 at 14.9; random control 2.628 / 0.762. Same conclusion as MNGU0.
- Greedy non-contiguous selection on 1B: prefix is optimal from 10-24 layers; at 8-9 layers skipping layers 5/6
  helps (8 layers: 2.972 vs prefix 3.056 mm seen), still only tying WavLM k7's probe at more FLOPs.
- Overall: in-domain (seen speakers) the compression picture is unchanged (WavLM most efficient, no XLS-R
  compression wins on both metrics). Under speaker shift there are hints that larger and deeper
  representations carry more (zero-shot ranking favours 1B/2B on one checkpoint each; the deepest 1B, k36, is best
  on unseen speakers in one seed), at 2-5x the FLOPs. Unseen-speaker mm RMSE is calibrated with each held-out
  speaker's own statistics; rmse_z/PCC are the primary metrics from here on. LOSO cross-validation tests this.

### Pooling decision, shift grid, wav2vec2 (2026-10-08)
- Pooling discarded (pre-registered rule, `python -m sparc.compression.pooling_decision`, output
  `analysis_multi/pooling_decision.md`). On ema_multi, 3 seeds per arm, arms chosen on validation, the pooled arm
  never beats unpooled h256 on seen-test rmse_z by > 2 SE: WavLM k7 -0.001, 300M k18 -0.002, 1B k10 +0.003 rmse_z
  (SE 0.002-0.004). 0/3 models, so LOSO uses the plain recipe. Exception, single speaker only: on MNGU0 1B k16,
  static per-articulator pooling h48 beats unpooled 0.734 vs 0.758 mm (+0.024, SE 0.004), not explained by the head
  size (unpooled h48 0.752).
- Shift grid -2..3 on ema_multi (probe at the selected layer): WavLM k7 0, 300M k18 +1, 1B k10 +1; none at the grid
  edge, so the old -1..1 grid did not clip.
- wav2vec2-large-lv60 (English LV-60k; 24 x 1024, same FLOPs per layer as XLS-R 300M). MNGU0 probe: best layer 18,
  0.890 mm valid (300M: 18, 0.860). ema_multi probe: flat and bimodal, layer 3 0.6742 / layer 5 0.6744 / layer 21
  0.6800 valid rmse_z; layer 3 transfers badly to unseen speakers (test_unseen rmse_z 0.939 vs 0.777 at layer 18).
  LOSO runs it at k=3 (validation rule) and k=18 (matched to 300M k18, isolates pretraining data/language).
- LOSO option A submitted 2026-10-08 (35 jobs `losoA_*` on preempt + `w2v2_mngu0_cc`): per fold and model, LoRA r8
  q/v + causal conv (3 seeds) and a frozen-encoder causal conv control (1 seed). Summarize with
  `python -m sparc.compression.loso --arms <label>=<model>/prefix<k>_independent_r8_q_proj+v_proj_conv_causal ...
  <label>=<model>/prefix<k>_none_conv_causal`.

### Leave-one-speaker-out CV, option A (2026-10-08)
Summary: `/data/user_data/xoy/xlsr_ema/analysis_loso/summary.md` (`python -m sparc.compression.loso`, arms in issue
#25). 7 folds, one recipe for all models (LoRA r8 q/v + causal conv, 3 seeds; frozen causal-conv control, 1 seed).
Held-out speaker, mean over folds (fold sd ~0.07-0.13 rmse_z, ~0.06-0.07 PCC; folds differ far more than models):
| model | k | GFLOPs/s | LoRA rmse_z | LoRA PCC | frozen rmse_z | frozen PCC | seen rmse_z / PCC (LoRA) |
|---|---|---|---|---|---|---|---|
| WavLM Large | 7 | 14.9 | 0.818 | 0.696 | 0.798 | 0.672 | 0.546 / 0.804 |
| XLS-R 300M | 18 | 29.1 | 0.816 | 0.708 | 0.790 | 0.684 | 0.549 / 0.816 |
| XLS-R 1B | 10 | 26.4 | 0.849 | 0.689 | 0.802 | 0.682 | 0.544 / 0.807 |
| wav2vec2-large-lv60 | 18 | 29.1 | 0.811 | 0.688 | 0.821 | 0.667 | 0.563 / 0.804 |
| wav2vec2-large-lv60 | 3 | 9.7 | 0.853 | 0.647 | 0.832 | 0.633 | 0.582 / 0.775 |
Paired across folds (mean difference, 95% t CI, folds won; with 7 folds and 90 comparisons nothing survives Holm,
so the intervals are the evidence):
- LoRA vs frozen: PCC rises for every model (WavLM +0.024 [0.013, 0.036] 7/7, 300M +0.024 [0.010, 0.037] 6/7,
  w2v2 k18 +0.022 7/7) but held-out rmse_z does not improve (WavLM -0.020, 300M -0.025, CIs span 0) and gets worse
  for 1B (-0.047 [-0.114, +0.021], 2/7). Adaptation improves trajectory shape on new speakers but not their
  normalized amplitude/offset; on seen speakers it helps both (0.546 vs 0.570 rmse_z for WavLM).
- Model size: XLS-R 1B k10 is no better than the native models under speaker shift. WavLM k7 beats it on rmse_z
  (+0.031 [0.006, 0.056], 6/7) at 56% of its FLOPs; 300M beats it on PCC (+0.019 [0.000, 0.037], 6/7).
- WavLM k7 vs 300M k18: rmse_z tie (-0.002 [-0.033, +0.029]); 300M PCC higher by 0.012 [-0.002, +0.025] (5/7)
  at 2x the FLOPs.
- Pretraining language at matched architecture/FLOPs: multilingual XLS-R 300M beats English w2v2 k18 on PCC
  (+0.019 [0.005, 0.034], 6/7), rmse_z tie; MNGU0 in-domain also favours 300M (0.739 vs 0.769 +/- 0.007 mm, 3 seeds).
  Zero-shot from MNGU0 they tie (PCC 0.673 vs 0.673; calibrated 0.730 vs 0.735).
- Validation-best layer selection fails for w2v2 on ema_multi: k3 (a near-tie with k5 and k21 on valid) is the
  worst LoRA arm and has the highest held-out rmse_z of all 10 arms (vs k18: rmse_z -0.042 [-0.074, -0.010] 6/7,
  PCC -0.041 [-0.060, -0.023] 7/7; its frozen control has lower PCC still, 0.633). It is also 3x cheaper (9.7 GFLOPs/s).
- Hardest folds: usc_M3 (DTW-segmented) and 5EMO (emotion, frame offsets); 5EMO held-out rmse_z ~0.85-0.94 vs
  USC ~0.74-0.82.
Bottom line for SPARC (normalized space, unseen speakers): WavLM Large truncated to 7 layers is the most
compute-efficient encoder; XLS-R 300M at 2x the FLOPs gives at most +0.012 PCC; compressed XLS-R 1B gives nothing.
The 1B zero-shot transfer edge (MNGU0-trained k16) did not carry over to multi-speaker-trained k10.

### PCA reparameterization of the articulators (2026-10-08)
Code: `datasets.target_matrices` (`--dataset ema_loso_<spk>_<t>` / `ema_multi_<t>`), `target_probe.py`. Each target
space is an invertible per-normalization-group linear map fitted on that group's training frames; predictions are
mapped back to per-channel z for every metric, so all numbers are comparable with target z. Like the z-score, a new
speaker's map needs that speaker's own EMA statistics (enrollment data); zca12 uses more of them (12x12 covariance).
- Diagnostic: tongue sensors are nearly isotropic in x/y (sd ratio 1.03-1.5), lips and incisor elongated (2-4).
  Matching each group's principal axes to shared axes needs 20-45 deg rotations, and for the same group the
  rotation can flip sign between folds (axes near 45 deg from the reference, or near-round distributions).
- Probe screen (ridge at the LOSO layer/shift, 7 folds, held-out speaker; difference vs z, + = better):
  | target | rmse_z | folds | PCC |
  |---|---|---|---|
  | zca (per-articulator whitening, orientation kept) | +0.007 to +0.010 | 4-6/7 | +0.008 to +0.010 |
  | pca (per-articulator principal axes aligned to shared axes) | -0.032 to -0.039 | 0-1/7 | -0.044 to -0.050 |
  | zca12 (joint 12-D whitening) | +0.030 to +0.040 (CIs exclude 0) | 7/7 every model | +0.021 to +0.028 (6/7) |
  Ranges are over WavLM k7, 300M k18, 1B k10, w2v2 k18. Per-speaker axis alignment (pca) hurts seen speakers too;
  removing each speaker's cross-articulator correlation structure (zca12) is the one that helps, by more than any
  model difference in LOSO. But this screen mixes two things (see the next section): per-speaker whitening of the
  training targets, and re-coloring predictions with the held-out speaker's own covariance at test time.
- LoRA + causal conv on zca12 targets (`losoZ_*`, issue #26, closed): all 7 folds finished; the gain is test-time
  re-coloring, not whitened training (see "Whitened training vs test-time re-coloring" below).

### Speaker scaling and per-speaker whitening (2026-10-08)
`speaker_scaling.py`; results `/data/user_data/xoy/xlsr_ema/analysis_scaling/summary.md`. Fixed model (ridge probe at
the LOSO layer/shift; WavLM k7, 300M k18, 1B k10, w2v2 k18), every subset of 1-6 training speakers, scored on every
excluded speaker in its per-channel z units. A design review showed that ridge is exactly equivariant to linear target
mixing, so "the space" itself cannot matter for a linear probe; a 2 x 2 separates what can: training targets z vs
each training group whitened with its own correlation matrix ("w"), and mapping back with the subset's average
whitening ("shared", uses only the new speaker's means/stds) vs the new speaker's own covariance ("own" = enrollment
data). The train effect is exactly 0 with one training group (checked to 1e-15). Conditions: all data, and a
frame-matched budget (constant total frames across k). Unseen speakers, rmse_z / PCC (mean over held-out speakers and
subsets; effects are improvements, + = better):
| encoder | z rmse_z k=1 -> 6 | z PCC k=1 -> 6 | zca12 train effect k=2 / 3 / 6, rmse_z | same, PCC | k=6 frame-matched rmse_z / PCC | own-covariance recolor k=1 / 6, rmse_z |
|---|---|---|---|---|---|---|
| WavLM k7 | 0.908 -> 0.789 | 0.581 -> 0.667 | +0.007 / +0.012 / +0.013 (6/7) | +0.005 / +0.007 / +0.008 (6/7) | +0.011 / +0.006 | +0.036 / +0.024 |
| XLS-R 300M k18 | 0.887 -> 0.777 | 0.608 -> 0.683 | +0.007 / +0.010 / +0.011 (6/7) | +0.003 / +0.005 / +0.006 (7/7) | +0.005 / +0.005 | +0.039 / +0.019 |
| XLS-R 1B k10 | 0.907 -> 0.793 | 0.595 -> 0.665 | +0.008 / +0.011 / +0.013 (6/7) | +0.004 / +0.006 / +0.008 (6/7) | +0.005 / +0.006 | +0.035 / +0.023 |
| w2v2 k18 | 0.916 -> 0.785 | 0.592 -> 0.675 | +0.009 / +0.010 / +0.005 (4/7) | +0.003 / +0.004 / +0.003 (4/7) | +0.007 / +0.005 | +0.038 / +0.022 |
- More training speakers is by far the largest effect (7/7 speakers, every arm): rmse_z -0.11 to -0.13, PCC +0.07 to
  +0.09 from 1 to 6 speakers, and about as much with total frames fixed, so it is speaker diversity, not data amount.
- Whitening each training speaker jointly over all 12 channels (zca12) before pooling makes pooled speakers modestly
  more useful to a new speaker: about -0.011 to -0.013 rmse_z and +0.006 to +0.008 PCC at k=6 with all data (w2v2
  half that), smaller when frames are matched. Per speaker (k=6, shared map) it is largest for usc_F5 (~+0.036 rmse_z),
  +0.006 to +0.019 for the other USC speakers and 5emo_jn, ~0 for 5emo_kf and negative for 5emo_jr. Per-articulator
  whitening (zca) does nothing (<= 0.006 rmse_z).
- Re-coloring predictions with the new speaker's own covariance at test time is the larger effect (-0.035 to -0.040
  rmse_z, +0.023 to +0.027 PCC with one training speaker; -0.019 to -0.024 / +0.016 to +0.019 with six), but it needs
  that speaker's EMA (enrollment) and shrinks as more speakers are pooled. Both together (= the earlier zca12 targets)
  give rmse_z 0.744-0.759 / PCC 0.691-0.711 at k=6 vs 0.777-0.793 / 0.665-0.683 for z.
- For SPARC: a per-speaker 12-D whitened articulatory space is a small, consistent improvement for pooling speakers;
  adding speakers matters much more than the choice of space.

### Whitened training vs test-time re-coloring with real models; SPARC-style single speaker (2026-10-12)
LOSO, LoRA r8 + causal conv (3 seeds) and frozen + causal conv (1 seed), all 7 folds (`loso_recolor.py`, results
`/data/user_data/xoy/xlsr_ema/analysis_loso_recolor/summary.md`). z-trained checkpoints were re-run on the held-out
speaker (reproduce stored rmse_z to <= 5e-4) and re-colored with that speaker's covariance; zca12-trained runs
(`ema_loso_<h>_zca12`) are mapped back with the speaker's own transform. Held-out rmse_z / PCC changes, + = better:
| encoder | recolor effect, LoRA | train effect (zca12-trained - z+recolor), LoRA | recolor effect, frozen | train effect, frozen |
|---|---|---|---|---|
| WavLM k7 | +0.035 (7/7) / +0.017 | +0.007 / -0.006 | +0.023 / +0.017 | -0.004 / -0.005 |
| XLS-R 300M k18 | +0.034 (6/7) / +0.016 | -0.004 / -0.011 | +0.023 / +0.018 | +0.000 / +0.005 |
| XLS-R 1B k10 | +0.037 (6/7) / +0.015 | +0.021 (6/7) / -0.003 | +0.018 / +0.012 | +0.008 / -0.003 |
| w2v2 k18 | +0.022 (6/7) / +0.017 | -0.011 / -0.005 | +0.022 / +0.019 | -0.000 / +0.000 |
- With real models, the benefit of the zca12 targets is re-coloring with the new speaker's covariance at test time;
  training on whitened targets adds nothing consistent (XLS-R 1B rmse_z the one partial exception; PCC changes -0.011
  to +0.005). Most fold CIs span 0 (7 of 8 recolor-effect CIs; only 300M frozen excludes 0); the direction of the
  recolor effect is consistent across encoders.
- Plain ridge probes beat the adapted models on held-out rmse_z (same folds, z targets): probe 0.777-0.793 vs frozen
  causal conv 0.790-0.821 vs LoRA causal conv 0.811-0.849 (LoRA - probe -0.026 to -0.055, CIs span 0, LoRA better on
  1-3/7 folds), while LoRA has higher PCC (+0.013 to +0.029, 6-7/7). Under speaker shift adaptation improves
  trajectory shape but fits the training speakers' amplitudes; the causal conv head, a large gain within MNGU0, adds
  nothing for a new speaker.

SPARC-style single-speaker training (MNGU0 -> 7 new speakers zero-shot; `crossspeaker.py --z-units`, results
`features_ema_multi/crossspeaker_unpaired.json`; 3 seeds per LoRA model). Mappings needing only the new speaker's EMA
statistics: z_transfer (means/stds) and recolor (zca / zca12, A_h^-1 A_MNGU0); paired references fit on 120
utterances of the speaker. Mean over speakers, rmse_z / PCC:
| model | z_transfer | recolor zca12 | paired affine | paired linear |
|---|---|---|---|---|
| SPARC shipped (WavLM k9 linear) | 0.951 / 0.656 | 0.995 / 0.632 | 0.786 / 0.656 | 0.736 / 0.720 |
| LoRA+cc WavLM k9, mm-trained | 0.914 / 0.672 | 0.938 / 0.672 | 0.765 / 0.672 | 0.705 / 0.727 |
| LoRA+cc WavLM k9, whitened-trained | 0.896 / 0.677 | 0.913 / 0.678 | 0.763 / 0.677 | 0.703 / 0.736 |
| LoRA+cc 300M k18, mm / whitened | 0.881 / 0.673, 0.878 / 0.669 | 0.898 / 0.673, 0.899 / 0.667 | 0.764 / 0.670 | 0.703 / 0.733 |
| LoRA+cc 1B k16, mm / whitened | 0.885 / 0.682, 0.865 / 0.680 | 0.903 / 0.680, 0.887 / 0.679 | 0.762 / 0.682 | 0.700 / 0.749 |
| LoRA+cc w2v2 k18, mm / whitened | 0.890 / 0.673, 0.883 / 0.671 | 0.911 / 0.673, 0.905 / 0.670 | 0.767 / 0.665 | 0.698 / 0.739 |
- From a single training speaker, re-coloring with the new speaker's covariance does not help: mean rmse_z is worse
  by 0.017-0.024 for the trained models (shipped SPARC 0.044, worse for 6/7 speakers), PCC unchanged for them (shipped
  SPARC 0.656 -> 0.632); per speaker 4/7 worse for every trained model (the three EMA_5EMO speakers and usc_F5), so
  the mean is driven by large losses on some speakers. The symmetric-whitening map is only a
  good speaker map when the source is an average over many speakers (LOSO above), not one speaker.
- Training on whitened MNGU0 targets costs 0.012-0.033 mm in-domain (all 12 seeds; e.g. WavLM 0.753 -> 0.772 mm) but
  slightly improves zero-shot rmse_z for WavLM (+0.018, 7/7), 1B (+0.019, 7/7) and w2v2 (+0.007, 6/7); 300M flat;
  PCC flat. Paired linear calibration with 120 utterances (rmse_z ~0.70, PCC 0.72-0.75) beats every unpaired mapping.

### LoRA + probe blend, enrollment curve, emotion and phone breakdowns (2026-10-12)
LoRA readout variants under LOSO (`lora_readout.py`, `analysis_readout/summary.md`; 7 folds, 3 seeds; held-out
rmse_z / PCC). Blend = 0.75 x LoRA + 0.25 x frozen-encoder ridge probe (weight chosen on the training speakers'
validation; 0.75 every time):
| encoder | probe | LoRA | ridge refit on LoRA features | blend | 2-seed LoRA ensemble (same 2x cost) |
|---|---|---|---|---|---|
| WavLM k7 | 0.789 / 0.667 | 0.818 / 0.696 | 0.822 / 0.688 | 0.787 / 0.709 | 0.806 / 0.707 |
| XLS-R 300M k18 | 0.777 / 0.683 | 0.816 / 0.708 | 0.804 / 0.708 | 0.780 / 0.721 | 0.804 / 0.717 |
| XLS-R 1B k10 | 0.793 / 0.665 | 0.849 / 0.689 | 0.851 / 0.690 | 0.811 / 0.705 | 0.835 / 0.703 |
| w2v2 k18 | 0.785 / 0.675 | 0.811 / 0.688 | 0.790 / 0.698 | 0.780 / 0.705 | 0.799 / 0.699 |
- The blend matches the probe's rmse_z (-0.018 to +0.005) and gives the best PCC (+0.030 to +0.042 over the probe, 7/7
  folds every encoder). At equal inference cost it beats a 2-seed LoRA ensemble on rmse_z by 0.019-0.024 (CIs exclude
  0 for all four, 5-7/7) and on PCC by 0.002-0.006 (ensemble numbers from a scratch analysis, not in a repo file).
- Review correction (2026-10-15): a seed ensemble does not separate shrinkage from information. A single gain on the
  LoRA output, chosen on training-speaker validation, recovers ~63% (usc_F5) and ~100% (5emo_jn) of the blend's
  rmse_z gain (WavLM k7, seed 0, 2 folds checked). So the blend's rmse_z gain is mostly shrinkage (damping is
  MSE-optimal under speaker uncertainty), not amplitude information LoRA lost; only its PCC gain over LoRA
  (+0.009 to +0.013 there) is beyond shrinkage. The blend weight grid (0..1 by 0.25) always picked 0.75.
  Cost: two encoder passes (frozen + LoRA). The single-pass alternative (ridge refit on LoRA features) does not get
  the probe's rmse_z.

Enrollment curve (`loso_posthoc.py enroll`, `analysis_posthoc/enroll.md`; LoRA, 7 folds; the new speaker's statistics
from N random training-split utterances, re-scored in its true z units). rmse_z, z-score / covariance re-color:
| encoder | N=1 | N=5 | N=10 | N=20 | all |
|---|---|---|---|---|---|
| WavLM k7 | 0.977 / 0.963 | 0.847 / 0.818 | 0.830 / 0.795 | 0.822 / 0.788 | 0.818 / 0.783 |
| XLS-R 300M k18 | 0.913 / 0.903 | 0.849 / 0.821 | 0.842 / 0.817 | 0.829 / 0.801 | 0.816 / 0.782 |
| XLS-R 1B k10 | 0.909 / 0.902 | 0.868 / 0.839 | 0.861 / 0.827 | 0.856 / 0.820 | 0.849 / 0.812 |
| w2v2 k18 | 0.924 / 0.921 | 0.845 / 0.830 | 0.820 / 0.801 | 0.828 / 0.809 | 0.811 / 0.789 |
- Roughly 10-20 utterances (well under a minute of EMA) recover most of both the means/stds and the re-coloring
  benefit (N=10-50 differences are within single-draw noise); N=1-2 is not enough for re-coloring. Without any of
  the speaker's EMA (training speakers' average statistics) rmse_z is ~6.4: absolute sensor positions differ between
  speakers by many standard deviations (hypothesis consistent with the per-speaker offsets; not separately checked).
  PCC does not depend on means/stds. Caveat: N is per normalization group, so usc_F5 (two groups) enrolls 2N.

Emotion (held-out EMA_5EMO speakers jn, jr, kf; `analysis_posthoc/emotion.md`; LoRA rmse_z / PCC over encoders):
fear 0.72-0.76 / 0.65-0.67, sadness 0.78-0.82 / 0.58-0.60, happiness 0.90-0.93 / 0.56-0.60, anger 0.91-0.95 /
0.66-0.67, neutral 1.03-1.05 / 0.66-0.67. Pooled neutral is worst only because of speaker jr's neutral block (frame
offset ~7.6 mm, rmse_z ~1.10); for jn and kf neutral is not the worst, and kf's highest-offset block (anger) is its
worst -- consistent with block frame offsets driving the rmse_z differences, though emotion and session are
confounded, so this cannot be separated. LoRA >= frozen on PCC in 19/20 encoder x
emotion cells (one tie; frozen is 1 seed); re-coloring raises anger PCC by 0.019-0.025.

MNGU0 phone classes (`analysis_posthoc/phones.md`; mm, single speaker; no phone alignments exist for the multi-speaker
corpora): coronal frames are easiest for every encoder (0.66-0.71 mm with LoRA); dorsal, silence and labial are
hardest (0.80-0.90; order varies by encoder). LoRA's largest gain over the frozen encoder is on silence frames (e.g.
WavLM 1.02 -> 0.86 mm). Whitened-target training is worse in 23 of 24 class x encoder cells (w2v2 glottal, 121 frames,
the exception).

### LOSO depth sweep: multi-speaker FLOPs frontier and leak-free depth selection (2026-10-12)
`depth_sweep.py` (`analysis_depth/summary.md`): ridge probe at every cached depth of each encoder, 7 LOSO folds; shift
(-2..3) and ridge strength chosen on the fold's training speakers only. Held-out frontier (rmse_z / PCC):
| GFLOPs/s | best encoder, depth | held-out rmse_z / PCC |
|---|---|---|
| 13.6 | w2v2 k6 | 0.794 / 0.656 |
| 17.4 | XLS-R 300M k9 ~ WavLM k9 | 0.772 / 0.679, 0.771 / 0.675 |
| 21-23 | XLS-R 300M k12 ~ WavLM k13 | 0.763 / 0.678, 0.762 / 0.675 |
| 44-50 | XLS-R 1B k19-22 | 0.754-0.759 / 0.686-0.688 |
| 78.5 | XLS-R 1B k36 | 0.751 / 0.696 |
- Beyond ~22 GFLOPs/s, 2-3.5x more compute buys only 0.008-0.012 rmse_z and +0.01-0.02 PCC, all from deep XLS-R 1B.
  Compressed 1B reaches the best probe scores under speaker shift, but only at its deep layers, never cheaply.
- The best held-out depth is deeper than the best seen-speaker depth (WavLM: k9 for seen, k12-13 for new speakers).
- Depth selection: choosing depth on the training speakers' validation sentences (nested, leak-free) picks shallow,
  speaker-specific depths (WavLM k6-9, held-out 0.787; 1B k10, 0.793; 300M k18, 0.777; w2v2 k21, 0.800) -- the same
  depths used in LOSO option A, which were chosen the same way on ema_multi. Choosing depth by held-out performance
  on the other six folds (cross-fold) is better for every encoder (WavLM k10-13 0.766; 300M k12-17 0.767; 1B k35-37
  0.757; w2v2 k16-17 0.784): to select for generalization, validate on held-out speakers. Caveats: cross-fold
  selection uses models trained with the evaluated speaker's data (not fully clean), and it costs 1.4x (WavLM) and
  3x (1B) the FLOPs of the nested choice (300M and w2v2 get cheaper); ~0.02 of selection optimism sits in k12.
- LoRA + causal conv at new frontier depths (`analysis_loso_frontier/summary.md`; 3 seeds x 7 folds; held-out
  rmse_z / PCC): WavLM k12 (21.3 GFLOPs/s) 0.803 / 0.706 vs k7 0.818 / 0.696 (PCC +0.010 [0.001, 0.019], 6/7);
  XLS-R 1B k19 (44.4) 0.814 / 0.704 vs k10 0.849 / 0.689 (rmse_z +0.035, PCC +0.015 [0.003, 0.028], 6/7);
  XLS-R 300M k10 (18.7) 0.830 / 0.697 is worse than its k18 (29.1) 0.816 / 0.708 (PCC -0.011, 7/7). WavLM k12,
  300M k18 and 1B k19 are tied (all paired CIs span 0); WavLM k12 is the cheapest of the three (73% of 300M's
  FLOPs, 48% of 1B's). As before, every LoRA arm's held-out rmse_z is worse than the probe's; the probe blend fixes it.

### Multi-speaker inverter heads dropped into SPARC (2026-10-12)
`sparc_multihead.py` (`/data/user_data/xoy/xlsr_ema/sparc_multihead/summary.md`, UTMOS in `utmos/`). Ridge heads on
WavLM layer 9 computed through the deployed `Inversion` pipeline, trained on the multi-speaker EMA (per held-out
speaker 'loso', or all seven 'all'), mapped into the shipped head's output space (per-channel sign/scale, or a full
12x12 least-squares adapter '_lin' fitted on the head's own training audio, no EMA), then resynthesized with the
shipped `load_model('en')` vocoder (shared pitch / loudness / speaker embedding). WER: whisper-large-v3 on USC-TIMIT
and LibriTTS only (many EMA_5EMO reference texts are prompt codes such as digit strings).
| system | USC-TIMIT WER | LibriTTS-R WER | SIM ECAPA (EMA / Libri) | UTMOSv2 (EMA / Libri) |
|---|---|---|---|---|
| original audio | 3.4 % | 2.3 % | - | 2.47 / 3.24 |
| shipped head | 8.6 % | 3.1 % | 0.45 / 0.81 | 2.74 / 3.15 |
| loso head, sign/scale map | 94.7 % | - | 0.31 | 2.06 |
| loso head, 12x12 adapter | 48.1 % | - | 0.35 | 2.38 |
| all-speaker head, sign/scale map | 82.6 % | 42.2 % | 0.30 / 0.56 | 2.08 / 2.50 |
| all-speaker head, 12x12 adapter | 38.8 % | 15.2 % | 0.35 / 0.65 | 2.43 / 2.85 |
| measured EMA (approximate mapping) | 145.7 % | - | 0.25 | 1.85 |
- These linear mappings do not make the shipped vocoder work with multi-speaker heads: measured EMA from these
  speakers gives the worst speech, and the best linear adapter explains only 53-78% of the shipped head's output
  variance per channel. This is consistent with the vocoder being coupled to its own inverter's output space, but it
  does not separate that from mapping, shrinkage or smoothing error; retraining the vocoder on the new inverter's
  outputs (or an oracle-mapping test) is needed to show it. Either way, a multi-speaker inverter cannot be judged or
  deployed with the shipped vocoder as is.
- The heads' held-out articulation is normal (rmse_z 0.64-0.88, PCC 0.61-0.78 per speaker, as the layer-9 probes).
- Vocoder controls (2026-10-16, `sparc_multihead.py controls`; WER whisper-large-v3, USC-TIMIT / LibriTTS):
  shipped outputs shrunk to the adapter's per-channel amplitude 9.9 % / 3.0 % (shipped 8.6 / 3.1): amplitude does
  not matter. Shipped outputs plus the all-speaker adapter's own residual, circularly time-shifted (same error size
  and spectrum, not aligned with the speech) 30.4 % / 9.9 % vs the adapter itself 38.8 % / 15.2 %: the size of the
  error alone accounts for ~55-70 % of the degradation; the vocoder is very sensitive to output deviations of the size
  a linear map from a multi-speaker head leaves (22-47 % unexplained variance). Oracle on the shipped inverter's own
  speaker: measured MNGU0 EMA mapped into its output space (12x12 linear map, R^2 0.74-0.86 per channel, fitted on
  training audio) gives 13.2 % WER vs 7.6 % for the shipped inverter (61 test utterances, WER vs Whisper on the
  original). So the vocoder does accept real articulation of its training speaker -- it is not tied to the inverter's
  outputs as such. The multi-speaker failure is mapping error of that size plus other speakers' different
  articulatory space (their measured EMA, 145.7 %). Retraining the vocoder on the new inverter's output space is still
  needed to deploy a multi-speaker inverter.

### Head screen: temporal inductive biases on frozen encoders (2026-10-14)
Question (user): an extra transformer/attention head on WavLM is redundant with keeping another encoder layer, so a
head must bring a different inductive bias; and FLOPs matter. `head_screen.py` + `seqheads.py`
(`analysis_heads/summary.md`): heads trained on cached, low-passed features of WavLM k7/9/12 and XLS-R 300M k9/18,
7 LOSO folds; fold-internal shift, learning rate from {1e-4, 3e-4, 1e-3, 3e-3} on training-speaker validation, then 3
seeds; primary arm without the output smoother. Heads: linear; MLP (same LayerNorm + 1024->256 GELU front end and
capacity as the sequence heads, no temporal context: the capacity control); causal conv 9 (the head used so far);
centered conv 9 / 17 (learned FIR); BiGRU; bidirectional diagonal linear SSM (S4D); BiMamba. Head cost is
0.001-0.17 GFLOPs/s (widest conv, 65 frames, 0.45) against ~1.3 per extra encoder layer. Heads no longer leak padding into valid frames (fixed in
`heads.py`; tested batched == unbatched).
- Temporal context raises held-out PCC beyond capacity (vs the MLP control, all 5 depths): BiGRU +0.024 to +0.041
  (7/7 folds everywhere), S4D +0.010 to +0.026, centered conv +0.008 to +0.019 (mostly 7/7), causal conv +0.005 to
  +0.011. Centered beats causal at every depth; recurrence is the strongest bias.
- No head improves held-out rmse_z: the ridge probe stays best at every depth (trained heads -0.0015 to -0.031, the
  best ones with CIs spanning 0; even the trained linear head trails ridge by ~0.005, the optimizer's calibration
  gap); temporal heads are within +-0.021 of the MLP. Every head with hidden capacity fits training speakers far better (seen rmse_z ~0.57 vs probe 0.67)
  without gaining on new speakers -- the same amplitude overfitting as LoRA.
- Head vs more encoder layers at matched FLOPs: for PCC a head wins -- WavLM k7 + BiGRU (14.9 GFLOPs/s, frozen)
  0.698, above any probe depth up to 23 GFLOPs/s (<= 0.681) and equal to LoRA + causal conv at k7 (0.696); WavLM
  k12 + BiGRU 0.706 = LoRA at k12. For rmse_z depth wins (probe k12 0.764 vs k7 + BiGRU 0.805).
- The +-1 s output smoother has small, head-dependent effects on low-passed features: typically 0.003-0.008 rmse_z,
  up to 0.015 (single seed; sign varies by head), PCC within ~0.006.
- Conv width (centered; PCC gain over the MLP; WavLM k7 / WavLM k12 / 300M k18): 3 frames +0.005 / +0.007 / -0.001,
  5: +0.010 / +0.012 / 0.000, 9: +0.015 / +0.016 / +0.008, 17: +0.019 / +0.017 / +0.011, 33: +0.014 / +0.020 / +0.014,
  65: +0.015 / +0.021 / +0.015. Context helps up to ~17-33 frames (0.3-0.7 s) then saturates; on shallow WavLM k7 widths
  33-65 hurt rmse_z (~-0.018 vs probe, CIs exclude 0). No width reaches BiGRU at these depths (+0.026 to +0.035).
- Cascades (held-out rmse_z / PCC; WavLM k7 / k12 / 300M k18): BiGRU 0.805/0.698, 0.774/0.706, 0.794/0.707; two-layer
  BiGRU no better (0.808/0.698, 0.773/0.703, 0.795/0.701); conv -> BiGRU slightly worse than BiGRU alone on both metrics
  (conv9 0.812/0.694, 0.774/0.700, 0.799/0.703); BiGRU -> conv9 0.805/0.694, 0.765/0.709, 0.791/0.707 (best head at
  WavLM k12, mixed elsewhere).
- BiMamba (parallel scan; 0.114 GFLOPs/s): PCC close to BiGRU and rmse_z better than BiGRU at every depth (by
  0.005-0.011; no paired CI, and per-fold seed sd of these heads is 0.007-0.019, so within seed noise), near the probe
  for WavLM: k9 0.773/0.707 (probe 0.771/0.675), k12 0.766/0.705 (probe 0.764/0.676); 0.010-0.017 behind it for 300M
  (k9 0.789/0.705, k18 0.787/0.702); WavLM k7 0.799/0.694. A good balanced head; the ordering among BiMamba, BiGRU->conv
  and BiGRU is not resolved.
- A frozen WavLM k12 + BiGRU->conv (0.765 / 0.709, 21.4 GFLOPs/s, one pass) ties the WavLM-k7 LoRA + probe blend's
  PCC (0.787 / 0.709, ~30 GFLOPs/s, two passes) and beats its rmse_z by 0.022, but that rmse_z edge is mostly depth
  (the k12 probe alone is 0.764), k12 itself was chosen by cross-fold held-out selection, the comparison is unpaired,
  the head is the best of many configurations scored on the held-out speakers, and it is 0.012 PCC below the 300M-k18
  blend (0.721). Frozen BiMamba vs LoRA + causal conv at k12 (+0.037 rmse_z, 6/7) mixes the head with LoRA's
  amplitude overfit and pipeline differences (frozen heads see 10 Hz low-passed cached features without the output
  smoother; LoRA sees unfiltered encoder outputs with dropout on, the smoother on and a fixed shift of 1); the clean
  frozen-vs-frozen head contrast is smaller (BiMamba vs causal conv, 0.012 rmse_z).
- LoRA under a strong head (WavLM k12, 7 folds x 3 seeds; `head_lora.py`, `analysis_heads/lora_heads_wavlm-large_k12.md`;
  held-out rmse_z / PCC): probe 0.764 / 0.676; causal conv frozen 0.778 / 0.682, LoRA 0.803 / 0.706, blend
  0.769 / 0.718; BiMamba frozen 0.766 / 0.705, LoRA 0.795 / 0.713, blend 0.770 / 0.722; BiGRU->conv frozen 0.765 / 0.709,
  LoRA 0.791 / 0.714, blend 0.762 / 0.724. With a causal conv head LoRA raises PCC by +0.024 [0.013, 0.036] (7/7); with
  BiMamba or BiGRU->conv only +0.005 to +0.008 (CIs span 0), while rmse_z still worsens by 0.026-0.029 (BiMamba CI
  excludes 0). So most of LoRA's held-out PCC gain is temporal context that a sequence head on the frozen encoder also
  provides. The probe blend still gives the best PCC (BiGRU->conv blend +0.048 [0.032, 0.063] over the probe, 7/7, rmse_z
  tie), but costs a second encoder pass (~43 vs 21.4 GFLOPs/s) for ~+0.015 PCC over the frozen head (unpaired), and its
  rmse_z gain over LoRA is likely mostly shrinkage (review, 2026-10-15). Caveat: the LoRA and frozen pipelines still
  differ (input low-pass, smoother, shift, LR grid, dropout); the matched-pipeline contrast is rigor-plan item 6.

### Why multi-speaker numbers are worse than single-speaker SPARC; short and long inputs (2026-10-15)
Same metric for all (rmse_z in each speaker's own z units / PCC), WavLM LoRA + causal conv: MNGU0 (train and test on one
speaker) 0.36 / 0.93 (ridge probe 0.43 / 0.90); multi-speaker seen speakers 0.55 / 0.80 (USC-TIMIT ~0.86 PCC); held-out
speakers 0.80-0.82 / 0.70-0.71 for LoRA + causal conv (0.76-0.79 for probes and frozen sequence heads). A plausible
reading, not tested by any data-size or session control: a data gap (MNGU0 is one clean session with ~1000 training
utterances vs 180-370 per multi-speaker speaker with session offsets, emotional speech and harder alignment) and a
speaker-shift gap (0.55 -> 0.80) that no model or head closed. z units are not strictly comparable across corpora
(MNGU0 statistics come from silence-trimmed data; EMA_5EMO std is inflated by block offsets). Single-speaker AAI
benchmarks overstate accuracy on a new person.

Input length (`length_eval.py`, `analysis_length/summary_k12.md`; LOSO, WavLM k12, seed 0, every system rerun from
audio and scored on the same EMA frames; audio trimmed to a multiple of 320 samples, which mainly changes the last
frames' filter edge -- untrimmed live features equal the cache to 3.5e-4 on a checked EMA_5EMO utterance, trimmed ones
differ by 0.14-0.18 relative at the final frames; one utterance per fold checked; heads and LoRA single seed). rmse_z / PCC:
| system | native | long (>= 20 s concatenations) | 1 s chunks | 0.5 s chunks |
|---|---|---|---|---|
| ridge probe | 0.763 / 0.676 | 0.773 / 0.666 | 0.809 / 0.640 | 0.836 / 0.613 |
| causal conv 9 | 0.774 / 0.682 | 0.797 / 0.652 | 0.812 / 0.659 | 0.822 / 0.636 |
| centered conv 9 | 0.772 / 0.687 | 0.800 / 0.653 | 0.804 / 0.665 | 0.813 / 0.640 |
| BiGRU | 0.775 / 0.705 | 0.789 / 0.691 | 0.814 / 0.668 | 0.824 / 0.648 |
| BiGRU -> conv 9 | 0.759 / 0.714 | 0.783 / 0.700 | 0.806 / 0.668 | 0.811 / 0.645 |
| BiMamba | 0.763 / 0.703 | 0.784 / 0.684 | 0.810 / 0.658 | 0.823 / 0.631 |
| LoRA + causal conv | 0.809 / 0.707 | 0.813 / 0.699 | 0.832 / 0.663 | 0.838 / 0.641 |
- Long inputs: every system loses a little (probe 0.009 rmse_z; recurrent/SSM heads 0.014-0.024 and 0.014-0.018 PCC;
  conv heads ~0.03 PCC) but none collapses; heads trained on 1-5 s clips hold up on 20-30 s. LoRA is the most
  length-stable but stays worst on rmse_z.
- Short inputs cost more: 1 s chunks lose 0.023-0.048 rmse_z and 0.022-0.046 PCC, 0.5 s chunks more. The frame-wise
  probe degrades most on rmse_z (+0.073 at 0.5 s), the recurrent/SSM heads most on PCC. The cause is not separated:
  encoder context, the 10 Hz low-pass re-applied to short pieces (edge effects) and per-chunk waveform normalization
  are confounded (no control). Temporal heads keep +0.018 to +0.035 PCC over the probe at 0.5 s. Relevant for chunked /
  streaming use.
- Natural duration (native): PCC falls with length for all systems (~0.74-0.76 under 2 s, n=18, to ~0.60-0.63 above
  6 s); rmse_z is not monotone (probe 0.795 at 3-4 s, 0.758 at 4-6 s, 0.843 above 6 s). The long utterances are mostly
  EMA_5EMO passage phrases, so length is confounded with corpus. Recurrent/SSM heads keep a PCC lead over the probe in
  every bin (only +0.005 to +0.023 above 6 s); conv heads tie or fall below the probe above 4 s.

### Rigor re-analysis, tier 1 (2026-10-15; pre-registered in `docs/rigor_prereg.md`)
`rigor.py` (`analysis_rigor/summary.md`, `results.json`, `latency.md`): every arm re-scored from cached held-out
predictions (retrained frozen heads reproduce the head screen to 1e-4; recomputed blends match lora_readout to 1.4e-4).
All configurations were already looked at on these folds, so nothing here is an untouched number.
Pre-registered contrasts (WavLM k12 unless noted; exact sign-flip test, Holm over 3; hierarchical bootstrap 95%):
| id | contrast | metric | diff | bootstrap 95% | folds | p | Holm |
|---|---|---|---|---|---|---|---|
| P1 | blend BiGRU->conv vs LoRA BiGRU->conv x one gain fitted on the other folds | rmse_z | +0.009 | [-0.021, +0.041] | 4/7 | 0.59 | n.s. |
| P2 | frozen BiGRU->conv vs ridge probe | PCC | +0.033 | [+0.022, +0.045] | 7/7 | 0.016 | significant |
| P3 | frozen BiGRU->conv, WavLM k12 vs XLS-R 300M k18 | PCC | +0.003 | [-0.009, +0.014] | 4/7 | 0.55 | n.s. |
- The second encoder pass is not worth it for rmse_z: scaling LoRA's output by one gain (~0.62-0.67, fitted on the other
  six folds; oracle mean ~0.68-0.76) matches the blend for every encoder and head (blend - shrunk LoRA -0.003 to
  +0.009 rmse_z; per-channel gain -0.011 to +0.004, all p > 0.5). The blend's only real gain over LoRA is PCC (+0.009 to +0.017,
  7/7 everywhere), which a gain cannot produce. Shrinking LoRA alone improves rmse_z by 0.019-0.041 (CIs span 0).
- rmse_z rankings are driven by per-utterance offsets: arm rankings under rmse_z and centered rmse_z (each utterance's
  mean error removed) barely agree (Kendall tau 0.30; speaker-centered 0.59; PCC vs concatenated PCC 0.70). On centered
  rmse_z every WavLM k12 LoRA arm (0.571-0.582) beats the k12 probe (0.593); across all encoders LoRA spans
  0.571-0.604 vs probes 0.591-0.601 (XLS-R 1B k10 LoRA the one arm above its probe). 44-49% of LoRA's squared error
  is per-utterance bias vs 39-42% for probes. Std ratios (prediction / target) 0.72-0.83 for raw arms.
- LoRA vs frozen at k12 is an offset effect: with BiMamba / BiGRU->conv heads they tie on centered rmse_z (+-0.005) and
  PCC (CIs span 0); with causal conv LoRA is better on centered rmse_z by 0.017 (frozen wins 1/7) and PCC by 0.024.
- Temporal heads vs the MLP control at k12: BiGRU->conv +0.039 PCC, BiMamba +0.035, causal conv +0.011 (7/7, 7/7, 6/7);
  BiMamba vs BiGRU->conv -0.004 PCC, -0.001 rmse_z (no difference).
- Encoder (P3): under a frozen BiGRU->conv head 300M k18 brings no PCC gain and is worse on rmse_z (0.791 vs 0.765) at
  1.37x the FLOPs; WavLM k12 stays the recommended encoder.
- Post-hoc +-1 s output smoother on every arm: |d rmse_z| <= 0.0034, |d PCC| <= 0.0038. Negligible.
- Enrollment for every arm: N = 0 (training speakers' average statistics) gives rmse_z 6.41-6.49; N = 10 utterances
  is within 0.017 of the full-statistics number, N = 20 within 0.011 (10 random draws).
- Noise and power: seed sd 0.008-0.023 rmse_z / 0.004-0.009 PCC per fold; fold sd 0.06-0.13 / 0.06-0.075; with 7 folds
  the effect detectable at 80% power is ~0.06 rmse_z and ~0.017-0.018 PCC, so most head-vs-head and encoder rmse_z
  differences cannot be resolved by this design.
- Latency (batch 1, `latency.md`; job landed on an RTX A6000): GPU times are launch-bound at batch 1 (9-18 ms for every
  encoder, so not informative). CPU (8 threads, 4 s audio): WavLM k7 145 ms, k9 174, k12 207, XLS-R 300M k18 277, 1B
  k19 350; heads on CPU: linear/MLP/conv <= 1.1 ms, BiGRU 11 ms, BiGRU->conv 12 ms, BiMamba 19 ms (the parallel scan is
  slower than the GRU on CPU). Heads cost <= 10% of the encoder; the two-pass blend doubles the encoder cost.

### Rigor tier 2 (2026-10-16; pre-registered addendum in `docs/rigor_prereg.md`; complete)
Final pre-registered family (Holm over T1, T2, N1; exact sign-flip test): none significant. T1 +0.010 PCC (p 0.078),
T2 +0.005 rmse_z (p 0.69), N1 -0.004 PCC (p 0.50, nested better on 0/7 folds).
- Item 4 nested selection (`nested.py`, `analysis_rigor/nested/summary.md`): depth chosen by inner leave-one-speaker-out
  on the training speakers only. Frozen BiGRU->conv (depths 7/9/12/15/18, PCC rule) picks k12 on 5/7 folds (k9, k15
  on the others) and scores 0.771 / 0.705 vs k12 0.765 / 0.709 (N1 above); BiMamba picks 9-15, 0.773 / 0.704 vs k12
  0.766 / 0.705. WavLM probe (depths 1-24) nested 0.769 / 0.674 vs k12 0.764 / 0.676 (n.s.); XLS-R 300M probe nested
  picks k12-13 (0.763 / 0.681) vs its k18 0.777 / 0.683. So the k12 recommendation is not an artifact of choosing depth
  on the held-out speakers: leak-free selection lands on (or next to) k12 and does no better.

`rigor.py tier2` (`analysis_rigor/tier2.md`). Matched pipeline (`matched.py`): LoRA + BiGRU->conv at WavLM k12 trained
like the frozen heads (encoder output low-passed in-graph, no smoother, fold shift, encoder dropout off, fp32, head
lr from the head screen, LoRA lr on validation; 3 seeds x 7 folds). Every fold chose the largest LoRA lr in the grid
(5e-4), so the matched LoRA may be under-tuned. Held-out rmse_z / PCC / centered rmse_z: matched LoRA 0.786 / 0.720 /
0.568; frozen BiGRU->conv 0.765 / 0.709 / 0.574; original-pipeline LoRA 0.791 / 0.714 / 0.576; probe 0.764 / 0.676 / 0.593.
| id | contrast | metric | diff | bootstrap 95% | folds | p |
|---|---|---|---|---|---|---|
| T1 | matched LoRA vs frozen BiGRU->conv | PCC | +0.010 | [+0.001, +0.021] | 6/7 | 0.078 (n.s.) |
| T2 | matched vs original-pipeline LoRA | rmse_z | +0.005 | [-0.021, +0.032] | 3/7 | 0.69 (n.s.) |
(N1 pending; Holm will be over T1, T2, N1.)
- With identical pipelines LoRA adds at most ~+0.01 PCC over a frozen BiGRU->conv head (not significant; bootstrap
  interval barely excludes 0) and still loses 0.021 rmse_z (3/7). Its rmse_z deficit is not a pipeline artifact (T2 ~0):
  the pipeline changes only PCC (+0.005, 6/7) and centered rmse_z (+0.008, 7/7). LoRA's extra error is per-utterance
  offset (bias share 0.475 vs 0.431 frozen). Conclusion stands: a frozen encoder + sequence head is the efficient choice.
- Causal pipeline (frozen heads, 3 seeds x 7 folds, no smoother): a causal 10 Hz low-pass (one-sided Butterworth;
  validation picks shift +3 frames = 60 ms lookahead) or no low-pass at all costs nothing vs zero-phase features
  (|d PCC| <= 0.003, |d rmse_z| <= 0.002 for causal conv 9 and a unidirectional GRU), and the unidirectional GRU on
  causal features is within 0.009 PCC of the BiGRU on zero-phase features (n.s.). By the pre-registered rule (> 0.01 PCC
  loss) the causal heads keep their label; the encoder itself is still bidirectional (full-utterance attention).
- Item 9 (alignment, `rigor_extra.py`): a +-1 frame test-time misalignment costs ~+0.02 rmse_z / -0.022 PCC for every
  arm with arm rankings nearly unchanged (Kendall tau 0.91-0.96); +-1 frame random training-speaker offsets cost probes
  ~+0.005 rmse_z / -0.007 PCC (7/7) with the encoder ranking unchanged in 5/5 draws.
- Item 10 (data curve; `rigor_extra.py`, `analysis_rigor/extra/summary.md`), WavLM k9, rmse_z / PCC: MNGU0 probe 0.426
  / 0.904 (all 1064 utterances) -> 0.473 / 0.884 (100); MNGU0 LoRA + causal conv (3 seeds) 0.362 / 0.933 (all) -> 0.388
  (400) -> 0.412 (200) -> 0.438 / 0.904 (100), so LoRA needs data more than the probe. Within-speaker probes on the
  multi-speaker speakers (own train -> own test) 0.588 / 0.772 at 100 utterances, 0.561 / 0.788 with all 184-368; a
  new (held-out) speaker 0.771 / 0.675. Of the MNGU0 -> new-speaker gap (+0.344 rmse_z) about 14% is data size, 33%
  corpus/speaker difficulty at matched N=100 (sensor placement, sessions, emotion, alignment) and 61% speaker shift
  (shares overlap). The multi-speaker vs single-speaker gap is mainly speaker shift, not too little data. Descriptive.
- Item 8 (length controls; `length_eval.py`, `analysis_length/summary_k12.md`; 7 folds x 3 seeds): decomposition of the
  1 s chunk loss (rmse_z / PCC, + = worse): total +0.023 to +0.047 / +0.023 to +0.044; head context (full-utterance
  features cut into pieces) ~0 rmse_z (-0.010 to +0.001) / +0.006 to +0.010 PCC for the heads; encoder context (each
  piece encoded alone, low-passed over the whole utterance) +0.038 to +0.067 / +0.018 to +0.033 -- the whole loss;
  per-chunk waveform normalization <= 0.007; re-filtering each piece is not a cost (it partly offsets the encoder term,
  -0.013 to -0.020 rmse_z for the heads). 0.5 s pieces: same pattern, encoder +0.052 to +0.095. So chunked or
  streaming use needs encoder context (overlapping windows / left context), not a different head or filter.

### Architecture conclusions (2026-10-12; claims checked against the raw results by an independent two-lens pass)
- Update (depth sweep, 2026-10-12): for new speakers the recommended depth is deeper than the seen-speaker optimum:
  WavLM Large k12 (21.3 GFLOPs/s) with LoRA + causal conv ties XLS-R 300M k18 and XLS-R 1B k19 at 48-73% of their
  FLOPs; blended 0.75/0.25 with the frozen-encoder probe it also gets the probe's rmse_z.
- Encoder / FLOPs: WavLM Large truncated to 7-9 layers is the cheapest encoder that is never clearly beaten. LOSO held
  out: WavLM k7 (14.9 GFLOPs/s) ties XLS-R 300M k18 (29.1) on rmse_z (0.818 vs 0.816), 300M +0.012 PCC (CI spans
  0). MNGU0: 300M k18 is better by 0.014 mm (0.739 vs 0.753) at 1.7x FLOPs. Each model ran at one depth in LOSO, so
  there is no multi-speaker efficiency frontier, only these points.
- Compressing large SSL models: no truncation, non-contiguous selection or head/FFN pruning of XLS-R 300M/1B/2B beats
  WavLM at matched FLOPs; XLS-R 1B k10 is the weakest LOSO model (WavLM better on rmse_z +0.031 [+0.006, +0.056] at
  56% of its FLOPs). 1B's only edge is zero-shot from MNGU0 (+0.009 PCC over 300M, 3 seeds) and it is not in rmse_z.
  1B was run at one validation-chosen depth in LOSO (k10), so "k10 was a bad pick" is not excluded.
- Adaptation and heads: within one speaker, the causal conv head (MNGU0 WavLM 0.889 -> 0.793 mm frozen) and LoRA
  (-> 0.753) are the large gains. Under speaker shift, the ridge probe has the best rmse_z at a given depth; the best
  PCC comes from the LoRA + probe blend (0.705-0.721) and frozen recurrent/SSM heads (0.705-0.709 at WavLM k12); the
  causal conv head adds little for new speakers. Layer pooling was discarded (0/3 models; MNGU0 1B the exception).
- Pretraining: multilingual XLS-R 300M beats English w2v2 at identical architecture/FLOPs on held-out PCC (+0.019
  [0.005, 0.034], 6/7) and on MNGU0 (0.739 vs 0.769 mm); rmse_z ties. All test speakers are English, so this says
  nothing about generalization to other languages.
- Data and target space matter more than architecture: 1 -> 6 training speakers lowers ridge rmse_z by 0.11-0.13
  (larger than any encoder difference, ~0.03); re-coloring with the new speaker's covariance (enrollment) gains
  0.02-0.04 rmse_z in pooled multi-speaker models; per-speaker whitening at training time is a small effect.
- Evidence limits: 7 LOSO folds from 2 English corpora, fold sd (0.07-0.13 rmse_z) exceeds most model differences, no
  LOSO contrast survives Holm over 90 comparisons; LoRA 3 seeds, frozen/2B/pruning/1B k36 single seed; layer depth and
  shift were selected on ema_multi validation, which contains sentences of speakers later held out in LOSO.

### Caveats and open directions
- Standing caveats from the adversarial reviews (2026-10-15), apply to every held-out table above:
  - Held-out rmse_z uses each held-out speaker's own full training-split EMA statistics (oracle calibration): ~6.4
    without any of its EMA, ~0.82 with 10-20 enrollment utterances. Relative rankings are fine; absolute numbers are
    not zero-shot.
  - rmse_z is dominated by per-utterance offset / gain (EMA_5EMO block offsets): with each utterance's mean error
    removed, LoRA beats the probe on usc_F5 (2 folds checked). Report centered rmse_z and prediction/target std
    ratios alongside rmse_z.
  - The selection chain ends on the test folds (depth chosen cross-fold, best heads highlighted on held-out scores); a
    fresh holdout speaker or corpus is needed for a clean final number.
  - Paired t CIs treat the 7 folds as independent, but folds share 5 of 6 training speakers and 3 share a corpus.
  - The multi-speaker latency correction used SPARC's WavLM-based inverter (a WavLM-family dependence avoided for
    MNGU0); residual lag is within +-6 ms.
- MNGU0 has one speaker and 61 test utterances. Speaker generalization and phonetic coverage need another
  EMA corpus (e.g. the group's `EMA_5EMO_NSF` set, or mocha-timit / HPRC). Per-phone-class errors are in
  each run's `results.json` but haven't been analyzed yet.
- All encoders are bidirectional transformers, so "causal" applies only to the head.
- Adaptation used one recipe (r=8 on q/v, lr 2e-4, 40 epochs, early stopping). It wasn't tuned per model.
- Directions the data does support: WavLM-based compact encoders; more training speakers; re-coloring with a new
  speaker's covariance when its EMA is available. Temporal heads and LoRA help within a speaker but not on held-out
  rmse_z. Open: depth sweeps under LOSO (no multi-speaker efficiency frontier yet), non-English test data,
  distillation.
