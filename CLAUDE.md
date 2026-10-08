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

### Caveats and open directions
- MNGU0 has one speaker and 61 test utterances. Speaker generalization and phonetic coverage need another
  EMA corpus (e.g. the group's `EMA_5EMO_NSF` set, or mocha-timit / HPRC). Per-phone-class errors are in
  each run's `results.json` but haven't been analyzed yet.
- All encoders are bidirectional transformers, so "causal" applies only to the head.
- Adaptation used one recipe (r=8 on q/v, lr 2e-4, 40 epochs, early stopping). It wasn't tuned per model.
- Directions the data does support: cheap temporal heads and WavLM-based compact encoders. Distillation, or structured pruning inside
  layers (heads/FFN width) rather than whole layers, would be the next way to test the large-model hypothesis.
