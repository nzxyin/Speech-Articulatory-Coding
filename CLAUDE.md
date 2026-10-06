# Speech-Articulatory-Coding (SPARC fork) — project state

Unofficial fork of SPARC (Berkeley). Changes are listed in `CHANGELOG.md`; this file tracks findings and
direction. Cluster conventions live in `~/.claude/CLAUDE.md`.

## Active direction: compressing large SSL encoders for speech-to-EMA (since 2026-10-06)

Question: does large-scale SSL pretraining followed by structural compression (layer truncation,
non-contiguous layer removal, LoRA adaptation) give a better compact encoder for articulatory (EMA)
prediction than a natively compact SSL model at the same inference compute? Primary model XLS-R 1B;
baselines XLS-R 300M and WavLM Large; XLS-R 2B only if the 1B result calls for a scaling test.

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
WavLM Large truncated to 9 layers is the most compute-efficient encoder throughout. The best absolute
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
  A windowed attention head (±25 frames) is no better than the causal conv (0.816).
- Frozen XLS-R 2B (k=11) + causal conv head is the best frozen encoder (0.772 vs 0.793 for 300M/WavLM, 0.810 for 1B),
  so larger pretraining does give richer features. But LoRA closes the gap for the smaller models, and 2B needs
  2-3x their FLOPs to get there.
- Cross-layer LoRA: with the causal conv head, shared-A/B with per-layer rank gates (41k LoRA params) matches
  independent LoRA (655k): 0.759 vs 0.758 on 1B k=16. With a linear head the shared variants trail
  (shared-gated 0.800, shared-A 0.769, independent 0.760). Merged LoRA weights reproduce the unmerged test
  RMSE in every run, so LoRA adds no inference cost.
- Non-contiguous selection on 1B (prefix 24 -> 8 layers): greedy backward elimination matches the prefix from
  10-24 layers (it picks exactly 1..16 at 16 layers). It only helps at aggressive compression (8 layers: 0.926
  vs prefix 0.981, by skipping layers 3 and 6 and reaching layer 10), which still only ties 300M k=8 at fewer
  FLOPs. Block-Influence (ShortGPT) selection is worse than the prefix at every size.
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
- Frame-wise attention pooling helps frozen encoders (-0.016 to -0.022 mm). The attention mixes the last
  retained layer with layer 1. Static pooling does not help frozen encoders. (On WavLM it never left its
  ridge initialization; with raw, non-normalized layers it reached 0.881.)
- With LoRA, the pools concentrate on the last retained layer (0.55-0.80 weight) and gains are seed-noise-sized.
  The best LoRA results for every model use static per-articulator pooling with a small head (hidden 48).
  That looks like regularization, not per-articulator specialization: all six articulators learn nearly
  the same layer weights.
- Per-articulator pooling gives no systematic gain. This matches the per-channel probe check (picking each
  channel's best layer on validation changes test RMSE by <= 0.003 mm).
- 1B still doesn't beat 300M (0.740 vs 0.737 mm, 38 vs 29 GFLOPs/s), so component pruning was run on 300M.

Component pruning of XLS-R 300M k=18 (LoRA + causal conv source; `components.py`: Taylor importance on
head/FFN masks, 4 iterative rounds with LoRA recovery, physical removal, 40-epoch final fine-tune):
| keep | heads | FFN neurons | params | GFLOPs/s | test RMSE | PCC | nearest truncation / native |
|---|---|---|---|---|---|---|---|
| 0.75 | 216/288 | 55k/74k | 183M | 23.3 | 0.754 | 0.932 | 1B k=9 0.780 @ 24.3 |
| 0.5 | 144/288 | 37k/74k | 127M | 17.4 | 0.786 | 0.929 | WavLM k=9 0.753 @ 17.4; 300M k=8 0.783 @ 16.1 |
| 0.5 random | 144/288 | 37k/74k | 127M | 17.4 | 0.858 | 0.916 | (control) |
| 0.4 | 115/288 | 29k/74k | 104M | 15.1 | 0.812 | 0.926 | 300M k=8 0.783 @ 16.1 |
| 0.3 | 86/288 | 22k/74k | 81M | 12.8 | 0.835 | 0.919 | WavLM k=6 0.785 @ 13.6 |
- Taylor importance clearly beats random (0.786 vs 0.858 at keep 0.5), so the scores carry real information.
- Mild pruning (keep 0.75) beats truncation at that cost. From keep 0.5 down, pruning inside the layers is no
  better than dropping late layers, and clearly worse than WavLM at the same FLOPs.
- Overall: no compression of XLS-R (1B or 300M; truncation, non-contiguous layers, pooling, head/FFN pruning)
  beats WavLM Large k=9 + LoRA + causal conv at its 17.4 GFLOPs/s.

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

### Caveats and open directions
- MNGU0 has one speaker and 61 test utterances. Speaker generalization and phonetic coverage need another
  EMA corpus (e.g. the group's `EMA_5EMO_NSF` set, or mocha-timit / HPRC). Per-phone-class errors are in
  each run's `results.json` but haven't been analyzed yet.
- All encoders are bidirectional transformers, so "causal" applies only to the head.
- Adaptation used one recipe (r=8 on q/v, lr 2e-4, 40 epochs, early stopping). It wasn't tuned per model.
- Directions the data does support: cheap temporal heads, cross-layer shared LoRA (16x fewer adapter
  parameters at equal accuracy), and WavLM-based compact encoders. Distillation, or structured pruning inside
  layers (heads/FFN width) rather than whole layers, would be the next way to test the large-model hypothesis.
