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

### Results so far (linear probes, test set; 2026-10-06)
| model | best k (valid) | GFLOPs/s audio | params | test RMSE mm (95% CI) | PCC |
|---|---|---|---|---|---|
| WavLM Large | 9 / 24 | 17.4 | 127M | 0.888 (0.853-0.925) | 0.904 |
| XLS-R 300M | 18 / 24 | 29.1 | 240M | 0.893 (0.858-0.931) | 0.902 |
| XLS-R 1B | 16 / 48 | 38.4 | 333M | 0.893 (0.859-0.931) | 0.901 |
Full models (last layer): 0.968 / 1.000 / 1.028 mm at 36.8 / 36.8 / 102.6 GFLOPs/s.

- WavLM's best layer (9) matches the SPARC paper's choice, which supports the new alignment/split.
- XLS-R 300M and 1B both have two peaks at the same relative depths (300M: layers 8-9 and 18;
  1B: layers 10-17 and 36), with a dip between. 1B reaches its ~0.90 mm plateau at k=10 (26% of its FLOPs).
- With linear probes, truncated XLS-R 1B does not beat the native models at matched compute: at the
  XLS-R 300M budget all three tie within CI, and WavLM gets there with less than half the FLOPs. At
  ~24 GFLOPs/s, 1B k=9 (0.934) does no better than 300M k=8 (0.926). So far the hypothesis is not supported
  by truncation alone; adaptation (LoRA variants), non-contiguous selection and XLS-R 2B are next.

### Running / next
- Phase 3 adaptation runs (LoRA independent / shared-A / shared-gated, temporal heads) on 1B k=16,
  fair baselines on 300M k=18 and WavLM k=9, lower budgets 1B k=9/12 and 300M k=8.
- Phase 5 greedy and Block-Influence layer selection on 1B from prefix 24 down to 8 (`prune.py`).
- Then: `python -m sparc.compression.analyze` to refresh figures and the compute-matched table.
