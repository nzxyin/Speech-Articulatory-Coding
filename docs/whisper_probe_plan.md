# Plan: do Whisper encoder layers carry articulatory (EMA) information as well as WavLM Large and XLS-R? (2026-11-05)

Fixed before any Whisper number exists. Everything in "Decisions" (models, padding, alignment, shift rule, selection,
compared quantities, thresholds) is fixed here; deviations go into a dated amendment at the end of this file. The only
numbers already known are the existing WavLM / XLS-R results (`analysis_depth/summary.md`, `CLAUDE.md` sections "Probe
results" and "LOSO depth sweep"), which are the comparison baselines and set no Whisper threshold.

## Question

Same as for the other encoders: how well does a linear (ridge) probe on the frozen hidden state of each encoder layer
predict the 12 EMA channels, in a single speaker (MNGU0) and for speakers never seen in training (7-fold leave-one-speaker-out
on USC-TIMIT + EMA_5EMO), and at what inference cost? Whisper is an encoder-decoder trained for supervised multilingual ASR
and translation (680k h for medium, about 5M h mixed labeled and pseudo-labeled for large-v3); WavLM Large is English
self-supervised masked prediction, XLS-R is multilingual self-supervised (contrastive). So the comparison mixes objective,
data and language; the within-Whisper language control (below) is the only attempt to separate them.

## What exists (inspected 2026-11-05)

- Encoder registry `compression/encoders.py` (`MODELS`, `load_full`, `subset_model`, `normalize_wav`, `frame_lengths`) is
  wav2vec2/WavLM-specific (`model.encoder.pos_conv_embed`, `_get_feat_extract_output_lengths`, per-utterance waveform z-score).
  Whisper needs its own wrapper; it must not be forced through that interface.
- `extract.py` writes, per model and dataset, `layer_XX.npy` (float16 memmap, `(N_frames, D)`, XX = 0 is the encoder input
  after the convolutions and positional embedding, XX = k is `hidden_states[k]`), `index.json` (`{stem: [start, n_frames]}`) and
  `meta.json`. The 10 Hz zero-phase low-pass (`butter_bandpass_filter(x[None], 10, 50, axis=1)`) is applied at extraction
  time over the whole untrimmed utterance, before alignment and trimming; an `OverflowError` guards fp16 overflow. The probe
  (`probe.py`) and the depth sweep (`depth_sweep.py`) only read the cache, `index.json` and `meta.json`, so any encoder that
  writes this layout works with them unchanged. Raw (unfiltered) caches are not used by the sweep.
- `probe.py`: ridge on standardized features, alpha grid `logspace(-1, 6, 15)`, shift and alpha chosen on the validation
  split; MNGU0 shift grid (0, 1, 2, 3) (`mngu0.SHIFT_CANDIDATES`), best layer by validation RMSE (mm), test with bootstrap CI.
- `depth_sweep.py`: ridge per layer 1..L and per LOSO fold (`EMAMulti(held_out=(h,), norm_by="norm_group")`), shift
  (-2..3) and alpha chosen on the fold's training speakers' validation sentences only, scored on the held-out speaker (rmse_z,
  PCC) and on the training speakers' test sentences (seen). `summary()` gives per-depth tables, nested depth (per fold, lowest
  training-speaker validation rmse_z), cross-fold depth (best mean held-out rmse_z on the other six folds) and a frontier
  against GFLOPs per second of audio from `features/<model>/compute_prefix.json` (4 s reference, 1 MAC = 2 FLOPs).
- `compute.py`: `FlopCounterMode` FLOPs, activation memory and latency (warm-up 5, median of 20, L40S) for prefix subsets.
- Cache sizes: MNGU0 234,284 frames (1188 utterances), ema_multi 469,423 frames (2719 utterances), fp16. XLS-R 1B ema_multi
  cache is 55 GB for 49 x 1280 (= frames x 2 bytes x D x layers, so the arithmetic below is reliable).
- Utterance durations (computed 2026-11-05, 16 kHz audio): MNGU0 mean 3.96 s, max 17.04 s; USC-TIMIT speakers mean 2.5-3.0 s,
  max 16.99 s (usc_F5), 9.49 s (usc_M3), 6.8 s (usc_M1), 5.16 s (usc_F1); EMA_5EMO mean 4.4-5.5 s, max 13.22 s. **No
  utterance exceeds 30 s** (0 of 3907), so no chunking is needed; extraction asserts `n_samples <= 480000`.
- Checkpoints: `openai/whisper-large-v3` is in the shared cache (`/data/hf_cache/hub`, 2.9 GB safetensors, 128 mel bins,
  32 encoder layers, d = 1280, 20 heads). `whisper-medium` and `whisper-medium.en` are not cached and must be downloaded.
  Some nodes cannot read the shared cache from jobs (the existing job scripts use the personal copy
  `/data/user_data/xoy/hf_hub_jobs` with `HF_HUB_OFFLINE=1`); the three Whisper repos are copied there too.
- transformers' `WhisperEncoder` (checked in the venv source): `hidden_states[0]` = convolutions + sinusoidal positional
  embedding, `hidden_states[k]` = residual stream after layer k (pre-LN blocks, so no per-layer normalization for k < L),
  `hidden_states[L]` = after the final `layer_norm`. This is the same convention as the XLS-R / WavLM caches (final LayerNorm
  only on the last layer). It raises unless the mel input has exactly 3000 frames (30 s).

## Decisions

### 1. Models

| name | checkpoint | encoder | mel | role |
|---|---|---|---|---|
| `whisper-large-v3` | openai/whisper-large-v3 | 32 x 1280, 20 heads, 635M params | 128 | primary; already cached |
| `whisper-medium` | openai/whisper-medium (multilingual) | 24 x 1024, 16 heads, about 305M params | 80 | size match to WavLM Large and XLS-R 300M (24 x 1024) |
| `whisper-medium-en` | openai/whisper-medium.en | 24 x 1024 | 80 | language control: same architecture and recipe as `whisper-medium`, English-only training data |

Why: the question is how Whisper compares to the three encoders already swept. Large-v3 is the model people use and is
width-matched to XLS-R 1B (1280); medium matches the 24 x 1024 geometry, so per-layer cost is comparable (1.3 GFLOPs/s per
layer idealized, like WavLM Large) and depth fractions are comparable. The test speakers are English, but XLS-R is
multilingual and the study's own framing is generalization across languages and styles, so the multilingual models are
primary; `medium.en` is the analogue of the existing wav2vec2-large-lv60 control (English-only vs XLS-R 300M) and separates
"English-only training data" from "Whisper objective" at fixed size. No English-only large model exists. Not run: whisper-small
and below (project rule: smaller models are not added), whisper-large-v3-turbo (its encoder is large-v3's; only the decoder
was pruned) and large-v2 (large-v3 supersedes it). Confound to state in the write-up: large-v3 uses 128 mel bins and
medium 80, and large-v3's training data differ from medium's, so large-v3 vs medium is not a clean size comparison.

### 2. Input, padding, chunking

- Audio is the 16 kHz waveform as stored, with no waveform z-score (the WavLM / XLS-R wrappers z-score each utterance because
  those models were trained that way; Whisper was trained on un-normalized audio and its front end has no level
  normalization except the log-mel dynamic-range clamp relative to the window maximum). Front end: the checkpoint's own
  `WhisperFeatureExtractor` (n_fft 400, hop 160, 80 or 128 mel as the config says, `padding="max_length"`, zero padding to
  480,000 samples, 3000 mel frames).
- Faithful use: pad to 30 s, run the whole encoder, slice the output. Positional embeddings are the fixed sinusoids for
  1500 positions, which the model only ever saw with a full 30 s window (speech followed by zeros). No chunking (no
  utterance reaches 30 s), no shortened input in the primary protocol.
- Frame geometry: conv1 stride 1, conv2 stride 2 (kernel 3, padding 1), so output frame j sits on mel frame 2j, whose STFT
  window (centered) is centered at sample 320 j (20 j ms): 50 Hz, same rate as EMA and as the other encoders. Real frames
  of an utterance of `n` samples: `n_w = ceil(n / 320)` (frames whose center lies inside the audio). The cache stores only
  these frames; the padding frames (which carry the zeros-after-speech state) are discarded before the low-pass so the filter
  never sees them. `index.json` stores `[start, n_w]`; `probe.aligned` truncates to the EMA length, so `n_w` differing from
  the WavLM frame count (WavLM has about one frame fewer) is harmless; extraction asserts `n_w` is at least the maximum
  `base_offset + shift + T_ema` needed (end of the shift grid).
- Numerics: fp32 forward, PyTorch defaults (the same as the existing extraction: TF32 matmul off, cuDNN conv default), SDPA
  attention, batch of 8 padded windows (rows are independent; checked by a test against batch 1).
- Features: all layers 0..L, fp16, same low-pass (zero-phase Butterworth 10 Hz at 50 Hz, whole real-frame sequence, as
  `extract.lowpass`). If a layer overflows fp16 (Whisper's residual stream has a few large outlier channels; `OverflowError`
  guard kept), that layer is stored as float32 (decided now, and recorded in `meta.json`); the probe's `astype(float32)`
  handles both.

### 3. Alignment and shift grid

Expected offset (derived from the geometry, not fit): WavLM / XLS-R frame i covers samples [320 i, 320 i + 400), centered at
320 i + 200 (12.5 ms after 20 i ms). Whisper frame i is centered at 320 i. At the same index Whisper therefore looks 12.5 ms
(0.625 frame) earlier in the audio, so for the same instant its index must be 0.625 higher: the Whisper shift should be
about 0.6 frames larger than WavLM's. WavLM / XLS-R selected 0-1 (mostly 1) on ema_multi and 0-3 was the MNGU0 grid, so the
expected Whisper shift is 1-2, inside both grids (LOSO -2..3: 1 frame of margin above the expected value; MNGU0 0..3: one
frame above, two below). Whisper's bidirectional 30 s context can smear or advance timing, so the geometry is a prior and
not a guarantee.

Rule (fixed now, applied once, after the first full pass over Whisper):
1. First pass uses the same grids as the baselines (MNGU0 0..3 via `probe.run`, LOSO -2..3 via `depth_sweep`), so the first
   pass is directly comparable.
2. Edge check, per model: (a) LOSO: the share of (layer, fold) cells over layers 1..L whose chosen shift is -2 or 3, and the
   chosen shifts at the nested-selected and cross-fold-selected layers; (b) MNGU0: the chosen shifts of the best layer by
   validation and of the three best layers. The grid is declared clipped if the LOSO edge share is at least 10 percent, or any
   fold at a selected layer chose an edge, or (MNGU0) any of the three best layers chose 3 (0 counts as an edge only if the
   expected value 1-2 is itself not preferred, i.e. never alone).
3. If clipped, Whisper is re-swept on a wider grid, LOSO -4..5 and MNGU0 -1..5, all layers, same folds and seeds. The wide
   grid is then the primary Whisper result, and the narrow-grid result stays in the report as the same-grid comparison. To
   keep grid width from becoming a confound, the baselines' compared layers (best by validation on MNGU0; nested and cross-fold
   selections and their neighbours k-1..k+1 on LOSO) are re-scored on the same wide grid, and the comparison table reports
   both. The wide grid is not adopted if the edge check passes (a wider grid only adds optimism).
4. Pre-extraction sanity test (no EMA involved): an impulse at a known sample must give its mel peak at frame `round(n / 160)`,
   and `n_w` must equal the frame count the encoder yields for the real part of the window.

### 4. Probe protocol (identical to WavLM / XLS-R)

- Low-pass 10 Hz zero-phase on features (above), ridge alphas `logspace(-1, 6, 15)`, per-feature standardization, fold-internal
  selection of shift and alpha on the training speakers' validation sentences, targets z-scored per normalization group.
- MNGU0: `python -m sparc.compression.probe <model>` on `features/<model>` (layers 0..L, layer 0 included as for the others);
  best layer by validation RMSE (mm), then test RMSE (mm) with bootstrap CI and PCC.
- LOSO: `depth_sweep run --model <model>` on `features_ema_multi/<model>`, layers 1..L, seven folds; `summary()` extended
  with the three Whisper names. Layer 0 is probed on MNGU0 only, as for the baselines.
- Layer-selection rule: unchanged. Primary: nested (per fold, the layer with the lowest validation rmse_z on that fold's
  training speakers). Secondary: cross-fold (best mean held-out rmse_z on the other six folds; known to be slightly
  optimistic, as stated in `CLAUDE.md`). Third: the full per-depth curve. Whisper has no fixed "LOSO option A" depth.
- Metrics: rmse_z and PCC (held-out speaker, in that speaker's z units; mm secondary on MNGU0). Companion: centered rmse_z
  (per-utterance mean error removed, `rigor.py` definition). `depth_sweep` stores only rmse_z and PCC, so centered rmse_z is
  computed by a short script that refits the probe with the stored shift and alpha (deterministic) at the selected layers of
  every encoder (the nested and cross-fold layers, per fold), for Whisper and the three baselines. Never r^2.

### 5. Compute accounting

- Unit: GFLOPs per second of audio for the encoder prefix (conv front end + positional embedding + layers 1..k, final
  LayerNorm only for k = L), `FlopCounterMode`, 1 MAC = 2 FLOPs, 4 s reference like all other encoders, so the frontier axis is
  unchanged. Mel computation is excluded (a small STFT; the baselines' waveform CNN is included, so the Whisper convolutions
  are, the STFT is not; stated).
- (A) Actual, stock use: the 30 s padded forward. FLOPs are constant per utterance (F_pad), whatever its length; per second
  of audio it is F_pad / duration. Reported at the 4 s reference (F_pad / 4) and, as the best case, at a full 30 s window
  (F_pad / 30); the evaluation corpora's mean durations (MNGU0 3.96 s, ema_multi 3.5 s) are reported too. A short utterance
  costs the same as a 30 s one, so cost per audio second grows as utterances get shorter.
- (B) Idealized: conv and positional embedding on the real frames only and attention over real frames only (a custom forward
  with positional embeddings sliced to `n_w`), at the 4 s reference. This is a cost lower bound that stock Whisper cannot
  deliver (the checkpoint requires 3000 frames), and this forward is never used to produce probe features (see optional X1).
- Hand estimates (to be replaced by the measured values, which must agree within 3 percent or the discrepancy is explained;
  1 MAC = 2 FLOPs; per-layer parameters 12 d^2):
  | model | params in layers | (A) per 30 s window | (A) per audio s at 4 s ref | (B) per audio s at 4 s ref (full depth) |
  |---|---|---|---|---|
  | large-v3 | 629M | 2.27 TFLOP | about 570 GFLOPs/s | about 65 GFLOPs/s |
  | medium | 302M | 1.14 TFLOP | about 285 GFLOPs/s | about 32 GFLOPs/s |
  For reference, WavLM Large and XLS-R 300M full depth are 36.8 GFLOPs/s (4 s ref), XLS-R 1B 102.6. So with stock padding,
  the shortest useful Whisper prefix costs far more than the whole baselines, while under idealization medium is comparable
  to a 24-layer baseline. The frontier is reported on both axes and labelled; no claim that Whisper is "cheap" is made from
  axis (B).
- Per-prefix profile for k = 1..L written to `features/<model>/compute_prefix.json` in the existing format, with
  `gflops_per_audio_s` = (A) at the 4 s reference (so unchanged `depth_sweep.gflops()` yields the actual cost) plus extra keys
  `gflops_per_audio_s_ideal` (B), `flops_per_window_padded`, `window_seconds: 30`. Prefix outputs are tested equal to
  `hidden_states[k]` of the full model.
- Latency: same GPU type as the other profiles (L40S; the job sets `--constraint` to L40S), batch 1, warm-up 5, median of 20,
  `latency()` protocol, padded forward (A) for all prefixes and the idealized forward (B) for a subset (k = 6, 12, 18, 24,
  L). Peak activation memory recorded as in `compute.py`. If no L40S is available, the report says which GPU was used and does
  not compare latencies with the L40S baselines.

### 6. What is compared

Primary (decision-bearing; written in the results before reading anything else):
- P1. LOSO held-out rmse_z and PCC at the nested-selected layers: whisper-large-v3 and whisper-medium against WavLM Large,
  XLS-R 300M, XLS-R 1B (baseline values: 0.7868 / 0.6699, 0.7767 / 0.6834, 0.7929 / 0.6655, with their mean GFLOPs/s).
- P2. The same at cross-fold-selected layers (baselines 0.7659 / 0.6755, 0.7672 / 0.6762, 0.7569 / 0.6916).
- P3. MNGU0: best layer by validation, test RMSE (mm) and PCC with bootstrap CI against 0.888 / 0.904 (WavLM), 0.893 / 0.902
  (XLS-R 300M), 0.893 / 0.901 (XLS-R 1B).
- P4. Where the best layer sits (layer and fraction k / L) per model.
Paired statements use the seven folds: mean paired difference in held-out rmse_z and PCC, number of folds in which Whisper is
better, and the bootstrap interval over folds. Fixed wording rule: "Whisper is better / worse" only if at least 6 of 7 folds
agree in sign and the mean gap exceeds the selection jitter (the mean absolute difference in held-out rmse_z, respectively PCC,
between the nested layer and its two neighbouring layers k-1, k+1, in the same encoder); otherwise "no supported difference". No
p-values (7 folds). The same rule is applied to the baselines among themselves so the comparison is symmetric.

Descriptive (reported, no verdict attached):
- D1. Per-layer curves of held-out rmse_z, PCC and seen-speaker rmse_z against layer index and against k / L, all encoders
  overlaid; selected shifts per layer.
- D2. Frontier of held-out rmse_z against GFLOPs/s (actual and idealized axes, labelled), the existing Pareto table extended.
- D3. Centered rmse_z at the selected layers; seen-vs-held-out gap.
- D4. Language control: whisper-medium vs whisper-medium-en curves and selected-layer scores (same rule as above, one pair).
- D5. Latency and activation memory table, with the padding fixed cost spelled out.
- D6. Optional X1 (only after everything above is written up): large-v3 at three layers (nested, cross-fold, last) with
  input truncated to the real length (no padding, positional embeddings sliced), LOSO held-out rmse_z / PCC, to say whether the
  padded window matters for the probe and whether axis (B) is attainable. Descriptive; its cost is under 0.5 GPU-h.

## Storage and GPU estimates

Frames: MNGU0 about 234k, ema_multi about 469k plus about 1 extra frame per utterance for Whisper's `ceil(n / 320)` (3907
utterances), so about 708k frames in total. fp16, all layers 0..L.
| model | layers x D | per layer (both datasets) | all layers |
|---|---|---|---|
| whisper-large-v3 | 33 x 1280 | 1.81 GB | about 60 GB |
| whisper-medium | 25 x 1024 | 1.45 GB | about 36 GB |
| whisper-medium-en | 25 x 1024 | 1.45 GB | about 36 GB |
Total about 132 GB under `/data/user_data/xoy` (5.2 TB free). Feature caches are regenerable, so they stay out of git; model
copies for jobs about 6 GB. If float32 fallback layers occur, large-v3 grows by at most 2x for those layers.

GPU time (L40S-class, one GPU per job):
- Extraction: large-v3 about 2.3 TFLOP x 3907 windows = 9 PFLOP (about 5 min of GPU at a realistic 30-40 TFLOPs fp32), but
  wall time is set by moving 33 layers off the GPU and low-passing 129k (layer, utterance) arrays and writing 60 GB: plan
  1.5 GPU-h for large-v3 and 1 GPU-h each for the medium models, 8 CPUs, the low-pass in a thread pool. About 3.5 GPU-h.
- MNGU0 probes (33 + 25 + 25 layers): about 0.3 GPU-h in total.
- LOSO depth sweep (32 + 24 + 24 layers x 7 folds x 6 shifts x 15 alphas, eigh on 1280^2 / 1024^2): about 1-2 GPU-min per layer from
  the earlier sweeps (not measured precisely here), so about 2-3 GPU-h in total, split by layer ranges into parallel parts as
  before (`--tag`). A wide-grid re-sweep, if triggered, doubles this and adds the baseline re-scoring (about 1 GPU-h).
- Compute profile and latency (FLOPs on CPU or GPU, latency on L40S): about 0.5 GPU-h.
- Total about 6-8 GPU-h without the wide re-sweep, about 10-12 with it, plus X1. Submission: `slurm-alloc submit --agent
  sparc-xlsr-ema --restartable --class eval --gpus 1 ...` (extraction and probes resume per layer; extraction restarts from the
  last complete model), logs under `/data/user_data/xoy/slurm_logs/`.

## Implementation (next stage, none of it written yet)

- `compression/whisper_encoders.py`: registry for the three checkpoints; `load_whisper(name)`, frame count `n_w`, front end
  (`features(wav)` -> (B, n_mels, 3000)), `hidden_states(batch)` for all layers, `PrefixEncoder(k)` (conv + positional embedding +
  layers 1..k, final norm only at k = L), idealized forward (B). The wav2vec2 `encoders.py` is unchanged.
- `compression/whisper_extract.py`: cache writer, same layout as `extract.py` (with the real-frame slicing before the
  low-pass, fp16 with float32 fallback), `meta.json` with `window_seconds`, `padded: true`, `n_mels`, `freqcut`.
- `compression/whisper_compute.py`: prefix profile (A) and (B), latency, `compute_prefix.json`.
- `depth_sweep.py`: add the three names to `MODELS`; a `--shifts` option for the wide grid, results to
  `<model>_wide*.json` so the narrow results are never overwritten; `summary()` frontier with a second axis for Whisper.
- `compression/whisper_report.py`: P1-P4 and D1-D5 tables and curves, centered rmse_z refit, paired fold statistics.
- Tests (`tests/test_whisper_probe.py`): n_w formula against the encoder output for the real part; `hidden_states[0]` and
  `[L]` identity and `PrefixEncoder(k)` equal to `hidden_states[k]`; batch of 8 equals batch 1 (fp32, atol 1e-4); real-frame
  slicing happens before the low-pass; impulse geometry; FLOP count within 3 percent of the hand estimates above; fp16 fallback
  path; refusal of utterances above 30 s.

## Outputs

- Caches: `/data/user_data/xoy/xlsr_ema/features/<model>/` (MNGU0, plus `probe_results.json`, `probe_heads.npz`,
  `compute_prefix.json`) and `/data/user_data/xoy/xlsr_ema/features_ema_multi/<model>/`, with `<model>` in
  `whisper-large-v3`, `whisper-medium`, `whisper-medium-en`.
- LOSO sweep: `/data/user_data/xoy/xlsr_ema/analysis_depth/<model>[_part<tag>].json` (and `_wide` variants); the existing
  `summary.md` is regenerated, with the three Whisper sections appended.
- Whisper report, tables, figures and the centered rmse_z refit: `/data/user_data/xoy/xlsr_ema/analysis_whisper/`
  (`report.md`, `p1_p4.json`, `fig_curves.png`, `fig_frontier.png`, `latency.json`).
- Findings go into `CLAUDE.md` (a new dated subsection) and code changes into `CHANGELOG.md`; the final comparison text
  states the model confounds listed above.

## Amendments

### Amendment 1 (2026-11-05, before any Whisper number): frame-sufficiency check at extraction

The plan said extraction asserts that `n_w` is at least `base_offset + shift + T_ema` at the end of the shift grid. Measured on
the existing data this cannot hold as a hard assertion: at shift 0 the utterances of ema_multi are short for 14.4 percent of
cases with the WavLM cache (largest shortfall 1 frame) and, at shift 3 or 5, for 2716 to 2717 of 2719 with both encoders, because
the EMA run of an utterance ends at the end of its audio and `probe.aligned` truncates the pair to the shorter length. MNGU0 is
never short (about 28 frames of margin). Whisper keeps one or two more frames than WavLM (`ceil(n / 320)` against
`floor((n - 400) / 320) + 1`), and at shift 0 no utterance is short for it (largest margin 1 frame). So extraction no longer
asserts; it records, in `meta.json`, the number of utterances short at shift 0 and the largest shortfall in the last run
(`short_at_shift0_last_run`, `max_shortfall_shift0_last_run`). Nothing else in the plan changes. The padded extraction runs on
the `rigor_job.sh` runner under `slurm-alloc` (the plan's `xlsr_ema_job_slurm.sh` carries placement headers the allocator refuses).
