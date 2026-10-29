# Phase 8 conditioning study: how the speaker code and utterance-level features enter the vocoder (draft pre-registration)

Status: implementation, unit tests and 300-step smoke runs done; no 100k run submitted. The screen in section 5 is the
pre-registration proposal (revised after review, 2026-10-29) and needs the user's go-ahead before training. All choices
among arms are made on the validation splits (`docs/spkemb_validation_protocol.md`,
`docs/spkemb_validation_splits.json`); test splits report the chosen arm once.

## 1. What the conditioning is today (24 kHz HiFi-GAN, `p5_fv`)

- **Frame input** (50 Hz, 15 channels, `models/frontend.py`): 12 EMA channels z-scored; pitch as `ln(ratio) / 0.19` where
  ratio = f0 / (median voiced f0 of the source utterance) (`pitch_mode: log_relative`); loudness as
  `(ln(loudness + 1e-4) - mean) / std` with global constants (absolute level kept); periodicity z-scored. They enter through
  one `input_conv` (k = 7, 512 channels).
- **Speaker code** (64-d): `z = SpeakerFFN(mean of 1-8 reference vectors) + PitchCode(ln of the references' geometric-mean
  median f0)`. `PitchCode` is a zero-initialized `Linear(1, 64)`, so the speaker's pitch *level* is the only
  utterance-level feature that already travels through the conditioning path (the Phase 4 speaker-relative-F0 gain,
  +0.035 to +0.04 ECAPA).
- **Modulation** (`models/hifigan.py`, `models/film.py`): the code enters only through FiLM, `x * (1 + gamma) + beta`, one
  zero-initialized `Linear(64, 2C)` per residual unit: 4 upsampling stages x 3 residual blocks x 3 units = 36 FiLM layers
  (36 x 64 x 2C parameters; C = 256, 128, 64, 32 across stages). The FiLM is applied to the output branch of the unit,
  `x + FiLM(conv2(conv1(x)))`. There is no modulation at the input convolution or on the transposed convolutions, no
  normalization layer anywhere (weight norm only), no gate beyond the FiLM scale. The code does not enter through the
  frame input.

## 2. Design notes (from the literature; details I recall from the papers are marked "not re-checked")

- **FiLM vs AdaLN-zero.** FiLM (Perez et al. 2018) predicts scale and shift per channel. Adaptive layer norm normalizes
  first and then modulates, so the modulation acts on a standardized signal and cannot be undone by upstream scale drift.
  AdaLN-zero (Peebles and Xie 2023, DiT; checked against the arXiv text) additionally regresses a gate `alpha` per residual
  branch and zero-initializes the projection, so every block starts as the identity; DiT found it the best of four
  conditioning designs (in-context, cross-attention, adaLN, adaLN-zero). For a GAN generator the identity start matters
  less than for a diffusion transformer (the current FiLM is already identity at init), but the gate can switch a branch
  off for some speakers and the normalization removes the per-channel gain that FiLM must otherwise learn to cancel.
  Risk: normalization layers in audio generators can remove loudness/level cues and cause artifacts (BigVGAN and HiFi-GAN
  use none for this reason, not re-checked); we normalize over channels per frame (not over time) so a frame's output does
  not depend on the utterance length (training crops are 64 frames, inference runs on whole utterances).
- **Where it is applied in other systems.** StyleTTS and StyleTTS 2 condition the decoder (and, for StyleTTS 2, the
  HiFi-GAN or iSTFTNet generator) with AdaIN driven by a style vector in every residual block (checked in the abstracts and
  summaries; the generator-stage detail not re-checked). Vocos uses an adaptive layer norm (a lookup table of scale and
  shift, per conditioning id) in its ConvNeXt blocks; the repository's `FiLMLayerNorm` generalizes this to a continuous
  code. VITS/YourTTS add the speaker embedding by *addition* to the flow and decoder inputs (YourTTS also to the
  HiFi-GAN generator input), i.e. conditioning once, not at every stage (not re-checked). AdaSpeech uses conditional layer
  norm in the decoder, with scale and shift from the speaker embedding (not re-checked). Our generator is already in the
  "every residual unit" family; what it lacks relative to these is conditioning *between* the units (input and
  upsampling outputs) and normalization.
- **Global or slowly varying features through the conditioning path.** Frame inputs that vary slowly (utterance
  median/range of pitch, loudness level, spectral tilt) are redundant at 50 Hz and must be inferred by the convolutional
  stack from a short window, which is the part that fails out of domain (the input convolution sees a 7-frame window).
  Conditioning gives every layer the global value directly and decouples *level* from *shape*. This is what the Phase 4
  speaker-relative F0 did for pitch level (the largest cloning gain so far), so the natural extensions are: (a) pitch
  range of the target speaker (matching F0 range gave about +0.06 in the Phase 4 inference variants, applied there as a
  hand-made contour transform; the generator never sees the range), (b) loudness level of the source utterance (frame
  loudness becomes relative, level goes through the code), (c) spectral tilt (not in the 15 cached features, so it
  would need a new extraction; left out). Expected effects: (a) can improve similarity on out-of-domain speakers
  whose range differs from training speakers; (b) mostly affects naturalness and training stability (a smaller dynamic
  range in the input), and Phase 4 found loudness matching at inference gives nothing; (c) not tested.
- **Mapping to the evidence.** The generator is strongly code-sensitive (another speaker's code lowers similarity by
  0.4, `docs/spkemb_ood_gap.md`) and a fitted code closes part of the out-of-domain gap, so the code path has capacity; the
  limit is what the code can express and how the generator uses it. The screen therefore varies how the code modulates a
  unit (AdaLN-zero, and its gate alone), what the code contains (pitch range) and which frame-level features reach the
  generator through the modulation instead of, or besides, the input (pitch and loudness contours). Naturalness and
  own-speaker losses with the combination recipe (Phases 4-5) are the guards that a mechanism change could worsen (more
  modulation capacity can overfit to the LibriTTS-R speakers) or fix (normalization can tame the GLOBE channel variation).
  These are hypotheses, not findings.

## 3. Variants implemented (all behind switches; defaults reproduce `p5_fv` bit for bit)

| switch | values | where |
|---|---|---|
| `vocoder.generator.cond_type` | `film` (current), `adaln_zero`, `film_gate` | `models/film.py` `AdaLNZero`, `GatedFiLM`; `models/hifigan.py` |
| `vocoder.generator.cond_sites` | `residual` (current), `all` | `all` adds a zero-initialized modulation after `input_conv` and each transposed convolution (implemented, not in the screen) |
| `vocoder.generator.cond_contour` | `none` (current), `added`, `only` | frame-level pitch and loudness drive a time-varying FiLM in every residual unit |
| `vocoder.generator.contour_smooth` | 5 | moving-average length (frames) of the contour channels |
| `speaker.cond_stats.f0_range` | false (current), true | `StatsCode`: ln(std of ln f0 of the references / 0.19) added to the code |
| `speaker.cond_stats.loudness` | `off` (current), `added`, `moved` | implemented, not in the screen (see below) |

What differs between the residual units (x = unit input, `c` = speaker code, `f` = conv2(conv1(.)) path):
- `film` (current): `x + FiLM_c(f(x))`; no normalization, no gate. The FiLM projection is zero-initialized, so the
  modulation is the identity at initialization, but the residual branch itself is on at initialization.
- `adaln_zero`: `x + gate_c * f(LN(x) * (1 + scale_c) + shift_c)`, `scale, shift, gate = Linear(c)` zero-initialized, LN over
  channels per frame without affine parameters (a frame's output does not depend on the utterance length). It differs
  from `film` in **three** things at once: (1) channel LayerNorm in front of each unit's convolutions, (2) the modulation
  acts on the convolution input instead of its output, (3) the gate (which, being zero at initialization, switches every
  residual branch off at the start, so the generator output does not depend on the code until the gates move; tested).
  An earlier version of this document counted "zero initialization" as a fourth difference; it is not one, since the FiLM
  projections are already zero-initialized. What zero-initialization adds in AdaLN-zero is only the zero gate (3).
- `film_gate` (new): `x + gate_c * FiLM_c(f(x))`, `gate` from the same zero-initialized projection. Same position, no
  normalization, plus the gate only. Together with `film` and `adaln_zero` it separates (3) from (1) + (2).
- `cond_contour` (new, `film` only): after the speaker FiLM, `h * (1 + gamma_t) + beta_t` with `gamma_t, beta_t` from a
  zero-initialized 1x1 convolution (2 -> 2C channels) of the contour, in all 36 residual units. The contour is the frontend's
  pitch channel (`ln f0-ratio / 0.19`, utterance-relative, unvoiced frames keep their filled-in contour) and loudness channel
  (`(ln loudness - mean) / std`, absolute level), smoothed by a 5-frame (100 ms) moving average and linearly resampled to each
  stage's frame rate. `added`: the two channels also stay in the input convolution's input (so the arm starts exactly as
  `p5_fv`; checked bit for bit). `only`: they are set to 0 in the input convolution's input, so the generator sees pitch and
  loudness only through the modulation (periodicity and the 12 EMA channels stay in the input). The speaker's pitch *level*
  still comes from the code (`PitchCode`).
- `StatsCode` is a zero-initialized `Linear(k, 64)` added to the code, like `PitchCode`. With `f0_range` only, the frame
  input is untouched. The statistics are computed on whole utterances (`CropDataset` / `FullUtteranceDataset`,
  `cond_stats: true`; `eval/clone.py`'s `ItemRenderer` recomputes them from the code's references). The statistic modules
  cannot be combined with swap-in-training or code mixing (error).

Tests (`tests/test_conditioning.py`, all pass; with `tests/test_training.py`: 57 passed, 1 skipped): default state-dict keys
and outputs equal the pre-change module (checksums taken from the old code, randomized modulation); AdaLN-zero and
`film_gate` speaker-independent at initialization; the gate gradient is nonzero at step 0; `GatedFiLM` with the gate fixed to 1
equals `FiLM`; contour `added` equals the default generator bit for bit at initialization and `only` does not depend on the
pitch and loudness input channels until the contour weights move; per-frame normalization; contour smoothing; statistics
helpers; relative loudness.

Not in the screen, implemented: `cond_sites: all` (FiLM or AdaLN-zero also between the stages), `loudness: added | moved`.
Not implemented: spectral tilt (not in the 15 cached features).

## 4. Smoke runs (300 steps, 50 mel warm-up steps, batch 16 x 64 frames, one L40S each)

All arms below ran on an L40S (job constraint `L40S`), so step times are comparable. Each trained once to the end (one
attempt, no restart). Step time is the median over consecutive logged discriminator steps from step 100 (adversarial phase;
it includes the validation pass at step 150 and data loading). The reference `p5_fv` smoke run (first batch of smokes) was
also on an L40S. Loss decreases and every logged scalar is finite in every run.

| arm | mel step 50 -> 300 | g_total 50 -> 300 | max grad norm (g) | s/step | trainable params (M) |
|---|---|---|---|---|---|
| `p5_fv` (reference) | 1.930 -> 1.136 | 86.8 -> 55.9 | 302.1 | 0.313 | 57.7 |
| `p8c_adaln` | 2.088 -> 1.152 | 93.9 -> 56.6 | 172.9 | 0.348 | 58.0 |
| `p8c_film_gate` | 2.099 -> 1.191 | 94.5 -> 59.2 | 181.1 | 0.321 | 58.0 |
| `p8c_film_f0range` | 1.930 -> 1.135 | 86.8 -> 56.0 | 218.7 | 0.310 | 57.7 |
| `p8c_film_contour_added` | 1.985 -> 1.123 | 89.3 -> 55.5 | 269.2 | 0.358 | 57.8 |
| `p8c_film_contour_only` | 1.988 -> 1.172 | 89.5 -> 58.0 | 233.2 | 0.356 | 57.8 |

(Params as Lightning prints them: generator, speaker network and discriminators.) The `film_gate` and AdaLN-zero starts are
higher in the first steps because every residual branch starts switched off, and they catch up by step 300 (mel 1.15-1.19
against 1.14); 300 steps say nothing about final quality, only that these arms train. `contour_only` is stable
(mel 1.172 at step 300, grad norm below the reference's), so it stays in the screen.

Earlier smoke attempt (first batch, recorded to correct the previous draft): the `p5_fv` reference and `p8c_film_stats` jobs
were requeued once by the scheduler, but only *after* the training had finished (they then logged "already done" and exited),
so all six first-batch runs trained in a single attempt. The earlier draft wrongly said "5 of 6 finish in one attempt". Its
A100 step times (0.37-0.46 s) are not used: they came from a different GPU type, and the draft's "pessimistic bound" of
0.46 s/step was not supported by any L40S measurement.

Synthesis check through `eval/clone.py synth` (the evaluation path) with the `p8c_film_f0range` smoke checkpoint
(`--pools libritts_dev --variants base`, 8 items from two shards): all 8 waveforms are written at 24 kHz, finite, non-silent
(peak 0.20-0.24, rms 0.031-0.044). This exercises `StatsCode` / `stats_term` in `ItemRenderer` end to end, including the
reference pitch-range computation; the audio of a 300-step model is not meaningful, and the loudness statistic is off in this
arm, so the loudness half of `stats_term` is covered by unit tests only.

## 5. Proposed screen (pre-registration; not submitted)

Reference: the `p5_fv` 100k run (`/data/user_data/xoy/spkemb/phase3/runs/p5_fv/hifigan`), seed 0 for every arm, with its
validation-pool outputs under `/data/user_data/xoy/spkemb/validation/ev*/p5_fv`. Every arm differs from `p5_fv` in exactly one
switch (checked by composing the configs and diffing them against `p5_fv`).

Arms (5 in total; each is a one-switch change, and the pairs are chosen so that the contrasts are interpretable):
1. `p8c_adaln`: full AdaLN-zero (normalization + modulation on the convolution input + zero gate).
2. `p8c_film_gate`: FiLM with the gate only. Contrasts: `film_gate` vs `p5_fv` is the gate (branches start off); `adaln` vs
   `film_gate` is normalization + moving the modulation to the convolution input (these two are not separated from each other:
   a position-only or LN-only arm would be a sixth arm; if `adaln` beats `film_gate`, a follow-up can split them).
3. `p8c_film_f0range`: the references' pitch range in the speaker code. The loudness statistic is **dropped** from the screen: in
   Phase 4 loudness matching at inference gave nothing, the earlier "moved" variant also changed the frame input (two changes at
   once), and the loudness contour is tested in arms 4 and 5.
4. `p8c_film_contour_added`: frame pitch and loudness contours also drive a time-varying modulation (input unchanged).
5. `p8c_film_contour_only`: the same, with the two channels removed from the input. Smoke run stable (section 4). Contrast with
   arm 4 isolates whether the input path is needed for these two features.
Not in the screen: `cond_sites: all` (site count; follow-up if arm 1 or 4 advances), loudness statistics, spectral tilt.

Decision rule (the Phase 4 / Phase 5 rule of `docs/speaker_embedding_plan.md`, applied to the **validation** pools; stated in
full): primary metric is the swap similarity to the target's reference-mean embedding (ECAPA, WavLM-SV and the held-out CAM++),
speaker-level paired bootstrap, inference variant `k5_f0_loud`, margin m = 0.0266. Pools: LibriTTS-R dev-clean, the validation
half of VCTK, GLOBE dev real speakers (scored with `eval/clone.py summarize-split` / `compare --split`). An arm advances if
its primary (ECAPA) gain over `p5_fv` is >= m with the bootstrap CI excluding 0 on LibriTTS-R dev-clean, the gain has the same
sign on VCTK and GLOBE and in CAM++, and all guards pass. Guards (relative to `p5_fv`): own-speaker T2/T3 similarity drop at most
0.01 per SV model; UTMOS drop at most 0.10 in every pool; similarity to the source rise at most 0.02; WER at most +0.5 points;
the reference-length curve and noisy-reference checks are reported. The best advancing arm by mean ECAPA gain over the three
pools is run once on the test pools (LibriTTS-R test.clean, the other VCTK half, GLOBE test real speakers) against `p5_fv`.
If no arm passes every guard, the results are reported and the screen stops (no combination, no rescue arm). Guards are
measured against `p5_fv`, so an arm is not required to repair guard failures that `p5_fv` itself has. Stability (gradient norms, discriminator
balance, NaN checks; AdaLN and gate values near 0 are reported as unused conditioning) is reported, not selected on. Single
seed per arm; only differences above m count. No test number is used for selection.

Cost, from measured step times: `p5_fv` ran 100k steps at a median 0.315 s/step (8.3 h wall) on its own run; the smoke medians on
an L40S are 0.31 (f0range), 0.32 (gate), 0.35 (adaln), 0.36 (both contour arms), which gives 8.6, 8.9, 9.7, 9.9 and 9.9 h for
the five arms: about 47 GPU-hours of training on L40S-class GPUs. The validation-pool synthesis and scoring cost per arm was not
measured for this setup (it is recorded from the first arm's evaluation jobs); the single test evaluation is one more
evaluation of the same kind. The runs are restartable (checkpoint every 20 minutes), so wall time is bounded by the
`slurm-alloc` per-agent GPU cap, not by the 47 hours. Statistics arm 3 uses the evaluation support in `eval/clone.py`
(`ItemRenderer`), checked end to end on a smoke checkpoint (section 4). `eval/ood_gap.py`, `ood_svfit*.py` and `ood_channel.py`
build their own batches and do not pass `utt_loud` / `ref_f0_std`, so Phase 8 code fitting is not available for statistics
checkpoints.
