# Out-of-domain gap diagnosis: why own-speaker resynthesis reaches only ~0.7 of ground truth on VCTK and GLOBE

Code: `src/sparc/vocoders/eval/ood_gap.py`. Outputs: `/data/user_data/xoy/spkemb/ood_gap/{summary.md,summary.json,choice.json,tune_grid.md}`.
No vocoder training; the checkpoint is the filtered LibriTTS-R + GLOBE + VCTK 100k HiFi-GAN (experiment `p5_fv`, up to 8
references, speaker-relative pitch).

## Question

Resynthesizing a speaker's own held-out utterance (no swap) reaches ~0.9 of the ground-truth same-speaker similarity on
LibriTTS-R but only ~0.6 to 0.7 on VCTK and GLOBE. Is the speaker code the bottleneck (the encoder maps the references to a
poor point), or can the generator and its conditioning not render these voices?

## Interpretation rule (fixed before the runs)

With A, C, D the speaker-averaged similarities of conditions (a), (c), (d) and G the ground-truth level:

- If (c) closes most of the gap from (a) to (d) (closure above 0.5, interval excluding 0), the code or encoder is the bottleneck.
- If (d) stays far below G (D / G about 0.8 or less) even though (D - A) is small, the generator or conditioning cannot render
  the voice.
- If (b) is far above (a), the references carry session or content information the encoder over-uses.

## Conditions

Same items in every condition: the own-speaker items of the cloning plans, the same ones in each condition (LibriTTS-R 59 items
from 20 speakers, VCTK 53 from 20, GLOBE 40 from 40 pseudo-speakers; GLOBE pseudo-speakers give 2 to 3 reference utterances, all used for the code, and
the target is a further utterance).

- (a) code from the speaker's other utterances (5 for LibriTTS-R and VCTK, 2 to 3 for GLOBE): the standard setting. It reproduces the stored
  `own_t3` similarities of the cloning evaluation on the same items to within 1e-3 (mean difference 3e-5 to 1e-4) and UTMOS exactly.
- (b) oracle: code from the target utterance itself.
- (c) test-time fitting starting at (a), on the reference utterances only, never the target: Adam on the 64-d code, or on an offset
  of the speaker network's pooled input vectors passed through the frozen FFN. Loss: the training reconstruction loss (mel + multi-resolution
  STFT, weights of the run's config, no adversarial terms) through the frozen generator on random 64-frame crops.
- (d) the same fitting on the target utterance itself (upper bound), with the tuned steps and learning rate, and an unregularized
  800-step variant (fixed learning rate 0.03 for the code, 0.01 for the FFN input; not tuned).
- Ground truth: the recorded audio of the same utterance, scored against the same reference set.

Similarity is the cosine of the output's embedding to the mean embedding of the speaker's reference utterances that neither gave
the code nor were fitted (VCTK, LibriTTS-R); GLOBE has none left, so it uses all of the speaker's references (the fitted
references that gave the code or were fitted are inside that set, which flatters (a) and (c) relative to (b) and (d); the conclusion does not depend on it since (d) does not exceed (a)). Embedding models: ECAPA, WavLM-SV, CAM++ (CAM++ was never used
in training or selection). Intervals are 95% percentile bootstraps over speakers.

Fitting steps and learning rate were chosen on a separate LibriTTS-R dev set (8 speakers, 15 items, never test items), maximizing
held-out similarity (mean of ECAPA and WavLM-SV): the 64-d code at learning rate 0.01 for 200 steps, the FFN input at 0.003 (in units
of the input std) for 100 steps. The dev gain over no fitting was only +0.003 (0.8513 to 0.8545), i.e. fitting hardly helps even
there; larger learning rates and more steps lowered held-out similarity (`tune_grid.md`).

## Results

### libritts (59 items, 20 speakers)

Cells: similarity [95% speaker-bootstrap CI] (fraction of ground truth). Last column: UTMOS.

| condition | ECAPA | WavLM-SV | CAM++ | UTMOS |
|---|---|---|---|---|
| (a) code from 5 other utterances | 0.729 [0.693, 0.763] (1.00) | 0.950 [0.935, 0.962] (1.00) | 0.736 [0.693, 0.773] (0.96) | 3.79 |
| (b) code from the target utterance | 0.672 [0.626, 0.713] (0.92) | 0.932 [0.914, 0.949] (0.98) | 0.686 [0.634, 0.731] (0.89) | 3.71 |
| (c) fit 64-d code on references | 0.722 [0.683, 0.755] (0.99) | 0.947 [0.935, 0.958] (1.00) | 0.733 [0.695, 0.768] (0.95) | 3.75 |
| (c) fit FFN input on references | 0.724 [0.686, 0.757] (1.00) | 0.949 [0.936, 0.959] (1.00) | 0.736 [0.699, 0.770] (0.96) | 3.80 |
| (d) fit 64-d code on target (tuned steps) | 0.687 [0.645, 0.723] (0.94) | 0.943 [0.930, 0.955] (1.00) | 0.700 [0.655, 0.739] (0.91) | 3.69 |
| (d) fit FFN input on target (tuned steps) | 0.711 [0.676, 0.745] (0.98) | 0.951 [0.940, 0.961] (1.00) | 0.721 [0.682, 0.756] (0.94) | 3.73 |
| (d) 64-d code on target, 800 steps | 0.665 [0.620, 0.706] (0.91) | 0.937 [0.921, 0.951] (0.99) | 0.682 [0.629, 0.726] (0.89) | 3.63 |
| (d) FFN input on target, 800 steps | 0.663 [0.616, 0.705] (0.91) | 0.936 [0.918, 0.952] (0.99) | 0.683 [0.638, 0.723] (0.89) | 3.63 |
| ground truth, same speaker | 0.727 [0.678, 0.770] (1.00) | 0.947 [0.929, 0.963] (1.00) | 0.770 [0.720, 0.813] (1.00) | 4.21 |

### vctk (53 items, 20 speakers)

Cells: similarity [95% speaker-bootstrap CI] (fraction of ground truth). Last column: UTMOS.

| condition | ECAPA | WavLM-SV | CAM++ | UTMOS |
|---|---|---|---|---|
| (a) code from 5 other utterances | 0.528 [0.495, 0.559] (0.69) | 0.936 [0.928, 0.944] (0.98) | 0.518 [0.493, 0.541] (0.63) | 3.67 |
| (b) code from the target utterance | 0.517 [0.488, 0.544] (0.67) | 0.930 [0.919, 0.942] (0.97) | 0.516 [0.484, 0.545] (0.63) | 3.69 |
| (c) fit 64-d code on references | 0.553 [0.520, 0.584] (0.72) | 0.937 [0.929, 0.945] (0.98) | 0.531 [0.505, 0.555] (0.65) | 3.61 |
| (c) fit FFN input on references | 0.546 [0.516, 0.573] (0.71) | 0.937 [0.930, 0.943] (0.98) | 0.523 [0.498, 0.545] (0.64) | 3.68 |
| (d) fit 64-d code on target (tuned steps) | 0.528 [0.499, 0.556] (0.69) | 0.934 [0.924, 0.943] (0.98) | 0.516 [0.485, 0.546] (0.63) | 3.63 |
| (d) fit FFN input on target (tuned steps) | 0.549 [0.521, 0.574] (0.71) | 0.941 [0.934, 0.949] (0.98) | 0.530 [0.506, 0.554] (0.65) | 3.69 |
| (d) 64-d code on target, 800 steps | 0.523 [0.487, 0.557] (0.68) | 0.927 [0.914, 0.940] (0.97) | 0.515 [0.484, 0.546] (0.63) | 3.52 |
| (d) FFN input on target, 800 steps | 0.526 [0.493, 0.557] (0.68) | 0.928 [0.917, 0.939] (0.97) | 0.521 [0.493, 0.549] (0.64) | 3.57 |
| ground truth, same speaker | 0.769 [0.750, 0.789] (1.00) | 0.958 [0.949, 0.965] (1.00) | 0.818 [0.799, 0.836] (1.00) | 3.96 |

### globe (40 items, 40 speakers)

Cells: similarity [95% speaker-bootstrap CI] (fraction of ground truth). Last column: UTMOS.

| condition | ECAPA | WavLM-SV | CAM++ | UTMOS |
|---|---|---|---|---|
| (a) code from 5 other utterances | 0.532 [0.497, 0.566] (0.70) | 0.937 [0.917, 0.952] (0.99) | 0.562 [0.528, 0.591] (0.71) | 3.05 |
| (b) code from the target utterance | 0.516 [0.487, 0.546] (0.68) | 0.925 [0.903, 0.941] (0.97) | 0.545 [0.517, 0.572] (0.69) | 3.01 |
| (c) fit 64-d code on references | 0.534 [0.497, 0.569] (0.70) | 0.932 [0.911, 0.948] (0.98) | 0.549 [0.517, 0.579] (0.70) | 2.85 |
| (c) fit FFN input on references | 0.537 [0.501, 0.570] (0.71) | 0.938 [0.919, 0.951] (0.99) | 0.560 [0.528, 0.588] (0.71) | 2.98 |
| (d) fit 64-d code on target (tuned steps) | 0.527 [0.495, 0.558] (0.69) | 0.927 [0.909, 0.941] (0.97) | 0.545 [0.517, 0.572] (0.69) | 2.84 |
| (d) fit FFN input on target (tuned steps) | 0.539 [0.505, 0.570] (0.71) | 0.935 [0.916, 0.949] (0.98) | 0.572 [0.543, 0.599] (0.73) | 2.95 |
| (d) 64-d code on target, 800 steps | 0.504 [0.472, 0.536] (0.66) | 0.926 [0.909, 0.939] (0.97) | 0.508 [0.476, 0.538] (0.64) | 2.71 |
| (d) FFN input on target, 800 steps | 0.503 [0.471, 0.534] (0.66) | 0.927 [0.911, 0.940] (0.97) | 0.516 [0.485, 0.545] (0.65) | 2.74 |
| ground truth, same speaker | 0.760 [0.742, 0.778] (1.00) | 0.951 [0.929, 0.966] (1.00) | 0.788 [0.755, 0.816] (1.00) | 3.01 |


## Reading

- Fitting the code to the target utterance itself ((d)) does not raise similarity above the standard code ((a)) in any pool: ECAPA
  0.53 for VCTK and GLOBE against 0.53 for (a), CAM++ 0.52 to 0.57 against 0.52 to 0.56 (the FFN-input variant is +0.01 to +0.02 at
  best). With 800 steps it is worse (and UTMOS falls by 0.15 to 0.35). So even a code optimized on the audio to be rendered does not bring this generator closer to the real
  speaker: (d) stays at 0.69 to 0.71 of ground truth (ECAPA) and 0.63 to 0.73 (CAM++) on VCTK and GLOBE, as (a) does. The
  interpretation rule's second branch holds: the generator or its conditioning, not the code, limits own-speaker similarity.
  (Closure, (c) - (a) over (d) - (a), is not meaningful because (d) - (a) is about 0; the intervals are in `summary.md`.)
- (c) gives small gains on VCTK: ECAPA +0.025 [0.011, 0.039] (64-d code) and +0.018 [0.008, 0.028] (FFN input), CAM++ +0.013 [0.005, 0.020]
  (64-d code), about 0 on GLOBE, and slightly negative on LibriTTS-R. These are 1 to 3 points of the 0.24 gap to ground truth (about 4% of
  it on VCTK ECAPA, and similar on CAM++).
- (b) is no better than (a): the oracle code from the target utterance is the same (VCTK, GLOBE) or worse (LibriTTS-R, -0.06 ECAPA).
  The references do not hide information the target would carry; a single utterance is a noisier speaker code than five.
- WavLM-SV is saturated (0.93 to 0.95 in every condition including ground truth 0.95 to 0.96) and does not discriminate; ECAPA and
  CAM++ carry the result.
- LibriTTS-R is the in-domain control: (a) equals ground truth (1.00 ECAPA), so there is no gap to explain there, and every fitting
  condition is at or below (a).

## Limits

- The fitting loss is the reconstruction loss (mel + STFT), not a speaker-verification loss. It barely decreased in 100 to 800 steps on crops
  (about 1%), so what is shown is that reconstruction-driven codes do not move toward the voice; an SV-loss-driven code was not tried (the
  Phase 4 SV identity loss during training did not help either).
- The rendering uses the model's own path (content features of the target utterance, speaker-relative pitch); if the gap is in how
  content features carry the original speaker's timbre (the "speaker" leaking through the content and F0 inputs), no code can fix that, which fits the result
  but was not isolated here.
- Subset sizes are small (20 speakers per pool; GLOBE 40 single-item speakers); the intervals reflect that.

## Follow-up: fitting the code to maximize speaker similarity through the frozen generator (2026-10-28)

Code: `src/sparc/vocoders/eval/ood_svfit.py`. Outputs: `/data/user_data/xoy/spkemb/ood_gap/sv_fit/{choice.json,sweep_table.md,summary.md,summary.json}`.

Why: the fit above used a reconstruction loss that is nearly flat in the code, so it could not show that no code renders the voice.
Here the loss is speaker similarity itself, through the frozen `p5_fv` HiFi-GAN generator.

**Method.** Own-speaker resynthesis of a held-out target utterance (content features and speaker-relative pitch as in the standard path).
- (a) standard code from the speaker's reference utterances.
- (e) the 64-d code (an offset on the standard code), or separately an offset on the speaker network's pooled input vectors (FFN input),
  fitted with Adam to maximize the differentiable ECAPA cosine of 4 s crops of the synthesized target content to the mean ECAPA
  embedding of the speaker's code-reference utterances (never the target utterance). Penalty: L2 toward the standard code (weight `lam`),
  plus the training reconstruction loss on reference crops (weight `w_rec`) as a quality anchor.
- (e, self) the same with the target utterance's own ECAPA embedding as the goal (an upper bound for the fitted judge only).
- (f) the same content rendered with a random other speaker's standard code (the code's leverage).
- Scoring: only models not in the fitting loss: CAM++ and the WeSpeaker ResNet34, cosine to the mean embedding of the speaker's reference
  utterances that did not give the code (none for GLOBE: all references, which include the code references). ECAPA is shown as the
  fitted objective, not as evidence. UTMOS and Whisper WER are the quality guard.

**Selection on validation speakers only** (`docs/spkemb_validation_splits.json`): LibriTTS-R dev-clean (8 speakers, 8 items), VCTK
validation half (10, 10), GLOBE real-speaker dev pool `globe_spk` (10, 10): 28 items. Grid per mode and goal: lr {0.01, 0.03}, `lam`
{0, 0.3, 3}, `w_rec` {0, 0.1}, steps {25, 50, 100, 200} (48 settings x 4 snapshots, `sweep_table.md`). Rule: maximize the mean of CAM++ and
WeSpeaker similarity (pools equally weighted) subject to a mean UTMOS drop <= 0.1 and a pooled WER increase <= 1 point against the
standard code. Chosen: FFN input lr 0.03, `lam` 0, `w_rec` 0, 50 steps (validation held-out similarity 0.599 -> 0.681, UTMOS -0.097);
64-d code lr 0.01, `lam` 0, `w_rec` 0, 200 steps (0.673, UTMOS -0.071). Both sit at an edge of the grid (largest step count for the code,
largest lr for the FFN input), so a longer or faster fit was not examined.

**Interpretation rule (fixed before any run).** Closure = (E - A) / (G - A) with A the standard code, E the fitted references-target
condition, G ground truth, on held-out verifiers. Closure above 0.5 (interval excluding 0) at acceptable quality: the code space can express
the voice and the encoder or inference is the bottleneck. E near A (closure below 0.2) even for the self-target fit at acceptable
quality: the generator or its other inputs are the limit. In between: partial.

**Test items (run once).** LibriTTS-R test-clean (36 items, 20 speakers), VCTK test half (34 items, 18 speakers), GLOBE: the real-speaker test
pool has one reference and no own-speaker items, so the GLOBE row uses the ECAPA-clustered pseudo-speakers' test half (30 speakers;
8 own-speaker items, 2 to 7 references, scored against all references; small and noisy). Mean of CAM++ and WeSpeaker similarity, fraction of
ground truth (95% speaker-bootstrap interval), closure of the standard-to-ground-truth gap:

| pool | (a) standard | (e) FFN input, refs | (e) 64-d code, refs | (e) FFN, self goal | (f) other speaker's code | ground truth (similarity) |
|---|---|---|---|---|---|---|
| LibriTTS-R | 0.953 [0.922, 0.988] | 1.001 [0.975, 1.034] | 0.999 [0.967, 1.040] | 0.958 | 0.411 | 0.782 |
| VCTK | 0.663 [0.623, 0.707] | 0.809 [0.781, 0.839] | 0.784 [0.756, 0.812] | 0.761 | 0.213 | 0.826 |
| GLOBE (pseudo, 8 items) | 0.754 [0.692, 0.831] | 0.913 [0.856, 0.987] | 0.894 [0.841, 0.970] | 0.817 | 0.262 | 0.781 |

Closure: VCTK 0.44 [0.38, 0.49] (FFN input) and 0.36 [0.32, 0.40] (code); GLOBE 0.64 [0.52, 0.93] and 0.57 [0.47, 0.83]; LibriTTS-R has no gap to close
(standard is already 0.95). CAM++ and WeSpeaker agree (VCTK closure 0.47 / 0.40 FFN input, 0.39 / 0.33 code; GLOBE 0.62 / 0.67 and 0.55 / 0.59).
The fitted ECAPA similarity rises far more (VCTK 0.555 -> 0.748 against ground truth 0.768; GLOBE 0.595 -> 0.806), which is what a fit on
that judge would give and is not evidence.

Quality guard (UTMOS / WER %), standard -> FFN input / 64-d code: LibriTTS-R 3.82 / 2.3 -> 3.70 / 2.2 and 3.78 / 2.3; VCTK 3.56 / 3.7 -> 3.46 / 3.0
and 3.48 / 3.3; GLOBE 3.36 / 3.9 -> 3.19 / 3.9 and 3.18 / 2.6. WER does not rise. UTMOS drops are within 0.1 in the validation pool mean,
but on the test pools the FFN-input fit exceeds 0.1 on LibriTTS-R (0.12) and GLOBE (0.17), and the 64-d code on GLOBE (0.18); VCTK is at 0.08 to 0.10.

**Self goal and leverage.** Fitting toward the target utterance's own embedding is a poorer code than fitting toward the other utterances' mean
on the held-out verifiers (VCTK closure 0.27 to 0.29, GLOBE 0.25 to 0.28, LibriTTS-R ~0.1 to 0.25): the target utterance's embedding carries its
session and content, so as a goal it is not an upper bound for held-out judges, only for the fitted ECAPA. The leverage check (f) shows the
generator is strongly code-sensitive: another speaker's standard code lowers held-out similarity to the target by 0.37 to 0.44 (to 0.2 to 0.4 of
ground truth) and raises it toward the code's speaker (donor similarity 0.41 to 0.73).

**Reading.** Raising the fitted ECAPA similarity also raises the two independent verifiers, by +0.10 to +0.12 on VCTK and +0.11 to +0.12 on GLOBE
(mean of CAM++ and WeSpeaker, intervals exclude 0), so the code space can move the output well toward the voice: the earlier "no code helps" was a
property of the reconstruction loss. By the pre-registered rule: GLOBE closes more than half of the gap (0.57 to 0.64, interval above 0.46) but at a
UTMOS cost above 0.1 on the test pool; VCTK closes 0.36 to 0.44 (interval upper ends 0.40 and 0.49, below 0.5), i.e. partial. A large part of the
gap, 56 to 64% on VCTK, remains after a fit that uses the verifier directly with unrestricted code freedom. Caveats: the three verifiers are all
VoxCeleb-trained speaker models, so part of the transfer could be shared sensitivity rather than voice (adversarial-style transfer), which the
quality drop is consistent with; the settings sit at grid edges; GLOBE is 8 items; UTMOS and WER are the only guard. Conclusion: the code space
carries substantial headroom (the encoder and its inference are a bottleneck, not only the generator), but a code fitted per speaker with the
verifier in the loop does not reach ground truth on VCTK, so the generator or its other inputs still limit part of the gap there.
