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
