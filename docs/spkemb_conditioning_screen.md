# Phase 8 conditioning study: how the speaker code and utterance-level features enter the vocoder (draft pre-registration)

Status: implementation and smoke runs done; no 100k run submitted. The screen below is a proposal and needs the
user's approval before any training. All choices among arms are made on the validation splits
(`docs/spkemb_validation_protocol.md`, `docs/spkemb_validation_splits.json`); test splits report the chosen arm once.

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
  limit is what the code can express and how the generator uses it. More modulation sites (arm B) and a different
  modulation form (arm A) address how it is used; added statistics (arm C) address what it contains. Naturalness and
  own-speaker losses with the combination recipe (Phases 4-5) are the guards that a mechanism change could worsen (more
  modulation capacity can overfit to the LibriTTS-R speakers) or fix (normalization can tame the GLOBE channel variation).
  These are hypotheses, not findings.

## 3. Variants implemented (all behind switches; defaults reproduce `p5_fv` bit for bit)

| switch | values | where |
|---|---|---|
| `vocoder.generator.cond_type` | `film` (current), `adaln_zero` | `models/film.py` `AdaLNZero`, `models/hifigan.py` |
| `vocoder.generator.cond_sites` | `residual` (current), `all` | `all` adds a zero-initialized modulation after `input_conv` and after each of the 4 transposed convolutions (FiLM, or `x + gate * (LN(x) (1 + scale) + shift)` for AdaLN-zero) |
| `speaker.cond_stats.f0_range` | false (current), true | `StatsCode`: ln(std of ln f0 of the references / 0.19) added to the code |
| `speaker.cond_stats.loudness` | `off` (current), `added`, `moved` | source utterance's mean `ln(loudness + 1e-4)` over voiced frames, normalized with the frontend's loudness constants, added to the code; `moved` also makes the frame loudness input relative to that level (frontend unchanged: the channel is rescaled before it) |

- AdaLN-zero residual unit: `x + gate * conv2(conv1(LN(x) (1 + scale) + shift))`, `scale, shift, gate = Linear(64, 3C)(code)`
  zero-initialized, LN over channels per frame without affine parameters. At initialization every unit is the identity, so
  the output does not depend on the speaker code until the gates move (tested); the gate gradient is nonzero at step 0.
- `StatsCode` is a zero-initialized `Linear(k, 64)` added to the code, like `PitchCode`; a run with these switches starts
  as `p5_fv`. Statistics are computed on whole utterances (training and evaluation alike): `CropDataset` and
  `FullUtteranceDataset` add `ref_f0_std` and `utt_loud` (`cond_stats: true`); `eval/clone.py`'s `ItemRenderer` computes
  them from the code's references and from the rendered content features (after matching), so the cloning evaluation works
  for these checkpoints. The statistic modules are not combined with swap-in-training or code mixing (error).
- Not implemented, deliberately: removing the *pitch contour* from the frame input (the vocoder needs the contour; the
  pitch *level* is already removed, `log_relative`), and spectral tilt (no cached feature).
- Configs (all = `p5_fv` plus the switches; seed 0, 100k): `p8c_adaln`, `p8c_film_all`, `p8c_adaln_all`,
  `p8c_film_stats` (f0_range + loudness moved), `p8c_film_stats_added` (control: same statistics, input unchanged).
- Tests: `tests/test_conditioning.py` (default state-dict keys and outputs equal the pre-change module, with checksums
  taken from the old code; AdaLN-zero speaker-independent at init; sites identity at init; gradients; per-frame
  normalization; statistics helpers; relative loudness).

## 4. Smoke runs (300 steps, 50 mel warm-up steps, batch 16 x 64 frames, one GPU)

Every variant: loss decreases, no NaN or Inf in any logged scalar, 5 of 6 finish in one attempt. Step times are wall-clock
over steps 100-300 (adversarial phase, includes one validation pass and data loading) and the jobs landed on different GPUs,
so only comparisons within a GPU type are meaningful.

| arm | GPU | mel loss step 50 -> 300 | g_total 50 -> 300 | finite | s/step | generator + speaker params (M) |
|---|---|---|---|---|---|---|
| `p5_fv` (reference) | L40S | 1.930 -> 1.136 | 86.8 -> 55.9 | yes | 0.336 | 57.7 |
| `p8c_film_stats` | L40S | 1.954 -> 1.172 | 87.9 -> 57.8 | yes | 0.335 | 57.7 |
| `p8c_film_stats_added` | L40S | 1.930 -> 1.134 | 86.8 -> 55.9 | yes | 0.338 | 57.7 |
| `p8c_adaln` | A100 80GB | 2.088 -> 1.147 | 93.9 -> 56.5 | yes | 0.371 | 58.2 |
| `p8c_film_all` | A100 80GB | 1.922 -> 1.142 | 86.5 -> 56.2 | yes | 0.414 | 57.9 |
| `p8c_adaln_all` | A100 80GB | 2.061 -> 1.099 | 92.8 -> 54.2 | yes | 0.457 | 58.2 |

(The params column is the whole trainable model as Lightning prints it, i.e. including the discriminators and the speaker
network.) Reading: the statistics arms add no measurable step time (0.335 vs 0.336 on the same GPU). On the A100 jobs the
ordering film_all (0.41) > adaln (0.37) is not explained by the architecture (both add similar work) and is most likely
data-loading or node noise; I have no same-GPU reference for them, so I take 0.34 s/step (L40S, reference) as the base and
0.46 s/step (the slowest, adaln_all on A100) as a pessimistic bound. 300 steps say nothing about final quality; they only
show the variants train and that AdaLN-zero's identity start is not slower in the first 300 steps (mel 1.10-1.15 vs 1.14).

## 5. Proposed screen (pre-registration draft; not submitted)

Reference: the `p5_fv` 100k run (`/data/user_data/xoy/spkemb/phase3/runs/p5_fv/hifigan`), seed 0 for all arms; margin and
guards as in Phases 4-5, evaluated with `eval/clone.py` on the **validation** pools (LibriTTS-R dev-clean, VCTK validation
half, GLOBE real-speaker dev pool).

Arms (at most four; chosen so that each tests one hypothesis and one is the combination of two):
1. `p8c_adaln`: AdaLN-zero instead of FiLM at the same sites (form of the modulation).
2. `p8c_film_all`: FiLM also at the input and after each upsampling (number of sites, cheap: 4 extra modulation layers).
3. `p8c_film_stats`: pitch range and loudness level through the code, frame loudness relative (what goes through the path).
4. `p8c_adaln_all`: AdaLN-zero at every site (arm 1 + arm 2; tests whether the two mechanism changes add up).
Not in the screen (follow-ups only if arm 3 advances): `p8c_film_stats_added` (separates moving the loudness level out of
the input from adding it), statistics on the best mechanism arm.

Decision rule (as Phase 4/5, on validation splits): an arm advances if its mean ECAPA gain in similarity to the target
over the three validation pools versus `p5_fv` is >= 0.0266 (twice the seed-to-seed difference, Phase 4) with the
speaker-level paired bootstrap interval excluding 0 on at least one of VCTK and GLOBE, CAM++ agreeing in sign, and no
guard failing: LibriTTS-R own-speaker resynthesis drop <= 0.01, UTMOS drop <= 0.1 on every pool (GLOBE included), WER
increase <= 0.5 points, GLOBE similarity to the source rise <= 0.02. Ties among advancing arms: highest mean gain. The chosen
arm (or `p5_fv` if none advances) is then run once on the test pools. The Phase 4-5 guards failed for the combination
recipe on GLOBE (UTMOS, source similarity) and LibriTTS-R own-speaker; with the same rule an arm cannot advance unless it
also repairs them, so a result of "no arm advances" is possible and informative.
Training stability is reported, not selected on: gradient norms, discriminator balance and NaN checks from the tensorboard
logs; AdaLN arms whose gates stay near 0 are reported as "unused conditioning".
Single seed per arm; only differences above the margin count, with seed noise as in Phase 4. No test number is used for
selection.

Cost (from the smoke step times above): 100k steps at 0.34-0.46 s/step = 9.4-12.8 GPU-hours per arm; 4 arms = 38-51 GPU-hours of training, plus about 1-2
GPU-hours per arm for the validation-pool cloning evaluation (synthesis, three verifiers, UTMOS, WER; my estimate from the
earlier phases, not measured here) = about 45-60 GPU-hours, and one test evaluation (about 2 GPU-hours) for the chosen arm. The runs are
restartable (checkpoint every 20 minutes) and fit `general`/`msp` under the per-agent cap one or two at a time, so
wall time is about 1-2 days. Arms 1, 2 and 4 need no evaluation-code change; arm 3 uses the statistics support added to
`eval/clone.py` (`ItemRenderer`), which has been exercised by unit-level logic but not yet end to end on a trained
checkpoint (the smoke checkpoints are too short to synthesize meaningful audio; a 5-utterance synthesis check on a smoke
checkpoint should be run before arm 3's evaluation). `eval/ood_gap.py`, `ood_svfit*.py` and `ood_channel.py` build their
own batches and do not yet pass `utt_loud`/`ref_f0_std`, so Phase 8 code-fitting experiments are not available for
statistics checkpoints.
