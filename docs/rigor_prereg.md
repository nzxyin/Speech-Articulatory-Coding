# Pre-registration: rigor re-analysis of the LOSO speech-to-EMA results (2026-10-15)

Written and committed before the analyses below were run. It fixes the primary contrasts, the metric definitions and
the tests, so that the new analyses do not add more choices tuned on the same seven held-out speakers. Code:
`src/sparc/compression/rigor.py`.

## Data and status of every number

- 7 leave-one-speaker-out folds (usc_M1, usc_F1, usc_M3, usc_F5, 5emo_jn, 5emo_jr, 5emo_kf), held-out speaker's
  test sentences (`test_unseen`). No untouched data exist: every encoder depth, head and learning-rate grid was
  already looked at on these folds. All results here are therefore labelled **selected on the test folds**; only a
  fresh speaker or corpus (rigor plan, tier 3) can give an untouched number.
- Contrasts P2 and P3 compare configurations whose held-out scores were seen in the head screen. Their pre-registered
  tests replace the earlier per-contrast t intervals; they are re-tests, not confirmations. P1 has not been computed.

## Metric definitions (all in the held-out speaker's own z units)

- **rmse_z (primary error metric):** frames of all test utterances pooled per channel, RMSE per channel, mean over the
  12 channels. z units use the held-out speaker's full training-split statistics (enrollment / oracle calibration).
- **PCC (primary shape metric):** Pearson correlation per utterance and channel, mean over channels and utterances.
- Diagnostic only, never used to pick a winner: centered rmse_z (each utterance's mean error per channel removed),
  the decomposition rmse_z^2 = per-utterance bias^2 + centered^2 (per channel, averaged), speaker-centered rmse_z,
  prediction / target std ratio, PCC of concatenated utterances per speaker, rmse_z under enrollment with N = 0, 10,
  20 utterances (10 random draws) and the post-hoc +-1 s output smoother.

## Primary contrasts (family of 3)

Fold value = mean over seeds (3 seeds for trained arms; the ridge probe is deterministic).

| id | contrast | metric | question |
|---|---|---|---|
| P1 | blend (0.75 LoRA + 0.25 probe), BiGRU->conv 9, WavLM k12 vs the same LoRA scaled by one gain chosen on the other six folds' held-out speakers | rmse_z | Does the second (frozen) encoder pass help beyond shrinkage? |
| P2 | frozen BiGRU->conv 9 vs ridge probe, WavLM k12 | PCC | Does a temporal head improve trajectory shape for new speakers? |
| P3 | frozen BiGRU->conv 9, WavLM Large k12 (21.4 GFLOPs/s) vs XLS-R 300M k18 (29.1 GFLOPs/s) | PCC | Does the larger-FLOPs encoder help under a strong head? |

## Tests

- Primary test: exact two-sided sign-flip permutation test over the 7 fold differences (128 sign patterns; the
  smallest attainable p is 2/128 = 0.016). Holm correction over the 3 contrasts at alpha = 0.05.
- Effect size: mean fold difference with a hierarchical bootstrap 95% interval (2000 resamples: speakers with
  replacement, then seeds within speaker, then utterances within speaker, paired between arms).
- Reported alongside: the paired t interval, folds won, the seed and fold variance components, and the minimum
  effect detectable with 80% power, (t_0.975,6 + t_0.8,6) x sd(fold differences) / sqrt(7).
- The 7 folds are not independent (each pair shares 5 training speakers; 4 USC and 3 EMA_5EMO speakers). Results are
  also reported per corpus, as a sensitivity check only.

## Shrinkage control (for P1 and the descriptive tables)

For each LoRA arm, fold and seed, the predictions are multiplied by g = sum(p y) / sum(p p) fitted on the other six
folds' held-out predictions of the same arm and seed (a scalar; also a per-channel variant). The held-out speaker
never contributes to its own gain. An oracle gain fitted on the speaker itself is reported as an upper bound and
labelled oracle. Zero in z units is the speaker's mean, so the gain shrinks toward the speaker's mean.

## What would change the conclusions

- P1 not significant and blend - shrunk LoRA < 0.005 rmse_z: the blend's rmse_z gain is shrinkage; recommend a single
  pass with a calibrated gain instead of the second encoder pass.
- P2 significant: temporal heads are confirmed to improve held-out PCC over the probe at k12.
- P3 not significant: no evidence that 300M k18 beats WavLM k12 under a strong head; recommend WavLM k12 on FLOPs.

---

# Addendum (2026-10-15, before any Tier 2 / Item 4 run): nested selection and controlled training

Same metric definitions, tests and status labels as above. Primary family (Holm over 3, exact sign-flip test over the
7 folds, hierarchical bootstrap for intervals):

| id | contrast | metric | question |
|---|---|---|---|
| T1 | LoRA + BiGRU->conv in the frozen-head pipeline ("matched") vs frozen BiGRU->conv, WavLM k12 | PCC | Does LoRA add trajectory shape once the pipelines are identical? |
| T2 | matched LoRA vs the original LoRA pipeline (same head, WavLM k12) | rmse_z | How much of LoRA's held-out rmse_z deficit is the pipeline? |
| N1 | frozen BiGRU->conv at the nested-selected depth vs at k12 (the cross-fold choice) | PCC | Is the k12 recommendation optimistic? (+ = nested better) |

Matched pipeline: encoder output low-passed with the same zero-phase 10 Hz filter as the cached features (in-graph),
no output smoother, per-fold shift from the depth sweep, encoder dropout off (encoder kept in eval mode), fp32, head
learning rate = the head screen's per-fold choice for BiGRU->conv at k12, LoRA learning rate chosen from
{1e-4, 2e-4, 5e-4} on the training speakers' validation (seed 0), then seeds 1 and 2. Only the endpoints are run (no
one-factor-at-a-time decomposition); which factor matters is not identified.

Nested selection (Item 4): for each outer held-out speaker, an inner leave-one-speaker-out over its 6 training speakers
(train on 5, score the 6th's test sentences). Probe: every WavLM and XLS-R 300M depth, shift = the outer fold's
depth-sweep shift at that depth, alpha on the inner training speakers' validation; depth chosen by mean inner rmse_z.
Heads (pre-declared: BiGRU->conv and BiMamba, WavLM depths {7, 9, 12, 15, 18}): one seed per inner fold at a fixed
learning rate per head (the head screen's most frequent choice); depth chosen by mean inner PCC; the outer score at the
chosen depth is the 3-seed head-screen result there. Reported: nested outer score vs the k12 / cross-fold score
(selection optimism), and the depths chosen.

Descriptive only (no tests): causal pipeline (causal 10 Hz low-pass or none, causal conv 9 and unidirectional GRU, no
smoother) vs the zero-phase pipeline; if causal heads lose more than 0.01 PCC, the word "causal" in earlier results is
replaced by "offline (zero-phase filtering)". Length controls (chunks cut from full-utterance features; chunk encoded
alone but low-passed over the full utterance; whole-utterance waveform normalization), 3 seeds. Alignment sensitivity:
held-out targets shifted by -2..+2 frames against cached predictions, and training speakers' alignment perturbed by a
random +-1 frame (probe), reporting rank agreement (Kendall tau) of the arms. Data curve: MNGU0 ridge probe and LoRA +
causal conv (WavLM k9) trained on 100 / 200 / 400 / all utterances, and within-speaker probes for each multi-speaker
speaker (its own train -> its own test), all in z units.
