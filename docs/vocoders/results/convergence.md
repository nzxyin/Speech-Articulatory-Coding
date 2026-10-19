# Compute and convergence (2026-10-14)

Sources: `convergence_data.json` (validation curves of every run from TensorBoard, generator FLOPs; script
`scripts/convergence_data.py`), `efficiency.md`, tables `final/tables/{seeds,ablations}/table.csv`. Two training seeds per
main arm; the seed difference below is the absolute difference of the two seeds, a crude noise estimate.

## 1. Compute: Vocos against HiFi-GAN

| quantity | HiFi-GAN | Vocos | Vocos advantage |
|---|---|---|---|
| generator parameters | 14.11 M | 13.78 M | about equal |
| generator FLOPs per second of audio (torch FlopCounterMode, batch 1, includes frontend and speaker FFN) | 30.3 G | 2.63 G | 11.5x |
| RTF on an L40S, median of 50 utterances | 0.0061 | 0.0008 | 7.6x |
| RTF on 1 CPU thread | 0.183 | 0.0266 | 6.9x |
| RTF on 8 CPU threads | 0.0671 | 0.0070 | 9.6x |
| training step, batch 16 x 1.28 s, fp32, L40S / 6000 Ada | 304-313 ms | 205-214 ms | 1.5x |
| GPU-hours for 400k steps (measured, before preemption losses) | about 34 | about 23 | 1.5x |
| end-to-end with the SPARC extractor, L40S | 0.0525 | 0.0471 | 1.1x |
| measured lookahead | 205 ms | 196 ms | about equal |

Vocos runs its ConvNeXt backbone at 100 frames per second and produces the waveform with an inverse STFT (n_fft 960,
hop 240); HiFi-GAN runs its convolution stacks through transposed-convolution upsampling up to 24,000 samples per second,
where most of its FLOPs are. Training cost is dominated by the shared discriminators (MPD + MRD, about 42 M parameters),
so the training advantage (1.5x) is much smaller than the inference advantage (7-11x). In the articulatory pipeline the
feature extractor (WavLM-large, CREPE; RTF 0.046 on the GPU, 3.5 on one CPU thread) dominates, so the end-to-end
advantage is about 10 percent. DDSP's FLOP count (0.93 G) excludes the oscillator bank's sine evaluations, which the
counter does not count; its measured CPU RTF (0.036 on one thread) is the better comparison.

## 2. Are 400k steps enough?

Each run uses a cosine schedule that decays to zero at its own end, so the flattening of a validation curve near the
end is annealing, not convergence. Only the end points of fully annealed runs of different lengths are comparable:
25k (Vocos pilots), 100k (ablation baselines) and 400k (main runs, two seeds).

| vocoder | dev mel L1 at 25k | 100k | 400k (mean of 2 seeds) | seed difference at 400k |
|---|---|---|---|---|
| Vocos (option B) | 0.5382 | 0.5061 | 0.4920 | 0.0028 |
| HiFi-GAN | n/a | 0.5174 | 0.4883 | 0.0006 |
| DDSP | n/a | 0.5233 | 0.5168 | 0.0052 |

Fit `L(B) = L_inf + A B^-alpha`. Vocos has three budgets four times apart, which fix the curve: the improvement shrinks
from 0.0320 (25k to 100k) to 0.0141 (100k to 400k), a ratio of 0.44 per factor four of steps, so alpha = 0.59. HiFi-GAN
and DDSP have two budgets; the table uses alpha from 0.4 to 0.8 around the Vocos value.

| vocoder | alpha | asymptote L_inf | distance at 400k | gain from 800k steps | gain / seed difference |
|---|---|---|---|---|---|
| Vocos | 0.59 (fitted) | 0.481 | 0.011 (2.3 %) | 0.0037 | 1.3 |
| HiFi-GAN | 0.4 to 0.8 | 0.449 to 0.474 | 0.014 to 0.039 (2.9 to 8.0 %) | 0.0061 to 0.0095 | 9 to 14 |
| DDSP | 0.4 to 0.8 | 0.508 to 0.514 | 0.003 to 0.009 (0.6 to 1.7 %) | 0.0014 to 0.0021 | 0.3 to 0.4 |

Inside each 400k run the unannealed curve reaches the annealed 100k value only at step 170k to 175k (HiFi-GAN), 235k to
240k (Vocos) and 235k to 285k (DDSP), which is the usual cost of a longer schedule.

Evaluation metrics on test.clean (T1), 100k baseline against the mean of the two 400k seeds, and the gain projected for
800k if each metric improves at the same rate per doubling as the Vocos dev mel L1 (0.265 times the 100k-to-400k change):

| vocoder | metric | 100k | 400k | change | projected gain at 800k | seed difference |
|---|---|---|---|---|---|---|
| HiFi-GAN | UTMOS | 3.95 | 4.16 | +0.22 | +0.06 (capped near the ground truth 4.21) | 0.004 |
| HiFi-GAN | Sim same | 0.801 | 0.858 | +0.057 | +0.015 | 0.001 |
| HiFi-GAN | F0 RMSE (cents) | 30.7 | 23.4 | -7.2 | -1.9 | 0.09 |
| HiFi-GAN | loudness RMSE (dB) | 1.81 | 1.49 | -0.32 | -0.09 | 0.004 |
| Vocos | UTMOS | 3.59 | 3.92 | +0.33 | +0.09 | 0.042 |
| Vocos | Sim same | 0.787 | 0.845 | +0.059 | +0.016 | 0.003 |
| Vocos | F0 RMSE (cents) | 39.4 | 34.2 | -5.2 | -1.4 | 1.5 |
| Vocos | loudness RMSE (dB) | 2.08 | 1.80 | -0.28 | -0.07 | 0.05 |
| DDSP | UTMOS | 3.77 | 3.83 | +0.06 | +0.02 | 0.011 |
| DDSP | Sim same | 0.771 | 0.802 | +0.031 | +0.008 | 0.001 |
| DDSP | F0 RMSE (cents) | 16.5 | 19.5 | +3.0 (worse) | worse | 2.7 |
| DDSP | loudness RMSE (dB) | 2.69 | 2.85 | +0.15 (worse) | worse | 0.09 |

WER changed by at most 0.2 percentage points from 100k to 400k for all three and is not budget-limited.

Reading:
- **DDSP has converged** at 400k for this recipe: the remaining dev mel gain is below the seed difference, UTMOS gained
  only 0.06 from 100k to 400k, and its F0, voicing and loudness errors grew from 100k to 400k.
- **Vocos is close**: about 2 percent above its fitted asymptote in dev mel L1; doubling to 800k is projected to gain
  about 0.09 UTMOS and 0.016 speaker similarity.
- **HiFi-GAN is the furthest from convergence** in dev mel L1 (3 to 8 percent above its asymptote) and in prosody, but its
  UTMOS (4.16) is already close to the ground truth (4.21), so most of its remaining gain would be in fidelity metrics.
- **Effect on the comparison**: longer training would probably widen HiFi-GAN's lead on fidelity and prosody and move
  Vocos ahead of DDSP on UTMOS (Vocos is projected to gain about 0.07 more than DDSP), which would settle the currently
  seed-sensitive Vocos-DDSP UTMOS ordering. A fixed step budget also favours the slower models: at equal training
  GPU-hours Vocos could take 1.5 times as many steps as HiFi-GAN.

Limits: one curve shape (fitted on Vocos dev mel L1) is used for every vocoder and metric; HiFi-GAN and DDSP have only two
budgets; two seeds give a crude noise estimate; UTMOS is bounded near the ground truth. The cheapest checks: evaluate the
25k Vocos pilots on test.clean (UTMOS-specific rate for Vocos, about 10 GPU-minutes); one 200k annealed run per vocoder
(third budget point for HiFi-GAN and DDSP, about 12-17 GPU-hours each); a direct test is one 800k run each for HiFi-GAN
and Vocos (about 70 and 46 GPU-hours).

## 3. Result of the 800k runs (2026-10-19)

Runs: experiment `main_800k` (seed 0, cosine schedule over 800k steps), one run each of HiFi-GAN and Vocos. Training cost
(wall time of all job segments, including restarts): HiFi-GAN 69.7 GPU-hours (8 segments, 316 ms per step on an L40S in the
longest one), Vocos 45.9 GPU-hours (one segment, L40S). Evaluation: `eval_systems=long`, test.clean, all three conditions,
tables `final/tables/long/` (copy of `$SV_ROOT/eval/tables/test.clean/long/`, aggregate job 10799215, no missing groups).
Scripts: `scripts/curves_800k.py` (dev curves, `curves_800k.json`), `scripts/convergence_refit.py` (end points and
fits), `scripts/long_report.py` (output `long_report_T1.md`, `_T2`, `_T3`), `scripts/long_extra_pairs.py` (pairs not in
`paired.csv`, output `long_extra_pairs.md`; its two check rows reproduce the `paired.csv` point estimates, with interval ends within 0.0005 UTMOS and 0.03 cents).
The interim note `800k_interim.md` (Vocos only) is superseded by this section.

### 3.1 Dev mel L1 (annealed end points)

| vocoder | 100k | 400k seed 0 | 400k seed 1 | 400k mean | 800k | gain 400k mean to 800k | projected gain (section 2) | seed difference |
|---|---|---|---|---|---|---|---|---|
| HiFi-GAN | 0.5174 | 0.4886 | 0.4880 | 0.4883 | 0.4809 | 0.0075 | 0.0061 to 0.0095 | 0.0007 |
| Vocos | 0.5061 | 0.4934 | 0.4906 | 0.4920 | 0.4899 | 0.0021 | 0.0037 | 0.0028 |

- **HiFi-GAN** gained 0.0075, inside the projected range and 11 times its seed difference (0.00067 unrounded; section 2
  prints it as 0.0006). The three budgets (100k, 400k,
  800k) fix the curve: alpha = 0.62 (close to the 0.59 fitted on Vocos in section 2), asymptote 0.467. HiFi-GAN was 4.6
  percent above it at 400k and is 3.0 percent above it at 800k; another doubling to 1.6M steps is projected to gain 0.0049.
- **Vocos** gained 0.0021 against the mean of the two 400k seeds (0.0035 against the same-seed 400k run), less than its
  seed difference (0.0028). The projection (0.0037) lies between the two readings, so one 800k run cannot tell whether
  Vocos follows the curve of section 2. Fits: all four budgets with the seed mean at 400k give alpha 0.65 and an asymptote
  of 0.484 (1.2 percent above it at 800k); the last three budgets alone give alpha 1.09 and 0.488 (0.4 percent). Vocos is
  within about 1 percent of its asymptote at 800k.

### 3.2 Evaluation metrics, test.clean T1

Paired differences, speaker-cluster bootstrap CI95, share of agreeing speakers; *supported* = CI excludes 0 and at least
75 percent agree. "Robust" applies the rule of RESULTS.md section 4.1 to 800k against each 400k seed: supported and better
against both seeds, and the smaller of the two differences larger than the seed-to-seed difference.

| vocoder | metric | 400k seed 0 | 400k seed 1 | 800k | 800k - seed 0 | 800k - seed 1 | seed difference | projected (against the seed mean) | robust |
|---|---|---|---|---|---|---|---|---|---|
| HiFi-GAN | UTMOS | 4.165 | 4.161 | 4.207 | +0.042 (+0.035, +0.050), 97 % | +0.046 (+0.037, +0.055), 92 % | 0.004 | +0.06 (capped) | yes |
| HiFi-GAN | PESQ | 2.091 | 2.091 | 2.165 | +0.075, 97 % | +0.075, 97 % | 0.000 | | yes |
| HiFi-GAN | WER (%) | 2.45 | 2.49 | 2.42 | -0.03, n.s. | -0.07, n.s. | 0.03 | none | no |
| HiFi-GAN | Sim same | 0.857 | 0.858 | 0.867 | +0.0098, 97 % | +0.0084, 95 % | 0.0014 | +0.015 | yes |
| HiFi-GAN | Sim spk | 0.731 | 0.732 | 0.733 | +0.0025, CI only | +0.0014, n.s. | 0.0011 | | no |
| HiFi-GAN | mel L1 | 0.4878 | 0.4879 | 0.4802 | -0.0076, 100 % | -0.0077, 97 % | 0.0001 | | yes |
| HiFi-GAN | F0 RMSE (cents) | 23.4 | 23.5 | 21.4 | -1.98 (-2.26, -1.70), 92 % | -2.07 (-2.33, -1.78), 90 % | 0.09 | -1.9 | yes |
| HiFi-GAN | VDE | 0.0388 | 0.0393 | 0.0370 | -0.0018, 87 % | -0.0023, 85 % | 0.0005 | | yes |
| HiFi-GAN | loudness RMSE (dB) | 1.49 | 1.49 | 1.39 | -0.103, 97 % | -0.108, 97 % | 0.004 | -0.09 | yes |
| HiFi-GAN | EMA r | 0.9710 | 0.9707 | 0.9724 | +0.0015, 97 % | +0.0017, 97 % | 0.0002 | | yes |
| Vocos | UTMOS | 3.896 | 3.939 | 3.969 | +0.072 (+0.058, +0.085), 90 % | +0.030 (+0.020, +0.040), 90 % | 0.042 | +0.09 | no (0.030 < 0.042) |
| Vocos | PESQ | 1.903 | 1.936 | 1.947 | +0.044, 95 % | +0.011, 77 % | 0.033 | | no |
| Vocos | WER (%) | 2.55 | 2.47 | 2.44 | -0.11, CI only | -0.03, n.s. | 0.08 | none | no |
| Vocos | Sim same | 0.844 | 0.847 | 0.852 | +0.0085, 92 % | +0.0058, 90 % | 0.0028 | +0.016 | yes |
| Vocos | Sim spk | 0.722 | 0.724 | 0.724 | +0.0021, CI only | +0.0002, n.s. | 0.0018 | | no |
| Vocos | mel L1 | 0.4940 | 0.4913 | 0.4902 | -0.0038, 97 % | -0.0011, 77 % | 0.0028 | | no |
| Vocos | F0 RMSE (cents) | 35.0 | 33.5 | 32.9 | -2.08 (-2.65, -1.66), 97 % | -0.55 (-1.48, +0.10), n.s. | 1.53 | -1.4 | no |
| Vocos | VDE | 0.0415 | 0.0416 | 0.0422 | +0.0006, n.s. | +0.0006, CI only (worse) | 0.0001 | | no |
| Vocos | loudness RMSE (dB) | 1.83 | 1.78 | 1.82 | -0.009, n.s. | +0.046, 87 % (worse) | 0.054 | -0.07 | no |
| Vocos | EMA r | 0.9685 | 0.9689 | 0.9695 | +0.0010, 90 % | +0.0006, 79 % | 0.0004 | | yes |

Source: `long_report_T1.md` (exact values and intervals). Under T2 and T3 the same HiFi-GAN rows are robust (UTMOS +0.044
and +0.049 against s0 and s1 under T2, +0.037 and +0.050 under T3), and Sim spk becomes robust under T3 only; for Vocos the same few rows pass (Sim same, EMA r, EMA
RMSE, mel L1 8-12 kHz; mel L1 4-8 kHz under T3 only) (`long_report_T2.md`, `long_report_T3.md`).

Reading:
- **HiFi-GAN improved robustly on every metric except WER, CER and Sim spk.** Its UTMOS (4.207) equals the ground truth
  (4.208). The F0 (-2.0 cents) and loudness (-0.11 dB) gains match the projection (-1.9 and -0.09); UTMOS (+0.044 against
  the seed mean) and Sim same (+0.009) gained about 74 and 61 percent of the projection.
- **Vocos improved less, and mostly within the seed spread.** Against the seed mean it gained +0.051 UTMOS (57 percent of
  the projection), +0.007 Sim same (45 percent) and -1.3 cents F0 (94 percent, but smaller than its seed difference);
  loudness and voicing did not improve. Only Sim same, the EMA consistency (EMA r, EMA RMSE) and mel L1 in the 4-8 and
  8-12 kHz bands pass the robustness rule.
- **WER does not depend on the training length** for either vocoder (all values 2.42 to 2.55 percent; no 800k difference
  supported against both seeds).

### 3.3 Effect on the comparison (T1)

| pair | metric | 400k s0 | 400k s1 | 800k |
|---|---|---|---|---|
| HiFi-GAN - Vocos | UTMOS | +0.268 | +0.222 | +0.239 (+0.210, +0.267), 97 % |
| HiFi-GAN - Vocos | PESQ | +0.188 | +0.155 | +0.219, 97 % |
| HiFi-GAN - Vocos | Sim same | +0.0130 | +0.0117 | +0.0144, 97 % |
| HiFi-GAN - Vocos | mel L1 | -0.0063 | -0.0034, CI only | -0.0100, 97 % |
| HiFi-GAN - Vocos | MCD (dB) | +0.025 (Vocos better) | +0.030 (Vocos better) | -0.010, n.s. |
| HiFi-GAN - Vocos | F0 RMSE (cents) | -11.6 | -10.0 | -11.5, 97 % |
| HiFi-GAN - Vocos | VDE | -0.0028 | -0.0023 | -0.0052, 97 % |
| HiFi-GAN - Vocos | loudness RMSE (dB) | -0.34 | -0.28 | -0.44, 97 % |

400k values from RESULTS.md Table 4.2 (seeds `paired.csv`); 800k from `long/paired.csv`.

- **At equal steps the HiFi-GAN lead holds or grows.** The UTMOS gap at 800k (+0.239) lies between the two 400k values;
  the gaps in PESQ, mel L1, VDE and loudness widen, and the MCD advantage of Vocos disappears. This is the direction
  projected in section 2.
- **At about equal training compute the HiFi-GAN lead also holds.** Vocos at 800k (45.9 GPU-hours) against HiFi-GAN at 400k
  (about 34 GPU-hours), recomputed from `per_speaker.csv`: UTMOS +0.196 (+0.172, +0.220), 97 percent (seed 0) and +0.192,
  97 percent (seed 1) for HiFi-GAN; F0 RMSE -9.5 cents; loudness RMSE -0.33 dB; Sim same +0.005 and +0.006 (77 and 82
  percent); mel L1 -0.0024 and -0.0023 (CI only). The 1.5 times cheaper training step of Vocos does not buy back the gap.
- **Vocos at 800k is ahead of DDSP at 400k on UTMOS** (DDSP has converged, section 2): +0.137 (+0.075, +0.209), 77 percent,
  against DDSP seed 0 and +0.148 (+0.098, +0.206), 79 percent, against DDSP seed 1, both supported; at 400k this order was
  not robust (RESULTS.md section 4.1). It rests on one Vocos 800k run, so it is not robust under the two-seed rule either;
  the smaller difference (0.137) is more than three times the larger seed difference (0.042, Vocos).
- **HiFi-GAN at 800k against DDSP at 400k on F0:** +3.2 cents (DDSP better, supported) against DDSP seed 0 and +0.5 cents
  (n.s.) against seed 1; the HiFi-GAN-DDSP F0 order stays unresolved.

Limits: one 800k run per vocoder, so the 800k results are single runs, like the ablations; the HiFi-GAN fit uses three
points and therefore has no residual; DDSP was not run for 800k steps; test.other was not evaluated at 800k.
