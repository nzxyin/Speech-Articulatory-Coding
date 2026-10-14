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
