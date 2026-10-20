# Probes: seed 0 against seed 1 (2026-10-20)

Question: do the controllability findings of RESULTS.md section 5.2 (measured on the seed-0 runs) hold for the second
training seed of each vocoder?

## Basis

- Probes (stage `probes`): the same 50 test.clean utterances of 3 to 10 s (28 speakers), the same edit grid (4 F0 edits,
  2 loudness edits, periodicity set to 0, plus or minus 1 training std on each of the 12 EMA channels; 1,550 rows per
  system) and the same probe configuration for all six systems; `probes_meta.json` differs only in the checkpoint and the
  system name. Seed-1 outputs: `$SV_ROOT/eval/probes/{hifigan,ddsp,vocos}_s1/` (written 2026-10-13 by the seed evaluation
  jobs 10752452, 10752454, 10752456; final checkpoints `runs/main_seed1/<vocoder>/ckpt/step000400000.ckpt`).
- Statistics: as `controllability.md` (median over the 50 utterances of the per-utterance value; speaker-cluster bootstrap
  CI95 of the median, 28 speakers, 10,000 resamples, seed 0). Seed differences and vocoder differences are medians of
  per-utterance differences on the same utterances. Script `scripts/probes_seeds.py` (outputs
  `controllability_seeds_long.csv`, `controllability_seeds_paired.csv`, `controllability_seeds_robust.csv`); it reproduces
  every seed-0 cell and pair of `controllability_long.csv` and `controllability_paired.csv` exactly (largest difference
  6e-14). Tables: `scripts/probes_seeds_report.py`.
- Robustness rule for vocoder differences (RESULTS.md section 4.1): same sign and CI95 excluding 0 in both seeds, and the
  smaller absolute difference larger than the larger absolute seed difference (seed 1 minus seed 0) of the two vocoders.
  There is no speaker-agreement criterion here because the probe statistics are medians over utterances.
- Natural variation (section 5 of `controllability.md`) for the seed-1 systems: `scripts/natural_variation.py
  hifigan_s1,ddsp_s1,vocos_s1 _s1` (outputs `natural_variation_s1.csv`, `natural_variation_matrix_s1.npz`; all 4,686
  re-extracted test.clean utterances, 1,512,210 frames, 39 speakers, T1).
- The F0 rows are quantized: the re-extracted pitch falls on a grid of 10 to 20 cents, so per-utterance median shifts are
  grid values and the intervals of the median collapse (caveat of `controllability.md` section 1).

### Table A. Per-system probe results, seed 0 and seed 1 (median over 50 utterances, speaker-cluster CI95)

| quantity | HiFi-GAN seed 0 | HiFi-GAN seed 1 | DDSP seed 0 | DDSP seed 1 | Vocos seed 0 | Vocos seed 1 |
|---|---|---|---|---|---|---|
| F0 edit -4 semitones (target -400 cents) | -400.0 (-400.0, -390.3) | -400.0 (-400.0, -400.0) | -400.0 (-400.0, -400.0) | -400.0 (-400.0, -400.0) | -390.0 (-390.0, -380.0) | -390.0 (-390.1, -380.0) |
| F0 edit -2 semitones (target -200 cents) | -200.0 (-200.0, -200.0) | -200.0 (-200.0, -200.0) | -200.0 (-200.0, -200.0) | -200.0 (-200.0, -200.0) | -197.5 (-200.0, -190.0) | -200.0 (-200.0, -199.8) |
| F0 edit +2 semitones (target +200 cents) | 200.0 (200.0, 200.0) | 200.0 (200.0, 200.0) | 200.0 (200.0, 200.0) | 200.0 (200.0, 200.0) | 200.0 (190.1, 200.0) | 200.0 (200.0, 200.0) |
| F0 edit +4 semitones (target +400 cents) | 400.0 (400.0, 400.0) | 400.0 (400.0, 400.0) | 400.0 (400.0, 400.0) | 400.0 (400.0, 400.0) | 399.8 (390.0, 400.0) | 400.0 (395.0, 400.0) |
| loudness x0.5 (target -6.02 dB) | -5.89 (-5.91, -5.87) | -5.90 (-5.92, -5.87) | -6.00 (-6.03, -5.97) | -5.99 (-6.02, -5.97) | -6.01 (-6.03, -5.99) | -5.97 (-6.01, -5.95) |
| loudness x2 (target +6.02 dB) | 5.78 (5.75, 5.81) | 5.78 (5.75, 5.81) | 5.79 (5.74, 5.87) | 5.80 (5.77, 5.87) | 5.87 (5.82, 5.90) | 5.82 (5.78, 5.87) |
| voiced fraction before periodicity = 0 | 0.56 (0.52, 0.59) | 0.56 (0.52, 0.59) | 0.61 (0.58, 0.63) | 0.61 (0.58, 0.63) | 0.56 (0.53, 0.59) | 0.55 (0.53, 0.59) |
| voiced fraction after periodicity = 0 (ideal 0) | 0.17 (0.15, 0.18) | 0.16 (0.13, 0.19) | 0.28 (0.24, 0.33) | 0.33 (0.27, 0.38) | 0.08 (0.05, 0.12) | 0.08 (0.06, 0.13) |
| single EMA channel: gain (mean of 12; 1 = reproduced) | 0.117 (0.112, 0.121) | 0.120 (0.115, 0.124) | 0.095 (0.093, 0.101) | 0.102 (0.096, 0.105) | 0.117 (0.115, 0.120) | 0.115 (0.112, 0.120) |
| single EMA channel: leakage (mean of 12, std) | 0.244 (0.239, 0.248) | 0.241 (0.238, 0.247) | 0.243 (0.239, 0.250) | 0.248 (0.244, 0.253) | 0.239 (0.235, 0.242) | 0.235 (0.231, 0.240) |
| EMA movement under F0 edits (std) | 0.12 (0.12, 0.13) | 0.12 (0.12, 0.13) | 0.16 (0.15, 0.16) | 0.15 (0.15, 0.16) | 0.14 (0.14, 0.15) | 0.14 (0.13, 0.14) |
| EMA movement under loudness edits (std) | 0.08 (0.08, 0.08) | 0.08 (0.08, 0.08) | 0.07 (0.07, 0.08) | 0.07 (0.07, 0.07) | 0.08 (0.07, 0.08) | 0.08 (0.07, 0.08) |
| EMA movement under periodicity = 0 (std) | 0.18 (0.16, 0.19) | 0.18 (0.16, 0.19) | 0.20 (0.19, 0.22) | 0.20 (0.18, 0.21) | 0.19 (0.18, 0.22) | 0.19 (0.18, 0.21) |

### Table B. EMA gain per channel (median over utterances, both signs averaged), seed 0 / seed 1

| channel | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| TDX | +0.26 / +0.28 | +0.24 / +0.27 | +0.29 / +0.27 |
| TDY | +0.05 / +0.05 | +0.05 / +0.02 | +0.07 / +0.07 |
| TBX | -0.15 / -0.12 | -0.16 / -0.17 | -0.16 / -0.15 |
| TBY | +0.32 / +0.32 | +0.26 / +0.26 | +0.31 / +0.30 |
| TTX | +0.25 / +0.26 | +0.25 / +0.26 | +0.24 / +0.25 |
| TTY | +0.12 / +0.13 | +0.11 / +0.11 | +0.12 / +0.13 |
| LIX | +0.11 / +0.10 | +0.08 / +0.07 | +0.09 / +0.09 |
| LIY | +0.08 / +0.08 | +0.07 / +0.09 | +0.08 / +0.08 |
| ULX | +0.06 / +0.06 | +0.06 / +0.06 | +0.06 / +0.06 |
| ULY | +0.05 / +0.05 | +0.04 / +0.05 | +0.05 / +0.06 |
| LLX | +0.11 / +0.13 | +0.11 / +0.13 | +0.13 / +0.13 |
| LLY | +0.09 / +0.09 | +0.07 / +0.06 | +0.08 / +0.08 |

### Table C. Seed 1 minus seed 0 (median of per-utterance differences, speaker-cluster CI95, share of utterances where seed 1 is larger)

| metric | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| per_zero voiced_frac_edit | -0.0085 (-0.0263, +0.0031), 0.36, n.s. | +0.0341 (+0.0190, +0.0534), 0.72 | +0.0167 (+0.0050, +0.0269), 0.70 |
| F0 abs error (cents, mean of 4 edits) | -0.0289 (-2.4927, +0.0000), 0.36, n.s. | +0.0000 (+0.0000, +0.0000), 0.62 | -1.8731 (-3.7391, +0.0145), 0.38, n.s. |
| loudness abs error (dB, mean of 2 edits) | +0.0037 (-0.0199, +0.0229), 0.52, n.s. | -0.0085 (-0.0305, +0.0082), 0.34, n.s. | +0.0071 (-0.0048, +0.0309), 0.56, n.s. |
| EMA gain (12 channels, mean) | +0.0022 (-0.0005, +0.0045), 0.64, n.s. | +0.0039 (+0.0012, +0.0061), 0.70 | -0.0019 (-0.0048, +0.0016), 0.44, n.s. |
| EMA leakage (12 channels, mean) | -0.0016 (-0.0025, -0.0002), 0.34 | +0.0024 (+0.0009, +0.0043), 0.72 | -0.0001 (-0.0035, +0.0013), 0.50, n.s. |
| EMA floor (non-EMA edits, mean) | +0.0003 (-0.0005, +0.0014), 0.54, n.s. | -0.0018 (-0.0033, -0.0007), 0.32 | -0.0014 (-0.0020, +0.0016), 0.42, n.s. |

### Table D. Vocoder differences in each seed and the robustness rule (summary metrics)

| metric | pair | seed 0 | seed 1 | larger seed difference | robust | reason |
|---|---|---|---|---|---|---|
| per_zero voiced_frac_edit | Vocos - DDSP | -0.1562 (-0.2857, -0.0704) | -0.1903 (-0.2889, -0.1392) | 0.0341 | yes |  |
| per_zero voiced_frac_edit | Vocos - HiFi-GAN | -0.0551 (-0.1074, -0.0158) | -0.0401 (-0.0821, -0.0177) | 0.0167 | yes |  |
| per_zero voiced_frac_edit | HiFi-GAN - DDSP | -0.1424 (-0.1667, -0.0869) | -0.1554 (-0.2086, -0.1056) | 0.0341 | yes |  |
| F0 abs error (cents, mean of 4 edits) | Vocos - DDSP | +8.12 (+5.00, +12.53) | +5.65 (+4.99, +9.98) | 1.87 | yes |  |
| F0 abs error (cents, mean of 4 edits) | Vocos - HiFi-GAN | +5.00 (+3.71, +9.99) | +5.03 (+3.73, +7.52) | 1.87 | yes |  |
| F0 abs error (cents, mean of 4 edits) | HiFi-GAN - DDSP | +2.49 (+0.01, +4.97) | +0.04 (+0.00, +2.49) | 0.03 | yes |  |
| loudness abs error (dB, mean of 2 edits) | Vocos - DDSP | -0.0435 (-0.0670, -0.0113) | -0.0268 (-0.0415, +0.0025) | 0.0085 | no | seed 1 CI includes 0 |
| loudness abs error (dB, mean of 2 edits) | Vocos - HiFi-GAN | -0.0697 (-0.0888, -0.0540) | -0.0541 (-0.0878, -0.0272) | 0.0071 | yes |  |
| loudness abs error (dB, mean of 2 edits) | HiFi-GAN - DDSP | +0.0323 (-0.0052, +0.0660) | +0.0233 (+0.0099, +0.0769) | 0.0085 | no | seed 0 CI includes 0 |
| EMA gain (12 channels, mean) | Vocos - DDSP | +0.0175 (+0.0141, +0.0223) | +0.0135 (+0.0101, +0.0195) | 0.0039 | yes |  |
| EMA gain (12 channels, mean) | Vocos - HiFi-GAN | -0.0014 (-0.0043, +0.0023) | -0.0055 (-0.0066, -0.0013) | 0.0022 | no | seed 0 CI includes 0; smaller than the seed difference |
| EMA gain (12 channels, mean) | HiFi-GAN - DDSP | +0.0192 (+0.0169, +0.0210) | +0.0191 (+0.0166, +0.0212) | 0.0039 | yes |  |
| EMA leakage (12 channels, mean) | Vocos - DDSP | -0.0070 (-0.0100, -0.0055) | -0.0104 (-0.0133, -0.0074) | 0.0024 | yes |  |
| EMA leakage (12 channels, mean) | Vocos - HiFi-GAN | -0.0062 (-0.0076, -0.0033) | -0.0057 (-0.0083, -0.0039) | 0.0016 | yes |  |
| EMA leakage (12 channels, mean) | HiFi-GAN - DDSP | -0.0010 (-0.0040, +0.0021) | -0.0036 (-0.0063, -0.0023) | 0.0024 | no | seed 0 CI includes 0; smaller than the seed difference |
| EMA floor (non-EMA edits, mean) | Vocos - DDSP | -0.0084 (-0.0136, -0.0063) | -0.0074 (-0.0111, -0.0050) | 0.0018 | yes |  |
| EMA floor (non-EMA edits, mean) | Vocos - HiFi-GAN | +0.0133 (+0.0110, +0.0151) | +0.0124 (+0.0107, +0.0142) | 0.0014 | yes |  |
| EMA floor (non-EMA edits, mean) | HiFi-GAN - DDSP | -0.0212 (-0.0257, -0.0192) | -0.0210 (-0.0228, -0.0178) | 0.0018 | yes |  |

Per-channel EMA gain: 10 of 36 channel-pair differences are robust: TDY Vocos - HiFi-GAN +0.014 / +0.014; TBY Vocos - DDSP +0.047 / +0.045; TBY HiFi-GAN - DDSP +0.065 / +0.061; TTY Vocos - DDSP +0.013 / +0.021; TTY HiFi-GAN - DDSP +0.021 / +0.013; LIX Vocos - DDSP +0.031 / +0.025; LIX HiFi-GAN - DDSP +0.031 / +0.016; ULY Vocos - DDSP +0.013 / +0.014; LLY Vocos - DDSP +0.014 / +0.024; LLY HiFi-GAN - DDSP +0.018 / +0.028

### Table E. Natural variation through the loop, seed 0 / seed 1 (mean over 12 channels; 4,686 utterances, 39 speakers)

| system | centred slope | centred r | between-utterance slope | regression diagonal | linear leakage | F0 slope (log2) |
|---|---|---|---|---|---|---|
| HiFi-GAN | 1.002 / 1.002 | 0.974 / 0.974 | 0.935 / 0.928 | 0.981 / 0.981 | 0.021 / 0.022 | 0.999 / 0.998 |
| DDSP | 0.959 / 0.958 | 0.964 / 0.964 | 0.903 / 0.898 | 0.929 / 0.926 | 0.034 / 0.031 | 0.999 / 0.996 |
| Vocos | 1.000 / 1.001 | 0.972 / 0.972 | 0.931 / 0.930 | 0.974 / 0.976 | 0.024 / 0.027 | 0.997 / 0.998 |

## Reading

- **Every finding of section 5.2 replicates in seed 1.**
  - F0 edits of plus or minus 2 and 4 semitones are followed to within 10 cents by all six runs.
  - Loudness edits are followed to within 0.24 dB (HiFi-GAN undershoots slightly in both seeds: -5.89 and -5.90, +5.78 and +5.78 dB).
  - Setting periodicity to 0 removes most voicing in Vocos (0.55 to 0.08 in seed 1), less in HiFi-GAN (0.56 to 0.16) and least in DDSP
    (0.61 to 0.33). None of the six runs reaches 0.
  - A single EMA channel edited by one training std is reproduced with a mean gain of 0.120 (HiFi-GAN), 0.102 (DDSP) and 0.115 (Vocos),
    against 0.117, 0.095 and 0.117 in seed 0. The leakage is 0.235 to 0.248 std, and the EMA noise floor under non-EMA edits changes by at most 0.002 std.
  - Tongue body X has a negative gain in all six runs (-0.12 to -0.17).
  - Natural EMA variation passes the loop as in seed 0. The centred slope is 1.00 for HiFi-GAN and Vocos and 0.96 for DDSP, and the
    regression diagonal is 0.98, 0.93 and 0.98. Every mean over channels changes by at most 0.007 between the seeds.
- **Seed differences are small.** The largest supported change is the voiced fraction of DDSP after periodicity = 0
  (+0.034, 72 percent of utterances, from 0.28 to 0.33). It agrees with the higher VDE of DDSP seed 1 (+0.0018,
  RESULTS.md Table 4.1). The mean EMA gain changes by at most 0.004 per vocoder, the leakage by at most 0.002, and the per-channel gains by at most
  0.04 (medians of per-utterance differences; largest HiFi-GAN TBX +0.037 and DDSP TDX +0.033).
- **Robust vocoder differences (both seeds, larger than the seed difference):**
  - *Periodicity control.* Vocos removes the most voicing and DDSP the least, for all three pairs. After the edit, the voiced fraction of
    Vocos is 0.16 and 0.19 lower than that of DDSP and 0.06 and 0.04 lower than that of HiFi-GAN.
  - *EMA gain.* HiFi-GAN and Vocos follow single-channel EMA edits slightly more than DDSP: +0.019 and +0.019 for HiFi-GAN
    minus DDSP, and +0.017 and +0.013 for Vocos minus DDSP. Vocos minus HiFi-GAN is not robust (-0.001, not significant, and -0.005).
    Per channel, 10 of the 36 channel-pair differences are robust. Nine of them have DDSP lower, by 0.013 to 0.065 (largest:
    tongue body Y). All of these differences are small next to the distance of every gain from 1.
  - *Leakage and noise floor.* Vocos has 0.006 to 0.010 std less leakage than both others. The EMA movement under non-EMA
    edits is lowest for HiFi-GAN and highest for DDSP, in all three pairs (0.007 to 0.021 std).
  - *F0 control.* Vocos follows F0 edits 5 to 8 cents less exactly than HiFi-GAN and DDSP in both seeds. This is below the
    resolution of the pitch estimate (10 to 20 cents) but consistent with its larger F0 error in synthesis (section 5.1).
    HiFi-GAN minus DDSP passes the rule numerically (2.49 and 0.04 cents) but is far below that resolution, so it is not
    counted.
  - *Loudness control.* Vocos is 0.05 to 0.07 dB more exact than HiFi-GAN. The other two loudness pairs are not robust.
- **What this does not change.** The conclusion of section 5.2 holds for both seeds. F0 and loudness are controlled
  closely. Constant single-channel EMA offsets are reproduced to about 10 to 12 percent while natural dynamics are
  reproduced with slope near 1. The vocoders differ by a few hundredths at most. The cause is still untested, apart from
  the X-Y probes of section 5.3.
- **Not repeated for seed 1:** the principal-component probe of `controllability.md` section 6 (6 utterances, CPU). The
  paired X-Y probes are in the next section.

## Paired X-Y probes (stage `probes_xy`), seed 0 against seed 1 (added 2026-10-20)

Basis. Job 10865560 (submitted through slurm-alloc 2026-10-20; tracking issue nzxyin/Speech-Articulatory-Coding#30) ran
`probes_xy` for hifigan_s1, ddsp_s1 and vocos_s1: 192 edits each (6 sensors x 4 directions x 4 profiles x 2 signs) on the
same 50 utterances, finished 12:49 after a failed GPU start and one preemption on `preempt` (resumed from finished
chunks; a last restart on `general` found every item done). Outputs `$SV_ROOT/eval/probes_xy/<system>_s1/`. Reports: `xy_probes_s1.md` and
`xy_probes_long_s1.csv` (the seed-0 report script with `--systems hifigan_s1 ddsp_s1 vocos_s1 --matrix
natural_variation_matrix_s1.npz`), and the comparison `xy_probes_seeds.md` / `.csv` (script `scripts/xy_probes_seeds.py`,
the same speaker resamples for all six systems, so seed and vocoder differences are paired; its self-test with the
seed-0 outputs under both names, `xy_probes_seeds_selftest.md`, reproduces the seed-0 report with zero seed differences).

Reading (numbers from `xy_probes_seeds.md` unless stated):

- **Gains barely move between seeds.** Each of the 48 headline gains (16 profile-direction cells, mean over sensors, three
  vocoders) changes by 0.010 or less between the seeds.
- **Pairing still does not help.** Paired principal minus X only and minus Y only (constant offsets) lie between -0.014 and
  +0.004 in seed 1, against -0.015 and +0.004 in seed 0. The largest positive value is +0.004 in both seeds. The signs of
  these near-zero contrasts are not stable between the seeds (for HiFi-GAN principal minus X only is +0.004 in seed 0 and
  -0.007 in seed 1, both with intervals excluding 0), so the strict replication criterion of the table fails for them. The
  conclusion that a paired edit is not followed better does replicate.
- **Time course still matters.** Dynamic scaling and local bumps are followed better than constant offsets in both seeds
  (seed 1: +0.053 to +0.127 for the three vocoders, against +0.059 to +0.137 in seed 0). Restricting the offset to speech frames adds
  +0.005 to +0.015 (seed 0: +0.007 to +0.015). All six of these contrasts replicate with the same sign and intervals
  excluding 0.
- **Direction within a sensor still matters.** Under dynamic scaling, tongue body along its principal axis gives 0.55,
  0.46 and 0.55 (HiFi-GAN, DDSP, Vocos; seed 0: 0.57, 0.44, 0.54), and X alone gives 0.05, -0.02 and 0.07 (seed 0: 0.07,
  0.00, 0.06). The tongue tip follows X (0.35 to 0.39) and its orthogonal direction (0.35 to 0.36) but not its principal
  axis (0.12 to 0.13). Shrinking movement (alpha = -0.5) is followed with gain 0.27 to 0.30 and exaggerating it with 0.09
  to 0.11, as in seed 0.
- **Leakage and noise are unchanged.** Leakage into the other sensors is 0.25 to 0.49 std in seed 1 (seed 0: 0.25 to 0.49)
  against a regression prediction of 0.02 to 0.05. It exceeds the on-target gain in 91 to 95 of the 96 cells per system in
  both seeds (`xy_probes_s1.md` section 1.2, `xy_probes_long*.csv`).
- **Robust vocoder differences.**
  - HiFi-GAN follows the X-Y edits more than DDSP in all five tested cells. These are constant principal, constant X,
    dynamic principal, bumps principal and the mean of all 16 cells, with differences of 0.012 to 0.034.
  - Vocos follows them more than DDSP in four of the five (0.006 to 0.040; not constant X only).
  - HiFi-GAN against Vocos is robust only for the constant principal edit (+0.010 and +0.007). It is not robust for the
    mean of all cells (+0.001 in both seeds).
  - As for the single-channel probes, all of these differences are a few hundredths, against gains of 0.1 to 0.6.
