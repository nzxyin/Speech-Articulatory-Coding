# Paired X-Y EMA probes: seed 0 against seed 1

Basis: stage `probes_xy` for the seed-0 systems (2026-10-12) and the seed-1 systems (main_seed1, step 400,000). Utterances per system: hifigan 50, ddsp 50, vocos 50, hifigan_s1 50, ddsp_s1 50, vocos_s1 50; 28 speakers; complete runs: hifigan, ddsp, vocos, hifigan_s1, ddsp_s1, vocos_s1. Every number is a gain (1 = the edit is reproduced) in the definitions of `xy_probes.md`, pooled over the two signs and averaged over the six sensors unless a sensor is named. Intervals are speaker-cluster bootstrap CI95 (10000 resamples, seed 0) with the same resamples for all six systems, so seed and vocoder differences are paired.

## 1. Gain by profile and direction (mean over sensors)

| profile | direction | HiFi-GAN seed 0 | HiFi-GAN seed 1 | HiFi-GAN seed 1 - seed 0 | DDSP seed 0 | DDSP seed 1 | DDSP seed 1 - seed 0 | Vocos seed 0 | Vocos seed 1 | Vocos seed 1 - seed 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| constant offset | X only | 0.11 (0.11, 0.12) | 0.12 (0.12, 0.12) | +0.008 (+0.004, +0.013) | 0.09 (0.09, 0.10) | 0.10 (0.10, 0.11) | +0.009 (+0.004, +0.014) | 0.11 (0.11, 0.12) | 0.11 (0.10, 0.11) | -0.003 (-0.007, -0.000) |
| constant offset | Y only | 0.12 (0.12, 0.12) | 0.12 (0.12, 0.12) | -0.002 (-0.005, +0.000), n.s. | 0.10 (0.10, 0.10) | 0.10 (0.09, 0.10) | -0.002 (-0.004, -0.000) | 0.12 (0.12, 0.12) | 0.12 (0.12, 0.12) | +0.001 (-0.002, +0.003), n.s. |
| constant offset | principal | 0.12 (0.11, 0.12) | 0.11 (0.11, 0.12) | -0.002 (-0.005, +0.001), n.s. | 0.10 (0.10, 0.10) | 0.10 (0.10, 0.11) | +0.003 (-0.000, +0.006), n.s. | 0.11 (0.10, 0.11) | 0.11 (0.10, 0.11) | +0.001 (-0.002, +0.004), n.s. |
| constant offset | orthogonal | 0.11 (0.11, 0.12) | 0.11 (0.11, 0.12) | +0.000 (-0.002, +0.003), n.s. | 0.10 (0.10, 0.10) | 0.10 (0.09, 0.10) | -0.003 (-0.007, +0.000), n.s. | 0.11 (0.11, 0.11) | 0.11 (0.11, 0.11) | +0.000 (-0.003, +0.003), n.s. |
| constant offset on speech frames | X only | 0.12 (0.12, 0.13) | 0.13 (0.13, 0.13) | +0.006 (+0.001, +0.010) | 0.10 (0.10, 0.11) | 0.11 (0.10, 0.11) | +0.006 (+0.002, +0.011) | 0.12 (0.12, 0.13) | 0.12 (0.12, 0.12) | -0.001 (-0.005, +0.002), n.s. |
| constant offset on speech frames | Y only | 0.13 (0.13, 0.14) | 0.13 (0.13, 0.14) | -0.002 (-0.005, +0.000), n.s. | 0.11 (0.11, 0.12) | 0.11 (0.11, 0.11) | -0.002 (-0.005, +0.000), n.s. | 0.13 (0.13, 0.14) | 0.13 (0.13, 0.14) | +0.001 (-0.001, +0.003), n.s. |
| constant offset on speech frames | principal | 0.13 (0.13, 0.13) | 0.13 (0.12, 0.13) | -0.004 (-0.007, -0.001) | 0.11 (0.11, 0.12) | 0.11 (0.11, 0.12) | -0.000 (-0.004, +0.003), n.s. | 0.12 (0.12, 0.13) | 0.12 (0.12, 0.13) | +0.001 (-0.003, +0.004), n.s. |
| constant offset on speech frames | orthogonal | 0.13 (0.12, 0.13) | 0.13 (0.12, 0.13) | -0.001 (-0.003, +0.002), n.s. | 0.11 (0.11, 0.11) | 0.11 (0.10, 0.11) | -0.004 (-0.007, -0.001) | 0.12 (0.12, 0.13) | 0.12 (0.12, 0.13) | +0.002 (-0.001, +0.004), n.s. |
| dynamic scaling around the mean | X only | 0.17 (0.17, 0.18) | 0.17 (0.17, 0.18) | +0.002 (-0.002, +0.006), n.s. | 0.16 (0.15, 0.16) | 0.16 (0.15, 0.16) | -0.002 (-0.007, +0.004), n.s. | 0.18 (0.17, 0.18) | 0.18 (0.17, 0.18) | +0.000 (-0.004, +0.005), n.s. |
| dynamic scaling around the mean | Y only | 0.20 (0.20, 0.20) | 0.20 (0.19, 0.21) | +0.001 (-0.005, +0.006), n.s. | 0.19 (0.18, 0.19) | 0.19 (0.18, 0.19) | -0.002 (-0.005, +0.002), n.s. | 0.20 (0.20, 0.20) | 0.20 (0.20, 0.21) | +0.004 (+0.001, +0.006) |
| dynamic scaling around the mean | principal | 0.21 (0.21, 0.22) | 0.21 (0.21, 0.22) | -0.002 (-0.005, +0.001), n.s. | 0.19 (0.19, 0.20) | 0.19 (0.19, 0.20) | +0.002 (-0.002, +0.006), n.s. | 0.21 (0.21, 0.22) | 0.21 (0.21, 0.22) | -0.004 (-0.008, +0.001), n.s. |
| dynamic scaling around the mean | orthogonal | 0.21 (0.20, 0.22) | 0.21 (0.20, 0.21) | -0.002 (-0.006, +0.002), n.s. | 0.19 (0.19, 0.20) | 0.19 (0.19, 0.20) | -0.000 (-0.005, +0.005), n.s. | 0.21 (0.21, 0.22) | 0.21 (0.21, 0.22) | +0.003 (-0.000, +0.007), n.s. |
| three local bumps (300 ms) | X only | 0.19 (0.18, 0.20) | 0.19 (0.18, 0.21) | +0.001 (-0.009, +0.011), n.s. | 0.18 (0.16, 0.19) | 0.17 (0.16, 0.18) | -0.009 (-0.016, -0.001) | 0.20 (0.19, 0.21) | 0.21 (0.19, 0.22) | +0.003 (-0.006, +0.012), n.s. |
| three local bumps (300 ms) | Y only | 0.27 (0.25, 0.28) | 0.27 (0.25, 0.28) | +0.001 (-0.006, +0.008), n.s. | 0.24 (0.22, 0.25) | 0.24 (0.23, 0.25) | +0.007 (+0.001, +0.013) | 0.26 (0.25, 0.28) | 0.26 (0.25, 0.27) | -0.001 (-0.008, +0.004), n.s. |
| three local bumps (300 ms) | principal | 0.24 (0.23, 0.25) | 0.23 (0.22, 0.24) | -0.005 (-0.014, +0.004), n.s. | 0.20 (0.19, 0.21) | 0.20 (0.19, 0.21) | -0.001 (-0.009, +0.008), n.s. | 0.24 (0.23, 0.25) | 0.23 (0.22, 0.24) | -0.010 (-0.015, -0.004) |
| three local bumps (300 ms) | orthogonal | 0.24 (0.22, 0.25) | 0.24 (0.23, 0.25) | +0.001 (-0.005, +0.007), n.s. | 0.22 (0.21, 0.23) | 0.22 (0.20, 0.23) | -0.003 (-0.009, +0.004), n.s. | 0.23 (0.22, 0.25) | 0.24 (0.23, 0.26) | +0.008 (+0.002, +0.015) |

## 2. Contrasts of section 5.3 in each seed

Difference of gains (mean over sensors), paired CI95. *Replicates* means: same sign in both seeds and either the CI excludes 0 in both (a supported effect) or includes 0 in both (no effect in either).

| contrast | HiFi-GAN seed 0 | HiFi-GAN seed 1 | DDSP seed 0 | DDSP seed 1 | Vocos seed 0 | Vocos seed 1 | replicates |
|---|---|---|---|---|---|---|---|
| paired principal minus X only (constant) | +0.004 (+0.000, +0.008) | -0.007 (-0.011, -0.002) | +0.004 (-0.000, +0.008), n.s. | -0.002 (-0.009, +0.004), n.s. | -0.006 (-0.010, -0.003) | -0.002 (-0.006, +0.002), n.s. | no (HiFi-GAN, DDSP, Vocos) |
| paired principal minus Y only (constant) | -0.006 (-0.009, -0.002) | -0.005 (-0.009, -0.002) | -0.002 (-0.005, +0.002), n.s. | +0.004 (-0.000, +0.007), n.s. | -0.015 (-0.018, -0.011) | -0.014 (-0.018, -0.011) | no (DDSP) |
| orthogonal minus X only (constant) | +0.002 (-0.002, +0.006), n.s. | -0.006 (-0.010, -0.003) | +0.006 (+0.001, +0.010) | -0.006 (-0.012, -0.001) | -0.003 (-0.006, +0.001), n.s. | +0.001 (-0.003, +0.005), n.s. | no (HiFi-GAN, DDSP, Vocos) |
| speech-only minus constant, principal | +0.014 (+0.011, +0.017) | +0.012 (+0.010, +0.014) | +0.014 (+0.012, +0.017) | +0.011 (+0.008, +0.014) | +0.015 (+0.013, +0.018) | +0.015 (+0.012, +0.018) | yes |
| speech-only minus constant, X only | +0.012 (+0.009, +0.015) | +0.009 (+0.006, +0.012) | +0.007 (+0.004, +0.010) | +0.005 (+0.002, +0.008) | +0.010 (+0.007, +0.013) | +0.012 (+0.009, +0.014) | yes |
| dynamic scaling minus constant, principal | +0.098 (+0.091, +0.105) | +0.098 (+0.090, +0.105) | +0.093 (+0.086, +0.100) | +0.092 (+0.087, +0.097) | +0.109 (+0.103, +0.115) | +0.105 (+0.098, +0.111) | yes |
| dynamic scaling minus constant, X only | +0.059 (+0.052, +0.066) | +0.053 (+0.045, +0.061) | +0.063 (+0.057, +0.070) | +0.053 (+0.045, +0.061) | +0.064 (+0.059, +0.069) | +0.067 (+0.060, +0.075) | yes |
| bumps minus constant, principal | +0.121 (+0.111, +0.132) | +0.118 (+0.106, +0.131) | +0.105 (+0.095, +0.113) | +0.101 (+0.093, +0.109) | +0.137 (+0.127, +0.148) | +0.127 (+0.117, +0.136) | yes |
| bumps minus constant, X only | +0.081 (+0.069, +0.093) | +0.073 (+0.060, +0.087) | +0.082 (+0.068, +0.096) | +0.064 (+0.051, +0.078) | +0.091 (+0.079, +0.102) | +0.097 (+0.084, +0.110) | yes |

## 3. Direction within a sensor and sign of the dynamic scaling

Gain under dynamic scaling (alpha = plus or minus 0.5), pooled over the two signs unless stated.

| quantity | HiFi-GAN seed 0 | HiFi-GAN seed 1 | DDSP seed 0 | DDSP seed 1 | Vocos seed 0 | Vocos seed 1 |
|---|---|---|---|---|---|---|
| TB, principal | 0.57 (0.54, 0.59) | 0.55 (0.53, 0.57) | 0.44 (0.42, 0.46) | 0.46 (0.44, 0.49) | 0.54 (0.51, 0.57) | 0.55 (0.53, 0.58) |
| TB, X only | 0.07 (0.04, 0.09) | 0.05 (0.02, 0.08) | 0.00 (-0.02, 0.02) | -0.02 (-0.05, 0.01) | 0.06 (0.03, 0.09) | 0.07 (0.04, 0.10) |
| TB, Y only | 0.42 (0.40, 0.44) | 0.41 (0.39, 0.43) | 0.38 (0.36, 0.40) | 0.38 (0.37, 0.40) | 0.39 (0.37, 0.41) | 0.40 (0.38, 0.42) |
| TT, X only | 0.36 (0.34, 0.38) | 0.36 (0.34, 0.38) | 0.37 (0.36, 0.39) | 0.39 (0.37, 0.41) | 0.35 (0.33, 0.37) | 0.35 (0.33, 0.37) |
| TT, orthogonal | 0.35 (0.33, 0.37) | 0.35 (0.33, 0.38) | 0.34 (0.32, 0.36) | 0.35 (0.33, 0.38) | 0.35 (0.33, 0.38) | 0.36 (0.33, 0.38) |
| TT, principal | 0.13 (0.12, 0.14) | 0.13 (0.12, 0.14) | 0.10 (0.09, 0.11) | 0.12 (0.11, 0.13) | 0.13 (0.12, 0.14) | 0.12 (0.11, 0.13) |
| shrinking (alpha = -0.5), all sensors and directions | 0.30 (0.29, 0.30) | 0.29 (0.29, 0.30) | 0.28 (0.27, 0.28) | 0.27 (0.27, 0.28) | 0.30 (0.29, 0.30) | 0.30 (0.29, 0.31) |
| exaggerating (alpha = +0.5), all sensors and directions | 0.10 (0.10, 0.11) | 0.11 (0.10, 0.11) | 0.09 (0.08, 0.10) | 0.09 (0.08, 0.10) | 0.11 (0.10, 0.11) | 0.10 (0.10, 0.11) |

## 4. Vocoder differences (robustness rule of RESULTS.md section 4.1)

Gain difference a - b (mean over sensors), paired CI95, in each seed. *Robust*: same sign and CI excluding 0 in both seeds, and the smaller absolute difference larger than the larger absolute seed difference (seed 1 - seed 0) of the two vocoders in the same cell.

| cell | pair | seed 0 | seed 1 | larger seed difference | robust |
|---|---|---|---|---|---|
| constant, principal | HiFi-GAN - DDSP | +0.017 (+0.013, +0.021) | +0.012 (+0.009, +0.015) | 0.003 | yes |
| constant, principal | HiFi-GAN - Vocos | +0.010 (+0.007, +0.013) | +0.007 (+0.004, +0.010) | 0.002 | yes |
| constant, principal | Vocos - DDSP | +0.007 (+0.003, +0.012) | +0.006 (+0.002, +0.009) | 0.003 | yes |
| constant, X only | HiFi-GAN - DDSP | +0.017 (+0.013, +0.022) | +0.017 (+0.011, +0.023) | 0.009 | yes |
| constant, X only | HiFi-GAN - Vocos | -0.000 (-0.004, +0.004), n.s. | +0.012 (+0.008, +0.015) | 0.008 | no |
| constant, X only | Vocos - DDSP | +0.017 (+0.013, +0.021) | +0.005 (-0.001, +0.011), n.s. | 0.009 | no |
| dynamic, principal | HiFi-GAN - DDSP | +0.022 (+0.017, +0.027) | +0.019 (+0.012, +0.025) | 0.002 | yes |
| dynamic, principal | HiFi-GAN - Vocos | -0.001 (-0.006, +0.003), n.s. | +0.000 (-0.003, +0.003), n.s. | 0.004 | no |
| dynamic, principal | Vocos - DDSP | +0.024 (+0.018, +0.029) | +0.018 (+0.013, +0.024) | 0.004 | yes |
| bumps, principal | HiFi-GAN - DDSP | +0.034 (+0.027, +0.042) | +0.029 (+0.019, +0.041) | 0.005 | yes |
| bumps, principal | HiFi-GAN - Vocos | -0.006 (-0.013, +0.001), n.s. | -0.002 (-0.011, +0.007), n.s. | 0.010 | no |
| bumps, principal | Vocos - DDSP | +0.040 (+0.033, +0.047) | +0.031 (+0.022, +0.040) | 0.010 | yes |
| all 16 cells | HiFi-GAN - DDSP | +0.019 (+0.017, +0.022) | +0.019 (+0.017, +0.021) | 0.000 | yes |
| all 16 cells | HiFi-GAN - Vocos | +0.001 (-0.001, +0.003), n.s. | +0.001 (-0.001, +0.002), n.s. | 0.000 | no |
| all 16 cells | Vocos - DDSP | +0.018 (+0.016, +0.021) | +0.019 (+0.016, +0.021) | 0.000 | yes |

