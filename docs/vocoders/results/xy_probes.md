# Paired X-Y EMA probes

Question: are paired X-Y edits of one sensor necessary for the vocoders to follow EMA edits, or do they fail for another reason?

## Basis

Probe utterances: 50 test.clean utterances of 3 to 10 s from 28 speakers (the subset of the single-channel probes), T1 speaker vectors, eval gain. Each edit is applied to the 15 raw input channels of one sensor (TD = channels 0 and 1, TB = 2 and 3, TT = 4 and 5, LI = 6 and 7, UL = 8 and 9, LL = 10 and 11, as X and Y), the edited and the unedited input are synthesized with the same speaker vector and synthesis RNG, both syntheses are re-extracted (refit head, no dither), and the difference of the re-extracted EMA is compared with the difference of the input EMA. All quantities are in training-std units of the individual channels. Final checkpoints: HiFi-GAN step 400000, DDSP step 400000, Vocos step 400000. One training run per vocoder.

Directions in the sensor plane (unit vectors in z-scored coordinates, so all constant edits have the same Euclidean norm of one training std):

- X only: (1, 0); Y only: (0, 1);
- principal: the leading eigenvector of the 2 x 2 covariance of the two z-scored channels on training frames (train.clean.100, every 20th frame), sign so that the larger component is positive;
- orthogonal: the principal direction rotated by +90 degrees (the minor axis).

| sensor | principal direction (X, Y) | angle from X (deg) | variance principal / orthogonal |
|---|---|---|---|
| TD | (+0.72, -0.70) | -44 | 1.36 / 0.66 |
| TB | (+0.72, -0.69) | -44 | 1.42 / 0.59 |
| TT | (+0.77, -0.63) | -39 | 1.11 / 0.92 |
| LI | (+0.71, -0.70) | -45 | 1.83 / 0.18 |
| UL | (+0.70, +0.71) | 45 | 1.57 / 0.43 |
| LL | (+0.71, -0.71) | -45 | 1.83 / 0.19 |

Temporal profiles of the edit along the direction (the signed amplitude is `p(t)`; the edit adds `p(t)` times the unit direction):

- **constant offset** (`const`): plus or minus 1 on every frame, the protocol of the single-channel and principal-component probes;
- **constant offset on speech frames** (`speech`): plus or minus 1 on frames with loudness above 20 dB below the utterance 95th percentile (gaps up to 100 ms closed), with a 100 ms Hann ramp; silence and pauses are not edited;
- **dynamic scaling** (`dyn`): the component of the input along the direction is scaled by 1 + alpha around its utterance mean, alpha = plus or minus 0.5; the utterance mean is kept exactly, the component orthogonal to the direction and all other channels are untouched;
- **local bumps** (`bump`): plus or minus 1 times a 300 ms Hann window, three windows per utterance centred at 20, 50 and 80 percent of the speech frames.

Metrics, per edit and utterance, pooled over utterances (ratio of sums) and over the two signs. With `dt` the change of the input EMA and `dm` the change of the re-extracted EMA (12 channels, std units):

- **gain** = sum over frames of the inner product of `dm` and `dt`, divided by the sum of |dt|^2; 1 means the edit is reproduced, 0 means no response;
- **angle** = the mean rotation (degrees, counter-clockwise in the X-Y plane) of the measured 2-D displacement of the sensor against the target, atan2 of the cross and dot sums; 0 means the direction is followed;
- **leakage** = root-mean-square of `dm` over frames and the 10 channels of the other five sensors, divided by the RMS norm of `dt` (both over all frames of the utterance). The leakage of an edit that the vocoder ignores is not zero: the re-extracted EMA of two different syntheses differs by a noise floor (0.07 to 0.20 std for edits that do not touch the EMA, RESULTS.md section 5.2), so leakage values near that level carry no information about the sensor.

Intervals are speaker-cluster bootstrap CI95 (28 speakers, 10000 resamples, shared across cells so that contrasts are paired). Aggregates over sensors or directions are the mean of the per-sensor (per-cell) values with equal weight. `/ p` after an interval is the regression prediction: the linear map fitted on natural dynamics (the 12 x 12 centred regression of RESULTS.md section 5.2, per vocoder) applied to the same targets; it is what a vocoder-plus-inversion loop that treats an edit like natural articulation would give. The prediction ignores noise and uses the centred-regression matrix for constant offsets too; the between-utterance slope for utterance means of natural speech is 0.90 to 0.94.

## 1. Headline: profile and direction (mean over the six sensors)

### 1.1 Gain (1 = followed)

| profile | direction | HiFi-GAN measured (CI95) / prediction | DDSP measured (CI95) / prediction | Vocos measured (CI95) / prediction |
|---|---|---|---|---|
| constant offset | X only | 0.11 (0.11, 0.12) / 0.98 | 0.09 (0.09, 0.10) / 0.93 | 0.11 (0.11, 0.12) / 0.97 |
| constant offset | Y only | 0.12 (0.12, 0.12) / 0.98 | 0.10 (0.10, 0.10) / 0.93 | 0.12 (0.12, 0.12) / 0.98 |
| constant offset | principal | 0.12 (0.11, 0.12) / 0.98 | 0.10 (0.10, 0.10) / 0.93 | 0.11 (0.10, 0.11) / 0.97 |
| constant offset | orthogonal | 0.11 (0.11, 0.12) / 0.98 | 0.10 (0.10, 0.10) / 0.93 | 0.11 (0.11, 0.11) / 0.98 |
| constant offset on speech frames | X only | 0.12 (0.12, 0.13) / 0.98 | 0.10 (0.10, 0.11) / 0.93 | 0.12 (0.12, 0.13) / 0.97 |
| constant offset on speech frames | Y only | 0.13 (0.13, 0.14) / 0.98 | 0.11 (0.11, 0.12) / 0.93 | 0.13 (0.13, 0.14) / 0.98 |
| constant offset on speech frames | principal | 0.13 (0.13, 0.13) / 0.98 | 0.11 (0.11, 0.12) / 0.93 | 0.12 (0.12, 0.13) / 0.97 |
| constant offset on speech frames | orthogonal | 0.13 (0.12, 0.13) / 0.98 | 0.11 (0.11, 0.11) / 0.93 | 0.12 (0.12, 0.13) / 0.98 |
| dynamic scaling around the mean | X only | 0.17 (0.17, 0.18) / 0.98 | 0.16 (0.15, 0.16) / 0.93 | 0.18 (0.17, 0.18) / 0.97 |
| dynamic scaling around the mean | Y only | 0.20 (0.20, 0.20) / 0.98 | 0.19 (0.18, 0.19) / 0.93 | 0.20 (0.20, 0.20) / 0.98 |
| dynamic scaling around the mean | principal | 0.21 (0.21, 0.22) / 0.98 | 0.19 (0.19, 0.20) / 0.93 | 0.21 (0.21, 0.22) / 0.97 |
| dynamic scaling around the mean | orthogonal | 0.21 (0.20, 0.22) / 0.98 | 0.19 (0.19, 0.20) / 0.93 | 0.21 (0.21, 0.22) / 0.98 |
| three local bumps (300 ms) | X only | 0.19 (0.18, 0.20) / 0.98 | 0.18 (0.16, 0.19) / 0.93 | 0.20 (0.19, 0.21) / 0.97 |
| three local bumps (300 ms) | Y only | 0.27 (0.25, 0.28) / 0.98 | 0.24 (0.22, 0.25) / 0.93 | 0.26 (0.25, 0.28) / 0.98 |
| three local bumps (300 ms) | principal | 0.24 (0.23, 0.25) / 0.98 | 0.20 (0.19, 0.21) / 0.93 | 0.24 (0.23, 0.25) / 0.97 |
| three local bumps (300 ms) | orthogonal | 0.24 (0.22, 0.25) / 0.98 | 0.22 (0.20, 0.23) / 0.93 | 0.23 (0.22, 0.25) / 0.98 |

### 1.2 Leakage into the other sensors (std)

| profile | direction | HiFi-GAN measured (CI95) / prediction | DDSP measured (CI95) / prediction | Vocos measured (CI95) / prediction |
|---|---|---|---|---|
| constant offset | X only | 0.38 (0.37, 0.38) / 0.04 | 0.39 (0.38, 0.40) / 0.05 | 0.37 (0.36, 0.37) / 0.04 |
| constant offset | Y only | 0.26 (0.26, 0.27) / 0.02 | 0.25 (0.25, 0.26) / 0.03 | 0.26 (0.25, 0.26) / 0.02 |
| constant offset | principal | 0.32 (0.32, 0.33) / 0.03 | 0.34 (0.34, 0.34) / 0.03 | 0.31 (0.31, 0.32) / 0.03 |
| constant offset | orthogonal | 0.33 (0.33, 0.34) / 0.03 | 0.33 (0.33, 0.34) / 0.05 | 0.33 (0.32, 0.33) / 0.03 |
| constant offset on speech frames | X only | 0.38 (0.37, 0.39) / 0.04 | 0.40 (0.40, 0.41) / 0.05 | 0.38 (0.37, 0.38) / 0.04 |
| constant offset on speech frames | Y only | 0.26 (0.26, 0.27) / 0.02 | 0.26 (0.25, 0.27) / 0.03 | 0.26 (0.25, 0.27) / 0.02 |
| constant offset on speech frames | principal | 0.33 (0.32, 0.34) / 0.03 | 0.35 (0.35, 0.36) / 0.03 | 0.32 (0.31, 0.33) / 0.03 |
| constant offset on speech frames | orthogonal | 0.34 (0.33, 0.34) / 0.03 | 0.34 (0.34, 0.35) / 0.05 | 0.33 (0.32, 0.34) / 0.03 |
| dynamic scaling around the mean | X only | 0.47 (0.46, 0.48) / 0.04 | 0.49 (0.48, 0.50) / 0.05 | 0.47 (0.46, 0.48) / 0.04 |
| dynamic scaling around the mean | Y only | 0.32 (0.31, 0.33) / 0.02 | 0.33 (0.33, 0.34) / 0.03 | 0.32 (0.32, 0.33) / 0.02 |
| dynamic scaling around the mean | principal | 0.38 (0.38, 0.39) / 0.03 | 0.40 (0.39, 0.41) / 0.03 | 0.38 (0.38, 0.39) / 0.03 |
| dynamic scaling around the mean | orthogonal | 0.45 (0.44, 0.46) / 0.03 | 0.47 (0.46, 0.48) / 0.05 | 0.46 (0.45, 0.47) / 0.03 |
| three local bumps (300 ms) | X only | 0.47 (0.45, 0.48) / 0.04 | 0.49 (0.48, 0.50) / 0.05 | 0.47 (0.46, 0.49) / 0.04 |
| three local bumps (300 ms) | Y only | 0.32 (0.31, 0.33) / 0.02 | 0.34 (0.33, 0.35) / 0.03 | 0.32 (0.31, 0.33) / 0.02 |
| three local bumps (300 ms) | principal | 0.40 (0.39, 0.41) / 0.03 | 0.42 (0.41, 0.42) / 0.03 | 0.41 (0.40, 0.42) / 0.03 |
| three local bumps (300 ms) | orthogonal | 0.40 (0.39, 0.42) / 0.03 | 0.43 (0.42, 0.44) / 0.05 | 0.41 (0.40, 0.42) / 0.03 |

### 1.3 Angle between measured and target 2-D displacement (degrees)

| profile | direction | HiFi-GAN measured (CI95) | DDSP measured (CI95) | Vocos measured (CI95) |
|---|---|---|---|---|
| constant offset | X only | -15 (-16, -13) | -19 (-22, -17) | -17 (-19, -15) |
| constant offset | Y only | -14 (-16, -11) | -6 (-9, -3) | -10 (-12, -9) |
| constant offset | principal | 8 (7, 10) | 6 (5, 7) | 6 (4, 7) |
| constant offset | orthogonal | 13 (10, 15) | 15 (13, 18) | 9 (7, 11) |
| constant offset on speech frames | X only | -9 (-11, -7) | -14 (-16, -12) | -12 (-14, -9) |
| constant offset on speech frames | Y only | -13 (-16, -10) | -7 (-10, -5) | -10 (-12, -8) |
| constant offset on speech frames | principal | 10 (8, 13) | 8 (7, 10) | 7 (6, 9) |
| constant offset on speech frames | orthogonal | 15 (13, 18) | 18 (16, 21) | 14 (12, 16) |
| dynamic scaling around the mean | X only | -1 (-4, 1) | -3 (-6, 1) | -3 (-5, 0) |
| dynamic scaling around the mean | Y only | 3 (1, 4) | 2 (0, 3) | 2 (1, 3) |
| dynamic scaling around the mean | principal | -4 (-6, -1) | -2 (-4, 0) | -2 (-4, 0) |
| dynamic scaling around the mean | orthogonal | 18 (16, 20) | 22 (21, 24) | 17 (15, 18) |
| three local bumps (300 ms) | X only | 3 (-1, 7) | 9 (4, 14) | -1 (-5, 2) |
| three local bumps (300 ms) | Y only | 5 (2, 7) | 3 (1, 6) | 5 (3, 7) |
| three local bumps (300 ms) | principal | -2 (-7, 6) | 1 (-22, 23) | 1 (-4, 8) |
| three local bumps (300 ms) | orthogonal | 19 (17, 22) | 21 (19, 23) | 17 (15, 20) |

### 1.4 Size of the edits

Target RMS norm of `dt` over all frames (training std; median over rows). A constant edit has norm 1.0; speech-only and bump edits are smaller because they cover part of the frames; dynamic scaling is a fraction of the natural dynamics.

| profile | X only | Y only | principal | orthogonal |
|---|---|---|---|---|
| constant offset | 1.00 | 1.00 | 1.00 | 1.00 |
| constant offset on speech frames | 0.92 | 0.92 | 0.92 | 0.92 |
| dynamic scaling around the mean | 0.50 | 0.50 | 0.61 | 0.35 |
| three local bumps (300 ms) | 0.26 | 0.26 | 0.26 | 0.26 |

## 2. Contrasts that answer the question

Differences of gains (mean over sensors, pooled over signs), with paired speaker-cluster bootstrap CI95. The regression prediction of the same difference is after the slash.

| contrast | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| paired principal minus X only (constant) | 0.004 (0.000, 0.008) / -0.000 | 0.004 (-0.000, 0.008) / 0.002 | -0.006 (-0.010, -0.003) / -0.000 |
| paired principal minus Y only (constant) | -0.006 (-0.009, -0.002) / -0.006 | -0.002 (-0.005, 0.002) / -0.002 | -0.015 (-0.018, -0.011) / -0.007 |
| orthogonal minus X only (constant) | 0.002 (-0.002, 0.006) / 0.006 | 0.006 (0.001, 0.010) / 0.002 | -0.003 (-0.006, 0.001) / 0.007 |
| speech-only minus constant, principal | 0.014 (0.011, 0.017) / -0.000 | 0.014 (0.012, 0.017) / -0.000 | 0.015 (0.013, 0.018) / -0.000 |
| speech-only minus constant, X only | 0.012 (0.009, 0.015) / 0.000 | 0.007 (0.004, 0.010) / 0.000 | 0.010 (0.007, 0.013) / 0.000 |
| dynamic scaling minus constant, principal | 0.098 (0.091, 0.105) / -0.000 | 0.093 (0.086, 0.100) / -0.000 | 0.109 (0.103, 0.115) / -0.000 |
| dynamic scaling minus constant, X only | 0.059 (0.052, 0.066) / 0.000 | 0.063 (0.057, 0.070) / 0.000 | 0.064 (0.059, 0.069) / 0.000 |
| bumps minus constant, principal | 0.121 (0.111, 0.132) / -0.000 | 0.105 (0.095, 0.113) / -0.000 | 0.137 (0.127, 0.148) / -0.000 |
| bumps minus constant, X only | 0.081 (0.069, 0.093) / 0.000 | 0.082 (0.068, 0.096) / 0.000 | 0.091 (0.079, 0.103) / 0.000 |

## 3. Where the edit is followed: speech frames against silence and pauses

Gain computed on the speech frames only and on the other frames only (frames in the speech mask of the unedited input; mean over sensors and directions, pooled over signs). Edits of the speech-only and bump profiles are zero outside the speech mask, so their 'other frames' gain is undefined or reflects ramps only.

| profile | frames | HiFi-GAN gain | DDSP gain | Vocos gain |
|---|---|---|---|---|
| constant offset | speech frames | 0.12 (0.11, 0.12) / 0.98 | 0.10 (0.10, 0.10) / 0.93 | 0.11 (0.11, 0.11) / 0.97 |
| constant offset | other frames | 0.12 (0.11, 0.12) / 0.98 | 0.10 (0.10, 0.11) / 0.93 | 0.11 (0.11, 0.12) / 0.97 |
| constant offset on speech frames | speech frames | 0.13 (0.13, 0.13) / 0.98 | 0.11 (0.11, 0.11) / 0.93 | 0.12 (0.12, 0.13) / 0.97 |
| constant offset on speech frames | other frames | -0.05 (-0.07, -0.03) / 0.98 | -0.06 (-0.07, -0.04) / 0.93 | -0.05 (-0.06, -0.04) / 0.97 |
| dynamic scaling around the mean | speech frames | 0.20 (0.20, 0.20) / 0.98 | 0.18 (0.18, 0.19) / 0.93 | 0.20 (0.20, 0.20) / 0.97 |
| dynamic scaling around the mean | other frames | 0.19 (0.18, 0.20) / 0.98 | 0.17 (0.17, 0.18) / 0.93 | 0.20 (0.19, 0.21) / 0.97 |
| three local bumps (300 ms) | speech frames | 0.23 (0.23, 0.24) / 0.98 | 0.21 (0.20, 0.22) / 0.93 | 0.24 (0.23, 0.25) / 0.97 |
| three local bumps (300 ms) | other frames | 0.21 (0.19, 0.24) / 0.98 | 0.19 (0.17, 0.22) / 0.93 | 0.21 (0.20, 0.22) / 0.97 |

## 4. By sensor

### 4.1 Gain, constant offset (measured with CI95 / regression prediction)

| sensor | direction | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|---|
| TD | X only | 0.26 (0.25, 0.27) / 0.94 | 0.24 (0.22, 0.26) / 0.88 | 0.29 (0.27, 0.30) / 0.94 |
| TD | Y only | 0.05 (0.04, 0.06) / 1.00 | 0.05 (0.03, 0.06) / 0.94 | 0.07 (0.06, 0.08) / 0.98 |
| TD | principal | 0.12 (0.11, 0.14) / 0.95 | 0.15 (0.13, 0.16) / 0.86 | 0.15 (0.13, 0.17) / 0.94 |
| TD | orthogonal | 0.18 (0.17, 0.20) / 0.99 | 0.14 (0.13, 0.15) / 0.96 | 0.18 (0.17, 0.19) / 0.99 |
| TB | X only | -0.14 (-0.15, -0.13) / 0.99 | -0.16 (-0.18, -0.14) / 0.95 | -0.16 (-0.17, -0.14) / 0.97 |
| TB | Y only | 0.33 (0.31, 0.34) / 1.00 | 0.26 (0.24, 0.27) / 0.92 | 0.30 (0.29, 0.32) / 1.00 |
| TB | principal | 0.24 (0.22, 0.26) / 0.96 | 0.16 (0.14, 0.18) / 0.94 | 0.16 (0.14, 0.17) / 0.97 |
| TB | orthogonal | -0.09 (-0.11, -0.08) / 1.03 | -0.07 (-0.09, -0.05) / 0.92 | -0.10 (-0.11, -0.09) / 1.00 |
| TT | X only | 0.25 (0.23, 0.28) / 1.02 | 0.25 (0.24, 0.27) / 0.99 | 0.25 (0.23, 0.27) / 1.02 |
| TT | Y only | 0.13 (0.12, 0.14) / 0.96 | 0.11 (0.10, 0.13) / 0.94 | 0.13 (0.11, 0.14) / 0.96 |
| TT | principal | 0.07 (0.06, 0.09) / 1.00 | 0.06 (0.04, 0.07) / 0.98 | 0.07 (0.06, 0.08) / 0.99 |
| TT | orthogonal | 0.35 (0.33, 0.36) / 0.98 | 0.33 (0.31, 0.34) / 0.96 | 0.33 (0.31, 0.34) / 1.00 |
| LI | X only | 0.11 (0.10, 0.12) / 0.93 | 0.07 (0.06, 0.08) / 0.86 | 0.10 (0.09, 0.10) / 0.91 |
| LI | Y only | 0.08 (0.08, 0.09) / 1.01 | 0.07 (0.07, 0.08) / 0.93 | 0.09 (0.08, 0.10) / 1.01 |
| LI | principal | 0.11 (0.10, 0.12) / 0.99 | 0.09 (0.08, 0.09) / 0.92 | 0.09 (0.08, 0.10) / 0.98 |
| LI | orthogonal | 0.09 (0.08, 0.10) / 0.95 | 0.07 (0.06, 0.07) / 0.87 | 0.10 (0.09, 0.11) / 0.94 |
| UL | X only | 0.06 (0.06, 0.07) / 1.00 | 0.06 (0.05, 0.07) / 0.95 | 0.06 (0.05, 0.07) / 0.99 |
| UL | Y only | 0.05 (0.05, 0.06) / 0.97 | 0.04 (0.04, 0.05) / 0.93 | 0.06 (0.05, 0.06) / 0.96 |
| UL | principal | 0.06 (0.06, 0.07) / 1.00 | 0.05 (0.05, 0.06) / 0.95 | 0.06 (0.06, 0.07) / 0.98 |
| UL | orthogonal | 0.06 (0.05, 0.06) / 0.98 | 0.05 (0.05, 0.06) / 0.92 | 0.06 (0.05, 0.06) / 0.96 |
| LL | X only | 0.12 (0.11, 0.14) / 0.98 | 0.11 (0.10, 0.12) / 0.94 | 0.14 (0.13, 0.14) / 0.99 |
| LL | Y only | 0.09 (0.08, 0.09) / 0.97 | 0.07 (0.06, 0.07) / 0.93 | 0.08 (0.07, 0.09) / 0.96 |
| LL | principal | 0.08 (0.08, 0.09) / 0.97 | 0.09 (0.08, 0.10) / 0.93 | 0.10 (0.09, 0.11) / 0.97 |
| LL | orthogonal | 0.10 (0.09, 0.11) / 0.98 | 0.08 (0.08, 0.09) / 0.94 | 0.09 (0.09, 0.10) / 0.98 |

### 4.2 Gain, constant offset on speech frames (measured with CI95 / regression prediction)

| sensor | direction | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|---|
| TD | X only | 0.25 (0.24, 0.27) / 0.94 | 0.23 (0.21, 0.25) / 0.88 | 0.29 (0.28, 0.30) / 0.94 |
| TD | Y only | 0.04 (0.03, 0.05) / 1.00 | 0.04 (0.03, 0.06) / 0.94 | 0.06 (0.04, 0.07) / 0.98 |
| TD | principal | 0.09 (0.07, 0.11) / 0.95 | 0.11 (0.09, 0.12) / 0.86 | 0.13 (0.11, 0.15) / 0.94 |
| TD | orthogonal | 0.20 (0.18, 0.21) / 0.99 | 0.16 (0.15, 0.17) / 0.96 | 0.20 (0.19, 0.21) / 0.99 |
| TB | X only | -0.09 (-0.11, -0.08) / 0.99 | -0.13 (-0.16, -0.11) / 0.95 | -0.13 (-0.15, -0.11) / 0.97 |
| TB | Y only | 0.38 (0.37, 0.40) / 1.00 | 0.30 (0.29, 0.32) / 0.92 | 0.35 (0.34, 0.37) / 1.00 |
| TB | principal | 0.33 (0.31, 0.35) / 0.96 | 0.25 (0.24, 0.27) / 0.94 | 0.24 (0.23, 0.26) / 0.97 |
| TB | orthogonal | -0.06 (-0.07, -0.05) / 1.03 | -0.04 (-0.05, -0.02) / 0.92 | -0.07 (-0.08, -0.06) / 1.00 |
| TT | X only | 0.27 (0.25, 0.29) / 1.02 | 0.26 (0.24, 0.27) / 0.99 | 0.26 (0.24, 0.28) / 1.02 |
| TT | Y only | 0.14 (0.13, 0.15) / 0.96 | 0.11 (0.10, 0.12) / 0.94 | 0.14 (0.12, 0.15) / 0.96 |
| TT | principal | 0.08 (0.07, 0.10) / 1.00 | 0.06 (0.05, 0.08) / 0.98 | 0.08 (0.07, 0.09) / 0.99 |
| TT | orthogonal | 0.36 (0.35, 0.37) / 0.98 | 0.32 (0.31, 0.34) / 0.96 | 0.33 (0.32, 0.35) / 1.00 |
| LI | X only | 0.09 (0.08, 0.10) / 0.93 | 0.07 (0.06, 0.08) / 0.86 | 0.09 (0.08, 0.09) / 0.91 |
| LI | Y only | 0.09 (0.08, 0.10) / 1.01 | 0.08 (0.07, 0.09) / 0.93 | 0.10 (0.09, 0.11) / 1.01 |
| LI | principal | 0.11 (0.10, 0.12) / 0.99 | 0.10 (0.09, 0.11) / 0.92 | 0.09 (0.09, 0.10) / 0.98 |
| LI | orthogonal | 0.08 (0.08, 0.09) / 0.95 | 0.07 (0.06, 0.08) / 0.87 | 0.09 (0.09, 0.10) / 0.94 |
| UL | X only | 0.09 (0.09, 0.10) / 1.00 | 0.09 (0.08, 0.10) / 0.95 | 0.09 (0.08, 0.09) / 0.99 |
| UL | Y only | 0.06 (0.05, 0.07) / 0.97 | 0.05 (0.05, 0.06) / 0.93 | 0.06 (0.06, 0.07) / 0.96 |
| UL | principal | 0.08 (0.07, 0.09) / 1.00 | 0.08 (0.07, 0.08) / 0.95 | 0.08 (0.07, 0.09) / 0.98 |
| UL | orthogonal | 0.07 (0.07, 0.08) / 0.98 | 0.07 (0.06, 0.07) / 0.92 | 0.07 (0.07, 0.08) / 0.96 |
| LL | X only | 0.13 (0.12, 0.14) / 0.98 | 0.10 (0.09, 0.11) / 0.94 | 0.14 (0.13, 0.14) / 0.99 |
| LL | Y only | 0.09 (0.09, 0.10) / 0.97 | 0.08 (0.07, 0.09) / 0.93 | 0.09 (0.08, 0.10) / 0.96 |
| LL | principal | 0.08 (0.07, 0.09) / 0.97 | 0.08 (0.07, 0.09) / 0.93 | 0.10 (0.09, 0.11) / 0.97 |
| LL | orthogonal | 0.12 (0.11, 0.12) / 0.98 | 0.09 (0.09, 0.10) / 0.94 | 0.11 (0.10, 0.12) / 0.98 |

### 4.3 Gain, dynamic scaling around the mean (measured with CI95 / regression prediction)

| sensor | direction | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|---|
| TD | X only | 0.09 (0.08, 0.11) / 0.94 | 0.15 (0.13, 0.16) / 0.88 | 0.13 (0.11, 0.16) / 0.94 |
| TD | Y only | 0.13 (0.12, 0.14) / 1.00 | 0.11 (0.10, 0.12) / 0.94 | 0.14 (0.13, 0.15) / 0.98 |
| TD | principal | 0.07 (0.05, 0.09) / 0.95 | 0.09 (0.07, 0.12) / 0.86 | 0.08 (0.06, 0.10) / 0.94 |
| TD | orthogonal | 0.16 (0.14, 0.17) / 0.99 | 0.16 (0.14, 0.17) / 0.96 | 0.17 (0.16, 0.18) / 0.99 |
| TB | X only | 0.07 (0.04, 0.09) / 0.99 | 0.00 (-0.02, 0.02) / 0.95 | 0.06 (0.03, 0.09) / 0.97 |
| TB | Y only | 0.42 (0.40, 0.44) / 1.00 | 0.38 (0.36, 0.40) / 0.92 | 0.39 (0.37, 0.41) / 1.00 |
| TB | principal | 0.57 (0.54, 0.59) / 0.96 | 0.44 (0.42, 0.46) / 0.94 | 0.54 (0.51, 0.57) / 0.97 |
| TB | orthogonal | 0.23 (0.21, 0.25) / 1.03 | 0.19 (0.17, 0.21) / 0.92 | 0.21 (0.18, 0.23) / 1.00 |
| TT | X only | 0.36 (0.34, 0.38) / 1.02 | 0.37 (0.36, 0.39) / 0.99 | 0.35 (0.33, 0.37) / 1.02 |
| TT | Y only | 0.17 (0.16, 0.19) / 0.96 | 0.14 (0.13, 0.15) / 0.94 | 0.17 (0.16, 0.18) / 0.96 |
| TT | principal | 0.13 (0.12, 0.14) / 1.00 | 0.10 (0.09, 0.11) / 0.98 | 0.13 (0.12, 0.14) / 0.99 |
| TT | orthogonal | 0.35 (0.33, 0.37) / 0.98 | 0.34 (0.32, 0.36) / 0.96 | 0.35 (0.33, 0.38) / 1.00 |
| LI | X only | 0.07 (0.06, 0.08) / 0.93 | 0.04 (0.03, 0.05) / 0.86 | 0.06 (0.05, 0.08) / 0.91 |
| LI | Y only | 0.20 (0.19, 0.21) / 1.01 | 0.20 (0.19, 0.22) / 0.93 | 0.20 (0.19, 0.21) / 1.01 |
| LI | principal | 0.14 (0.13, 0.16) / 0.99 | 0.12 (0.11, 0.14) / 0.92 | 0.14 (0.12, 0.15) / 0.98 |
| LI | orthogonal | 0.14 (0.13, 0.15) / 0.95 | 0.13 (0.12, 0.14) / 0.87 | 0.15 (0.14, 0.16) / 0.94 |
| UL | X only | 0.20 (0.19, 0.21) / 1.00 | 0.20 (0.19, 0.21) / 0.95 | 0.20 (0.19, 0.21) / 0.99 |
| UL | Y only | 0.11 (0.10, 0.13) / 0.97 | 0.12 (0.11, 0.13) / 0.93 | 0.13 (0.11, 0.14) / 0.96 |
| UL | principal | 0.12 (0.11, 0.13) / 1.00 | 0.15 (0.14, 0.16) / 0.95 | 0.13 (0.12, 0.14) / 0.98 |
| UL | orthogonal | 0.19 (0.18, 0.20) / 0.98 | 0.17 (0.16, 0.18) / 0.92 | 0.20 (0.19, 0.21) / 0.96 |
| LL | X only | 0.24 (0.22, 0.25) / 0.98 | 0.18 (0.17, 0.20) / 0.94 | 0.25 (0.23, 0.26) / 0.99 |
| LL | Y only | 0.17 (0.15, 0.18) / 0.97 | 0.19 (0.17, 0.20) / 0.93 | 0.17 (0.16, 0.19) / 0.96 |
| LL | principal | 0.25 (0.24, 0.27) / 0.97 | 0.24 (0.22, 0.25) / 0.93 | 0.27 (0.26, 0.29) / 0.97 |
| LL | orthogonal | 0.19 (0.18, 0.20) / 0.98 | 0.17 (0.16, 0.18) / 0.94 | 0.19 (0.18, 0.20) / 0.98 |

### 4.4 Gain, three local bumps (300 ms) (measured with CI95 / regression prediction)

| sensor | direction | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|---|
| TD | X only | 0.13 (0.10, 0.17) / 0.94 | 0.11 (0.08, 0.14) / 0.88 | 0.17 (0.14, 0.21) / 0.94 |
| TD | Y only | 0.19 (0.16, 0.22) / 1.00 | 0.16 (0.13, 0.19) / 0.94 | 0.21 (0.18, 0.24) / 0.98 |
| TD | principal | 0.06 (0.02, 0.10) / 0.95 | 0.02 (-0.02, 0.06) / 0.86 | 0.07 (0.03, 0.10) / 0.94 |
| TD | orthogonal | 0.25 (0.22, 0.29) / 0.99 | 0.24 (0.21, 0.28) / 0.96 | 0.28 (0.25, 0.32) / 0.99 |
| TB | X only | 0.08 (0.03, 0.13) / 0.99 | 0.09 (0.05, 0.13) / 0.95 | 0.05 (0.00, 0.09) / 0.97 |
| TB | Y only | 0.61 (0.57, 0.66) / 1.00 | 0.51 (0.47, 0.55) / 0.92 | 0.57 (0.53, 0.62) / 1.00 |
| TB | principal | 0.55 (0.50, 0.60) / 0.96 | 0.43 (0.38, 0.48) / 0.94 | 0.53 (0.48, 0.58) / 0.97 |
| TB | orthogonal | 0.24 (0.18, 0.31) / 1.03 | 0.21 (0.15, 0.28) / 0.92 | 0.18 (0.11, 0.26) / 1.00 |
| TT | X only | 0.42 (0.39, 0.45) / 1.02 | 0.43 (0.39, 0.47) / 0.99 | 0.44 (0.40, 0.47) / 1.02 |
| TT | Y only | 0.20 (0.16, 0.23) / 0.96 | 0.16 (0.12, 0.20) / 0.94 | 0.21 (0.18, 0.25) / 0.96 |
| TT | principal | 0.21 (0.18, 0.23) / 1.00 | 0.19 (0.17, 0.22) / 0.98 | 0.24 (0.22, 0.26) / 0.99 |
| TT | orthogonal | 0.41 (0.37, 0.44) / 0.98 | 0.38 (0.34, 0.42) / 0.96 | 0.39 (0.36, 0.43) / 1.00 |
| LI | X only | 0.11 (0.09, 0.14) / 0.93 | 0.07 (0.05, 0.09) / 0.86 | 0.11 (0.09, 0.13) / 0.91 |
| LI | Y only | 0.24 (0.22, 0.27) / 1.01 | 0.21 (0.18, 0.23) / 0.93 | 0.22 (0.20, 0.24) / 1.01 |
| LI | principal | 0.20 (0.18, 0.22) / 0.99 | 0.16 (0.14, 0.19) / 0.92 | 0.19 (0.17, 0.22) / 0.98 |
| LI | orthogonal | 0.15 (0.13, 0.16) / 0.95 | 0.12 (0.11, 0.14) / 0.87 | 0.15 (0.13, 0.16) / 0.94 |
| UL | X only | 0.25 (0.22, 0.28) / 1.00 | 0.23 (0.21, 0.26) / 0.95 | 0.24 (0.21, 0.27) / 0.99 |
| UL | Y only | 0.12 (0.10, 0.14) / 0.97 | 0.13 (0.12, 0.15) / 0.93 | 0.13 (0.11, 0.15) / 0.96 |
| UL | principal | 0.17 (0.15, 0.20) / 1.00 | 0.19 (0.17, 0.21) / 0.95 | 0.16 (0.14, 0.19) / 0.98 |
| UL | orthogonal | 0.19 (0.17, 0.21) / 0.98 | 0.17 (0.15, 0.19) / 0.92 | 0.20 (0.18, 0.21) / 0.96 |
| LL | X only | 0.16 (0.13, 0.20) / 0.98 | 0.13 (0.10, 0.16) / 0.94 | 0.21 (0.17, 0.24) / 0.99 |
| LL | Y only | 0.23 (0.21, 0.25) / 0.97 | 0.24 (0.22, 0.26) / 0.93 | 0.24 (0.22, 0.26) / 0.96 |
| LL | principal | 0.22 (0.19, 0.25) / 0.97 | 0.22 (0.18, 0.25) / 0.93 | 0.26 (0.22, 0.30) / 0.97 |
| LL | orthogonal | 0.19 (0.17, 0.21) / 0.98 | 0.18 (0.16, 0.20) / 0.94 | 0.21 (0.18, 0.23) / 0.98 |

### 4.5 Leakage and angle by sensor (constant offset and dynamic scaling; leakage, angle in degrees)

| profile | sensor | direction | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|---|---|
| constant offset | TD | X only | 0.48 (pred 0.05); 2 deg | 0.51 (pred 0.05); -3 deg | 0.48 (pred 0.04); 1 deg |
| constant offset | TD | Y only | 0.30 (pred 0.02); -43 deg | 0.29 (pred 0.02); -26 deg | 0.29 (pred 0.02); -22 deg |
| constant offset | TD | principal | 0.45 (pred 0.04); 37 deg | 0.49 (pred 0.03); 31 deg | 0.44 (pred 0.03); 32 deg |
| constant offset | TD | orthogonal | 0.36 (pred 0.04); -39 deg | 0.36 (pred 0.04); -39 deg | 0.34 (pred 0.03); -39 deg |
| constant offset | TB | X only | 0.60 (pred 0.06); -121 deg | 0.61 (pred 0.07); -139 deg | 0.57 (pred 0.07); -133 deg |
| constant offset | TB | Y only | 0.38 (pred 0.02); -1 deg | 0.36 (pred 0.04); -6 deg | 0.36 (pred 0.02); 0 deg |
| constant offset | TB | principal | 0.54 (pred 0.04); -52 deg | 0.59 (pred 0.04); -61 deg | 0.52 (pred 0.04); -62 deg |
| constant offset | TB | orthogonal | 0.51 (pred 0.04); 132 deg | 0.51 (pred 0.06); 123 deg | 0.49 (pred 0.05); 129 deg |
| constant offset | TT | X only | 0.38 (pred 0.03); 42 deg | 0.39 (pred 0.04); 42 deg | 0.36 (pred 0.04); 39 deg |
| constant offset | TT | Y only | 0.29 (pred 0.01); -22 deg | 0.29 (pred 0.03); -11 deg | 0.28 (pred 0.02); -21 deg |
| constant offset | TT | principal | 0.34 (pred 0.03); 67 deg | 0.36 (pred 0.04); 73 deg | 0.33 (pred 0.03); 64 deg |
| constant offset | TT | orthogonal | 0.33 (pred 0.02); -1 deg | 0.33 (pred 0.04); 1 deg | 0.32 (pred 0.03); -2 deg |
| constant offset | LI | X only | 0.28 (pred 0.02); -7 deg | 0.28 (pred 0.06); -5 deg | 0.28 (pred 0.03); 5 deg |
| constant offset | LI | Y only | 0.25 (pred 0.03); -5 deg | 0.24 (pred 0.05); 5 deg | 0.25 (pred 0.03); -4 deg |
| constant offset | LI | principal | 0.21 (pred 0.02); -3 deg | 0.22 (pred 0.02); -5 deg | 0.21 (pred 0.02); -1 deg |
| constant offset | LI | orthogonal | 0.30 (pred 0.03); -17 deg | 0.30 (pred 0.07); 9 deg | 0.31 (pred 0.04); -9 deg |
| constant offset | UL | X only | 0.18 (pred 0.01); 6 deg | 0.18 (pred 0.03); 9 deg | 0.17 (pred 0.02); 5 deg |
| constant offset | UL | Y only | 0.14 (pred 0.01); -4 deg | 0.14 (pred 0.01); 16 deg | 0.15 (pred 0.01); -7 deg |
| constant offset | UL | principal | 0.15 (pred 0.01); -2 deg | 0.15 (pred 0.02); 1 deg | 0.14 (pred 0.01); -2 deg |
| constant offset | UL | orthogonal | 0.18 (pred 0.01); 9 deg | 0.17 (pred 0.02); 20 deg | 0.18 (pred 0.01); 3 deg |
| constant offset | LL | X only | 0.34 (pred 0.03); -9 deg | 0.36 (pred 0.07); -20 deg | 0.33 (pred 0.04); -18 deg |
| constant offset | LL | Y only | 0.22 (pred 0.01); -8 deg | 0.20 (pred 0.03); -12 deg | 0.21 (pred 0.01); -8 deg |
| constant offset | LL | principal | 0.24 (pred 0.02); 3 deg | 0.24 (pred 0.04); -2 deg | 0.23 (pred 0.02); 3 deg |
| constant offset | LL | orthogonal | 0.33 (pred 0.03); -7 deg | 0.33 (pred 0.06); -23 deg | 0.32 (pred 0.04); -28 deg |
| dynamic scaling around the mean | TD | X only | 0.58 (pred 0.05); 36 deg | 0.61 (pred 0.05); 23 deg | 0.59 (pred 0.04); 28 deg |
| dynamic scaling around the mean | TD | Y only | 0.36 (pred 0.02); -35 deg | 0.37 (pred 0.02); -40 deg | 0.36 (pred 0.02); -32 deg |
| dynamic scaling around the mean | TD | principal | 0.52 (pred 0.04); -13 deg | 0.54 (pred 0.03); -11 deg | 0.52 (pred 0.03); 1 deg |
| dynamic scaling around the mean | TD | orthogonal | 0.44 (pred 0.04); 4 deg | 0.45 (pred 0.04); 7 deg | 0.44 (pred 0.03); 5 deg |
| dynamic scaling around the mean | TB | X only | 0.75 (pred 0.06); -75 deg | 0.75 (pred 0.07); -90 deg | 0.74 (pred 0.07); -78 deg |
| dynamic scaling around the mean | TB | Y only | 0.46 (pred 0.02); 10 deg | 0.46 (pred 0.04); 13 deg | 0.44 (pred 0.02); 11 deg |
| dynamic scaling around the mean | TB | principal | 0.67 (pred 0.04); -28 deg | 0.66 (pred 0.04); -29 deg | 0.66 (pred 0.04); -29 deg |
| dynamic scaling around the mean | TB | orthogonal | 0.62 (pred 0.04); 25 deg | 0.64 (pred 0.06); 27 deg | 0.62 (pred 0.05); 31 deg |
| dynamic scaling around the mean | TT | X only | 0.46 (pred 0.03); 38 deg | 0.50 (pred 0.04); 40 deg | 0.46 (pred 0.04); 35 deg |
| dynamic scaling around the mean | TT | Y only | 0.34 (pred 0.01); -2 deg | 0.35 (pred 0.03); -0 deg | 0.34 (pred 0.02); -6 deg |
| dynamic scaling around the mean | TT | principal | 0.40 (pred 0.03); 44 deg | 0.42 (pred 0.04); 58 deg | 0.39 (pred 0.03); 42 deg |
| dynamic scaling around the mean | TT | orthogonal | 0.40 (pred 0.02); -5 deg | 0.41 (pred 0.04); -5 deg | 0.40 (pred 0.03); -8 deg |
| dynamic scaling around the mean | LI | X only | 0.35 (pred 0.02); 10 deg | 0.38 (pred 0.06); 14 deg | 0.35 (pred 0.03); 17 deg |
| dynamic scaling around the mean | LI | Y only | 0.30 (pred 0.03); 11 deg | 0.34 (pred 0.05); 14 deg | 0.31 (pred 0.03); 10 deg |
| dynamic scaling around the mean | LI | principal | 0.25 (pred 0.02); -13 deg | 0.28 (pred 0.02); -17 deg | 0.25 (pred 0.02); -16 deg |
| dynamic scaling around the mean | LI | orthogonal | 0.48 (pred 0.03); 44 deg | 0.52 (pred 0.07); 45 deg | 0.51 (pred 0.04); 36 deg |
| dynamic scaling around the mean | UL | X only | 0.23 (pred 0.01); -7 deg | 0.24 (pred 0.03); -2 deg | 0.24 (pred 0.02); -6 deg |
| dynamic scaling around the mean | UL | Y only | 0.18 (pred 0.01); 25 deg | 0.18 (pred 0.01); 10 deg | 0.19 (pred 0.01); 21 deg |
| dynamic scaling around the mean | UL | principal | 0.18 (pred 0.01); -15 deg | 0.19 (pred 0.02); -14 deg | 0.18 (pred 0.01); -12 deg |
| dynamic scaling around the mean | UL | orthogonal | 0.25 (pred 0.01); 18 deg | 0.26 (pred 0.02); 20 deg | 0.27 (pred 0.01); 17 deg |
| dynamic scaling around the mean | LL | X only | 0.44 (pred 0.03); -11 deg | 0.47 (pred 0.07); -3 deg | 0.44 (pred 0.04); -13 deg |
| dynamic scaling around the mean | LL | Y only | 0.29 (pred 0.01); 7 deg | 0.31 (pred 0.03); 16 deg | 0.30 (pred 0.01); 8 deg |
| dynamic scaling around the mean | LL | principal | 0.28 (pred 0.02); 3 deg | 0.31 (pred 0.04); 3 deg | 0.29 (pred 0.02); 3 deg |
| dynamic scaling around the mean | LL | orthogonal | 0.52 (pred 0.03); 20 deg | 0.55 (pred 0.06); 39 deg | 0.54 (pred 0.04); 19 deg |

## 5. Sign asymmetry

Gain by sign of the edit (negative / positive direction, or alpha = -0.5 / +0.5), mean over sensors.

| profile | direction | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|---|
| constant offset | X only | 0.09 / 0.13 | 0.08 / 0.11 | 0.11 / 0.11 |
| constant offset | Y only | 0.15 / 0.09 | 0.13 / 0.07 | 0.14 / 0.10 |
| constant offset | principal | 0.09 / 0.14 | 0.08 / 0.12 | 0.09 / 0.12 |
| constant offset | orthogonal | 0.10 / 0.13 | 0.09 / 0.11 | 0.11 / 0.11 |
| constant offset on speech frames | X only | 0.10 / 0.14 | 0.09 / 0.11 | 0.12 / 0.12 |
| constant offset on speech frames | Y only | 0.16 / 0.11 | 0.15 / 0.08 | 0.15 / 0.11 |
| constant offset on speech frames | principal | 0.10 / 0.16 | 0.09 / 0.13 | 0.10 / 0.14 |
| constant offset on speech frames | orthogonal | 0.11 / 0.15 | 0.10 / 0.12 | 0.11 / 0.13 |
| dynamic scaling around the mean | X only | 0.31 / 0.04 | 0.28 / 0.03 | 0.31 / 0.04 |
| dynamic scaling around the mean | Y only | 0.27 / 0.13 | 0.26 / 0.12 | 0.27 / 0.13 |
| dynamic scaling around the mean | principal | 0.33 / 0.09 | 0.30 / 0.08 | 0.33 / 0.10 |
| dynamic scaling around the mean | orthogonal | 0.28 / 0.15 | 0.26 / 0.12 | 0.28 / 0.14 |
| three local bumps (300 ms) | X only | 0.20 / 0.18 | 0.19 / 0.16 | 0.20 / 0.20 |
| three local bumps (300 ms) | Y only | 0.26 / 0.27 | 0.24 / 0.23 | 0.26 / 0.26 |
| three local bumps (300 ms) | principal | 0.23 / 0.24 | 0.21 / 0.20 | 0.24 / 0.24 |
| three local bumps (300 ms) | orthogonal | 0.19 / 0.29 | 0.17 / 0.26 | 0.19 / 0.28 |

## 6. Reference values

Natural dynamics through the same loop (RESULTS.md section 5.2): centred slope 1.00 (HiFi-GAN), 0.96 (DDSP), 1.00 (Vocos); regression diagonal 0.98, 0.93, 0.97; between-utterance slope 0.94, 0.90, 0.93. Single-channel constant edits of one std: gain 0.12 (HiFi-GAN), 0.10 (DDSP), 0.12 (Vocos), leakage 0.24. Constant edits along principal components of the 12-D EMA: gain 0.09 to 0.14 (6 utterances).

## Interpretation

**Answer to the question.** Paired X-Y edits are not what is missing. Averaged over the six sensors, an edit along the
sensor's natural principal direction (both channels together) is followed no better than an edit of X alone or Y alone:
the paired-minus-single contrasts are between -0.015 and +0.004 under constant offsets in all three vocoders (section 2).
The same holds for the orthogonal (minor-axis) direction.

**What does matter.**

1. *Time course.* Edits that vary in time are followed about twice as well as constant offsets: dynamic scaling around
   the utterance mean adds +0.06 to +0.11 and local 300 ms bumps add +0.08 to +0.14 of gain over the constant offset of
   the same direction (section 2). Restricting a constant offset to speech frames adds only +0.01 (sections 2 and 3),
   so the offset landing on silence is not the reason constant edits fail.
2. *Direction within a sensor.* The mean over sensors hides large, consistent differences between directions (section
   4). Under dynamic scaling, the tongue body is followed best along its natural principal axis (gain 0.57, 0.44, 0.54
   for HiFi-GAN, DDSP, Vocos) and hardly at all along X alone (0.07, 0.00, 0.06); the tongue tip is followed along X
   (0.35 to 0.37) and the orthogonal direction (0.34 to 0.35) but not along its principal axis (0.10 to 0.13). So a
   paired edit helps where the paired direction is the acoustically effective one, and hurts where it is not. The three
   vocoders agree closely on which directions work, which points to a property of the feature-to-acoustics mapping (or
   of the inversion) and not of one architecture.
3. *Sign.* Shrinking the movement range (alpha = -0.5) is followed with gain about 0.3, exaggerating it (alpha = +0.5)
   with 0.03 to 0.15 (section 5): the loop does not produce articulation beyond the range it has seen.

**What it does not show.** Even the best cell stays at about 0.57 against a natural-dynamics prediction near 1.0, and
leakage into the other sensors (0.25 to 0.49 std) exceeds the on-target gain in most cells. The probe measures the whole
loop (vocoder, then the linear WavLM inversion), so it cannot say whether the vocoder ignores an isolated edit or
produces acoustics that the inversion maps to a different articulation (the articulatory-to-acoustic map is many to
one). Separating the two legs needs an inversion-free acoustic measure of the edit (for example formant or spectral
shifts against the unedited synthesis), or edits that move all articulators coherently along natural trajectories
(for example dynamic scaling of the full 12-D EMA, or replacing one sensor's trajectory with that of another
utterance of the same phone sequence). Each vocoder is a single training run; the seed-1 runs will show whether the
direction pattern is stable.

