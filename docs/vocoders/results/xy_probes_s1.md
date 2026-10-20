# Paired X-Y EMA probes

Question: are paired X-Y edits of one sensor necessary for the vocoders to follow EMA edits, or do they fail for another reason?

## Basis

Probe utterances: 50 test.clean utterances of 3 to 10 s from 28 speakers (the subset of the single-channel probes), T1 speaker vectors, eval gain. Each edit is applied to the 15 raw input channels of one sensor (TD = channels 0 and 1, TB = 2 and 3, TT = 4 and 5, LI = 6 and 7, UL = 8 and 9, LL = 10 and 11, as X and Y), the edited and the unedited input are synthesized with the same speaker vector and synthesis RNG, both syntheses are re-extracted (refit head, no dither), and the difference of the re-extracted EMA is compared with the difference of the input EMA. All quantities are in training-std units of the individual channels. Final checkpoints: HiFi-GAN (seed 1) step 400000, DDSP (seed 1) step 400000, Vocos (seed 1) step 400000. One training run per vocoder.

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

| profile | direction | HiFi-GAN (seed 1) measured (CI95) / prediction | DDSP (seed 1) measured (CI95) / prediction | Vocos (seed 1) measured (CI95) / prediction |
|---|---|---|---|---|
| constant offset | X only | 0.12 (0.12, 0.12) / 0.98 | 0.10 (0.10, 0.11) / 0.92 | 0.11 (0.10, 0.11) / 0.97 |
| constant offset | Y only | 0.12 (0.12, 0.12) / 0.98 | 0.10 (0.09, 0.10) / 0.93 | 0.12 (0.12, 0.12) / 0.98 |
| constant offset | principal | 0.11 (0.11, 0.12) / 0.98 | 0.10 (0.10, 0.10) / 0.93 | 0.11 (0.10, 0.11) / 0.97 |
| constant offset | orthogonal | 0.11 (0.11, 0.12) / 0.99 | 0.10 (0.09, 0.10) / 0.93 | 0.11 (0.11, 0.11) / 0.98 |
| constant offset on speech frames | X only | 0.13 (0.13, 0.13) / 0.98 | 0.11 (0.10, 0.11) / 0.92 | 0.12 (0.12, 0.12) / 0.97 |
| constant offset on speech frames | Y only | 0.13 (0.13, 0.14) / 0.98 | 0.11 (0.11, 0.11) / 0.93 | 0.13 (0.13, 0.14) / 0.98 |
| constant offset on speech frames | principal | 0.13 (0.12, 0.13) / 0.98 | 0.11 (0.11, 0.12) / 0.93 | 0.12 (0.12, 0.13) / 0.97 |
| constant offset on speech frames | orthogonal | 0.13 (0.12, 0.13) / 0.99 | 0.11 (0.10, 0.11) / 0.93 | 0.12 (0.12, 0.13) / 0.98 |
| dynamic scaling around the mean | X only | 0.17 (0.17, 0.18) / 0.98 | 0.16 (0.15, 0.16) / 0.92 | 0.18 (0.17, 0.18) / 0.97 |
| dynamic scaling around the mean | Y only | 0.20 (0.19, 0.21) / 0.98 | 0.19 (0.18, 0.19) / 0.93 | 0.20 (0.20, 0.21) / 0.98 |
| dynamic scaling around the mean | principal | 0.21 (0.21, 0.22) / 0.98 | 0.19 (0.19, 0.20) / 0.93 | 0.21 (0.21, 0.22) / 0.97 |
| dynamic scaling around the mean | orthogonal | 0.21 (0.20, 0.21) / 0.99 | 0.19 (0.19, 0.20) / 0.93 | 0.21 (0.21, 0.22) / 0.98 |
| three local bumps (300 ms) | X only | 0.19 (0.18, 0.21) / 0.98 | 0.17 (0.16, 0.18) / 0.92 | 0.21 (0.19, 0.22) / 0.97 |
| three local bumps (300 ms) | Y only | 0.27 (0.25, 0.28) / 0.98 | 0.24 (0.23, 0.25) / 0.93 | 0.26 (0.25, 0.27) / 0.98 |
| three local bumps (300 ms) | principal | 0.23 (0.22, 0.24) / 0.98 | 0.20 (0.19, 0.21) / 0.93 | 0.23 (0.22, 0.24) / 0.97 |
| three local bumps (300 ms) | orthogonal | 0.24 (0.23, 0.25) / 0.99 | 0.22 (0.20, 0.23) / 0.93 | 0.24 (0.23, 0.26) / 0.98 |

### 1.2 Leakage into the other sensors (std)

| profile | direction | HiFi-GAN (seed 1) measured (CI95) / prediction | DDSP (seed 1) measured (CI95) / prediction | Vocos (seed 1) measured (CI95) / prediction |
|---|---|---|---|---|
| constant offset | X only | 0.37 (0.37, 0.38) / 0.04 | 0.40 (0.39, 0.40) / 0.05 | 0.36 (0.35, 0.37) / 0.04 |
| constant offset | Y only | 0.26 (0.26, 0.27) / 0.02 | 0.25 (0.25, 0.26) / 0.03 | 0.26 (0.25, 0.26) / 0.02 |
| constant offset | principal | 0.33 (0.32, 0.33) / 0.03 | 0.34 (0.33, 0.34) / 0.03 | 0.32 (0.31, 0.32) / 0.03 |
| constant offset | orthogonal | 0.33 (0.32, 0.34) / 0.03 | 0.34 (0.33, 0.35) / 0.04 | 0.32 (0.31, 0.33) / 0.04 |
| constant offset on speech frames | X only | 0.38 (0.37, 0.39) / 0.04 | 0.41 (0.40, 0.42) / 0.05 | 0.37 (0.36, 0.38) / 0.04 |
| constant offset on speech frames | Y only | 0.26 (0.26, 0.27) / 0.02 | 0.26 (0.26, 0.27) / 0.03 | 0.26 (0.25, 0.27) / 0.02 |
| constant offset on speech frames | principal | 0.33 (0.33, 0.34) / 0.03 | 0.35 (0.34, 0.35) / 0.03 | 0.32 (0.32, 0.33) / 0.03 |
| constant offset on speech frames | orthogonal | 0.33 (0.33, 0.34) / 0.03 | 0.35 (0.34, 0.36) / 0.04 | 0.32 (0.32, 0.33) / 0.04 |
| dynamic scaling around the mean | X only | 0.47 (0.46, 0.48) / 0.04 | 0.49 (0.48, 0.50) / 0.05 | 0.47 (0.46, 0.48) / 0.04 |
| dynamic scaling around the mean | Y only | 0.32 (0.32, 0.33) / 0.02 | 0.34 (0.33, 0.34) / 0.03 | 0.33 (0.32, 0.33) / 0.02 |
| dynamic scaling around the mean | principal | 0.38 (0.38, 0.39) / 0.03 | 0.40 (0.39, 0.40) / 0.03 | 0.39 (0.38, 0.39) / 0.03 |
| dynamic scaling around the mean | orthogonal | 0.46 (0.44, 0.47) / 0.03 | 0.47 (0.46, 0.48) / 0.04 | 0.47 (0.45, 0.48) / 0.04 |
| three local bumps (300 ms) | X only | 0.47 (0.46, 0.48) / 0.04 | 0.48 (0.47, 0.49) / 0.05 | 0.47 (0.46, 0.49) / 0.04 |
| three local bumps (300 ms) | Y only | 0.32 (0.31, 0.33) / 0.02 | 0.34 (0.33, 0.35) / 0.03 | 0.32 (0.31, 0.33) / 0.02 |
| three local bumps (300 ms) | principal | 0.41 (0.40, 0.42) / 0.03 | 0.41 (0.40, 0.42) / 0.03 | 0.41 (0.40, 0.42) / 0.03 |
| three local bumps (300 ms) | orthogonal | 0.41 (0.40, 0.42) / 0.03 | 0.42 (0.41, 0.44) / 0.04 | 0.41 (0.40, 0.42) / 0.04 |

### 1.3 Angle between measured and target 2-D displacement (degrees)

| profile | direction | HiFi-GAN (seed 1) measured (CI95) | DDSP (seed 1) measured (CI95) | Vocos (seed 1) measured (CI95) |
|---|---|---|---|---|
| constant offset | X only | -16 (-18, -14) | -20 (-22, -18) | -17 (-19, -16) |
| constant offset | Y only | -13 (-15, -10) | -9 (-14, -5) | -12 (-14, -10) |
| constant offset | principal | 6 (5, 8) | 7 (6, 9) | 6 (5, 8) |
| constant offset | orthogonal | 9 (7, 12) | 11 (9, 13) | 7 (5, 10) |
| constant offset on speech frames | X only | -11 (-13, -9) | -16 (-18, -14) | -12 (-14, -10) |
| constant offset on speech frames | Y only | -14 (-16, -12) | -11 (-14, -8) | -12 (-15, -10) |
| constant offset on speech frames | principal | 9 (7, 10) | 10 (8, 11) | 8 (7, 10) |
| constant offset on speech frames | orthogonal | 13 (11, 16) | 15 (13, 18) | 12 (10, 15) |
| dynamic scaling around the mean | X only | -2 (-4, 1) | -4 (-7, -0) | -1 (-3, 2) |
| dynamic scaling around the mean | Y only | 3 (2, 4) | 2 (0, 3) | 3 (2, 5) |
| dynamic scaling around the mean | principal | -4 (-6, -1) | 1 (-1, 3) | -1 (-4, 2) |
| dynamic scaling around the mean | orthogonal | 18 (16, 19) | 20 (18, 22) | 18 (16, 19) |
| three local bumps (300 ms) | X only | -1 (-5, 4) | 9 (6, 13) | 1 (-2, 4) |
| three local bumps (300 ms) | Y only | 4 (2, 7) | 2 (-1, 5) | 6 (4, 9) |
| three local bumps (300 ms) | principal | -0 (-7, 11) | 17 (10, 23) | 8 (-5, 18) |
| three local bumps (300 ms) | orthogonal | 21 (19, 24) | 22 (21, 24) | 17 (15, 19) |

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

| contrast | HiFi-GAN (seed 1) | DDSP (seed 1) | Vocos (seed 1) |
|---|---|---|---|
| paired principal minus X only (constant) | -0.007 (-0.011, -0.002) / -0.001 | -0.002 (-0.009, 0.004) / 0.002 | -0.002 (-0.006, 0.002) / 0.002 |
| paired principal minus Y only (constant) | -0.005 (-0.009, -0.002) / -0.007 | 0.004 (-0.000, 0.007) / -0.002 | -0.014 (-0.018, -0.010) / -0.008 |
| orthogonal minus X only (constant) | -0.006 (-0.010, -0.003) / 0.007 | -0.006 (-0.012, -0.000) / 0.002 | 0.001 (-0.003, 0.005) / 0.008 |
| speech-only minus constant, principal | 0.012 (0.010, 0.014) / -0.000 | 0.011 (0.008, 0.014) / 0.000 | 0.015 (0.012, 0.018) / 0.000 |
| speech-only minus constant, X only | 0.009 (0.006, 0.012) / 0.000 | 0.005 (0.002, 0.008) / 0.000 | 0.012 (0.009, 0.014) / 0.000 |
| dynamic scaling minus constant, principal | 0.098 (0.090, 0.105) / -0.000 | 0.092 (0.087, 0.097) / -0.000 | 0.105 (0.098, 0.111) / -0.000 |
| dynamic scaling minus constant, X only | 0.053 (0.045, 0.061) / 0.000 | 0.053 (0.044, 0.061) / 0.000 | 0.067 (0.060, 0.075) / 0.000 |
| bumps minus constant, principal | 0.118 (0.106, 0.131) / -0.000 | 0.101 (0.093, 0.109) / -0.000 | 0.127 (0.117, 0.136) / -0.000 |
| bumps minus constant, X only | 0.073 (0.060, 0.087) / 0.000 | 0.064 (0.051, 0.078) / 0.000 | 0.097 (0.083, 0.110) / 0.000 |

## 3. Where the edit is followed: speech frames against silence and pauses

Gain computed on the speech frames only and on the other frames only (frames in the speech mask of the unedited input; mean over sensors and directions, pooled over signs). Edits of the speech-only and bump profiles are zero outside the speech mask, so their 'other frames' gain is undefined or reflects ramps only.

| profile | frames | HiFi-GAN (seed 1) gain | DDSP (seed 1) gain | Vocos (seed 1) gain |
|---|---|---|---|---|
| constant offset | speech frames | 0.12 (0.11, 0.12) / 0.98 | 0.10 (0.10, 0.10) / 0.93 | 0.11 (0.11, 0.11) / 0.98 |
| constant offset | other frames | 0.11 (0.11, 0.12) / 0.98 | 0.10 (0.10, 0.11) / 0.93 | 0.11 (0.10, 0.12) / 0.98 |
| constant offset on speech frames | speech frames | 0.13 (0.13, 0.13) / 0.98 | 0.11 (0.11, 0.11) / 0.93 | 0.13 (0.12, 0.13) / 0.98 |
| constant offset on speech frames | other frames | -0.05 (-0.07, -0.04) / 0.98 | -0.05 (-0.07, -0.04) / 0.93 | -0.05 (-0.06, -0.04) / 0.98 |
| dynamic scaling around the mean | speech frames | 0.20 (0.19, 0.20) / 0.98 | 0.18 (0.18, 0.19) / 0.93 | 0.20 (0.20, 0.20) / 0.98 |
| dynamic scaling around the mean | other frames | 0.19 (0.19, 0.20) / 0.98 | 0.17 (0.16, 0.18) / 0.93 | 0.20 (0.19, 0.21) / 0.98 |
| three local bumps (300 ms) | speech frames | 0.23 (0.22, 0.24) / 0.98 | 0.21 (0.20, 0.22) / 0.93 | 0.24 (0.23, 0.25) / 0.98 |
| three local bumps (300 ms) | other frames | 0.22 (0.20, 0.24) / 0.98 | 0.17 (0.15, 0.19) / 0.93 | 0.22 (0.21, 0.24) / 0.98 |

## 4. By sensor

### 4.1 Gain, constant offset (measured with CI95 / regression prediction)

| sensor | direction | HiFi-GAN (seed 1) | DDSP (seed 1) | Vocos (seed 1) |
|---|---|---|---|---|
| TD | X only | 0.28 (0.26, 0.29) / 0.95 | 0.27 (0.26, 0.29) / 0.89 | 0.27 (0.26, 0.28) / 0.95 |
| TD | Y only | 0.05 (0.04, 0.07) / 1.00 | 0.02 (0.01, 0.04) / 0.92 | 0.06 (0.05, 0.08) / 0.98 |
| TD | principal | 0.16 (0.14, 0.18) / 0.96 | 0.16 (0.14, 0.17) / 0.85 | 0.14 (0.12, 0.15) / 0.95 |
| TD | orthogonal | 0.17 (0.16, 0.19) / 0.99 | 0.14 (0.13, 0.15) / 0.96 | 0.17 (0.16, 0.18) / 0.98 |
| TB | X only | -0.12 (-0.13, -0.10) / 0.98 | -0.17 (-0.19, -0.15) / 0.93 | -0.15 (-0.17, -0.14) / 0.95 |
| TB | Y only | 0.32 (0.31, 0.33) / 1.00 | 0.26 (0.25, 0.27) / 0.94 | 0.31 (0.30, 0.32) / 1.01 |
| TB | principal | 0.20 (0.18, 0.22) / 0.95 | 0.16 (0.14, 0.18) / 0.96 | 0.17 (0.16, 0.19) / 0.95 |
| TB | orthogonal | -0.07 (-0.08, -0.06) / 1.02 | -0.09 (-0.11, -0.07) / 0.92 | -0.07 (-0.08, -0.06) / 1.00 |
| TT | X only | 0.27 (0.25, 0.29) / 1.03 | 0.26 (0.24, 0.27) / 1.00 | 0.25 (0.23, 0.27) / 1.04 |
| TT | Y only | 0.13 (0.11, 0.14) / 0.96 | 0.11 (0.10, 0.12) / 0.93 | 0.13 (0.12, 0.14) / 0.96 |
| TT | principal | 0.07 (0.06, 0.08) / 1.00 | 0.07 (0.05, 0.08) / 0.97 | 0.07 (0.06, 0.09) / 1.00 |
| TT | orthogonal | 0.34 (0.32, 0.35) / 0.99 | 0.31 (0.29, 0.33) / 0.96 | 0.32 (0.31, 0.33) / 1.00 |
| LI | X only | 0.10 (0.09, 0.11) / 0.93 | 0.08 (0.07, 0.09) / 0.87 | 0.10 (0.09, 0.11) / 0.91 |
| LI | Y only | 0.08 (0.07, 0.09) / 1.01 | 0.09 (0.08, 0.10) / 0.93 | 0.08 (0.07, 0.09) / 1.01 |
| LI | principal | 0.11 (0.10, 0.12) / 0.99 | 0.09 (0.08, 0.10) / 0.93 | 0.10 (0.09, 0.11) / 0.97 |
| LI | orthogonal | 0.09 (0.09, 0.10) / 0.95 | 0.08 (0.07, 0.09) / 0.87 | 0.09 (0.08, 0.10) / 0.95 |
| UL | X only | 0.06 (0.05, 0.07) / 1.01 | 0.06 (0.05, 0.06) / 0.93 | 0.06 (0.05, 0.07) / 0.98 |
| UL | Y only | 0.05 (0.04, 0.06) / 0.97 | 0.05 (0.04, 0.05) / 0.93 | 0.06 (0.06, 0.07) / 0.96 |
| UL | principal | 0.06 (0.05, 0.06) / 1.00 | 0.05 (0.04, 0.06) / 0.94 | 0.06 (0.05, 0.07) / 0.98 |
| UL | orthogonal | 0.05 (0.05, 0.06) / 0.98 | 0.05 (0.05, 0.06) / 0.92 | 0.06 (0.05, 0.06) / 0.96 |
| LL | X only | 0.13 (0.12, 0.14) / 0.97 | 0.13 (0.12, 0.14) / 0.93 | 0.13 (0.12, 0.14) / 0.99 |
| LL | Y only | 0.08 (0.08, 0.09) / 0.97 | 0.06 (0.05, 0.07) / 0.92 | 0.08 (0.08, 0.09) / 0.96 |
| LL | principal | 0.08 (0.08, 0.09) / 0.96 | 0.08 (0.07, 0.09) / 0.91 | 0.10 (0.09, 0.10) / 0.98 |
| LL | orthogonal | 0.10 (0.09, 0.10) / 0.98 | 0.09 (0.08, 0.10) / 0.94 | 0.09 (0.08, 0.09) / 0.98 |

### 4.2 Gain, constant offset on speech frames (measured with CI95 / regression prediction)

| sensor | direction | HiFi-GAN (seed 1) | DDSP (seed 1) | Vocos (seed 1) |
|---|---|---|---|---|
| TD | X only | 0.27 (0.25, 0.28) / 0.95 | 0.28 (0.26, 0.29) / 0.89 | 0.27 (0.26, 0.29) / 0.95 |
| TD | Y only | 0.04 (0.03, 0.05) / 1.00 | 0.02 (0.00, 0.03) / 0.92 | 0.05 (0.04, 0.07) / 0.98 |
| TD | principal | 0.11 (0.09, 0.13) / 0.96 | 0.12 (0.10, 0.14) / 0.85 | 0.11 (0.09, 0.12) / 0.95 |
| TD | orthogonal | 0.19 (0.18, 0.21) / 0.99 | 0.16 (0.15, 0.18) / 0.96 | 0.19 (0.18, 0.20) / 0.98 |
| TB | X only | -0.10 (-0.12, -0.07) / 0.98 | -0.15 (-0.18, -0.13) / 0.93 | -0.11 (-0.13, -0.10) / 0.95 |
| TB | Y only | 0.37 (0.36, 0.39) / 1.00 | 0.31 (0.29, 0.32) / 0.94 | 0.36 (0.34, 0.37) / 1.01 |
| TB | principal | 0.29 (0.27, 0.31) / 0.95 | 0.25 (0.22, 0.27) / 0.96 | 0.26 (0.24, 0.29) / 0.95 |
| TB | orthogonal | -0.05 (-0.06, -0.04) / 1.02 | -0.06 (-0.08, -0.04) / 0.92 | -0.04 (-0.06, -0.03) / 1.00 |
| TT | X only | 0.29 (0.27, 0.31) / 1.03 | 0.26 (0.25, 0.28) / 1.00 | 0.26 (0.24, 0.28) / 1.04 |
| TT | Y only | 0.14 (0.13, 0.15) / 0.96 | 0.12 (0.10, 0.13) / 0.93 | 0.14 (0.13, 0.15) / 0.96 |
| TT | principal | 0.08 (0.07, 0.10) / 1.00 | 0.07 (0.06, 0.09) / 0.97 | 0.08 (0.07, 0.09) / 1.00 |
| TT | orthogonal | 0.35 (0.33, 0.36) / 0.99 | 0.31 (0.29, 0.32) / 0.96 | 0.34 (0.32, 0.35) / 1.00 |
| LI | X only | 0.09 (0.08, 0.10) / 0.93 | 0.08 (0.07, 0.09) / 0.87 | 0.09 (0.08, 0.10) / 0.91 |
| LI | Y only | 0.09 (0.08, 0.10) / 1.01 | 0.09 (0.08, 0.10) / 0.93 | 0.09 (0.08, 0.10) / 1.01 |
| LI | principal | 0.11 (0.10, 0.12) / 0.99 | 0.10 (0.09, 0.11) / 0.93 | 0.10 (0.09, 0.11) / 0.97 |
| LI | orthogonal | 0.09 (0.08, 0.10) / 0.95 | 0.07 (0.07, 0.08) / 0.87 | 0.09 (0.08, 0.10) / 0.95 |
| UL | X only | 0.09 (0.08, 0.10) / 1.01 | 0.08 (0.07, 0.09) / 0.93 | 0.09 (0.08, 0.09) / 0.98 |
| UL | Y only | 0.06 (0.05, 0.06) / 0.97 | 0.05 (0.05, 0.06) / 0.93 | 0.07 (0.06, 0.07) / 0.96 |
| UL | principal | 0.08 (0.07, 0.08) / 1.00 | 0.07 (0.06, 0.08) / 0.94 | 0.08 (0.07, 0.09) / 0.98 |
| UL | orthogonal | 0.07 (0.06, 0.07) / 0.98 | 0.07 (0.06, 0.07) / 0.92 | 0.07 (0.06, 0.07) / 0.96 |
| LL | X only | 0.13 (0.12, 0.14) / 0.97 | 0.11 (0.10, 0.12) / 0.93 | 0.12 (0.11, 0.13) / 0.99 |
| LL | Y only | 0.10 (0.09, 0.10) / 0.97 | 0.07 (0.07, 0.08) / 0.92 | 0.10 (0.09, 0.10) / 0.96 |
| LL | principal | 0.08 (0.08, 0.09) / 0.96 | 0.06 (0.05, 0.07) / 0.91 | 0.09 (0.09, 0.10) / 0.98 |
| LL | orthogonal | 0.11 (0.10, 0.12) / 0.98 | 0.10 (0.09, 0.11) / 0.94 | 0.10 (0.10, 0.11) / 0.98 |

### 4.3 Gain, dynamic scaling around the mean (measured with CI95 / regression prediction)

| sensor | direction | HiFi-GAN (seed 1) | DDSP (seed 1) | Vocos (seed 1) |
|---|---|---|---|---|
| TD | X only | 0.12 (0.10, 0.14) / 0.95 | 0.15 (0.13, 0.17) / 0.89 | 0.12 (0.10, 0.15) / 0.95 |
| TD | Y only | 0.14 (0.13, 0.16) / 1.00 | 0.09 (0.08, 0.11) / 0.92 | 0.14 (0.13, 0.16) / 0.98 |
| TD | principal | 0.06 (0.04, 0.08) / 0.96 | 0.06 (0.05, 0.08) / 0.85 | 0.06 (0.05, 0.08) / 0.95 |
| TD | orthogonal | 0.16 (0.15, 0.17) / 0.99 | 0.16 (0.14, 0.18) / 0.96 | 0.17 (0.15, 0.18) / 0.98 |
| TB | X only | 0.05 (0.02, 0.08) / 0.98 | -0.02 (-0.05, 0.01) / 0.93 | 0.07 (0.04, 0.10) / 0.95 |
| TB | Y only | 0.41 (0.39, 0.43) / 1.00 | 0.38 (0.37, 0.40) / 0.94 | 0.40 (0.38, 0.42) / 1.01 |
| TB | principal | 0.55 (0.53, 0.57) / 0.95 | 0.46 (0.44, 0.49) / 0.96 | 0.55 (0.53, 0.58) / 0.95 |
| TB | orthogonal | 0.22 (0.21, 0.24) / 1.02 | 0.16 (0.15, 0.18) / 0.92 | 0.22 (0.20, 0.24) / 1.00 |
| TT | X only | 0.36 (0.34, 0.38) / 1.03 | 0.39 (0.37, 0.41) / 1.00 | 0.35 (0.33, 0.37) / 1.04 |
| TT | Y only | 0.17 (0.15, 0.19) / 0.96 | 0.14 (0.13, 0.16) / 0.93 | 0.17 (0.16, 0.19) / 0.96 |
| TT | principal | 0.13 (0.12, 0.14) / 1.00 | 0.12 (0.11, 0.13) / 0.97 | 0.12 (0.11, 0.13) / 1.00 |
| TT | orthogonal | 0.35 (0.33, 0.38) / 0.99 | 0.35 (0.33, 0.38) / 0.96 | 0.36 (0.33, 0.38) / 1.00 |
| LI | X only | 0.06 (0.05, 0.08) / 0.93 | 0.04 (0.03, 0.06) / 0.87 | 0.07 (0.05, 0.08) / 0.91 |
| LI | Y only | 0.21 (0.19, 0.22) / 1.01 | 0.20 (0.19, 0.21) / 0.93 | 0.20 (0.18, 0.21) / 1.01 |
| LI | principal | 0.14 (0.13, 0.16) / 0.99 | 0.14 (0.12, 0.15) / 0.93 | 0.14 (0.13, 0.15) / 0.97 |
| LI | orthogonal | 0.14 (0.13, 0.15) / 0.95 | 0.13 (0.12, 0.14) / 0.87 | 0.14 (0.13, 0.15) / 0.95 |
| UL | X only | 0.19 (0.18, 0.21) / 1.01 | 0.20 (0.19, 0.21) / 0.93 | 0.20 (0.19, 0.22) / 0.98 |
| UL | Y only | 0.11 (0.10, 0.13) / 0.97 | 0.12 (0.11, 0.13) / 0.93 | 0.12 (0.11, 0.13) / 0.96 |
| UL | principal | 0.12 (0.11, 0.13) / 1.00 | 0.15 (0.13, 0.16) / 0.94 | 0.13 (0.12, 0.14) / 0.98 |
| UL | orthogonal | 0.19 (0.18, 0.20) / 0.98 | 0.17 (0.16, 0.19) / 0.92 | 0.21 (0.19, 0.22) / 0.96 |
| LL | X only | 0.25 (0.23, 0.27) / 0.97 | 0.17 (0.16, 0.19) / 0.93 | 0.24 (0.23, 0.26) / 0.99 |
| LL | Y only | 0.17 (0.16, 0.18) / 0.97 | 0.18 (0.17, 0.19) / 0.92 | 0.18 (0.17, 0.20) / 0.96 |
| LL | principal | 0.26 (0.25, 0.28) / 0.96 | 0.23 (0.21, 0.25) / 0.91 | 0.26 (0.25, 0.28) / 0.98 |
| LL | orthogonal | 0.19 (0.18, 0.20) / 0.98 | 0.17 (0.16, 0.18) / 0.94 | 0.19 (0.18, 0.20) / 0.98 |

### 4.4 Gain, three local bumps (300 ms) (measured with CI95 / regression prediction)

| sensor | direction | HiFi-GAN (seed 1) | DDSP (seed 1) | Vocos (seed 1) |
|---|---|---|---|---|
| TD | X only | 0.16 (0.12, 0.20) / 0.95 | 0.10 (0.07, 0.14) / 0.89 | 0.14 (0.11, 0.18) / 0.95 |
| TD | Y only | 0.19 (0.17, 0.22) / 1.00 | 0.15 (0.12, 0.18) / 0.92 | 0.21 (0.17, 0.24) / 0.98 |
| TD | principal | 0.05 (0.00, 0.09) / 0.96 | -0.01 (-0.05, 0.03) / 0.85 | 0.02 (-0.01, 0.04) / 0.95 |
| TD | orthogonal | 0.28 (0.25, 0.31) / 0.99 | 0.25 (0.21, 0.29) / 0.96 | 0.28 (0.25, 0.32) / 0.98 |
| TB | X only | 0.01 (-0.04, 0.07) / 0.98 | 0.06 (0.02, 0.10) / 0.93 | 0.12 (0.06, 0.17) / 0.95 |
| TB | Y only | 0.60 (0.56, 0.64) / 1.00 | 0.53 (0.49, 0.57) / 0.94 | 0.59 (0.54, 0.63) / 1.01 |
| TB | principal | 0.53 (0.48, 0.59) / 0.95 | 0.48 (0.45, 0.52) / 0.96 | 0.57 (0.53, 0.61) / 0.95 |
| TB | orthogonal | 0.18 (0.13, 0.24) / 1.02 | 0.20 (0.14, 0.28) / 0.92 | 0.23 (0.17, 0.30) / 1.00 |
| TT | X only | 0.45 (0.42, 0.49) / 1.03 | 0.41 (0.37, 0.45) / 1.00 | 0.43 (0.40, 0.47) / 1.04 |
| TT | Y only | 0.20 (0.16, 0.23) / 0.96 | 0.18 (0.14, 0.21) / 0.93 | 0.20 (0.17, 0.24) / 0.96 |
| TT | principal | 0.21 (0.19, 0.24) / 1.00 | 0.18 (0.15, 0.21) / 0.97 | 0.22 (0.19, 0.25) / 1.00 |
| TT | orthogonal | 0.43 (0.40, 0.47) / 0.99 | 0.38 (0.34, 0.42) / 0.96 | 0.41 (0.37, 0.45) / 1.00 |
| LI | X only | 0.10 (0.07, 0.12) / 0.93 | 0.08 (0.06, 0.10) / 0.87 | 0.12 (0.09, 0.15) / 0.91 |
| LI | Y only | 0.25 (0.22, 0.27) / 1.01 | 0.21 (0.19, 0.23) / 0.93 | 0.22 (0.20, 0.24) / 1.01 |
| LI | principal | 0.21 (0.19, 0.23) / 0.99 | 0.18 (0.15, 0.20) / 0.93 | 0.20 (0.18, 0.22) / 0.97 |
| LI | orthogonal | 0.13 (0.12, 0.15) / 0.95 | 0.11 (0.10, 0.13) / 0.87 | 0.14 (0.12, 0.16) / 0.95 |
| UL | X only | 0.25 (0.23, 0.28) / 1.01 | 0.24 (0.22, 0.26) / 0.93 | 0.24 (0.22, 0.27) / 0.98 |
| UL | Y only | 0.13 (0.11, 0.15) / 0.97 | 0.14 (0.12, 0.15) / 0.93 | 0.12 (0.11, 0.14) / 0.96 |
| UL | principal | 0.17 (0.15, 0.19) / 1.00 | 0.20 (0.17, 0.22) / 0.94 | 0.16 (0.14, 0.18) / 0.98 |
| UL | orthogonal | 0.20 (0.19, 0.22) / 0.98 | 0.18 (0.16, 0.19) / 0.92 | 0.20 (0.18, 0.21) / 0.96 |
| LL | X only | 0.18 (0.15, 0.21) / 0.97 | 0.11 (0.08, 0.13) / 0.93 | 0.18 (0.14, 0.21) / 0.99 |
| LL | Y only | 0.23 (0.20, 0.25) / 0.97 | 0.25 (0.23, 0.27) / 0.92 | 0.24 (0.22, 0.26) / 0.96 |
| LL | principal | 0.22 (0.19, 0.25) / 0.96 | 0.19 (0.16, 0.22) / 0.91 | 0.23 (0.20, 0.27) / 0.98 |
| LL | orthogonal | 0.20 (0.18, 0.22) / 0.98 | 0.18 (0.16, 0.20) / 0.94 | 0.20 (0.17, 0.22) / 0.98 |

### 4.5 Leakage and angle by sensor (constant offset and dynamic scaling; leakage, angle in degrees)

| profile | sensor | direction | HiFi-GAN (seed 1) | DDSP (seed 1) | Vocos (seed 1) |
|---|---|---|---|---|---|
| constant offset | TD | X only | 0.48 (pred 0.05); -3 deg | 0.52 (pred 0.05); -2 deg | 0.49 (pred 0.05); 4 deg |
| constant offset | TD | Y only | 0.30 (pred 0.02); -33 deg | 0.28 (pred 0.02); -42 deg | 0.30 (pred 0.02); -26 deg |
| constant offset | TD | principal | 0.46 (pred 0.04); 29 deg | 0.48 (pred 0.04); 34 deg | 0.46 (pred 0.04); 33 deg |
| constant offset | TD | orthogonal | 0.35 (pred 0.04); -41 deg | 0.36 (pred 0.04); -42 deg | 0.34 (pred 0.04); -39 deg |
| constant offset | TB | X only | 0.59 (pred 0.06); -118 deg | 0.60 (pred 0.06); -141 deg | 0.57 (pred 0.07); -129 deg |
| constant offset | TB | Y only | 0.37 (pred 0.02); 2 deg | 0.36 (pred 0.03); -2 deg | 0.37 (pred 0.02); -0 deg |
| constant offset | TB | principal | 0.55 (pred 0.04); -60 deg | 0.56 (pred 0.04); -60 deg | 0.52 (pred 0.05); -60 deg |
| constant offset | TB | orthogonal | 0.49 (pred 0.04); 123 deg | 0.51 (pred 0.05); 121 deg | 0.48 (pred 0.06); 124 deg |
| constant offset | TT | X only | 0.38 (pred 0.03); 38 deg | 0.38 (pred 0.04); 43 deg | 0.36 (pred 0.05); 39 deg |
| constant offset | TT | Y only | 0.29 (pred 0.01); -28 deg | 0.30 (pred 0.03); -6 deg | 0.28 (pred 0.01); -25 deg |
| constant offset | TT | principal | 0.35 (pred 0.03); 68 deg | 0.37 (pred 0.04); 72 deg | 0.33 (pred 0.04); 66 deg |
| constant offset | TT | orthogonal | 0.33 (pred 0.02); -3 deg | 0.33 (pred 0.03); 1 deg | 0.31 (pred 0.03); -3 deg |
| constant offset | LI | X only | 0.28 (pred 0.03); -5 deg | 0.29 (pred 0.05); -3 deg | 0.27 (pred 0.04); -3 deg |
| constant offset | LI | Y only | 0.25 (pred 0.03); -5 deg | 0.24 (pred 0.05); 8 deg | 0.24 (pred 0.03); -8 deg |
| constant offset | LI | principal | 0.22 (pred 0.02); -0 deg | 0.22 (pred 0.02); -6 deg | 0.21 (pred 0.02); -2 deg |
| constant offset | LI | orthogonal | 0.31 (pred 0.03); -10 deg | 0.31 (pred 0.07); 11 deg | 0.30 (pred 0.04); -14 deg |
| constant offset | UL | X only | 0.18 (pred 0.02); 7 deg | 0.19 (pred 0.02); 5 deg | 0.17 (pred 0.02); 3 deg |
| constant offset | UL | Y only | 0.14 (pred 0.01); -3 deg | 0.14 (pred 0.01); 12 deg | 0.15 (pred 0.01); -4 deg |
| constant offset | UL | principal | 0.14 (pred 0.01); -2 deg | 0.15 (pred 0.02); 6 deg | 0.14 (pred 0.01); 1 deg |
| constant offset | UL | orthogonal | 0.18 (pred 0.01); 8 deg | 0.18 (pred 0.02); 13 deg | 0.17 (pred 0.01); -2 deg |
| constant offset | LL | X only | 0.34 (pred 0.03); -15 deg | 0.39 (pred 0.06); -22 deg | 0.32 (pred 0.04); -16 deg |
| constant offset | LL | Y only | 0.21 (pred 0.02); -9 deg | 0.21 (pred 0.02); -25 deg | 0.21 (pred 0.01); -7 deg |
| constant offset | LL | principal | 0.23 (pred 0.02); 3 deg | 0.24 (pred 0.03); -2 deg | 0.23 (pred 0.03); 1 deg |
| constant offset | LL | orthogonal | 0.33 (pred 0.03); -21 deg | 0.37 (pred 0.05); -38 deg | 0.31 (pred 0.04); -23 deg |
| dynamic scaling around the mean | TD | X only | 0.58 (pred 0.05); 29 deg | 0.61 (pred 0.05); 33 deg | 0.61 (pred 0.05); 39 deg |
| dynamic scaling around the mean | TD | Y only | 0.36 (pred 0.02); -32 deg | 0.38 (pred 0.02); -46 deg | 0.36 (pred 0.02); -31 deg |
| dynamic scaling around the mean | TD | principal | 0.52 (pred 0.04); -17 deg | 0.53 (pred 0.04); 7 deg | 0.53 (pred 0.04); -4 deg |
| dynamic scaling around the mean | TD | orthogonal | 0.44 (pred 0.04); 6 deg | 0.45 (pred 0.04); 6 deg | 0.45 (pred 0.04); 10 deg |
| dynamic scaling around the mean | TB | X only | 0.75 (pred 0.06); -79 deg | 0.74 (pred 0.06); -95 deg | 0.75 (pred 0.07); -76 deg |
| dynamic scaling around the mean | TB | Y only | 0.46 (pred 0.02); 9 deg | 0.47 (pred 0.03); 13 deg | 0.45 (pred 0.02); 10 deg |
| dynamic scaling around the mean | TB | principal | 0.67 (pred 0.04); -30 deg | 0.65 (pred 0.04); -29 deg | 0.66 (pred 0.05); -29 deg |
| dynamic scaling around the mean | TB | orthogonal | 0.62 (pred 0.04); 28 deg | 0.63 (pred 0.05); 28 deg | 0.62 (pred 0.06); 21 deg |
| dynamic scaling around the mean | TT | X only | 0.46 (pred 0.03); 38 deg | 0.50 (pred 0.04); 39 deg | 0.46 (pred 0.05); 38 deg |
| dynamic scaling around the mean | TT | Y only | 0.35 (pred 0.01); -3 deg | 0.35 (pred 0.03); 1 deg | 0.34 (pred 0.01); -3 deg |
| dynamic scaling around the mean | TT | principal | 0.40 (pred 0.03); 47 deg | 0.42 (pred 0.04); 54 deg | 0.39 (pred 0.04); 48 deg |
| dynamic scaling around the mean | TT | orthogonal | 0.40 (pred 0.02); -7 deg | 0.41 (pred 0.03); -5 deg | 0.40 (pred 0.03); -6 deg |
| dynamic scaling around the mean | LI | X only | 0.35 (pred 0.03); 21 deg | 0.38 (pred 0.05); -3 deg | 0.35 (pred 0.04); 9 deg |
| dynamic scaling around the mean | LI | Y only | 0.30 (pred 0.03); 13 deg | 0.33 (pred 0.05); 14 deg | 0.31 (pred 0.03); 11 deg |
| dynamic scaling around the mean | LI | principal | 0.25 (pred 0.02); -14 deg | 0.28 (pred 0.02); -16 deg | 0.25 (pred 0.02); -14 deg |
| dynamic scaling around the mean | LI | orthogonal | 0.49 (pred 0.03); 45 deg | 0.51 (pred 0.07); 40 deg | 0.51 (pred 0.04); 40 deg |
| dynamic scaling around the mean | UL | X only | 0.23 (pred 0.02); -7 deg | 0.24 (pred 0.02); -2 deg | 0.23 (pred 0.02); -4 deg |
| dynamic scaling around the mean | UL | Y only | 0.18 (pred 0.01); 25 deg | 0.18 (pred 0.01); 12 deg | 0.19 (pred 0.01); 23 deg |
| dynamic scaling around the mean | UL | principal | 0.17 (pred 0.01); -12 deg | 0.19 (pred 0.02); -13 deg | 0.18 (pred 0.01); -11 deg |
| dynamic scaling around the mean | UL | orthogonal | 0.26 (pred 0.01); 17 deg | 0.26 (pred 0.02); 12 deg | 0.27 (pred 0.01); 18 deg |
| dynamic scaling around the mean | LL | X only | 0.44 (pred 0.03); -14 deg | 0.48 (pred 0.06); 1 deg | 0.44 (pred 0.04); -11 deg |
| dynamic scaling around the mean | LL | Y only | 0.29 (pred 0.02); 6 deg | 0.31 (pred 0.02); 14 deg | 0.30 (pred 0.01); 10 deg |
| dynamic scaling around the mean | LL | principal | 0.28 (pred 0.02); 3 deg | 0.31 (pred 0.03); 4 deg | 0.29 (pred 0.03); 3 deg |
| dynamic scaling around the mean | LL | orthogonal | 0.52 (pred 0.03); 17 deg | 0.56 (pred 0.05); 41 deg | 0.54 (pred 0.04); 24 deg |

## 5. Sign asymmetry

Gain by sign of the edit (negative / positive direction, or alpha = -0.5 / +0.5), mean over sensors.

| profile | direction | HiFi-GAN (seed 1) | DDSP (seed 1) | Vocos (seed 1) |
|---|---|---|---|---|
| constant offset | X only | 0.10 / 0.14 | 0.08 / 0.13 | 0.11 / 0.11 |
| constant offset | Y only | 0.15 / 0.09 | 0.13 / 0.07 | 0.14 / 0.10 |
| constant offset | principal | 0.09 / 0.14 | 0.07 / 0.14 | 0.10 / 0.12 |
| constant offset | orthogonal | 0.10 / 0.12 | 0.09 / 0.11 | 0.11 / 0.11 |
| constant offset on speech frames | X only | 0.11 / 0.15 | 0.07 / 0.14 | 0.12 / 0.12 |
| constant offset on speech frames | Y only | 0.16 / 0.10 | 0.14 / 0.08 | 0.15 / 0.11 |
| constant offset on speech frames | principal | 0.09 / 0.16 | 0.07 / 0.15 | 0.10 / 0.14 |
| constant offset on speech frames | orthogonal | 0.11 / 0.15 | 0.09 / 0.13 | 0.12 / 0.13 |
| dynamic scaling around the mean | X only | 0.30 / 0.04 | 0.29 / 0.02 | 0.32 / 0.03 |
| dynamic scaling around the mean | Y only | 0.27 / 0.13 | 0.25 / 0.12 | 0.27 / 0.13 |
| dynamic scaling around the mean | principal | 0.32 / 0.10 | 0.29 / 0.10 | 0.33 / 0.09 |
| dynamic scaling around the mean | orthogonal | 0.27 / 0.15 | 0.26 / 0.12 | 0.28 / 0.15 |
| three local bumps (300 ms) | X only | 0.19 / 0.20 | 0.17 / 0.17 | 0.22 / 0.19 |
| three local bumps (300 ms) | Y only | 0.27 / 0.27 | 0.25 / 0.24 | 0.27 / 0.26 |
| three local bumps (300 ms) | principal | 0.21 / 0.25 | 0.20 / 0.20 | 0.22 / 0.25 |
| three local bumps (300 ms) | orthogonal | 0.19 / 0.29 | 0.16 / 0.27 | 0.22 / 0.27 |

## 6. Reference values

Seed-1 systems: the natural-dynamics and single-channel reference values of the seed-1 runs are in phase3/analysis/probes_seeds.md (natural_variation_s1.csv, controllability_seeds_long.csv). The values below are those of seed 0.

Natural dynamics through the same loop (RESULTS.md section 5.2): centred slope 1.00 (HiFi-GAN), 0.96 (DDSP), 1.00 (Vocos); regression diagonal 0.98, 0.93, 0.97; between-utterance slope 0.94, 0.90, 0.93. Single-channel constant edits of one std: gain 0.12 (HiFi-GAN), 0.10 (DDSP), 0.12 (Vocos), leakage 0.24. Constant edits along principal components of the 12-D EMA: gain 0.09 to 0.14 (6 utterances).

