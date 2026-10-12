# Controllability

## Key results (medians over 50 utterances; speaker-cluster bootstrap CI95 in the sections below)

| quantity | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| largest median F0 error over the four edits (cents) | 0.0 | 0.0 | 10.0 |
| loudness change for x0.5 / x2 (dB; targets -6.02 / +6.02) | -5.89 / +5.78 | -6.00 / +5.79 | -6.01 / +5.87 |
| voiced fraction before / after periodicity = 0 | 0.56 / 0.17 | 0.61 / 0.28 | 0.56 / 0.08 |
| single-channel EMA edit: gain (mean of 12 channels) | 0.12 | 0.10 | 0.12 |
| single-channel EMA edit: leakage (mean of 12 channels, std units) | 0.24 | 0.24 | 0.24 |
| natural variation: centred slope of re-extracted on input EMA (mean of 12 channels; gt floor 1.00) | 1.00 | 0.96 | 1.00 |
| natural variation: gain of one channel with the others fixed (regression diagonal) | 0.98 | 0.93 | 0.97 |

Reading. F0 and loudness are controlled closely by all three vocoders. Setting periodicity to 0 lowers the voiced fraction of the output most in Vocos (0.56 to 0.08), less in HiFi-GAN (0.56 to 0.17) and least in DDSP (0.61 to 0.28); none reaches 0. A constant plus or minus one-std edit of a single EMA channel is reproduced to only 4 to 32 percent (tongue body X has a negative gain) with off-target movement larger than the on-target gain, in all three vocoders; the same loop reproduces the natural EMA variation of speech with slope near 1 (section 5), and the low gain is not specific to single channels: a constant edit of one natural standard deviation along the principal components of natural EMA, which stays on the natural manifold in direction, is also reproduced to only 8 to 16 percent (section 6, 6 utterances per vocoder). So the failure is of a constant offset in the articulatory input, not of articulatory input as such. The data here do not say why. Candidate explanations not tested: the vocoder responds to articulatory change over time and not to the absolute position; the offset is added to silence and pause frames as well; the inversion (WavLM and a linear head) responds to phonetic content and reflects a shifted tongue position only weakly. A probe that scales the dynamics of an utterance around its mean, or shifts only speech frames, would separate these. Differences between the three vocoders in all of this are small (paired EMA gain differences of 0.02 or less) and each vocoder is a single training run.

Statistics. Section 1 to 4 are exactly those of EVALUATION.md section 6 on the stored probe results. The F0 intervals are degenerate because the pitch estimate is quantized (see section 1).

Not done: a probe that edits only dynamics, a repeat with another random seed of the vocoder, probes on other splits, and human listening. The EMA gain is also not corrected for any attenuation by the inversion itself: the gt row of section 5 shows the inversion is neutral on natural audio (slope 1.00), but it was not tested on audio with an artificially shifted tongue.


## Basis and statistics of sections 1 to 4

Probes of eval/probes/<system>/results.parquet: 50 test.clean utterances of 3 to 10 s (28 speakers, at most 2 per speaker), T1 speaker vectors, edits of the 15 raw input channels, unedited and edited synthesis re-extracted with the refit head without dither (EVALUATION.md section 6). Every cell is the median over the 50 utterances of the per-utterance value, with a speaker-cluster bootstrap CI95 of that median in brackets (28 speakers resampled with replacement, all their utterances kept, 10000 resamples, seed 0). Speaker-cluster bootstrap is used because 28 speakers are available. Each vocoder is a single training run, so differences between vocoders are differences between three trained models and not an estimate of an architecture effect; the intervals cover the sampling of speakers and utterances only, not training variance.

Caveat on the F0 rows. The re-extracted pitch comes from CREPE with a quantized pitch output, so per-utterance median shifts fall on a coarse grid and the bootstrap intervals of the median collapse onto grid points (for example exactly -200.0). The intervals therefore understate the true uncertainty of the shift; the useful reading is that the median error is at most 10 cents for every edit.

## 1. F0 control (frames voiced in both; target is the edit in cents)

| edit (semitones) | target (cents) | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|---|
| -4 | -400 | -400.0 (-400.0, -390.3) | -400.0 (-400.0, -400.0) | -390.0 (-390.0, -380.0) |
| -2 | -200 | -200.0 (-200.0, -200.0) | -200.0 (-200.0, -200.0) | -197.5 (-200.0, -190.0) |
| +2 | 200 | 200.0 (200.0, 200.0) | 200.0 (200.0, 200.0) | 200.0 (190.1, 200.0) |
| +4 | 400 | 400.0 (400.0, 400.0) | 400.0 (400.0, 400.0) | 399.8 (390.0, 400.0) |

Mean over the four edits of the absolute error of the per-utterance median shift (paired by utterance, median of the paired difference, speaker-cluster CI):

| comparison | difference (cents) | CI95 | fraction of utterances with a larger error for the first system |
|---|---|---|---|
| Vocos minus DDSP | 8.1 | (5.0, 12.5) | 0.98 |
| Vocos minus HiFi-GAN | 5.0 | (3.7, 10.0) | 0.94 |
| HiFi-GAN minus DDSP | 2.5 | (0.0, 5.0) | 0.94 |

## 2. Loudness control (dB change of the raw loudness; targets -6.02 and +6.02 dB)

| edit | target (dB) | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|---|
| x0.5 | -6.02 | -5.89 (-5.91, -5.87) | -6.00 (-6.03, -5.97) | -6.01 (-6.03, -5.99) |
| x2 | 6.02 | 5.78 (5.75, 5.81) | 5.79 (5.74, 5.87) | 5.87 (5.82, 5.90) |

Paired difference of the mean absolute error over the two edits:

| comparison | difference (dB) | CI95 | fraction of utterances with a larger error for the first system |
|---|---|---|---|
| Vocos minus DDSP | -0.043 | (-0.067, -0.011) | 0.26 |
| Vocos minus HiFi-GAN | -0.070 | (-0.089, -0.054) | 0.08 |
| HiFi-GAN minus DDSP | 0.032 | (-0.005, 0.066) | 0.62 |

## 3. Periodicity set to 0 (voiced-frame fraction of the re-extracted audio; ideal after the edit is 0)

| system | voiced fraction before | voiced fraction after | drop |
|---|---|---|---|
| HiFi-GAN | 0.564 (0.524, 0.587) | 0.167 (0.147, 0.185) | 0.398 (0.352, 0.435) |
| DDSP | 0.607 (0.579, 0.631) | 0.282 (0.241, 0.332) | 0.319 (0.270, 0.358) |
| Vocos | 0.560 (0.525, 0.588) | 0.075 (0.051, 0.125) | 0.463 (0.342, 0.534) |

Paired difference of the voiced fraction after the edit:

| comparison | difference | CI95 | fraction of utterances where the first system is larger |
|---|---|---|---|
| Vocos minus DDSP | -0.156 | (-0.286, -0.070) | 0.16 |
| Vocos minus HiFi-GAN | -0.055 | (-0.107, -0.016) | 0.28 |
| HiFi-GAN minus DDSP | -0.142 | (-0.167, -0.087) | 0.18 |

## 4. Single EMA channel edits (plus or minus one training std)

Gain is the mean change of the re-inverted channel divided by the edit (1 means reproduced); leakage is the mean absolute change of the other 11 channels in training std. Per utterance the two signs are averaged before taking the median over utterances. The reference for leakage is the movement of the re-inverted EMA caused by edits that do not touch EMA (last block), which is the response of the loop vocoder to a change of F0, loudness or periodicity and gives the size of off-target movement that is not specific to the edited channel.

| channel | gain HiFi-GAN | gain DDSP | gain Vocos | leakage HiFi-GAN | leakage DDSP | leakage Vocos |
|---|---|---|---|---|---|---|
| TDX | 0.26 (0.24, 0.29) | 0.24 (0.22, 0.26) | 0.29 (0.28, 0.31) | 0.37 (0.36, 0.38) | 0.39 (0.38, 0.40) | 0.37 (0.36, 0.38) |
| TDY | 0.05 (0.03, 0.06) | 0.05 (0.02, 0.07) | 0.07 (0.04, 0.08) | 0.23 (0.23, 0.24) | 0.22 (0.21, 0.22) | 0.22 (0.22, 0.22) |
| TBX | -0.15 (-0.16, -0.11) | -0.16 (-0.18, -0.14) | -0.16 (-0.18, -0.14) | 0.46 (0.45, 0.47) | 0.47 (0.46, 0.48) | 0.44 (0.43, 0.45) |
| TBY | 0.32 (0.31, 0.33) | 0.26 (0.25, 0.28) | 0.31 (0.29, 0.32) | 0.29 (0.29, 0.30) | 0.28 (0.27, 0.29) | 0.28 (0.27, 0.28) |
| TTX | 0.25 (0.23, 0.27) | 0.25 (0.22, 0.26) | 0.24 (0.23, 0.27) | 0.29 (0.28, 0.30) | 0.31 (0.30, 0.31) | 0.28 (0.27, 0.29) |
| TTY | 0.12 (0.12, 0.14) | 0.11 (0.10, 0.12) | 0.12 (0.11, 0.14) | 0.22 (0.21, 0.22) | 0.22 (0.21, 0.23) | 0.21 (0.21, 0.22) |
| LIX | 0.11 (0.10, 0.12) | 0.08 (0.06, 0.08) | 0.09 (0.09, 0.10) | 0.21 (0.20, 0.22) | 0.21 (0.20, 0.21) | 0.21 (0.20, 0.22) |
| LIY | 0.08 (0.07, 0.09) | 0.07 (0.07, 0.08) | 0.08 (0.07, 0.10) | 0.18 (0.18, 0.19) | 0.17 (0.17, 0.18) | 0.18 (0.18, 0.19) |
| ULX | 0.06 (0.06, 0.07) | 0.06 (0.05, 0.07) | 0.06 (0.05, 0.07) | 0.13 (0.13, 0.14) | 0.14 (0.14, 0.14) | 0.13 (0.13, 0.14) |
| ULY | 0.05 (0.05, 0.06) | 0.04 (0.04, 0.05) | 0.05 (0.05, 0.06) | 0.11 (0.10, 0.11) | 0.10 (0.10, 0.11) | 0.11 (0.10, 0.11) |
| LLX | 0.11 (0.11, 0.13) | 0.11 (0.10, 0.12) | 0.13 (0.13, 0.15) | 0.26 (0.26, 0.26) | 0.27 (0.26, 0.28) | 0.25 (0.25, 0.25) |
| LLY | 0.09 (0.08, 0.10) | 0.07 (0.06, 0.08) | 0.08 (0.07, 0.09) | 0.16 (0.16, 0.17) | 0.15 (0.15, 0.16) | 0.16 (0.15, 0.16) |
| mean of 12 channels | 0.12 (0.11, 0.12) | 0.10 (0.09, 0.10) | 0.12 (0.11, 0.12) | 0.24 (0.24, 0.25) | 0.24 (0.24, 0.25) | 0.24 (0.23, 0.24) |

Off-target movement of the re-inverted EMA (mean absolute change over all 12 channels, training std) under edits that do not touch EMA:

| edit type | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| f0 | 0.12 (0.12, 0.13) | 0.16 (0.15, 0.16) | 0.14 (0.14, 0.15) |
| loudness | 0.08 (0.08, 0.08) | 0.07 (0.07, 0.08) | 0.08 (0.07, 0.08) |
| periodicity | 0.18 (0.16, 0.19) | 0.20 (0.19, 0.22) | 0.19 (0.18, 0.22) |

Paired differences between vocoders (median of per-utterance differences of the mean over 12 channels and two signs):

| metric | comparison | difference | CI95 | fraction of utterances where the first is larger |
|---|---|---|---|---|
| EMA gain (12 channels, mean) | Vocos minus DDSP | 0.017 | (0.014, 0.022) | 0.96 |
| EMA gain (12 channels, mean) | Vocos minus HiFi-GAN | -0.001 | (-0.004, 0.002) | 0.44 |
| EMA gain (12 channels, mean) | HiFi-GAN minus DDSP | 0.019 | (0.017, 0.021) | 0.98 |
| EMA leakage (12 channels, mean) | Vocos minus DDSP | -0.007 | (-0.010, -0.005) | 0.16 |
| EMA leakage (12 channels, mean) | Vocos minus HiFi-GAN | -0.006 | (-0.008, -0.003) | 0.12 |
| EMA leakage (12 channels, mean) | HiFi-GAN minus DDSP | -0.001 | (-0.004, 0.002) | 0.46 |

## 5. Natural-variation baseline for the EMA gains

Basis. All test.clean frames of the T1 re-extraction (arrays/.../reextract) against the cached input EMA: 4686 utterances, 1,512,210 frames, 39 speakers (908_157963_000024_000000 has no re-extraction for any system and is absent; the re-extraction is one frame shorter than the input and is compared on the common frames). CI95 by speaker-cluster bootstrap over the 39 speakers (10000 resamples, seed 0) from per-speaker sufficient statistics. The input is the regressor (it is noise free), the re-extracted EMA is the response.

Definitions. Pooled: all frames. Centred: input and output minus their utterance means (dynamics within an utterance). Between: regression on utterance means (what a constant offset resembles). Diagonal: the entry B[c,c] of the 12 by 12 regression of the centred re-extracted EMA on all 12 centred input channels, which is the gain of a channel with the other eleven held fixed, the quantity that an isolated single-channel edit measures. Linear leakage: mean over c' not equal to c of |B[c',c]| std_c / std_c', the leakage that the same regression predicts for a one-std edit of channel c.

Mean over the 12 channels of each measure (the range over channels is in natural_variation.csv):

| system | pooled slope | centred slope | centred r | between slope | diagonal B[c,c] | linear leakage | F0 slope (log2 f0) |
|---|---|---|---|---|---|---|---|
| gt (floor) | 1.000 (0.999 to 1.003) | 1.000 (0.999 to 1.003) | 0.999 (0.997 to 1.000) | 0.990 (0.979 to 0.995) | 1.001 (0.986 to 1.015) | 0.008 (0.001 to 0.038) | 1.000 (1.000, 1.000) |
| Vocos mel copy synthesis | 0.972 (0.960 to 0.987) | 0.973 (0.960 to 0.987) | 0.983 (0.979 to 0.986) | 0.957 (0.928 to 0.976) | 0.957 (0.929 to 1.006) | 0.020 (0.005 to 0.044) | 0.998 (0.997, 0.999) |
| HiFi-GAN | 1.001 (0.995 to 1.008) | 1.002 (0.996 to 1.010) | 0.974 (0.968 to 0.978) | 0.935 (0.915 to 0.955) | 0.981 (0.932 to 1.024) | 0.021 (0.005 to 0.047) | 0.999 (0.999, 1.000) |
| DDSP | 0.958 (0.932 to 0.978) | 0.959 (0.933 to 0.979) | 0.964 (0.957 to 0.970) | 0.903 (0.879 to 0.933) | 0.929 (0.860 to 0.990) | 0.034 (0.005 to 0.056) | 0.999 (0.998, 1.000) |
| Vocos | 0.998 (0.979 to 1.016) | 1.000 (0.980 to 1.017) | 0.972 (0.965 to 0.976) | 0.931 (0.914 to 0.949) | 0.974 (0.913 to 1.022) | 0.024 (0.005 to 0.050) | 0.997 (0.994, 0.999) |

Each cell is the mean over channels with the minimum and maximum channel in brackets. Per-channel values with bootstrap CI are in natural_variation.csv.

Per-channel centred slope and diagonal (bootstrap CI95 is at most plus or minus 0.02 for every cell):

| channel | centred Vocos mel copy synthesis | centred HiFi-GAN | centred DDSP | centred Vocos | diag. Vocos mel copy synthesis | diag. HiFi-GAN | diag. DDSP | diag. Vocos |
|---|---|---|---|---|---|---|---|---|
| TDX | 0.97 | 1.00 | 0.96 | 1.00 | 0.96 | 0.94 | 0.88 | 0.94 |
| TDY | 0.96 | 1.00 | 0.93 | 0.99 | 0.94 | 1.00 | 0.94 | 0.98 |
| TBX | 0.97 | 1.00 | 0.96 | 0.99 | 0.93 | 0.99 | 0.95 | 0.97 |
| TBY | 0.97 | 1.00 | 0.94 | 0.99 | 0.98 | 1.00 | 0.92 | 1.00 |
| TTX | 0.97 | 1.00 | 0.96 | 0.99 | 1.01 | 1.02 | 0.99 | 1.02 |
| TTY | 0.98 | 1.00 | 0.96 | 1.00 | 0.95 | 0.96 | 0.94 | 0.96 |
| LIX | 0.99 | 1.01 | 0.98 | 1.01 | 0.93 | 0.93 | 0.86 | 0.91 |
| LIY | 0.98 | 1.00 | 0.95 | 1.01 | 0.97 | 1.01 | 0.93 | 1.01 |
| ULX | 0.97 | 1.01 | 0.97 | 1.00 | 0.96 | 1.00 | 0.95 | 0.99 |
| ULY | 0.96 | 1.00 | 0.96 | 0.98 | 0.95 | 0.97 | 0.93 | 0.96 |
| LLX | 0.98 | 1.01 | 0.98 | 1.02 | 0.94 | 0.98 | 0.94 | 0.99 |
| LLY | 0.98 | 1.00 | 0.97 | 1.01 | 0.96 | 0.97 | 0.93 | 0.96 |

## 6. Probe along the principal components of the natural EMA

Done (CPU synthesis). Directions: PCA of the 12 EMA channels of train.clean.100 frames (every 20th frame); variances of PC1 to PC4 are 3.49, 3.12, 1.26, 1.11 (92 percent of the total). The edit adds a constant vector s sqrt(lambda_k) v_k to the 12 EMA channels of every frame (the same constant-offset protocol as section 4). Subset: 6 of the 50 probe utterances (every 8th of the sorted list, 6 utterances; fewer than the main probe because CPU extraction costs about 4 minutes per utterance and vocoder), so the intervals are utterance bootstrap (percentile, 10000 resamples), not speaker-cluster, and they cover few utterances. Gain along the direction is mean_frames(delta . v_k) / (s sqrt(lambda_k)); leakage is the mean absolute change of the component of delta orthogonal to v_k, over the 12 channels, in training std. PC10 is a direction with little natural variance, scaled to Euclidean norm 0.9 (the norm of a single-channel edit of one training std).

| direction | edit size | metric | HiFi-GAN | DDSP | Vocos | regression prediction HiFi-GAN / DDSP / Vocos |
|---|---|---|---|---|---|---|
| PC1 | 1 natural SD | gain | 0.14 (0.12, 0.16) | 0.10 (0.08, 0.13) | 0.14 (0.12, 0.16) | 1.00 / 0.96 / 1.00 |
| PC1 | 1 natural SD | leakage | 0.12 (0.12, 0.13) | 0.11 (0.10, 0.11) | 0.13 (0.12, 0.13) | 0.00 / 0.01 / 0.00 |
| PC2 | 1 natural SD | gain | 0.13 (0.10, 0.15) | 0.11 (0.07, 0.13) | 0.13 (0.09, 0.14) | 1.01 / 0.97 / 1.02 |
| PC2 | 1 natural SD | leakage | 0.11 (0.10, 0.11) | 0.10 (0.09, 0.10) | 0.11 (0.10, 0.11) | 0.00 / 0.01 / 0.00 |
| PC3 | 1 natural SD | gain | 0.13 (0.12, 0.16) | 0.11 (0.10, 0.13) | 0.14 (0.12, 0.15) | 1.00 / 0.96 / 0.99 |
| PC3 | 1 natural SD | leakage | 0.09 (0.09, 0.10) | 0.08 (0.08, 0.09) | 0.09 (0.09, 0.09) | 0.01 / 0.01 / 0.01 |
| PC4 | 1 natural SD | gain | 0.11 (0.09, 0.14) | 0.09 (0.07, 0.10) | 0.11 (0.10, 0.13) | 1.00 / 0.96 / 0.99 |
| PC4 | 1 natural SD | leakage | 0.11 (0.11, 0.12) | 0.10 (0.09, 0.11) | 0.12 (0.11, 0.12) | 0.00 / 0.01 / 0.00 |
| PC1 | 2 natural SD | gain | 0.16 (0.15, 0.17) | 0.13 (0.11, 0.14) | 0.16 (0.13, 0.18) | 1.00 / 0.96 / 1.00 |
| PC1 | 2 natural SD | leakage | 0.23 (0.22, 0.24) | 0.21 (0.20, 0.22) | 0.22 (0.22, 0.24) | 0.01 / 0.03 / 0.01 |
| PC10 | norm 0.9 | gain | 0.09 (0.08, 0.10) | 0.08 (0.07, 0.11) | 0.08 (0.07, 0.09) | 0.98 / 0.92 / 0.97 |
| PC10 | norm 0.9 | leakage | 0.29 (0.28, 0.31) | 0.33 (0.32, 0.34) | 0.28 (0.27, 0.29) | 0.03 / 0.03 / 0.02 |

For comparison, the single-channel edits of section 4 give a median gain of 0.12 / 0.10 / 0.12 and leakage of 0.24 / 0.24 / 0.24 (HiFi-GAN / DDSP / Vocos, mean of 12 channels). The regression prediction is what the linear map fitted on natural dynamics (section 5) would give for the same constant edit.

