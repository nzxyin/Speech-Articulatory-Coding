# Failure cases and headline findings (test.clean)

Scope: HiFi-GAN, DDSP and Vocos at their final checkpoints (400k steps), all numbers from the per-utterance Phase 3 results (`master.parquet`, built by `scripts/build_master.py` and `scripts/add_sims.py`; it reproduces the table means) and the re-extracted feature arrays. **Each vocoder arm is a single training run.** Differences between vocoders below describe these three trained models; they cannot be attributed to the architecture alone, and no run-to-run variance is estimated. The paired evidence is in `paired.md`.

Audio, spectrogram PNGs and a `README.txt` per case are under `/data/user_data/xoy/sparc-vocoders/eval/failures/<case>/` (16-bit PCM, 24 kHz; the PNG shows the four spectrograms, the re-extracted F0 of the voiced frames, loudness relative to the ground-truth maximum, and periodicity; ground truth in black, HiFi-GAN blue, DDSP orange, Vocos green). Per-utterance numbers of all cases: `failure_cases_metrics.csv`. No listening was done; every statement rests on metrics, transcripts, spectrograms and feature tracks.

## Headline findings

1. **HiFi-GAN is the strongest of the three on perceived quality and speaker similarity, and the evidence is the same under T1, T2 and T3.** UTMOS 4.165 against 3.832 (DDSP) and 3.896 (Vocos); paired differences +0.33 (CI +0.28 to +0.40, all 39 speakers agree) and +0.27 (CI +0.24 to +0.30, 97 percent agree). It sits 0.04 below the ground truth (4.208). DDSP and Vocos are not distinguishable on UTMOS (difference -0.065, CI -0.146 to +0.002, 64 percent of speakers agree). The caveat is the single training run: this is a statement about three trained models, not about three architectures.
2. **Intelligibility does not separate the vocoders.** Corpus WER is 2.45, 2.52 and 2.55 percent (HiFi-GAN, DDSP, Vocos) against 1.89 percent for the ground truth. No WER or CER pair is supported in any condition (at most 65 percent of speakers agree with the sign; largest difference 0.14 percentage points). The word errors are shared: the per-utterance excess over the ground truth correlates 0.55 to 0.63 between vocoders, and the worst utterances are the same ones (case `shared_content_errors`). Utterances of five reference words or fewer contribute only about 5 percent of the word errors.
3. **The vocoders fail on different prosodic quantities, and the differences are supported.** DDSP follows F0 best (F0 RMSE 18.2 cents, 98 percent of voiced frames within 50 cents), but it has the worst voicing decisions (VDE 0.066 against 0.039 and 0.042; it marks 13.5 percent of ground-truth unvoiced frames as voiced against 4.8 percent for the other two) and the worst loudness (RMSE 2.80 dB, concentrated in pauses and quiet frames, where it is too quiet by 10 to 20 dB in the examples). Vocos has the largest F0 error (35.0 cents, 86 percent within 50 cents), but this is a widespread deviation of 50 to 150 cents, strongest on male voices, and not octave errors (6 of 797816 voiced frames lie within 100 cents of an octave). Below about 70 Hz both HiFi-GAN and Vocos overshoot (median +180 and +420 cents below 60 Hz), DDSP does not. All three pairs are supported on F0 RMSE under T1, T2 and T3.
4. **DDSP loses on male speakers.** Its UTMOS is 3.69 for male and 3.93 for female speakers (HiFi-GAN 4.15 and 4.17), 77 of its utterances of 3 s or longer fall below UTMOS 3 (HiFi-GAN 3, Vocos 14), and its speaker similarity drops most for low-pitched male speakers (speaker 7729: Sim spk 0.726 against 0.867 for the ground truth and 0.826 for HiFi-GAN). Spectral band levels and noise floor do not explain it; false voicing and quiet pauses are correlated with the UTMOS drop (r = -0.40 for VDE) but a causal link was not tested.
5. **T2 and T3 mainly show the limits of the similarity metric, not a different vocoder ranking.** The paired differences between vocoders are almost unchanged across conditions. Sim same under T2 correlates 0.80 to 0.86 with the ECAPA cosine between the ground truth of the target and of its reference utterance (mean 0.595; 239 utterances below 0.3), so a large part of the T1-to-T2 fall (0.857 to 0.724 for HiFi-GAN) reflects how different two utterances of one speaker are. Under T3 all vocoders exceed the ground-truth Sim spk ceiling (0.783 to 0.787 against 0.749) because the speaker-mean vector moves the output to the centre of the speaker's embedding cloud; Sim spk under T3 should not be read as fidelity.

## Case overview

| # | case | system(s) | what goes wrong | key numbers |
|---|---|---|---|---|
| 1 | `vocos_f0_jitter_low_pitch` | Vocos | segments of 50 to 150 cents F0 offset and voicing dropouts on male speech | F0 within 50 cents 0.58 and 0.495 against 0.79 and 0.92 (HiFi-GAN), 0.99 and 0.98 (DDSP) |
| 2 | `vocos_octave_error` | Vocos | one octave-high frame defines the whole utterance F0 RMSE | 1170 cents from 1 frame in both-voiced set |
| 3 | `f0_floor_and_octave_frames` | Vocos, HiFi-GAN | pitch floor near 75 to 80 Hz, octave-sized frames | Vocos 189 cents, HiFi-GAN 220 cents |
| 4 | `ddsp_loudness_unvoiced` | DDSP | pauses and quiet frames far too quiet | loudness RMSE 7.20 and 5.94 dB against 2.0 to 2.7 dB |
| 5 | `ddsp_low_utmos_male` | DDSP | UTMOS 2.4 to 2.5 where HiFi-GAN is at ground-truth level | DDSP 2.48 and 2.41, HiFi-GAN 4.35 and 4.02 |
| 6 | `ddsp_male_speaker_similarity` | DDSP | low speaker similarity for male speaker 7729 | Sim same 0.632 against 0.882 (HiFi-GAN) |
| 7 | `short_utterances` | all | utterances below 1 s: metrics are not meaningful, errors are large | ground-truth UTMOS 1.82 and 1.77 |
| 8 | `too_short_to_reextract` | all | re-extraction fails | 0.40 s, 1 utterance, all prosody cells n = 4686 |
| 9 | `t2_reference_mismatch` | all | Sim same collapses under T2 for speaker 1284 | 0.816 (T1) to 0.129 (T2) for HiFi-GAN |
| 10 | `t3_exceeds_ground_truth_ceiling` | all | T3 Sim spk above the ground-truth ceiling | 0.788 to 0.795 against 0.557 |
| 11 | `shared_content_errors` | all | the same words are wrong for all three vocoders | 5, 7, 7 word errors against 0 |

## Case details

### 1. Vocos F0 deviations on low-pitched male speech (`vocos_f0_jitter_low_pitch`)

Utterances 7729_102255_000002_000010 and 908_31957_000017_000001 (T1).

The task description gives two figures for Vocos: F0 RMSE about twice that of DDSP and 14 percent of frames off by more than 50 cents. Over all voiced-in-both frames (797816) the numbers are 13.9 percent above 50 cents, 1.9 percent above 100 cents, 0.04 percent (298 frames) above 300 cents, and 6 frames within 100 cents of an octave (1100 to 1300 cents). The signed errors are symmetric (mean +1.6 cents, median 0, 5th and 95th percentile -60 and +60 cents). **So the large F0 RMSE is not caused by octave errors; it is a broad deviation of one to several CREPE bins (20 cents per bin).** Frames above 300 cents carry only 5.7 percent of Vocos' squared F0 error; 63 percent comes from frames between 50 and 300 cents. Removing frames above 300 cents lowers the RMSE only from 36.97 to 35.91 cents (these are frame-pooled values; the table value of 35.0 is the mean of per-utterance RMSE).

The errors are slowly varying, not frame-to-frame noise: the lag-1 autocorrelation of the error over consecutive voiced frames is 0.81 for Vocos (HiFi-GAN 0.70, DDSP 0.53). In 7729_102255_000002_000010 the Vocos track runs roughly 5 to 10 percent above the ground truth for about 0.7 s (1.2 to 1.9 s), and its periodicity drops to zero at about 0.95 s and 1.1 s while the ground truth stays voiced.

Where it happens (share of voiced-in-both frames with an error above 50 cents, Vocos / HiFi-GAN / DDSP):

| ground-truth F0 | Vocos | HiFi-GAN | DDSP |
|---|---|---|---|
| up to 80 Hz | 0.56 (n = 7252) | 0.26 | 0.01 |
| 80 to 100 Hz | 0.25 | 0.11 | 0.01 |
| 100 to 130 Hz | 0.18 | 0.07 | 0.02 |
| 130 to 160 Hz | 0.20 | 0.05 | 0.03 |
| 160 to 200 Hz | 0.095 | 0.045 | 0.02 |
| 200 to 300 Hz | 0.06 to 0.07 | 0.03 | 0.02 |
| 300 to 400 Hz | 0.10 | 0.06 | 0.02 |
| male / female speakers | 0.195 / 0.100 | 0.075 / 0.041 | 0.022 / 0.018 |

The error rate is not elevated at utterance ends (0.20 within 2 frames of the end, but only 1280 such frames, against 0.14 elsewhere), and not near voicing boundaries for Vocos (0.13 within 1 frame of a ground-truth unvoiced frame, 0.10 to 0.19 further inside); for DDSP and HiFi-GAN the boundary frames are somewhat worse (0.058 and 0.060). The speakers with the most Vocos off-frames are all male: 7729 (28 percent; median F0 94 Hz), 908, 7176, 8455 and 7021 (20 to 26 percent); speaker 61 (one utterance) has none. UTMOS and WER of the affected utterances are normal (7729_102255_000002_000010: UTMOS 4.27; 908_31957_000017_000001: 4.30, no word errors), so the loss is in pitch precision, which UTMOS does not register. Whether the cause is the 24 kHz iSTFT head, the frame rate or the amount of low-pitched training speech was not tested.

Files: `vocos_f0_jitter_low_pitch/`. Statistics: `f0_error_decomposition.csv`, `f0_error_by_gt_f0_band.csv`.

### 2. Vocos octave error on a 0.52 s utterance (`vocos_octave_error`)

Utterance 7176_92135_000076_000001 (T1, the word 'sugar').

The largest per-utterance Vocos F0 RMSE in the test set is 1170 cents (within 50 cents: 0.00, VDE 0.25). The ground truth is voiced for about five frames (0.17 to 0.27 s, F0 about 88 Hz); Vocos is voiced only at the first of these, there at about 180 Hz, one octave too high. Exactly one frame is voiced in both streams (`n_voiced_both` = 1), so **the utterance values rest on a single frame**. This is the clearest case of why per-utterance F0 RMSE on very short or sparsely voiced utterances should not be ranked. The utterance is also hard for the other systems: Whisper hears 'Show me your...' (HiFi-GAN, 4 word errors), 'Cheerio.' (DDSP) and 'Shavuot.' (Vocos), against 'Sugar?' for the ground truth, and the ground-truth UTMOS is only 2.96.

Files: `vocos_octave_error/`.

### 3. Pitch floor and octave-sized frames on longer utterances (`f0_floor_and_octave_frames`)

Utterances 2300_131720_000026_000007 (Vocos 189 cents) and 7021_79740_000017_000002 (HiFi-GAN 220 cents), both T1.

In 2300_131720_000026_000007 the ground truth drops to creaky voice at about 57 Hz (1.05 to 1.2 s and 3.05 to 3.2 s). The Vocos track stays at a nearly constant 75 to 80 Hz in both stretches. This is a general behaviour, not one event. Median signed F0 error by ground-truth F0 (cents; n voiced frames Vocos / HiFi-GAN / DDSP):

| ground-truth F0 | Vocos | HiFi-GAN | DDSP | n (Vocos, HiFi-GAN, DDSP) |
|---|---|---|---|---|
| up to 60 Hz | +420 | +180 | 0 | 171, 99, 258 |
| 60 to 70 Hz | +200 | +20 | 0 | 883, 698, 993 |
| 70 to 80 Hz | +30 | 0 | 0 | 6198, 6222, 6950 |

RMS error in the lowest bin is 443 (Vocos), 211 (HiFi-GAN) and 16 cents (DDSP). In the DDSP design the F0 channel sets the phase of the harmonic oscillators directly (Phase 1 report, section 6), which is consistent with DDSP being unaffected; the two neural vocoders have to learn the mapping from the input channel and fail below about 70 Hz. The affected bins hold about 0.1 percent of the voiced frames, but a handful of such frames dominates the largest per-utterance F0 RMSE values (HiFi-GAN's worst, 220 cents, is 7021_79740_000017_000002, where a few frames near 2.65 s are tracked at about 180 Hz against about 85 Hz in the ground truth; the exact frame count was not tabulated). Frames more than 300 cents off occur in 298 (Vocos), 32 (HiFi-GAN) and 8 (DDSP) frames, in 83, 23 and 5 utterances. Whether the neural vocoders saw little training speech below 70 Hz was not checked. 7021_79730_000056_000002 (DDSP, 102 cents, the largest DDSP F0 RMSE) was plotted but not analysed.

Files: `f0_floor_and_octave_frames/`. Statistic: `f0_error_by_gt_f0_band.csv`.

### 4. DDSP: pauses and quiet frames too quiet (`ddsp_loudness_unvoiced`)

Utterances 4507_16021_000036_000001 and 8455_210777_000033_000000 (T1).

DDSP's loudness RMSE is 2.80 dB (HiFi-GAN 1.49, Vocos 1.83; paired difference to HiFi-GAN +1.31 dB, CI +1.21 to +1.43, 100 percent of speakers) and its mean bias is -0.97 dB (HiFi-GAN -0.34, Vocos -0.44). The bias explains only 12 percent of its mean squared loudness error; the rest is concentrated where the signal is quiet:

| frames | DDSP RMSE / bias (dB) | HiFi-GAN | Vocos |
|---|---|---|---|
| ground truth voiced | 1.23 / -0.45 | 0.83 / -0.20 | 1.12 / -0.30 |
| ground truth unvoiced | 4.40 / -1.82 | 2.13 / -0.49 | 2.56 / -0.60 |
| 30 to 40 dB below the utterance maximum | 5.29 / -2.43 | 2.70 / -0.44 | 3.01 / -0.48 |
| 20 to 30 dB below | 4.22 / -1.61 | 1.89 / -0.50 | 2.24 / -0.60 |
| within 10 dB of the maximum | about 1.5 to 1.6 / -0.7 to -0.9 | 0.9 to 1.0 | 1.3 to 1.65 |

(Mean over frames above the 5th percentile of the utterance, as in the metric; males -1.26 dB, females -0.82 dB bias for DDSP.) In 4507_16021_000036_000001 the pause from 0.5 to 1.35 s contains room noise at about -25 dB relative to the loudest frame; DDSP renders it at about -40 dB and the spectrogram is visibly darker there, while HiFi-GAN and Vocos keep the level. Loudness RMSE is 7.20 dB for DDSP against 2.28 and 2.74 dB. DDSP therefore produces cleaner, emptier pauses than the recording. A sample of 200 utterances showed no excess noise: in quiet frames DDSP is 0.8 dB below the ground truth in the 8 to 12 kHz band (HiFi-GAN 1.1, Vocos 1.0), so the level error is not a high-frequency effect. The low-frequency band (0 to 1 kHz) in unvoiced frames is 2.0 dB too low for DDSP against 0.6 to 0.7 dB for the others.

Files: `ddsp_loudness_unvoiced/`.

### 5. DDSP: low UTMOS on male speakers (`ddsp_low_utmos_male`)

Utterances 7021_85628_000020_000000 (DDSP 2.48, HiFi-GAN 4.35, Vocos 4.09, ground truth 4.36) and 260_123440_000002_000000 (DDSP 2.41, HiFi-GAN 4.02, Vocos 3.41, ground truth 3.72), both T1.

DDSP has 196 utterances with UTMOS below 3 (HiFi-GAN 35, Vocos 95, ground truth 27); for utterances of 3 s or longer the counts are 77, 3 and 14 of 3266. 130 of the 196 DDSP utterances are male. Mean UTMOS by sex: DDSP 3.69 (male) and 3.93 (female), HiFi-GAN 4.15 and 4.17, Vocos 3.95 and 3.86, ground truth 4.18 and 4.23. The largest speaker-level UTMOS drops against the ground truth are 1.00 (speaker 7021) and 0.90 (2300) for DDSP, against 0.25 and 0.14 for HiFi-GAN.

What the evidence shows for DDSP: (a) higher false-voicing rate (13.5 percent of ground-truth unvoiced frames against 4.8 percent for HiFi-GAN and 4.8 percent for Vocos; the identical 32873 frames for HiFi-GAN and Vocos were checked and are a coincidence: the counts differ in 3193 of the first 4000 utterances); in 7021_85628_000020_000000 the plot shows spurious voiced runs at about 0.2 s, 2.05 s, 4.7 s and 8.6 s where the ground truth is unvoiced; (b) pauses 10 to 15 dB too quiet (case 4); (c) a slightly more harmonic spectrum (harmonic peak-to-valley ratio below 4 kHz 27.2 dB against 25.7 dB for the ground truth, HiFi-GAN 25.8, Vocos 25.3; 200 utterances), the difference being larger for male speakers (24.0 against 22.3 dB). Within DDSP the per-utterance UTMOS correlates with VDE at -0.40, with MCD at -0.50, with EMA r at +0.46 and with loudness bias at +0.29 (for Vocos the VDE correlation is only -0.06). These are correlations over utterances and do not identify the cause; a comparison of band levels found nothing that explains the UTMOS gap by itself. The DDSP design has no explicit voicing gate (Phase 1 report, section 6: the network must switch the harmonic gain off), which is consistent with the false voicing, but this was not tested. Because the DDSP arm is a single run, a data, optimisation or conditioning effect specific to this run cannot be separated from a design effect.

Files: `ddsp_low_utmos_male/`.

### 6. DDSP: speaker similarity for male speaker 7729 (`ddsp_male_speaker_similarity`)

Utterance 7729_102255_000012_000001 (T1, 6.4 s).

For speaker 7729 (97 utterances) the T1 Sim spk is 0.867 (ground truth), 0.826 (HiFi-GAN), 0.799 (Vocos) and 0.726 (DDSP); Sim same is 0.883 (HiFi-GAN), 0.857 (Vocos) and 0.782 (DDSP). In the listed utterance Sim same is 0.882 (HiFi-GAN), 0.831 (Vocos) and 0.632 (DDSP), and DDSP and Vocos both transcribe 'Lawrence' as 'Loris' (HiFi-GAN and the ground truth are correct). The same deficit appears for the other low-pitched male speakers: Sim spk against the ground truth in T2 is lower by 0.138 (7729), 0.110 (7127), 0.100 (1089), 0.089 (8455) and 0.078 (1320) for DDSP, while HiFi-GAN is within 0.04 of the ground truth for all five. It persists under T2 and T3 conditioning, so it is not tied to the own-utterance speaker vector. Averaged over speakers, DDSP's Sim spk is 0.040 below HiFi-GAN under T1 (CI 0.033 to 0.049, 100 percent of speakers).

Files: `ddsp_male_speaker_similarity/`.

### 7. Utterances below 1 s (`short_utterances`)

Utterances 1580_141084_000040_000000 ('Aha', 0.68 s) and 672_122797_000028_000000 ('Help', 0.48 s), T1.

Every extreme of the per-utterance lists is a very short utterance. For 'Aha' the ground truth has UTMOS 1.82 and Sim spk -0.004; all three vocoders mishear the word (Whisper: 'A fath.', 'La Fave.', 'The Father'; WER 2.0 because two words replace one), loudness RMSE is 3.4 to 7.5 dB (DDSP bias -4.71 dB), the mean EMA correlation falls to 0.78 to 0.79 (test-set mean 0.96 to 0.97), and DDSP and Vocos reach only 0.75 and 0.50 F0 within 50 cents on the few voiced frames. 'Help' is recognised by all systems, but DDSP has a loudness bias of -4.89 dB (RMSE 7.65 dB), Sim same 0.26 and VDE 0.136. Across all utterances, the correlation of UTMOS with duration is 0.12 (HiFi-GAN), 0.00 (DDSP) and 0.02 (Vocos), so very short utterances are a tail effect, not a general trend; the five lowest ground-truth UTMOS values (1.77 to 2.23) also belong to utterances of 0.48 to 1.28 s. Word errors in utterances of five reference words or fewer are only 119 (HiFi-GAN), 123 (DDSP) and 127 (Vocos) of 2221, 2284 and 2309, so these cases matter for the tails of the distributions and not for corpus WER. For these utterances UTMOS, Sim spk and the F0 metrics should be interpreted with caution (few frames, ground truth already outside the range for which the metrics were designed).

Files: `short_utterances/`.

### 8. The utterance too short to re-extract (`too_short_to_reextract`)

Utterance 908_157963_000024_000000 (T1, 0.40 s, reference text '3').

The feature re-extraction fails for the ground truth and for every system (the 480 T-sample cut is shorter than the padding length of the zero-phase filter in the extractor), so the arrays contain no features for this id and all F0, VDE, periodicity, loudness and EMA metrics are undefined. The prosody cells of the tables for T1 therefore use 4686 of 4687 utterances. Audio-based metrics exist: UTMOS 2.48 (ground truth), 2.12 (HiFi-GAN), 1.55 (DDSP), 1.63 (Vocos); Whisper reads the ground truth as '3.' and the vocoders as 'Brie.', 'breathe.' and 'Where he...' (word errors 1, 1, 3). Sim same is 0.475 to 0.493 for all three. The utterance has a minor effect on the means (one utterance in 4687). The spectrogram PNG has no F0, loudness and periodicity panels for this case.

Files: `too_short_to_reextract/`.

### 9. T2: Sim same collapse for speaker 1284 (`t2_reference_mismatch`)

Utterances 1284_1181_000040_000005 (reference 1284_1180_000043_000000) and 1284_1181_000027_000000 (reference 1284_1180_000039_000000), both T2; the reference utterance audio is included.

Under T1 HiFi-GAN reaches Sim same 0.816 and 0.857 on these utterances; under T2 it falls to 0.129 and 0.254 (DDSP 0.111 and 0.203, Vocos 0.171 and 0.261). The outputs are otherwise sound (UTMOS 4.28 and 4.15, no extra word errors, F0 and loudness follow the target). The T2 output follows the reference: its ECAPA cosine with the ground truth of the reference is 0.599 and 0.648, with the target 0.129 and 0.254. The cause lies in ECAPA space, not in the vocoder: the cosine between the ground truth of the target and of its reference is only 0.114 and 0.247.

Over all T2 utterances, the correlation between Sim same and this target-reference cosine is 0.86 (HiFi-GAN), 0.80 (DDSP) and 0.85 (Vocos); the mean target-reference cosine is 0.595, and for the 239 utterances (5.1 percent) with a cosine below 0.3 the mean T2 Sim same is 0.485, 0.486 and 0.486 against 0.799, 0.764 and 0.794 for the 2490 utterances with a cosine of 0.6 or more. Speaker 1284 is an extreme case: even within one chapter the mean cosine between its ground-truth utterances is 0.40 (0.399 and 0.419 in the two chapters, 0.392 across them), which is why its mean T1-to-T2 fall in HiFi-GAN Sim same (0.829 to 0.547) is the largest of all speakers. The lowest per-speaker mean target-reference cosines are 1221 (0.383), 1284 (0.405), 672 (0.420), 3729 (0.471) and 260 (0.472). **A T2 fall of Sim same therefore partly measures how different two utterances of the same speaker are for ECAPA, not only a conditioning weakness of the vocoder.** Under T2 the same-chapter subset (n = 875) has higher similarity than the different-chapter subset (3811) for all three vocoders (for example HiFi-GAN Sim same 0.797 against 0.708), as the table reports.

Files: `t2_reference_mismatch/`. Statistics: `t2_case_embeddings.csv`, `t2_gt_gt_cosine.csv`.

### 10. T3: speaker-mean conditioning above the ground-truth ceiling (`t3_exceeds_ground_truth_ceiling`)

Utterance 3729_6852_000044_000000 (T3, 4.04 s).

The ground-truth Sim spk for this utterance is 0.557, the T3 outputs reach 0.788 (HiFi-GAN), 0.795 (DDSP) and 0.771 (Vocos); Sim same is only 0.663, 0.667 and 0.644 and the output is otherwise faultless (UTMOS 4.18 to 4.45, no word errors). On the whole set, Sim spk under T3 is 0.787 (HiFi-GAN), 0.731 (DDSP) and 0.783 (Vocos) against the ground-truth ceiling of 0.749; the T3 output has a higher Sim spk than the ground truth of the same utterance in 70 percent (HiFi-GAN), 36 percent (DDSP) and 66 percent (Vocos) of the utterances, and Sim spk rises from T1 to T3 for 89, 84 and 91 percent of utterances. Sim same falls from 0.857, 0.802 and 0.844 (T1) to 0.756, 0.722 and 0.755 (T3). The speaker-mean vector (an average over all utterances of the speaker) moves the output towards the centre of the speaker's embedding cloud, which is exactly what Sim spk (cosine to the mean of the speaker's other utterances) rewards. Speakers with a scattered embedding cloud profit most (3729: ground truth 0.633, HiFi-GAN T3 0.741; 1284: 0.626 and 0.698; Vocos 0.730 and 0.719). The metric therefore cannot rank T3 against T1 as a measure of fidelity. DDSP stays below the ceiling (0.731), consistent with case 6.

Files: `t3_exceeds_ground_truth_ceiling/`.

### 11. Word errors shared by all three vocoders (`shared_content_errors`)

Utterances 260_123440_000011_000000 and 121_127105_000008_000000 (T1).

Total word errors on test.clean (90538 reference words): ground truth 1708, HiFi-GAN 2221, DDSP 2284, Vocos 2309. The per-utterance excess over the ground truth correlates between vocoders (HiFi-GAN with DDSP 0.55, with Vocos 0.63, DDSP with Vocos 0.55). In 260_123440_000011_000000 ('I am sure I am not Ada ... I can not be Mabel', Alice-in-Wonderland names) HiFi-GAN, DDSP and Vocos have 5, 7 and 7 word errors against 0 for the ground truth: 'shrun' replaces 'sure I am', and 'mean' or 'kit me' replaces 'be'; DDSP additionally hears 'ate' for 'Ada'. In 121_127105_000008_000000 the three systems make 5, 4 and 5 errors, omitting or altering 'at so tender an age'. Whisper reads the ground truth of both correctly, so the errors arise in the vocoded audio; because three independently trained vocoders fail on the same content, a common cause in the shared input (features or speaker vector) is plausible, but it was not tested. These utterances contain rare or unusual words, which are the least robust to small articulation errors.

Files: `shared_content_errors/`.

## Supporting numbers

Voiced-in-both frames (re-extracted F0, ground truth with CREPE dither off; the ground-truth floor against the cached input is 5.8 cents RMSE). The re-extracted F0 is quantised to 20 cents (CREPE bin width), so the percentile values of the errors are multiples of 10 or 20 cents.

| | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| frames | 802299 | 819930 | 797816 |
| RMS error pooled over frames (cents) | 24.9 | 18.4 | 37.0 |
| share above 50 cents | 0.055 | 0.020 | 0.139 |
| share above 100 cents | 0.0058 | 0.0008 | 0.0188 |
| frames above 300 cents | 32 | 8 | 298 |
| frames within 100 cents of an octave | 19 | 4 | 6 |
| share of squared error from frames above 300 cents | 0.069 | 0.028 | 0.057 |
| share of squared error from 50 to 300 cents | 0.500 | 0.248 | 0.633 |
| lag-1 autocorrelation of the error | 0.70 | 0.53 | 0.81 |
| false voicing (share of ground-truth unvoiced frames) | 0.048 | 0.135 | 0.048 |
| missed voicing (share of ground-truth voiced frames) | 0.032 | 0.011 | 0.037 |

(The pooled RMS is over frames; the table value in `table.md` is the mean of per-utterance RMS values, hence 23.4, 18.2 and 35.0 cents.) HiFi-GAN is not free of the Vocos pattern: it shows the same dependence on pitch (0.26 of frames above 50 cents below 80 Hz) with about half the rate.

## Files

- `failures.md`, `paired.md` (this folder), `paired_vocoders.csv`, `failure_cases_metrics.csv` (per-utterance numbers of the case utterances), `f0_error_decomposition.csv`, `f0_error_by_gt_f0_band.csv`, `voicing_confusion.csv`, `t2_case_embeddings.csv`, `t2_gt_gt_cosine.csv` (target-reference ECAPA cosine and Sim same per T2 utterance), `spectral_sample.csv` (band offsets and harmonic peak-to-valley, 200 utterances), `master.parquet` (all per-utterance results of all systems and conditions with Sim same and Sim spk).
- Scripts in `scripts/`: `paired_report.py`, `build_master.py`, `add_sims.py`, `common.py`, `cases.py`, `build_cases.py`, `explore_*.py`, `failure_stats.py`, `f0_low.py`, `t2_ceiling.py` (exploration scripts print their results; the numbers quoted here come from them and from `failure_stats.py`).
- Case audio and PNGs: `/data/user_data/xoy/sparc-vocoders/eval/failures/<case>/` (11 cases, 18 utterances).
