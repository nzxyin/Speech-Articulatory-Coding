# Phase 3 write-up: three vocoders for SPARC articulatory features

Status: 2026-10-12. The main evaluation and the ablation evaluation are complete (sections 3 to 8).

Every number in this document is copied from one of the files listed below; the source is given in square brackets after the number or at the head of the table. All files are under `~/sparc-vocoders-work/` unless stated otherwise.

| tag | file |
|---|---|
| [table] | `phase3/final/tables/test.clean/table.md` (copy of `$SV_ROOT/eval/tables/test.clean/`), with `paired.csv` and `meta.json` in the same folder |
| [paired] | `phase3/analysis/paired.md` and `paired_vocoders.csv` |
| [failures] | `phase3/analysis/failures.md` |
| [control] | `phase3/analysis/controllability.md` |
| [eff] | `phase3/analysis/efficiency.md` and `phase3/final/efficiency/*.json` |
| [abl] | `phase3/E_ablations.md` (ablation design and evaluation jobs) |
| [abl-table] | `phase3/final/tables/ablations/{table.md, table.csv, paired.csv, per_speaker.csv, meta.json}` (copy of `$SV_ROOT/eval/tables/test.clean/ablations/`) |
| [abl-extra] | paired differences not in `paired.csv`, recomputed from `per_speaker.csv`; method and script in `phase3/I_ablation_section.md` |
| [xy] | `phase3/analysis/xy_probes.md` (paired X-Y probes; copy in `docs/vocoders/results/xy_probes.md`), `xy_probes_long.csv`, notes `phase3/M_xyprobes.md` |
| [status] | `STATUS.md` |
| [p1] | `phase1_report.md` |
| [eval] | `~/sparc-eval-wt/docs/vocoders/EVALUATION.md` (the evaluation contract) |

`$SV_ROOT` is `/data/user_data/xoy/sparc-vocoders`.

## 1. Summary

Three vocoders (HiFi-GAN, DDSP, Vocos) were trained once each, for 400,000 generator steps, to synthesize 24 kHz speech from SPARC articulatory features (12 EMA channels, pitch, loudness and periodicity at 50 Hz) with FiLM speaker conditioning, and were evaluated on LibriTTS-R test-clean (4,687 utterances, 39 speakers). **Each arm is a single training run.** Every difference below is therefore a statement about three trained models. The speaker-cluster intervals cover the sampling of test speakers only, not the sampling of training runs, so no difference can be attributed to the architecture alone.

1. **HiFi-GAN is ahead on perceived quality and speaker similarity (strong evidence for these three models, same result under T1, T2 and T3; at 100k steps the lead over Vocos holds under both discriminators, and the lead over DDSP is not supported at about 6 M parameters, section 8).** UTMOS is 4.165 for HiFi-GAN, 3.896 for Vocos and 3.832 for DDSP, against 4.208 for the ground truth [table]. The paired UTMOS difference of HiFi-GAN is +0.33 over DDSP (CI +0.28 to +0.40, all 39 speakers agree) and +0.27 over Vocos (CI +0.24 to +0.30, 97 percent agree) under T1 [paired]. DDSP against Vocos on UTMOS is not supported (difference -0.065, CI -0.146 to +0.002, 64 percent agree) [paired].
2. **Word and character error rates do not separate the vocoders (strong evidence of no resolvable difference).** Corpus WER is 2.45, 2.52 and 2.55 percent (HiFi-GAN, DDSP, Vocos) against 1.89 percent for the ground truth. No pair is supported in any condition; the largest difference is 0.14 percentage points and at most 66 percent of speakers agree [paired]. The word errors are largely shared: the per-utterance excess over the ground truth correlates 0.55 to 0.63 between vocoders [failures].
3. **The vocoders fail on different prosodic quantities (strong evidence; every pair is supported on F0 RMSE).** DDSP follows F0 best (18.2 cents RMSE) but has the worst voicing decisions (VDE 0.066) and loudness (2.80 dB RMSE). Vocos has the largest F0 error (35.0 cents), caused by broad deviations of 50 to 150 cents and not by octave errors (6 of 797,816 voiced frames lie near an octave), and it is strongest on male voices; the 100k ablations show that the F0 error is not caused by the multi-resolution discriminator (47 cents with the multi-scale one) and that Vocos option A lowers it by only 2 cents (section 8). HiFi-GAN is the best of the three on VDE and loudness and second on F0 [table, paired, failures].
4. **DDSP loses on male speakers (moderate evidence; correlational).** Its UTMOS is 3.69 for male and 3.93 for female speakers, against 4.15 and 4.17 for HiFi-GAN. It marks 13.5 percent of ground-truth unvoiced frames as voiced, against 4.8 percent for each of the other two, and renders pauses too quietly. These properties correlate with the UTMOS drop within DDSP (VDE, r = -0.40), but a causal link was not tested [failures]. A DDSP with twice the parameters (448 channels) or with the multi-scale discriminator shows the same UTMOS drop on the two speakers with the largest drops, so capacity and the discriminator do not explain it (section 8.7).
5. **Controllability: F0 and loudness edits are followed closely by all three vocoders; single-channel EMA edits are not (strong evidence for the finding, no explanation).** A constant plus or minus one-std edit of one EMA channel is reproduced to a mean gain of 0.12, 0.10 and 0.12 (HiFi-GAN, DDSP, Vocos), with a mean off-target movement of 0.24 std in all three, although the same loop reproduces natural EMA variation with a slope of 1.00 (gt floor, HiFi-GAN, Vocos) or 0.96 (DDSP). A constant edit of one natural standard deviation along the first four natural principal components also fails (gain 0.09 to 0.14, six utterances per vocoder). The data do not say why [control].
6. **Efficiency: Vocos is the cheapest and the most streaming-friendly (single timing session per device).** Vocos has a median RTF of 0.0008 on an L40S GPU and 0.027 on one CPU thread, with a measured lookahead of 196 ms. HiFi-GAN is about 7 times slower on one CPU thread (0.18). DDSP has the fewest parameters (7.0 M against 14.9 M and 15.2 M) but a lookahead of about 0.9 s and a phase state that spans the whole utterance. The SPARC feature extractor dominates the cost of the full pipeline on every device [eff].
7. **T2 and T3 mostly expose the limits of the similarity metric and do not change the ranking (strong evidence).** Sim same under T2 correlates 0.80 to 0.86 with the ECAPA cosine between the ground truth of the target and of its reference. Under T3 HiFi-GAN and Vocos exceed the ground-truth Sim spk ceiling (0.787 and 0.783 against 0.749 for the ground truth; DDSP, at 0.731, stays below it), because the speaker-mean vector moves the output to the centre of the speaker's embedding cloud [failures, table].
8. **Ablations at 100k steps (single runs; supported differences listed, section 8).** (a) *Training length*: the main runs gained 0.22 (HiFi-GAN), 0.31 (Vocos) and 0.07 (DDSP) UTMOS from 100k to 400k, and DDSP's F0, voicing and loudness errors were worse at 400k; at 100k the UTMOS order of DDSP and Vocos is reversed (DDSP ahead by 0.18, CI only), so the 100k results cannot be used to rank the vocoders at 400k. (b) *Vocos option A against B*: B is better on UTMOS (+0.12) and speaker similarity (+0.014); A is only 0.6 percent lower in mel L1, so the pilot choice of B holds. (c) *`p_cross` = 0*: improves the self-conditioned T1 result (mel L1 -2.8 percent, Sim same +0.028) and worsens the T2 mel L1 (+3.3 percent) and PESQ (-0.026), but under T2 UTMOS, WER and speaker similarity do not change in a supported way (under T1 UTMOS rises by a supported +0.019). (d) *Native SPARC recipe*: higher UTMOS (+0.11 to +0.14) but worse mel L1 (+8.8 percent), PESQ (-0.14) and WER (+0.4 percentage points); the recipe changes several things at once. (e) *MSD instead of MRD*: lowers Vocos and DDSP on speaker similarity, F0 and voicing, raises HiFi-GAN UTMOS (+0.07) and leaves the ranking unchanged (the HiFi-GAN lead grows). (f) *Size-matched*: at about 6 M the HiFi-GAN lead over DDSP is not supported on UTMOS (+0.08, CI only) and Sim same is equal for all three, while the lead over Vocos (+0.36 UTMOS) is unchanged; doubling DDSP does not help it (UTMOS -0.05, F0 +4.4 cents).

## 2. Setup in brief

**Task and data.** The input is a 15-channel feature track at 50 Hz (12 EMA channels from the per-utterance MNGU0 refit of the linear inversion head, pitch, loudness, periodicity) and a 64-dimensional speaker vector, and the output is 24 kHz audio of length 480 times the number of frames [p1 sections 0 and 11; eval section 1]. Training data are the three filtered LibriTTS-R training subsets (`parler-tts/libritts_r_filtered`; 542 h, 2,299 speakers in the plan [p1 section 9]). The packed training split holds 339,007 utterances and 97.2 million frames [status, entry 2026-10-05 15:50]. All splits are speaker-disjoint [p1 section 9]. The test set is test.clean after the 171 not-encodable rows across all splits were removed: 4,687 utterances, 39 speakers [eval section 1; table].

**Training.** One shared recipe for all three vocoders: multi-period plus multi-resolution discriminators, hinge loss, mel L1 loss at 24 kHz (weight 45), a 10,000-step mel-only warm-up, AdamW with learning rate 2e-4 and cosine decay, global batch 16 segments of 64 frames, 400,000 generator steps, fp32, seed 0, and speaker references taken from a different utterance of the same speaker with probability `p_cross` = 0.5 [p1 section 8; run configuration `runs/main/vocos/hydra/*/.hydra/config.yaml`: seed 0, lr 0.0002, `max_g_steps` 400000, `p_cross` 0.5, precision 32-true, batch size 16, crop 64]. Each speaker FFN is trained from scratch. The Vocos variant is option B (hop 240); it won the 25k-step A/B pilot on dev mel L1 (0.5382 against 0.5406, within the 2 percent rule) [status, entry 2026-10-06 01:55]. Parameter counts (generator plus speaker FFN): HiFi-GAN 15.22 M, DDSP 7.04 M, Vocos 14.90 M [eff]. The HiFi-GAN generator has 14.11 M, DDSP 5.92 M, Vocos 13.78 M [eff].

**Conditions.** T1: speaker vector of the utterance itself (an upper bound, because the vector carries utterance-level information; Phase 1 measured a 16.9 percent log-mel increase when a different chapter was used for the shipped en+ HiFi-GAN [p1 section 4]). T2: the vector of another utterance of the same speaker from a fixed seeded manifest (speaker 61 has a single utterance and is absent, giving 4,686 utterances and 38 speakers); T2 is also reported for the 875 same-chapter and the 3,811 different-chapter utterances. T3: the mean vector of the speaker [eval section 1; table].

**Metrics.** UTMOS22 strong, PESQ (wideband), WER and CER with Whisper large-v3 (corpus level), speaker similarity with ECAPA-TDNN (Sim same: cosine with the ground truth of the same utterance; Sim spk: cosine with the mean of the speaker's other ground-truth utterances, whose ground-truth value is a ceiling), MCD, multi-resolution STFT distance, mel L1 (all bands and three sub-bands), and prosody and articulation measured on features re-extracted from the output with CREPE dither disabled (F0 RMSE in cents on frames voiced in both, VDE, periodicity MAE, loudness RMSE in dB, EMA correlation and RMSE) [eval section 4]. Intervals are speaker-cluster bootstrap CI95 (10,000 resamples, seed 0) [table].

**Systems.** `gt` (ground truth, the first 480 T samples at -3 dBFS gain), `vocos_mel` (Vocos mel copy synthesis, `charactr/vocos-mel-24khz`, a reference without the articulatory bottleneck), `enplus16` (the shipped SPARC en+ decoder at 16 kHz, with its own shipped inversion head, a reference), and the three vocoders [eval section 1].

**Deviations from the Phase 1 plan and evaluation fixes.**

- Precision: the planned bf16 pilot was dropped and all training is fp32, approved by the user on 2026-10-05, because measured fp32 steps cost 214 to 304 ms on an RTX 6000 Ada [p1 header; status].
- Training layout: one GPU per vocoder at the same global batch, as recommended in Phase 1 decision D7, and not 4-GPU DDP [status]. The HiFi-GAN run started on an RTX A6000 (about four times slower in fp32), was checkpointed at 34,780 steps, requeued and then constrained to faster GPU types through `SBATCH_CONSTRAINT`; the final checkpoint is at step 400,000 [status, entry 2026-10-06 01:55; meta.json].
- WER normalization: Whisper's English text normalizer deletes text in parentheses and brackets. 40 test.clean references contain spoken parentheticals, so the first evaluation counted them as insertions. The normalizer was changed (commit 5e77425) and the stored counts were recomputed by a new stage `asr_rescore` (35 utterances changed per system). The corpus WER of the ground truth changed from 2.12 to 1.89 percent, DDSP from 2.75 to 2.52 and Vocos from 2.77 to 2.55. The HiFi-GAN results used the fixed normalizer from the start [status, entry 2026-10-07 12:25].
- Efficiency markers: the finished-stage markers of the efficiency runs on GPU and CPU used one file, so the CPU run could be skipped; they now carry a device suffix (commit 848840c) [status, entry 2026-10-07 10:20].
- One utterance, `908_157963_000024_000000` (0.40 s, reference text "3"), is too short for the zero-phase filter in the feature extractor. It has no re-extracted features for any system, so the prosody and EMA cells use 4,686 of 4,687 utterances; the en+ reference also failed to produce audio for it [table, failures case 8].
- Loudness: Phase 1 found that SPARC z-scores each waveform before computing loudness, so the cached loudness does not change with gain; an extra un-normalized loudness was cached [p1 section 0, finding 1].

## 3. Main results (test.clean)

**Caveats that apply to every row.**

- Single training run per arm (see section 1).
- The en+ reference runs at 16 kHz and is upsampled for the 24 kHz metrics. It has no content above 8 kHz, so its MCD (21.99), MR-STFT and mel L1 values are not comparable with the 24 kHz systems, and its 8 to 12 kHz band is not applicable. PESQ is computed at 16 kHz and is comparable [table].
- The en+ checkpoint's training splits are undocumented, so en+ may have seen test-clean. This can favour en+ and does not affect the new vocoders, which use none of its trained weights [p1 section 14, assumption 8].
- The EMA and the other features of the ground truth are compared with the cached input, so the "ground truth" row of the prosody table is an extraction floor: F0 RMSE 5.8 cents, EMA r 0.998, EMA RMSE 0.048 std, caused by dither in the cache (F0) and by truncation at the last frames (EMA) [table; status, 2026-10-06 entries].
- The metrics MCD, MR-STFT and mel L1 are close to the training loss; UTMOS, WER, speaker similarity, prosody and EMA consistency are the primary metrics [p1 section 8].
- UTMOS, Sim spk and the F0 metrics are unreliable for utterances under 1 s (failures case 7).

**Table 1. T1 (own-utterance speaker vector), mean over utterances with CI95 (speaker-cluster bootstrap). n = 4,687 utterances, 39 speakers; prosody cells n = 4,686.** Source: [table].

| system | UTMOS | WER (%) | Sim same | Sim spk | MCD (dB) | F0 RMSE (cents) | VDE | Loudness RMSE (dB) | EMA r |
|---|---|---|---|---|---|---|---|---|---|
| ground truth | 4.208 (4.169, 4.242) | 1.89 (1.60, 2.22) | n/a | 0.749 (0.728, 0.773) | n/a | 5.8 (floor) | 0.000 | 0.00 | 0.998 |
| Vocos mel copy synthesis | 3.967 (3.925, 4.005) | 1.89 (1.60, 2.23) | 0.982 (0.980, 0.984) | 0.744 (0.723, 0.769) | 1.348 (1.335, 1.359) | 18.2 (17.1, 19.5) | 0.035 | 1.72 | 0.981 |
| SPARC en+ (16 kHz) | 4.050 (4.017, 4.082) | 2.69 (2.37, 3.05) | 0.829 (0.820, 0.839) | 0.710 (0.691, 0.731) | 21.988 (not comparable) | 27.5 (25.9, 29.5) | 0.069 | 2.48 | 0.965 |
| HiFi-GAN | 4.165 (4.136, 4.191) | 2.45 (2.12, 2.83) | 0.857 (0.848, 0.867) | 0.731 (0.712, 0.752) | 3.992 (3.944, 4.045) | 23.4 (22.1, 24.9) | 0.039 (0.035, 0.043) | 1.49 (1.44, 1.54) | 0.971 (0.970, 0.972) |
| DDSP | 3.832 (3.749, 3.902) | 2.52 (2.18, 2.91) | 0.802 (0.790, 0.816) | 0.690 (0.669, 0.714) | 4.192 (4.145, 4.243) | 18.2 (17.7, 18.8) | 0.066 (0.061, 0.072) | 2.80 (2.67, 2.96) | 0.960 (0.959, 0.961) |
| Vocos | 3.896 (3.862, 3.933) | 2.55 (2.20, 2.94) | 0.844 (0.835, 0.854) | 0.722 (0.703, 0.744) | 3.967 (3.922, 4.016) | 35.0 (32.6, 37.9) | 0.042 (0.038, 0.046) | 1.83 (1.77, 1.90) | 0.969 (0.968, 0.970) |

Further T1 values for the three vocoders [table]: PESQ 2.091, 1.913 and 1.903 (HiFi-GAN, DDSP, Vocos); mel L1 0.488, 0.515 and 0.494; F0 within 50 cents 0.946, 0.979 and 0.863; periodicity MAE 0.044, 0.070 and 0.050; EMA RMSE 0.237, 0.271 and 0.247 std.

**Table 2. Speaker conditions T2 and T3 (the T1 rows are in Table 1).** Mean (CI95), speaker-cluster bootstrap. n = 4,686 utterances and 38 speakers for T2 and T3 (en+ has no T3). Source: [table].

| system | cond | UTMOS | WER (%) | Sim same | Sim spk |
|---|---|---|---|---|---|
| HiFi-GAN | T2 | 4.167 (4.138, 4.193) | 2.56 | 0.724 (0.705, 0.745) | 0.751 (0.733, 0.770) |
| HiFi-GAN | T2 same chapter (n = 875, 13 speakers) | 4.212 | 2.01 | 0.797 (0.775, 0.819) | 0.807 (0.779, 0.829) |
| HiFi-GAN | T2 different chapter (n = 3,811, 26 speakers) | 4.157 | 2.73 | 0.708 (0.688, 0.728) | 0.738 (0.719, 0.757) |
| HiFi-GAN | T3 | 4.171 (4.141, 4.198) | 2.54 | 0.756 (0.736, 0.777) | 0.787 (0.769, 0.806) |
| DDSP | T2 | 3.839 (3.757, 3.909) | 2.49 | 0.696 (0.676, 0.717) | 0.706 (0.685, 0.728) |
| DDSP | T3 | 3.848 (3.766, 3.919) | 2.55 | 0.722 (0.703, 0.744) | 0.731 (0.711, 0.754) |
| Vocos | T2 | 3.896 (3.860, 3.935) | 2.63 | 0.721 (0.702, 0.742) | 0.741 (0.723, 0.761) |
| Vocos | T3 | 3.899 (3.862, 3.938) | 2.60 | 0.755 (0.735, 0.775) | 0.783 (0.765, 0.802) |
| SPARC en+ (16 kHz) | T2 | 4.033 (3.997, 4.069) | 2.78 | 0.680 (0.660, 0.703) | 0.725 (0.708, 0.744) |

How to read the speaker columns:

- Sim same falls from T1 to T2 and T3 for every system because the output follows a vector that differs from the one that carried the target utterance's own information. The fall is partly a property of the metric (section 7, cases 9 and 10).
- Sim spk under T3 is above the ground-truth ceiling (0.749) for HiFi-GAN (0.787) and Vocos (0.783) and below it for DDSP (0.731), so Sim spk under T3 must not be read as fidelity [table; failures case 10].
- For all three vocoders the same-chapter subset of T2 scores higher than the different-chapter subset (for example HiFi-GAN Sim same 0.797 against 0.708) [table].

**Comparison with the references (paired, T1, speaker-cluster CI95, `paired.csv`).**

- HiFi-GAN UTMOS is above the Vocos mel copy synthesis by 0.198 (CI 0.173 to 0.224, 95 percent of speakers agree) and above en+ by 0.115 (CI 0.102 to 0.128, 97 percent). DDSP and Vocos are below en+ on UTMOS by 0.218 (CI -0.280 to -0.165) and 0.153 (CI -0.180 to -0.123).
- The copy-synthesis reference loses 0.24 UTMOS against the ground truth (3.967 against 4.208), so the Vocos architecture on mel input is not at the ground-truth level either. That HiFi-GAN scores above a copy-synthesis reference is a property of the trained models and of UTMOS, not a sign that articulatory features carry more than the mel spectrogram.
- All three vocoders have a lower corpus WER than en+ (2.69 percent), by 0.24 (HiFi-GAN, 82 percent of speakers agree), 0.17 (DDSP, 77 percent) and 0.14 percentage points (Vocos, 67 percent); the first two are supported and the Vocos difference is CI only [paired].
- All three vocoders have a lower loudness RMSE than en+ except DDSP, whose RMSE is higher (+0.32 dB, CI +0.20 to +0.45).
- The Vocos mel copy synthesis keeps a WER equal to the ground truth (1.89 percent) while every articulatory vocoder is 0.56 to 0.66 percentage points higher; the loss of intelligibility therefore enters at the feature bottleneck and not in the waveform generator. This reading rests on the equality of the two WER values and was not tested in isolation.

## 4. Paired comparisons of the vocoders

Method [paired]. Each row compares two systems on the same utterances. The difference is `a - b`. The interval is the speaker-cluster bootstrap CI95 (10,000 resamples, seed 0). A difference is called *supported* when the CI95 excludes 0 and at least 75 percent of the speakers have a mean difference with the sign of the overall difference. *CI only* marks a CI that excludes 0 with fewer than 75 percent agreeing speakers. No correction for multiple comparisons was applied; about 180 comparisons are listed in `paired.md`, and borderline results (CI edge within 0.001 of zero) should be read with that in mind. The criterion is a convention of this report and was not pre-registered. Because each arm is one training run, a supported difference shows that these two trained models differ on this test set; it does not show that the architecture is the cause.

**Table 3. Selected paired differences under T1** (n = 4,687 utterances, 39 speakers, prosody rows 4,686). Source: [paired] (`paired_vocoders.csv`).

| metric | HiFi-GAN - DDSP | HiFi-GAN - Vocos | DDSP - Vocos |
|---|---|---|---|
| UTMOS | +0.333 (+0.279, +0.398), 100 % | +0.27 (+0.24, +0.30), 97 % (T3: +0.272) | -0.065 (-0.146, +0.002), 64 %, not supported |
| WER | not supported | not supported | not supported |
| Sim same | supported | +0.013, supported | -0.041 (-0.049, -0.035), 100 % |
| Sim spk (T1) | supported | +0.008, supported | supported |
| F0 RMSE (cents) | +5.18 (+4.15, +6.45), DDSP better | -11.61 (-13.22, -10.30), HiFi-GAN better | -16.79 (-19.50, -14.59), DDSP better, 100 % |
| VDE | -0.0274 (-0.0304, -0.0244), HiFi-GAN better | -0.0028 (-0.0036, -0.0020), 87 % | +0.0246 (+0.0214, +0.0279), Vocos better |
| Loudness RMSE (dB) | -1.314 (-1.429, -1.214), HiFi-GAN better | -0.342 (-0.364, -0.318) | +0.972 (+0.870, +1.088), Vocos better |

Not every cell of this table is spelled out in the sources I quote here. Cells marked only with a word ("supported") are taken from the summary table of [paired], which gives the ordering and the ranges but not each interval; the full intervals are in `paired.md` and `paired_vocoders.csv`.

**Reading across the three conditions** [paired]:

- The conditions change only the speaker vector, and the paired differences between vocoders hardly move: the UTMOS advantage of HiFi-GAN is +0.33, +0.33 and +0.32 against DDSP and +0.27 in all three conditions against Vocos; Vocos F0 RMSE is 11.3 to 11.6 cents worse than HiFi-GAN and 16.4 to 16.8 cents worse than DDSP in all three.
- Only borderline rows change status between conditions (Sim same of HiFi-GAN against Vocos: +0.013 under T1, +0.003 and +0.001 under T2 and T3, not supported; VDE of HiFi-GAN against Vocos: supported under T1 and T2, CI only under T3).

**Size matters.** Many supported differences are small: HiFi-GAN against Vocos on Sim spk (+0.008) and on EMA r (+0.002). The raw differences must be compared with the scale of the metric (ground-truth UTMOS 4.21, WER 1.9 percent, ground-truth Sim spk 0.749) [paired].

**Spectral metrics** [paired]: HiFi-GAN is better than DDSP on almost every spectral metric. Vocos has the lowest MCD (3.97 dB, 0.025 below HiFi-GAN, supported but small), is better than HiFi-GAN in the 4 to 8 kHz and 8 to 12 kHz mel bands and worse in the 0 to 4 kHz band, in PESQ and in MR-STFT. DDSP against Vocos on MR-STFT and PESQ is not supported under T1.

## 5. Prosody and controllability

### 5.1 Prosody errors of the synthesis

All values from [table] and [failures]; re-extracted features, CREPE dither off. The pitch estimate is quantized to the CREPE bin width of 20 cents, so percentiles of F0 errors are multiples of 10 or 20 cents.

| | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| F0 RMSE (cents; mean of per-utterance RMSE) | 23.4 | 18.2 | 35.0 |
| F0 RMSE pooled over frames (cents) | 24.9 | 18.4 | 37.0 |
| voiced-in-both frames | 802,299 | 819,930 | 797,816 |
| share of frames more than 50 cents off | 0.055 | 0.020 | 0.139 |
| share more than 100 cents off | 0.0058 | 0.0008 | 0.0188 |
| frames more than 300 cents off | 32 | 8 | 298 |
| frames within 100 cents of an octave | 19 | 4 | 6 |
| lag-1 autocorrelation of the F0 error | 0.70 | 0.53 | 0.81 |
| false voicing (share of ground-truth unvoiced frames) | 0.048 | 0.135 | 0.048 |
| missed voicing (share of ground-truth voiced frames) | 0.032 | 0.011 | 0.037 |
| loudness bias (dB) | -0.34 | -0.97 | -0.44 |
| loudness RMSE on ground-truth unvoiced frames (dB) | 2.13 | 4.40 | 2.56 |

Findings [failures]:

- **Vocos F0.** The large F0 RMSE comes from slowly varying deviations of one to several CREPE bins, symmetric in sign (mean +1.6 cents). Frames above 300 cents carry only 5.7 percent of the squared F0 error; 63 percent comes from frames between 50 and 300 cents. The share of frames more than 50 cents off rises as the ground-truth F0 falls: 0.56 below 80 Hz (n = 7,252) against 0.06 to 0.07 at 200 to 300 Hz, and 0.195 for male against 0.100 for female speakers. UTMOS and WER of the affected utterances are normal, so the loss is in pitch precision that UTMOS does not register. The cause (iSTFT head, frame rate, amount of low-pitched training speech) was not tested.
- **Low pitch.** Below 60 Hz the median signed F0 error is +420 cents for Vocos and +180 cents for HiFi-GAN, and 0 for DDSP (n voiced frames 171, 99 and 258). In the DDSP design the F0 channel sets the oscillator phase directly, which is consistent with DDSP being unaffected; this was not tested.
- **DDSP voicing and loudness.** DDSP has no explicit voicing gate, which is consistent with its false voicing, but this was not tested. Its loudness error is concentrated in quiet frames (30 to 40 dB below the utterance maximum: RMSE 5.29 dB for DDSP against 2.70 for HiFi-GAN and 3.01 for Vocos). In an example pause DDSP renders room noise at about -40 dB where the recording has about -25 dB relative to the loudest frame.
- **Sex.** UTMOS by sex: DDSP 3.69 (male) and 3.93 (female), HiFi-GAN 4.15 and 4.17, Vocos 3.95 and 3.86, ground truth 4.18 and 4.23. Utterances of 3 s or longer with UTMOS below 3: 77 (DDSP), 3 (HiFi-GAN), 14 (Vocos) of 3,266. Within DDSP the per-utterance UTMOS correlates with VDE at -0.40, with MCD at -0.50, with EMA r at +0.46 and with loudness bias at +0.29 (for Vocos the VDE correlation is -0.06). These are correlations over utterances and do not identify a cause. Band levels and noise floor were compared on a sample of 200 utterances and did not explain the gap.

### 5.2 Controllability probes

Method [control, eval section 6]. 50 test.clean utterances of 3 to 10 s (28 speakers, at most 2 per speaker), T1 speaker vectors. The 15 raw input channels are edited, the output is re-extracted without dither, and the response is compared with the unedited synthesis. Medians over utterances; intervals are speaker-cluster bootstrap CI95 with 10,000 resamples. The F0 intervals are degenerate because of the quantized pitch output and understate the true uncertainty.

| quantity [control] | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| largest median F0 error over edits of -4, -2, +2, +4 semitones (cents) | 0.0 | 0.0 | 10.0 |
| loudness change for x0.5 and x2 (targets -6.02 and +6.02 dB) | -5.89 / +5.78 | -6.00 / +5.79 | -6.01 / +5.87 |
| voiced fraction before and after setting periodicity to 0 (ideal 0) | 0.56 / 0.17 | 0.61 / 0.28 | 0.56 / 0.08 |
| single EMA channel, plus or minus 1 training std: gain (mean over 12 channels; 1 = reproduced) | 0.12 (0.11, 0.12) | 0.10 (0.09, 0.10) | 0.12 (0.11, 0.12) |
| single EMA channel: leakage into the other channels (std units) | 0.24 | 0.24 | 0.24 |
| off-target movement of the re-inverted EMA under F0 / loudness / periodicity edits (std) | 0.12 / 0.08 / 0.18 | 0.16 / 0.07 / 0.20 | 0.14 / 0.08 / 0.19 |
| natural variation: centred slope of re-extracted on input EMA (gt floor 1.00; Vocos mel copy 0.97) | 1.00 | 0.96 | 1.00 |
| natural variation: gain of one channel with the others fixed (regression diagonal) | 0.98 | 0.93 | 0.97 |

Findings:

- **F0 and loudness are controlled closely** by all three vocoders. Setting periodicity to 0 does not remove voicing in any vocoder; it lowers it most in Vocos.
- **Single-channel EMA edits are reproduced to only 4 to 32 percent**; tongue body X has a negative gain of about -0.15 in all three vocoders [control section 4]. The leakage of 0.24 std is larger than the 0.12 std gain, but edits that do not touch EMA already move the re-inverted EMA by 0.07 to 0.20 std, which is the noise floor for leakage.
- **Differences between vocoders are small**: the paired EMA gain differences are 0.02 or less (Vocos minus DDSP 0.017, CI 0.014 to 0.022; HiFi-GAN minus DDSP 0.019, CI 0.017 to 0.021; Vocos minus HiFi-GAN -0.001, CI -0.004 to 0.002).
- **The loop itself preserves natural EMA variation.** On all 4,686 re-extracted test.clean utterances (1,512,210 frames, 39 speakers) the centred slope of re-extracted on input EMA is 1.00 for the ground-truth floor, HiFi-GAN and Vocos, 0.96 for DDSP and 0.97 for the mel copy synthesis. The regression of the centred re-extracted EMA on all 12 centred input channels has a diagonal of 0.93 to 0.98 and predicts a leakage of only 0.02 to 0.03 for an isolated one-channel edit [control section 5]. Isolated single-channel edits therefore fail while natural dynamics pass.
- **The failure is not only an off-manifold direction.** A constant edit of one natural standard deviation along principal components 1 to 4 of natural EMA gives a gain of 0.09 to 0.14 (DDSP 0.09 to 0.11), against 0.96 to 1.02 predicted by the regression. This probe used 6 of the 50 utterances per vocoder (CPU synthesis), so its intervals come from an utterance bootstrap and cover few utterances [control section 6].
- **Open explanation.** The data show that a constant offset of the articulatory input is poorly reproduced. They do not say why. Candidate explanations that were not tested: the vocoder responds to change over time and not to the absolute position; the offset is added to silence and pause frames as well; the inversion (WavLM and a linear head) responds mostly to phonetic content and reflects a shifted tongue position only weakly [control]. The EMA gain is also not corrected for attenuation by the inversion, which is neutral on natural audio (slope 1.00) but was not tested on audio with an artificially shifted tongue.

### 5.3 Paired X-Y edits of one sensor (follow-up, 2026-10-12)

Question: are the single-channel EMA edits followed poorly because a sensor must be moved in two dimensions at once?
Probe (`probes_xy`, same 50 utterances and protocol as 5.2, all three vocoders): for each of the six sensors, edits of
equal norm along X only, Y only, the sensor's natural principal direction and the orthogonal direction, each as a
constant offset, a constant offset on speech frames only, a dynamic scaling of the movement around the utterance mean
(alpha = plus or minus 0.5) and three 300 ms bumps; 192 edits per vocoder. Full tables:
`docs/vocoders/results/xy_probes.md` [xy].

| profile (mean of 6 sensors, both signs) | X only | Y only | principal | orthogonal |
|---|---|---|---|---|
| constant offset | 0.09-0.11 | 0.10-0.12 | 0.10-0.12 | 0.10-0.11 |
| constant offset, speech frames only | 0.10-0.12 | 0.11-0.13 | 0.11-0.13 | 0.11-0.13 |
| dynamic scaling around the mean | 0.16-0.18 | 0.19-0.20 | 0.19-0.21 | 0.19-0.21 |
| three local 300 ms bumps | 0.18-0.20 | 0.24-0.27 | 0.20-0.24 | 0.22-0.24 |

Cells give the range of the gain over the three vocoders (1 = followed; the natural-dynamics regression predicts 0.93
to 0.98 for every cell) [xy].

- **Pairing is not what is missing.** The paired principal direction is followed no better than X or Y alone (paired
  minus single gain between -0.015 and +0.004 under constant offsets, speaker-cluster CI95) [xy].
- **Time course matters.** Dynamic scaling and local bumps are followed about twice as well as constant offsets (+0.06
  to +0.14 gain); restricting an offset to speech frames adds only about +0.01 [xy].
- **Direction within a sensor matters, consistently across vocoders.** Under dynamic scaling the tongue body follows
  its principal axis (gain 0.57, 0.44, 0.54 for HiFi-GAN, DDSP, Vocos) but not X alone (0.07, 0.00, 0.06); the tongue
  tip follows X (0.35 to 0.37) and its orthogonal direction (0.34 to 0.35) but not its principal axis (0.10 to 0.13).
  Shrinking movement (alpha = -0.5) is followed with gain about 0.3, exaggerating it with 0.03 to 0.15 [xy].
- **Open.** The best cell reaches about 0.57 of the prediction and leakage into other sensors (0.25 to 0.49 std) exceeds
  the on-target gain in most cells. The loop includes the linear inversion, so the probe cannot separate a vocoder that
  ignores an isolated edit from one whose acoustics the inversion reads as a different articulation. An inversion-free
  acoustic measure (formant or spectral shift) or coherent 12-D edits would separate the two.

## 6. Efficiency and streaming

Source: [eff]. 50 test.clean utterances (290.2 s of audio), batch 1, fp32, five warm-up calls, one NVIDIA L40S for every system, one timing session per device. RTF is compute time divided by audio duration (median, with the 95th percentile in brackets); values below 1 are faster than real time. The RTF includes the front end and the speaker FFN. Differences of a few percent are not meaningful.

| system | total parameters (M) | RTF L40S | RTF CPU, 1 thread | RTF CPU, 8 threads | measured lookahead (ms) | measured past context (ms) |
|---|---|---|---|---|---|---|
| HiFi-GAN | 15.22 | 0.0061 (0.0120) | 0.183 (0.191) | 0.0671 (0.0702) | 205 | 217 |
| DDSP | 7.04 | 0.0050 (0.0088) | 0.0363 (0.0381) | 0.0188 (0.0210) | 929 | 3,980 (reaches the end of the 4 s window) |
| Vocos | 14.90 | 0.0008 (0.0013) | 0.0266 (0.0274) | 0.0070 (0.0076) | 196 | 250 |
| Vocos mel (reference, decode only) | 13.53 | 0.0006 | 0.0252 | 0.0047 | 192 | 235 |
| SPARC en+ 16 kHz (reference, decode only) | 16.50 | 0.0062 | 0.162 | 0.0457 | 189 | 219 |

Notes:

- The pooled RTF on the GPU in the raw JSON files (total time over total audio) is 0.0057, 0.0046 and 0.0007 for HiFi-GAN, DDSP and Vocos; the table uses the median. The `STATUS.md` entry of 2026-10-08 03:40 quotes the pooled values rounded as 0.0058, 0.0046 and 0.0007 (the 0.0058 for HiFi-GAN is a rounding of 0.00575 to the next digit).
- **Feature extractor** (SPARC en+ WavLM-large, CREPE full, refit head, batch 1, no dither): median RTF 0.0463 on the L40S, 0.738 with 8 CPU threads, 3.47 with one thread. The extractor dominates the speech-to-speech pipeline on every device: the sum of the medians is 0.0525 (HiFi-GAN), 0.0513 (DDSP) and 0.0471 (Vocos) on the L40S, 0.805, 0.757 and 0.745 with 8 CPU threads, and 3.66, 3.51 and 3.50 with one thread. These are offline whole-utterance timings, not streaming measurements.
- **Possible timing problem.** The measured encode-plus-decode RTF of en+ with 8 CPU threads (0.536) is 31 percent below the sum of the extractor and the en+ decode (0.738 + 0.0457 = 0.784). The files do not explain this; it is probably a difference between two timing runs.
- **Streaming** [eff]: Vocos needs about 0.2 s of future features and has a bounded past context of 250 ms, and runs faster than real time on one CPU core (0.027). HiFi-GAN has a similar lookahead (205 ms) and is slower on CPU but still faster than real time. DDSP is bounded for the EMA, loudness and periodicity channels (lookahead 0.7 to 1.1 s), but an F0 change moves every later harmonic sample because the phase is a cumulative sum of F0 (the measured influence reaches the end of the 4 s window), so a streaming DDSP needs the oscillator phase carried across chunks and a lookahead of about 0.9 to 1.1 s. The measured lookahead of 929 ms is below the analytic bound of 1.36 s.
- The lookahead was measured with a 0.5 std perturbation of one frame in one source utterance and a threshold of 1e-4 of the peak. These are not latency measurements of an implemented streaming system.

## 7. Failure cases

Eleven cases cover 18 utterances. For each case the folder contains a `README.txt`, 16-bit PCM WAV copies at 24 kHz of the ground truth, HiFi-GAN, DDSP and Vocos (T2 cases also the reference utterance), and a PNG with four spectrograms, the re-extracted F0 of the voiced frames, loudness and periodicity (ground truth black, HiFi-GAN blue, DDSP orange, Vocos green). **No listening was done: every statement rests on metrics, transcripts, spectrograms and re-extracted feature tracks.** Folders: `/data/user_data/xoy/sparc-vocoders/eval/failures/<case>/` [failures]. Per-utterance numbers: `phase3/analysis/failure_cases_metrics.csv`.

In the file names below, `<id>__T1__hifigan.wav` is the HiFi-GAN output for utterance `<id>` under T1, and `<id>__gt.wav` is the ground truth.

| # | case folder | utterances | what goes wrong | key numbers [failures] |
|---|---|---|---|---|
| 1 | `vocos_f0_jitter_low_pitch` | 7729_102255_000002_000010, 908_31957_000017_000001 (T1) | Vocos: segments with a 50 to 150 cent F0 offset and voicing dropouts on low-pitched male speech | F0 within 50 cents 0.58 and 0.495 for Vocos against 0.79 and 0.92 (HiFi-GAN), 0.99 and 0.98 (DDSP); UTMOS 4.27 and 4.30, so the loss is invisible to UTMOS |
| 2 | `vocos_octave_error` | 7176_92135_000076_000001 (T1, 0.52 s, "sugar") | one octave-high frame defines the F0 RMSE of the utterance | F0 RMSE 1170 cents from one voiced-in-both frame; per-utterance F0 RMSE must not be ranked on such utterances |
| 3 | `f0_floor_and_octave_frames` | 2300_131720_000026_000007 (Vocos), 7021_79740_000017_000002 (HiFi-GAN) (T1) | pitch floor near 75 to 80 Hz for creaky voice at about 57 Hz; octave-sized frames | Vocos 189 cents, HiFi-GAN 220 cents; RMS error below 60 Hz 443 (Vocos), 211 (HiFi-GAN), 16 cents (DDSP) |
| 4 | `ddsp_loudness_unvoiced` | 4507_16021_000036_000001, 8455_210777_000033_000000 (T1) | DDSP: pauses and quiet frames far too quiet | loudness RMSE 7.20 and 5.94 dB against 2.0 to 2.7 dB for the others |
| 5 | `ddsp_low_utmos_male` | 7021_85628_000020_000000, 260_123440_000002_000000 (T1) | DDSP: UTMOS near 2.4 to 2.5 where HiFi-GAN is at ground-truth level | DDSP 2.48 and 2.41; HiFi-GAN 4.35 and 4.02; ground truth 4.36 and 3.72 |
| 6 | `ddsp_male_speaker_similarity` | 7729_102255_000012_000001 (T1, 6.4 s) | DDSP: low speaker similarity for male speaker 7729 | Sim same 0.632 against 0.882 (HiFi-GAN); speaker-level Sim spk 0.726 (DDSP) against 0.867 (ground truth) and 0.826 (HiFi-GAN) |
| 7 | `short_utterances` | 1580_141084_000040_000000 ("Aha", 0.68 s), 672_122797_000028_000000 ("Help", 0.48 s) (T1) | utterances under 1 s: metrics are not meaningful | ground-truth UTMOS 1.82 and 1.77 |
| 8 | `too_short_to_reextract` | 908_157963_000024_000000 (T1, 0.40 s) | re-extraction fails for every system | all prosody cells use n = 4,686 |
| 9 | `t2_reference_mismatch` | 1284_1181_000040_000005, 1284_1181_000027_000000 (T2; reference audio included) | Sim same collapses under T2 for speaker 1284 | HiFi-GAN 0.816 (T1) to 0.129 (T2) on the first; the target-to-reference ECAPA cosine is only 0.114 and 0.247 |
| 10 | `t3_exceeds_ground_truth_ceiling` | 3729_6852_000044_000000 (T3, 4.04 s) | T3 Sim spk above the ground-truth ceiling | 0.788 (HiFi-GAN), 0.795 (DDSP), 0.771 (Vocos) against 0.557 for the ground truth |
| 11 | `shared_content_errors` | 260_123440_000011_000000, 121_127105_000008_000000 (T1) | the same words are wrong for all three vocoders | 5, 7 and 7 word errors against 0 for the ground truth in the first |

Reading of the cases:

- **Case 7 and 8** show that very short utterances dominate the tails of the per-utterance metrics but not the corpus values: the correlation of UTMOS with duration is only 0.12 (HiFi-GAN), 0.00 (DDSP) and 0.02 (Vocos), and utterances of five reference words or fewer contribute 119, 123 and 127 of the 2,221, 2,284 and 2,309 word errors.
- **Case 9** shows that a part of the T1-to-T2 fall of Sim same measures how different two utterances of one speaker are for ECAPA. The Sim same under T2 correlates 0.86 (HiFi-GAN), 0.80 (DDSP) and 0.85 (Vocos) with the cosine between the ground truth of the target and of its reference; for the 239 utterances (5.1 percent) with a cosine below 0.3 the mean T2 Sim same is 0.485, 0.486 and 0.486, against 0.799, 0.764 and 0.794 for the 2,490 utterances with a cosine of 0.6 or more. The output is otherwise sound (UTMOS 4.28 and 4.15).
- **Case 10** shows that Sim spk under T3 rewards centring: the T3 output has a higher Sim spk than the ground truth of the same utterance in 70 percent (HiFi-GAN), 36 percent (DDSP) and 66 percent (Vocos) of the utterances.
- **Case 11**: because three independently trained vocoders fail on the same content, a common cause in the shared input (features or speaker vector) is plausible. This was not tested. Whisper reads the ground truth of both utterances correctly, so the errors arise in the vocoded audio.
- The shared word-error total over 90,538 reference words is 1,708 (ground truth), 2,221 (HiFi-GAN), 2,284 (DDSP) and 2,309 (Vocos) [failures case 11].

Audio of 20 matched utterances (10 female, 10 male speakers, 3 to 12 s) for every system and condition is in `$SV_ROOT/eval/samples/<id>/<system>_<cond>.wav` with `samples/index.csv` (copy: `phase3/final/samples/index.csv`) [eval section 9]. Probe examples are in `$SV_ROOT/eval/probes/<system>/wav/<id>/<edit>.wav`.

## 8. Ablations

**Status: complete.** All twelve ablation runs trained to 100,000 generator steps and were evaluated with the same pipeline and conditions as the main set (T1, T2, T3). Every group of every system and condition is complete; there is no failed or missing run [abl-table, Data completeness]. Each ablation is a single training run, so every difference below is a statement about two trained models and includes run-to-run variance, which was not estimated.

### 8.1 Design and how to read the tables

The ablations follow Phase 1 decision D10 [p1 section 11; abl]. Each ablation is compared with the 100k baseline of the same vocoder, which was trained with the same recipe and a 100,000-step cosine schedule (the baselines are separate runs of experiment `ablation_100k`, not the step-100,000 checkpoint of the 400,000-step main runs).

| system name | what changes | compared with |
|---|---|---|
| `a100k_hifigan`, `a100k_vocos`, `a100k_ddsp` | the three main architectures at 100k steps (Vocos is option B) | baselines |
| `a100k_vocos_a` | Vocos option A (hop 480, n_fft 1920) instead of option B | `a100k_vocos` |
| `pcross0_hifigan` | speaker references always from the target utterance (`p_cross` = 0 instead of 0.5) | `a100k_hifigan` |
| `native_hifigan` | the native SPARC HiFi-GAN recipe: MPD + MSD, LSGAN, adversarial and feature-matching weights 5 and 10 (MPD) and 3 and 6 (MSD), Adam with betas 0.5 and 0.9, learning rate 1e-4, no mel-only warm-up | `a100k_hifigan` |
| `msd_hifigan`, `msd_vocos`, `msd_ddsp` | MPD + MSD in place of MPD + MRD (hinge loss, weight 1.0 for both sets instead of 0.1 for the MRD) | the baseline of the same vocoder |
| `a100k_hifigan_c320`, `a100k_vocos_d336` | about 6 M parameters: HiFi-GAN with 320 channels, Vocos with width 336 | the baseline of the same vocoder, and DDSP at 100k |
| `a100k_ddsp_c448` | DDSP with 448 channels (about 14.4 M parameters) | `a100k_ddsp`, and HiFi-GAN and Vocos B at 100k |

[abl; `~/sparc-eval-wt/src/sparc/conf/{loss,vocoder,experiment,eval_systems}/*.yaml`] The exact parameter counts of the size-matched variants were not extracted; the sizes "about 6 M" and "about 14.4 M" are taken from the comments in the configuration files, which refer to the generator. The baseline generators have 14.11 M (HiFi-GAN), 13.78 M (Vocos) and 5.92 M (DDSP) parameters (15.22 M, 14.90 M and 7.04 M in total with the speaker FFN; section 2). The DDSP baseline is therefore the "about 6 M" point and HiFi-GAN and Vocos B are the "about 14 M" points of the size-matched comparison.

**Method.** The differences are variant minus baseline from `paired.csv` of the ablation tables, with the speaker-cluster CI95 and the percentage of speakers with the sign of the overall difference, under the criterion of section 4 [abl-table]. Every difference is quoted from `paired.csv` except two groups that are not in it: the 100k-against-400k differences (section 8.2) and the comparisons between vocoders under MSD (the MSD columns of Table 8.6). I recomputed those from `per_speaker.csv` with the same method (speaker-cluster bootstrap, 10,000 resamples, seed 0, utterance-weighted speaker means) [abl-extra]. On six cells that exist in both, the recomputed point estimates equal those of `paired.csv` and the interval ends differ only by bootstrap resampling noise (at most 0.002 UTMOS or 0.02 cents). Corpus-level WER differences were not recomputed this way, so the 100k-against-400k WER differences are not tested. Cells are written "difference (CI95), agreeing speakers"; *n.s.* marks a CI that includes 0 and *CI only* a CI that excludes 0 with fewer than 75 percent of the speakers agreeing; both are not supported. T1 has 39 speakers and 4,687 utterances, T2 and T3 have 38 speakers and 4,686 utterances. The ablation `paired.csv` holds about 1,700 rows (17 pairs, 20 metrics, three conditions and the T2 strata) and no correction for multiple comparisons was applied.

### 8.2 Context: what the main runs gained from 100k to 400k steps

**Table 8.1. Main run (400k) minus 100k baseline, T1** (HiFi-GAN, DDSP and Vocos B; each cell: 100k value, 400k value, difference with CI95, agreeing speakers) [abl-table, table, abl-extra]. The 400k runs follow a 400,000-step cosine schedule and the 100k runs a 100,000-step schedule.

| metric | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| UTMOS | 3.946, 4.165: +0.218 (+0.197, +0.240), 97 % | 3.767, 3.832: +0.065 (+0.048, +0.085), 92 % | 3.588, 3.896: +0.308 (+0.287, +0.329), 100 % |
| Sim same | 0.801, 0.857: +0.056 (+0.053, +0.060), 100 % | 0.771, 0.802: +0.031 (+0.026, +0.036), 97 % | 0.787, 0.844: +0.057 (+0.053, +0.061), 100 % |
| Sim spk | 0.706, 0.731: +0.025 (+0.019, +0.030), 97 % | 0.678, 0.690: +0.013 (+0.008, +0.018), 92 % | 0.692, 0.722: +0.031 (+0.026, +0.037), 100 % |
| mel L1 | 0.5127, 0.4878: -0.0249 (-0.0266, -0.0233), 100 % | 0.5245, 0.5146: -0.0099 (-0.0107, -0.0091), 100 % | 0.5055, 0.4940: -0.0115 (-0.0127, -0.0103), 100 % |
| F0 RMSE (cents) | 30.7, 23.4: -7.3 (-8.2, -6.6), 100 % | 16.5, 18.2: **+1.7** (+1.3, +2.1), 95 % | 39.4, 35.0: -4.4 (-5.0, -3.9), 95 % |
| VDE | 0.0448, 0.0388: -0.0061 (-0.0070, -0.0052), 92 % | 0.0629, 0.0662: **+0.0032** (+0.0019, +0.0045), 79 % | 0.0478, 0.0415: -0.0063 (-0.0082, -0.0047), 100 % |
| loudness RMSE (dB) | 1.81, 1.49: -0.32 (-0.35, -0.30), 100 % | 2.69, 2.80: **+0.11** (+0.09, +0.13), 95 % | 2.08, 1.83: -0.25 (-0.27, -0.22), 97 % |

Every metric of every vocoder is supported; bold marks the three DDSP rows where the longer run is worse. T2 and T3 give the same signs. Corpus WER fell from 2.66 to 2.45 percent (HiFi-GAN), from 2.59 to 2.52 (DDSP) and from 2.64 to 2.55 (Vocos); these differences were not tested [table].

Reading: the gain from longer training differs by vocoder. HiFi-GAN and Vocos gain 0.22 and 0.31 UTMOS and about 0.06 Sim same, more than any ablation changes UTMOS (at most 0.14); DDSP gains little (+0.07 UTMOS, +0.03 Sim same) and its F0, voicing and loudness errors are worse at 400k than at 100k, although its mel L1 is lower. For HiFi-GAN and Vocos a 100k ablation difference of 0.05 to 0.1 UTMOS is therefore a small fraction of what training length changes, and the 100k ranking of the vocoders cannot be assumed to hold at 400k. The ranking by UTMOS at 100k is HiFi-GAN (3.946), DDSP (3.767), Vocos (3.588); at 400k it is HiFi-GAN, Vocos, DDSP. HiFi-GAN stays first in both. The DDSP-against-Vocos order is not stable: +0.18 (CI +0.08 to +0.27, 67 %, CI only) at 100k and -0.065 (not supported) at 400k [abl-table, paired]. The orders on F0 RMSE (DDSP, HiFi-GAN, Vocos) and VDE (HiFi-GAN, Vocos, DDSP) are the same at both lengths, and so is the order on Sim same (HiFi-GAN, Vocos, DDSP).

### 8.3 Vocos option A against option B

**Table 8.2. Vocos A minus Vocos B at 100k steps, T1** [abl-table]

| metric | B (baseline) | A | A - B (CI95), agreeing speakers |
|---|---|---|---|
| UTMOS | 3.588 | 3.466 | -0.122 (-0.154, -0.094), 90 % |
| PESQ | 1.789 | 1.805 | +0.015 (+0.002, +0.028), 56 %, CI only |
| WER (percent) | 2.64 | 2.60 | -0.04 pp (-0.10, +0.03), 49 %, n.s. |
| Sim same | 0.787 | 0.773 | -0.014 (-0.016, -0.012), 95 % |
| Sim spk | 0.692 | 0.679 | -0.013 (-0.015, -0.010), 97 % |
| mel L1 | 0.5055 | 0.5023 | -0.0032 (-0.0041, -0.0023), 87 % |
| F0 RMSE (cents) | 39.4 | 37.3 | -2.2 (-3.3, -0.9), 67 %, CI only |
| VDE | 0.0478 | 0.0483 | +0.0005 (-0.0004, +0.0013), 46 %, n.s. |

Under T2 and T3 the differences have the same signs and sizes (UTMOS -0.12 in both, Sim same -0.008 and -0.009, mel L1 -0.006 in both) [abl-table].

The pilot rule picked B at 25k steps (dev mel L1 0.5382 for B against 0.5406 for A; B is chosen when within 2 percent) [status]. The 100k runs agree with B on the metrics that matter for a listener and for the speaker: A has a lower UTMOS by 0.12 and lower speaker similarity (Sim same by 0.008 to 0.014, Sim spk by 0.013 to 0.015), supported under all three conditions. The advantages of A are small: its mel L1 is 0.6 percent lower (dev mel L1 at 100k: 0.5043 for A, 0.5061 for B, 0.4 percent lower [status]), and its F0 error is 2 cents lower (CI only). Under the rule fixed in advance (Phase 1) B is kept even at 100k, since A is within 2 percent. A reduces the Vocos F0 error by only 2 cents, so the larger FFT window of A (n_fft 1920) does not remove the F0 weakness of section 5: A is still at 37.3 cents against 30.7 cents for the 100k HiFi-GAN.

### 8.4 Speaker references from the target utterance (`p_cross` = 0)

**Table 8.3. HiFi-GAN with `p_cross` = 0 minus the `p_cross` = 0.5 baseline** [abl-table]. Baseline values are given as T1 / T2 / T3. The extra column is the T2 subset of the 3,811 utterances whose reference is from a different chapter.

| metric | baseline T1 / T2 / T3 | T1 | T2 | T2 different chapter | T3 |
|---|---|---|---|---|---|
| Sim same | 0.801 / 0.681 / 0.705 | +0.028 (+0.026, +0.030), 97 % | -0.007 (-0.010, -0.003), 66 %, CI only | -0.008 (-0.012, -0.004), 73 %, CI only | +0.010 (+0.007, +0.013), 87 % |
| Sim spk | 0.706 / 0.719 / 0.748 | +0.007 (+0.004, +0.010), 74 %, CI only | +0.004 (+0.0001, +0.008), 66 %, CI only | +0.004 (-0.0002, +0.008), n.s. | +0.023 (+0.020, +0.027), 97 % |
| UTMOS | 3.946 / 3.948 / 3.930 | +0.019 (+0.012, +0.026), 87 % | +0.001 (-0.005, +0.006), n.s. | +0.001 (-0.006, +0.008), n.s. | +0.005 (-0.002, +0.012), n.s. |
| PESQ | 1.893 / 1.782 / 1.801 | +0.050 (+0.043, +0.056), 100 % | -0.026 (-0.032, -0.019), 89 % | -0.030 (-0.036, -0.023), 96 % | +0.011 (+0.003, +0.019), 63 %, CI only |
| mel L1 | 0.5127 / 0.5671 / 0.5531 | -0.0144 (-0.0150, -0.0138), 100 % | +0.0188 (+0.0151, +0.0225), 97 % | +0.0210 (+0.0171, +0.0249), 100 % | +0.0083 (+0.0051, +0.0115), 82 % |
| WER (pp) | 2.66 / 2.76 / 2.78 | -0.03 (-0.09, +0.03), n.s. | +0.02 (-0.05, +0.09), n.s. | +0.02 (-0.07, +0.10), n.s. | 0.00 (-0.07, +0.07), n.s. |

Reading: the own-utterance shortcut is visible in the signal-level metrics. With `p_cross` = 0 the T1 result improves (mel L1 -2.8 percent, PESQ +0.050, Sim same +0.028) and the T2 result worsens (mel L1 +3.3 percent, PESQ -0.026, both supported); the gap in Sim same between T1 and T2 grows from 0.120 to 0.154 (0.829 against 0.675). The cost does not reach what a listener or the speaker model registers under T2: UTMOS and WER do not change under T2 (UTMOS under T1 rises by a supported +0.019), and the T2 changes in Sim same (-0.007) and Sim spk (+0.004) are CI only and at most 0.008. The rise of Sim spk under T3 (+0.023, to 0.772, above the ground-truth ceiling of 0.749) should not be read as better fidelity, because Sim spk under T3 rewards centring (section 3, case 10); why `p_cross` = 0 raises it was not tested. The dev mel L1 of the training log (0.5011 against 0.5174) has the sign of the T1 test result and is consistent with a dev loop that uses the own-utterance vector, which was not checked [status].

### 8.5 Native SPARC recipe against the shared recipe (HiFi-GAN)

**Table 8.4. Native recipe minus the shared-recipe baseline, HiFi-GAN at 100k** [abl-table]. The native value is for T1.

| metric | baseline T1 | native T1 | T1 | T2 | T3 |
|---|---|---|---|---|---|
| UTMOS | 3.946 | 4.060 | +0.114 (+0.093, +0.133), 87 % | +0.118 (+0.100, +0.137), 92 % | +0.141 (+0.123, +0.160), 95 % |
| PESQ | 1.893 | 1.755 | -0.139 (-0.152, -0.124), 100 % | -0.126 (-0.139, -0.112), 100 % | -0.111 (-0.125, -0.098), 100 % |
| WER (percent) | 2.66 | 3.07 | +0.41 pp (+0.32, +0.50), 85 % | +0.38 pp (+0.28, +0.49), 84 % | +0.38 pp (+0.29, +0.48), 87 % |
| Sim same | 0.801 | 0.796 | -0.004 (-0.008, -0.00005), 59 %, CI only | -0.007 (-0.010, -0.002), 61 %, CI only | +0.002 (-0.002, +0.006), n.s. |
| Sim spk | 0.706 | 0.706 | -0.0003 (-0.005, +0.005), n.s. | +0.001 (-0.003, +0.006), n.s. | +0.009 (+0.005, +0.014), 76 % |
| mel L1 | 0.5127 | 0.5579 | +0.0453 (+0.0436, +0.0469), 100 % | +0.0459 (+0.0439, +0.0477), 100 % | +0.0454 (+0.0434, +0.0474), 100 % |
| F0 RMSE (cents) | 30.7 | 32.6 | +1.9 (+1.3, +2.5), 90 % | +2.0 (+1.5, +2.5), 87 % | +1.9 (+1.1, +2.5), 87 % |
| loudness RMSE (dB) | 1.81 | 2.02 | +0.21 (+0.19, +0.24), 100 % | +0.23 (+0.20, +0.26), 100 % | +0.22 (+0.19, +0.26), 100 % |

Reading: the native recipe gives a higher UTMOS (+0.11 to +0.14, the same in all three conditions) but follows the input less closely: mel L1 is 8.8 percent higher, PESQ is 0.14 lower, WER is 0.4 percentage points (15 percent) higher and the F0 and loudness errors are larger, all supported; speaker similarity does not differ in a supported way under T1 and T2 (under T3 Sim spk is +0.009, supported, but T3 Sim spk is not a fidelity measure, section 3). The native recipe changes the discriminator set, the GAN loss form, the loss weights, the optimizer settings and the warm-up together, so this ablation cannot say which change causes which effect; the UTMOS direction is shared with the MSD-only ablation (section 8.6, +0.07) and may come in part from the discriminator. The native recipe at 100k (UTMOS 4.060) stays below the shared recipe at 400k (4.165) [table].

### 8.6 Multi-scale (MSD) against multi-resolution (MRD) discriminator

**Table 8.5. MSD minus MRD for each vocoder at 100k, T1** (difference with CI95, agreeing speakers) [abl-table]. Under T2 the signs and sizes are the same (for example F0 RMSE +2.5, +8.1 and +0.6 cents).

| metric | HiFi-GAN | Vocos | DDSP |
|---|---|---|---|
| UTMOS | +0.073 (+0.060, +0.087), 92 % | -0.048 (-0.080, -0.019), 69 %, CI only | -0.057 (-0.071, -0.043), 87 % |
| PESQ | -0.098 (-0.107, -0.089), 100 % | -0.199 (-0.214, -0.182), 100 % | -0.084 (-0.094, -0.076), 100 % |
| WER (pp) | +0.09 (+0.02, +0.17), 54 %, CI only | +0.26 (+0.18, +0.35), 72 %, CI only | +0.08 (+0.03, +0.14), 56 %, CI only |
| Sim same | -0.001 (-0.004, +0.002), n.s. | -0.025 (-0.030, -0.021), 97 % | -0.028 (-0.032, -0.023), 97 % |
| Sim spk | +0.001 (-0.003, +0.006), n.s. | -0.010 (-0.016, -0.004), 74 %, CI only | -0.020 (-0.026, -0.015), 92 % |
| mel L1 | +0.0277 (+0.0270, +0.0284), 97 % | +0.0353 (+0.0343, +0.0363), 100 % | +0.0079 (+0.0074, +0.0085), 97 % |
| F0 RMSE (cents) | +2.2 (+1.6, +2.7), 90 % | **+7.8** (+7.0, +8.7), 100 % | +0.6 (+0.4, +0.8), 90 % |
| VDE | +0.0009 (+0.0001, +0.0016), 64 %, CI only | +0.0046 (+0.0034, +0.0057), 90 % | **+0.0085** (+0.0069, +0.0104), 100 % |

**Table 8.6. Ranking of the three vocoders at 100k under each discriminator, T1** (first minus second, CI95, agreeing speakers; the MRD columns are quoted from `paired.csv`, with the sign reversed for DDSP - Vocos, and the MSD columns are recomputed [abl-table, abl-extra])

| metric | HiFi-GAN - Vocos, MRD | HiFi-GAN - Vocos, MSD | HiFi-GAN - DDSP, MRD | HiFi-GAN - DDSP, MSD | DDSP - Vocos, MRD | DDSP - Vocos, MSD |
|---|---|---|---|---|---|---|
| UTMOS | +0.359 (+0.328, +0.388), 97 % | +0.480 (+0.441, +0.523), 97 % | +0.180 (+0.109, +0.262), 82 % | +0.310 (+0.247, +0.380), 95 % | +0.179 (+0.078, +0.266), 67 %, CI only | +0.170 (+0.094, +0.239), 72 %, CI only |
| Sim same | +0.014 (+0.010, +0.018), 92 % | +0.038 (+0.034, +0.042), 100 % | +0.029 (+0.023, +0.035), 97 % | +0.056 (+0.048, +0.065), 100 % | -0.015 (-0.021, -0.010), 90 % | -0.018 (-0.026, -0.010), 72 %, CI only |
| F0 RMSE (cents) | -8.7 (-10.0, -7.7), 97 % | -14.4 (-16.1, -12.9), 97 % | +14.1 (+12.2, +16.5), 100 % | +15.7 (+13.8, +18.0), 100 % | -22.9 (-26.3, -20.1), 97 % | -30.1 (-34.0, -26.8), 100 % |
| VDE | -0.0030 (-0.0047, -0.0016), 79 % | -0.0067 (-0.0085, -0.0050), 92 % | -0.0181 (-0.0204, -0.0158), 97 % | -0.0257 (-0.0296, -0.0221), 100 % | +0.0151 (+0.0119, +0.0179), 87 % | +0.0190 (+0.0156, +0.0227), 97 % |

Reading: the Phase 1 concern that the MRD favours Vocos and DDSP is partly right about the effect and wrong about the ranking. Replacing the MRD hurts Vocos and DDSP on speaker similarity and voicing (Sim same -0.025 and -0.028; VDE +0.005 and +0.009) and raises the Vocos F0 error from 39.4 to 47.2 cents, whereas HiFi-GAN is unchanged on speaker similarity and gains 0.07 UTMOS. The ranking is the same under both discriminators on every metric in Table 8.6, and the HiFi-GAN lead grows under MSD (UTMOS lead over Vocos +0.36 to +0.48 and over DDSP +0.18 to +0.31). The discriminator therefore does not explain the HiFi-GAN advantage of section 3. The MSD arm also uses a larger adversarial weight (1.0 against 0.1) and no retuning, so it measures "this MSD configuration" and not a general discriminator effect. The Vocos F0 weakness and the DDSP voicing errors (VDE) are not caused by the MRD: both are worse without it.

### 8.7 Size-matched variants

**Table 8.7. Effect of width within each family at 100k, T1** (variant minus baseline of the same vocoder, CI95, agreeing speakers) [abl-table]

| metric | HiFi-GAN 320 channels (about 6 M) minus HiFi-GAN 100k | Vocos width 336 (about 6 M) minus Vocos B 100k | DDSP 448 channels (14.4 M) minus DDSP 100k |
|---|---|---|---|
| UTMOS | -0.096 (-0.107, -0.084), 100 % | -0.100 (-0.113, -0.085), 95 % | -0.049 (-0.062, -0.034), 85 % |
| Sim same | -0.029 (-0.032, -0.026), 100 % | -0.016 (-0.018, -0.014), 100 % | +0.004 (+0.002, +0.006), 74 %, CI only |
| Sim spk | -0.016 (-0.019, -0.013), 100 % | -0.008 (-0.010, -0.005), 89 % | +0.001 (-0.001, +0.004), n.s. |
| mel L1 | +0.0208 (+0.0201, +0.0215), 100 % | +0.0164 (+0.0157, +0.0171), 100 % | -0.0018 (-0.0027, -0.0010), 74 %, CI only |
| F0 RMSE (cents) | +1.4 (+0.9, +1.8), 90 % | +1.4 (+1.0, +1.9), 90 % | **+4.4** (+4.0, +5.0), 97 % |
| VDE | +0.0035 (+0.0027, +0.0044), 90 % | +0.0007 (-0.0006, +0.0018), n.s. | -0.0040 (-0.0053, -0.0029), 90 % |
| WER (pp) | +0.18 (+0.10, +0.27), 69 %, CI only | +0.16 (+0.09, +0.23), 69 %, CI only | -0.08 (-0.16, -0.01), 56 %, CI only |

**Table 8.8. The three vocoders at matched size, T1** (first minus second, CI95, agreeing speakers; for F0 RMSE and VDE a lower value is better). About 6 M: HiFi-GAN 320 channels, DDSP 100k, Vocos width 336. About 14 M: HiFi-GAN 100k, DDSP 448 channels, Vocos B 100k. All cells are quoted from `paired.csv` (signs reversed where the file lists the other order) [abl-table].

| size | metric | HiFi-GAN - DDSP | HiFi-GAN - Vocos | DDSP - Vocos |
|---|---|---|---|---|
| 6 M | UTMOS | +0.084 (+0.014, +0.164), 59 %, CI only | +0.363 (+0.329, +0.394), 97 % | +0.278 (+0.174, +0.367), 74 %, CI only |
| 6 M | Sim same | 0.000 (-0.006, +0.005), n.s. | 0.000 (-0.004, +0.004), n.s. | 0.000 (-0.004, +0.005), n.s. |
| 6 M | Sim spk | +0.012 (+0.006, +0.018), 84 % | +0.006 (+0.002, +0.011), 68 %, CI only | -0.006 (-0.012, -0.001), 58 %, CI only |
| 6 M | F0 RMSE (cents) | +15.5 (+13.5, +18.1), 100 % | -8.8 (-9.9, -7.7), 100 % | -24.3 (-27.9, -21.4), 100 % |
| 6 M | VDE | -0.0146 (-0.0171, -0.0120), 97 % | -0.0002 (-0.0008, +0.0004), n.s. | +0.0144 (+0.0117, +0.0170), 92 % |
| 14 M | UTMOS | +0.229 (+0.165, +0.300), 92 % | +0.359 (+0.328, +0.388), 97 % | +0.130 (+0.040, +0.208), 62 %, CI only |
| 14 M | Sim same | +0.025 (+0.019, +0.032), 95 % | +0.014 (+0.010, +0.018), 92 % | -0.011 (-0.018, -0.006), 72 %, CI only |
| 14 M | Sim spk | +0.027 (+0.019, +0.035), 95 % | +0.015 (+0.010, +0.019), 89 % | -0.012 (-0.019, -0.006), 74 %, CI only |
| 14 M | F0 RMSE (cents) | +9.7 (+7.6, +12.2), 100 % | -8.7 (-10.0, -7.7), 97 % | -18.4 (-22.0, -15.4), 97 % |
| 14 M | VDE | -0.0141 (-0.0162, -0.0117), 95 % | -0.0030 (-0.0047, -0.0016), 79 % | +0.0111 (+0.0075, +0.0141), 79 % |

Reading: shrinking HiFi-GAN or Vocos to about 6 M costs about 0.10 UTMOS and 0.016 to 0.029 Sim same (supported), while doubling DDSP changes little and makes its F0 error worse by 4.4 cents. The HiFi-GAN advantage over DDSP depends on size: at 14 M it is supported on UTMOS (+0.23) and on both similarity metrics, and at 6 M the UTMOS lead is CI only (+0.08) and Sim same is equal (0.000, with Sim spk +0.012 still supported). The HiFi-GAN lead over Vocos on UTMOS is the same at both sizes (+0.36), and the DDSP pattern of the best F0 error and the worst voicing and loudness errors holds at both. The two UTMOS comparisons that involve DDSP are CI only at 6 M, so the DDSP-against-Vocos order is as unsettled here as at 400k (section 8.2).

The DDSP deficit on the speakers with the largest DDSP drops in the failure analysis does not depend on width or on the discriminator. The UTMOS drop against the ground truth (T1) for speaker 7021 is 1.05 (DDSP 100k), 1.04 (448 channels), 0.99 (MSD) and 1.00 (400k main run), against 0.39 for HiFi-GAN 100k and 0.25 for the 400k HiFi-GAN; for speaker 2300 it is 0.94, 0.95, 0.91 and 0.90 against 0.33 and 0.14; for the male speaker 7729 it is 0.52, 0.58, 0.53 and 0.28 against 0.14 and 0.02, so for this speaker the 400k run has about half the drop of the three 100k runs [abl-table per-speaker, table, failures]. These are three speakers, chosen because the failure analysis names them, and not a test of the male-speaker deficit over all male speakers (speaker sex is not in the copied tables). A capacity explanation for the DDSP deficit is not supported by this result.

### 8.8 What the ablations do and do not show

- Supported for these runs: option B is better than option A for listener-facing quality at 100k; the own-utterance reference protocol trades cross-utterance signal fidelity for self-conditioned fidelity without a supported change in cross-utterance speaker similarity; the MRD helps Vocos and DDSP but does not produce the vocoder ranking; DDSP does not gain from width.
- Not shown: which component of the native recipe produces its UTMOS gain and its losses in fidelity; whether the 400k ranking of Vocos and DDSP can be inferred from 100k; run-to-run variance. Every effect of 0.1 UTMOS or smaller, and every WER difference of 0.2 percentage points or smaller, could be produced by run variance, which was not estimated.

## 9. Open questions and next steps

Open questions, each tied to what was and was not tested:

1. **Architecture or run variance.** Every difference in sections 3 to 7 comes from one run per arm. Training variance is not estimated. The cheapest way to bound it is a second seed of one 100k-step ablation arm (suggested in Phase 1 section 8 if budget allows); none has been run. The ablations make this more pressing: several supported effects (UTMOS 0.05 to 0.1, WER 0.1 to 0.2 percentage points) are of a size that run variance could produce, and the 100k ranking differs from the 400k ranking.
2. **Why isolated EMA offsets are not reproduced.** The candidate explanations in section 5.2 are untested. A probe that scales the dynamics of an utterance around its mean, or shifts only speech frames, would separate them. A probe with real articulatory change (a re-inversion of audio that is known to differ in tongue position) is not available.
3. **Vocos F0 precision.** Whether the 50 to 150 cent deviations and the error below 70 Hz come from the iSTFT head, the 50 Hz frame rate or the amount of low-pitched training speech is untested. The ablations rule out two causes: the multi-resolution discriminator (the F0 error is 47 cents without it) and the FFT window of option A (37 cents against 39). The HiFi-GAN comparison (half the error rate with the same dependence on pitch) remains partial evidence.
4. **DDSP voicing and the male-speaker deficit.** The false voicing and the quiet pauses correlate with the UTMOS drop but a causal link is untested; a voicing gate or a different loudness path would test it. The size-matched DDSP (448 channels) and MSD ablations show that neither capacity nor the discriminator removes the UTMOS drop on the speakers with the largest drops (section 8.7); the voicing error is worse with MSD (VDE 0.071 against 0.063; the false-voicing share itself was not measured for the MSD arm) and the 448-channel model has a slightly lower VDE (-0.004) and a larger F0 error (+4.4 cents). A causal test of a voicing gate is still open.
5. **Shared word errors.** A common cause in the shared input is plausible (three vocoders fail on the same content) and untested. The Vocos mel copy synthesis has WER equal to the ground truth, which points to the feature bottleneck, but the WER of the features themselves was not measured by a separate probe.
6. **Metrics.** Sim spk is not a fidelity measure under T3, and Sim same under T2 depends on the target-reference similarity. A speaker-similarity measure that corrects for this (for example, using several references per speaker) or a listening test would be needed. No human listening was done in this phase; UTMOS is a predictor, not a listener.
7. **Statistics.** The paired criterion (CI95 excludes 0 and at least 75 percent of speakers agree) is a convention; roughly 180 comparisons were made in the main set and about 1,700 paired rows were produced in the ablation set, without correction.
8. **Streaming.** Only offline timings and a single-utterance lookahead probe exist. A chunked implementation would be needed to measure real latency; for DDSP the oscillator phase state is the open item.
9. **Second split and the reference.** test.other was not evaluated. en+ may have seen test.clean.
10. **Unexplained timing difference** in the en+ pipeline on 8 CPU threads (section 6).
11. **Confounded ablations.** The native recipe changes the discriminators, loss form and weights, optimizer and warm-up together, and the MSD arm also changes the adversarial weight; the source of the native recipe's UTMOS gain and fidelity loss is not identified. Single-factor runs (the shared recipe with only the optimizer settings or only the loss form changed) would separate them.
12. **`p_cross` between 0 and 0.5, and the T3 effect.** Only 0 and 0.5 were run, so the trade between T1 fidelity and T2 fidelity is known at two points. Why `p_cross` = 0 raises Sim spk under T3 by 0.023 was not tested.
13. **DDSP at 400k.** Its F0, voicing and loudness errors are worse at 400k than at 100k although its mel L1 is lower; whether this is a property of the run, of the loss or of longer training was not tested.

Next steps: (a) and (b) are done: the ablation support (config group `eval_systems`, `table_name` in `io.py`, copied experiment configs, `tests/eval/test_ablations.py`) was committed as `ea550ad` and `phase3-eval` was merged into `vocoder-comparison` as `4af6f33` [status, 2026-10-12 entries]; (c) optionally run `test.other` with the same code (`eval.split=test.other`); (d) a second seed for one 100k arm to bound run variance (open question 1); (e) single-factor runs that separate the components of the native recipe (open question 11).

## 10. Reproducibility

**Code.**

- Training branch `vocoder-comparison` of the fork `nzxyin/Speech-Articulatory-Coding` (base `9777c30`). Commits: `603c58b` scaffolding, `85a65c1` pipeline, `a966a75` review fixes (`cudnn.benchmark` off), `c91ce98` documentation, `a11d1c4` ablation_100k configuration, `05bf159` D10 ablation configurations (`git log` of `~/Speech-Articulatory-Coding`). The main runs started on 2026-10-05 16:02 to 2026-10-06 01:48 (hydra directories), after `c91ce98`. The commit that each run used is not recorded in the run directories that I read; it was one of `c91ce98` or the later configuration-only commits, which do not change training code, but this was not verified.
- Evaluation branch `phase3-eval`, worktree `~/sparc-eval-wt`. The tables record `fork_commit` `374142c96bcd5039b799ac188f62dca864162da6` (`Changelog: evaluation fixes`) [meta.json]. Earlier commits: `da436ab` contract, `fb66135` pipeline, `848840c` efficiency markers, `5e77425` WER normalization. The ablation tables record `374142c96bcd5039b799ac188f62dca864162da6+dirty`, because the ablation support was uncommitted when they were built; it was committed afterwards as `ea550ad` and merged into `vocoder-comparison` as `4af6f33` [meta.json; status].
- Issues: tracking issue `nzxyin/Speech-Articulatory-Coding#19`; Phase 1 bugs #9 to #15; `pysptk`/`pkg_resources` issue #20 [p1 section 13; status].

**Data and features.**

- Feature cache: `$SPARC_VOC_CACHE` = `$SV_ROOT/cache` (358,332 valid utterances, 171 not encodable, 0 invalid), packed splits in `cache/packed/` (train 339,007 utterances, 97.2 million frames), training statistics `cache/stats/train_stats.json`, manifest `cache/manifest.parquet` [status, 2026-10-05 15:50].
- T2 reference manifests: `cache/sources/ref_manifest_test.clean.jsonl` (and `ref_manifest_dev.clean.jsonl`); the manifest generator is deterministic and was validated [p1 section 4; `ls cache/sources`].
- Inversion head: `/data/user_data/xoy/mngu0_linear_aai_refit/linear_aai_mngu0_refit_peralign.pkl`, sha256 `f6514ec405d311a9...` as recorded in [status] (`f6514ec4…` in [p1 section 13]; full value in `STATUS.md`).
- SPARC en+ Hugging Face snapshot `2e6a07d4b35022d4366a2896a6293fe16aaed78a`; `charactr/vocos-mel-24khz` revision `0feb3fdd…`; filtered list `parler-tts/libritts_r_filtered` snapshot `8980d653…`; torch 2.9.0, torchaudio 2.9.0, Lightning 2.6.5, transformers 4.57.1 [p1 section 13]. Evaluation models: UTMOS22 strong (`tarepan/SpeechMOS:v1.2.0`), `openai/whisper-large-v3`, `speechbrain/spkrec-ecapa-voxceleb`, `microsoft/wavlm-base-plus-sv` as secondary [eval section 4]. Model revisions and package versions are recorded in the `meta.json` sidecars of each stage.

**Runs and checkpoints.** Run directories `$SV_ROOT/runs/<experiment>/<vocoder>/` hold `ckpt/`, `hydra/`, `predictions/`, `tb/`, `status.json`. Seed 0 for every run (`seed: 0` in the run configuration and in `train_config.yaml`). Checkpoints used in the main evaluation [table, Checkpoints; efficiency JSON]:

| system | checkpoint | g_step |
|---|---|---|
| HiFi-GAN | `$SV_ROOT/runs/main/hifigan/ckpt/step000400000.ckpt` | 400,000 of 400,000 |
| DDSP | `$SV_ROOT/runs/main/ddsp/ckpt/step000400000.ckpt` | 400,000 of 400,000 |
| Vocos | `$SV_ROOT/runs/main/vocos/ckpt/step000400000.ckpt` | 400,000 of 400,000 |

Preemption history: HiFi-GAN was requeued at least twice and DDSP at least once (several hydra start directories in `runs/main/<vocoder>/hydra/`); training resumes at a batch boundary, and the final checkpoint is the one evaluated [status]. No checkpoint was selected on test metrics.

**Evaluation outputs** under `$SV_ROOT/eval/`: `audio/`, `results/test.clean/<system>/<cond>/<group>/part-*.parquet`, `embeddings/`, `arrays/`, `probes/<system>/`, `efficiency/<system>.json`, `tables/test.clean/{table.md, table.csv, paired.csv, per_speaker.csv, meta.json}`, `samples/`, `failures/<case>/`. Copies of the final tables, efficiency JSON files and sample index: `~/sparc-vocoders-work/phase3/final/`. Analysis files and scripts: `~/sparc-vocoders-work/phase3/analysis/` (`paired_report.py`, `build_master.py`, `add_sims.py`, `build_cases.py`, `cases.py`, `failure_stats.py`, `efficiency.py`, `probes_table.py`, `natural_variation.py`, `pc_probe.py`, `controllability_report.py` under `scripts/`). Per-utterance results of all systems with Sim same and Sim spk: `analysis/master.parquet`.

**Commands.** `python -m sparc.cli.eval_vocoder stage=<gt|refs|signal|utmos|asr|spk|reextract|prosody|probes|efficiency|aggregate|samples> system=<name|all> condition=<T1|T2|T3|all> eval.split=test.clean`, run as `~/sparc-eval-wt/.venv/bin/python` on a compute node after `source ~/sparc-vocoders-work/env.sh`. Slurm wrappers: `scripts/slurm/eval.sh` and `submit_eval.sh`. Ablation evaluation: add `eval_systems=ablations`; its tables are in `$SV_ROOT/eval/tables/test.clean/ablations/` (copy: `phase3/final/tables/ablations/`), the checkpoints are those of experiments `ablation_100k`, `abl_pcross0`, `abl_native` and `abl_msd` under `$SV_ROOT/runs/` (names in section 8.1; I did not inspect `/data`). The extra paired differences of section 8 are recomputed with the script reproduced in `phase3/I_ablation_section.md`.

## Appendix: inconsistencies found between sources

1. **GPU RTF values.** `STATUS.md` (2026-10-08 03:40) quotes 0.0058, 0.0007 and 0.0046 for HiFi-GAN, Vocos and DDSP; `efficiency.md` uses the medians 0.0061, 0.0008 and 0.0050. Both are correct for their definition: the JSON files hold the pooled value (total time over total audio: 0.00575, 0.00070, 0.00464) and the median. This document uses the medians and states the pooled values.
2. **Ceiling wording in the failures summary.** The hand-over summary says that under T3 the vocoders exceed the ceiling "0.783 to 0.787 against 0.749 for HiFi-GAN and Vocos". The value 0.749 is the ground-truth ceiling (table row "ground truth", Sim spk); DDSP under T3 (0.731) is below it. This document uses the table.
3. **Dev mel L1 of the Vocos B ablation.** `STATUS.md` gives 0.5061 at 100k (2026-10-07 03:00 entry) and 0.506 at 95k (2026-10-06 12:30 entry); the two are consistent.
4. **Efficiency timing mismatch** for the en+ pipeline on 8 CPU threads (31 percent) is stated in the source and left unexplained.
5. **Statement on the evaluation of the lookahead for DDSP.** `efficiency.md` gives a measured lookahead of 929 ms for all channels and 717 to 1,063 ms per channel group; the hand-over summary says "about 0.9 to 1.1 s". The per-group maximum is 1,063 ms and the all-channel value is 929 ms, so "about 0.9 to 1.1 s" is a fair rounding of the range.
6. **Paired-table cell detail.** Several cells of Table 3 give only the verdict ("supported") because `paired.md` summarizes them in words; the numeric values are in `paired_vocoders.csv`.
7. **Run commit.** The training commit used by each main run is not stored in the run directory (section 10).
