# Phase 3 write-up: three vocoders for SPARC articulatory features

Status: 2026-10-19. The main evaluation, the ablation evaluation, the evaluation of a second split (test.other, section 3.1), the evaluation of a second training seed of the three main vocoders (section 4.1) and the evaluation of runs of 800,000 generator steps of HiFi-GAN and Vocos (section 6.1.3) are complete (sections 3 to 8).

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
| [other] | `phase3/final/tables/test.other/{table.md, table.csv, paired.csv, per_speaker.csv, meta.json}` (copy of `$SV_ROOT/eval/tables/test.other/`); notes `phase3/L_testother.md` |
| [seeds] | `phase3/final/tables/seeds/{table.md, table.csv, paired.csv, per_speaker.csv, meta.json}` (copy of `$SV_ROOT/eval/tables/test.clean/seeds/`); `paired.csv` holds the seed-to-seed pairs (for example `hifigan_s1 - hifigan`) and the vocoder pairs within and across seeds; notes `phase3/K_seeds.md`; the robustness rule of section 4.1 was applied to `paired.csv` with the script reproduced in `phase3/O_results_followups.md` |
| [probes-seeds] | `phase3/analysis/probes_seeds.md` (controllability probes and natural variation of both seeds; scripts `probes_seeds.py`, `probes_seeds_report.py`, `natural_variation.py hifigan_s1,ddsp_s1,vocos_s1 _s1`; outputs `controllability_seeds_{long,paired,robust}.csv`, `natural_variation_s1.csv`, `natural_variation_matrix_s1.npz`) |
| [conv] | `phase3/analysis/convergence.md` and `convergence_data.json`; section 3 of `convergence.md` covers the 800k runs, with `curves_800k.json` and the scripts `curves_800k.py` and `convergence_refit.py` |
| [long] | `phase3/final/tables/long/{table.md, table.csv, paired.csv, per_speaker.csv, meta.json}` (copy of `$SV_ROOT/eval/tables/test.clean/long/`, the 800k runs next to the two 400k seeds); paired rows per condition in `phase3/analysis/long_report_{T1,T2,T3}.md` (script `long_report.py`); pairs not in `paired.csv` in `phase3/analysis/long_extra_pairs.md` (script `long_extra_pairs.py`, method of section 8) |
| [status] | `STATUS.md` |
| [p1] | `phase1_report.md` |
| [eval] | `~/sparc-eval-wt/docs/vocoders/EVALUATION.md` (the evaluation contract) |

`$SV_ROOT` is `/data/user_data/xoy/sparc-vocoders`.

## 1. Summary

Three vocoders (HiFi-GAN, DDSP, Vocos) were each trained twice (seed 0, the main runs, and seed 1), for 400,000 generator steps, to synthesize 24 kHz speech from SPARC articulatory features (12 EMA channels, pitch, loudness and periodicity at 50 Hz) with FiLM speaker conditioning. They were evaluated on LibriTTS-R test-clean (4,687 utterances, 39 speakers; both seeds) and on test-other (4,709 utterances, 33 speakers; seed 0 only). **The ablations (section 8) and the 800k-step runs (section 6.1.3) are single runs.** The speaker-cluster intervals cover the sampling of test speakers only, not the sampling of training runs. Two runs per vocoder give a first, crude estimate of run-to-run variation (section 4.1) and not a variance estimate. A difference between two vocoders is therefore called *robust* only if it has the same sign and is supported (section 4) in both seeds and is larger than the seed-to-seed difference of each of the two vocoders. A robust difference is still a statement about these trained models; the architecture is not shown to be the cause. Every finding below states whether it holds in both seeds and on test.other.

1. **HiFi-GAN is ahead on perceived quality and speaker similarity (strong evidence for these models; robust in both seeds; replicates on test.other; same result under T1, T2 and T3 for UTMOS; at 100k steps the lead over Vocos holds under both discriminators, and the lead over DDSP is not supported at about 6 M parameters, section 8).** UTMOS is 4.165 (seed 0) and 4.161 (seed 1) for HiFi-GAN, 3.896 and 3.939 for Vocos and 3.832 and 3.821 for DDSP, against 4.208 for the ground truth [table, seeds]. The paired UTMOS difference of HiFi-GAN under T1 is +0.33 over DDSP in both seeds (seed 0: CI +0.28 to +0.40, all 39 speakers agree; seed 1: +0.340, CI +0.295 to +0.392, 100 percent) and +0.27 (seed 0, 97 percent) and +0.22 (seed 1, 97 percent) over Vocos [paired, seeds]; each is more than five times the larger seed-to-seed difference (0.011 for DDSP, 0.042 for Vocos). Sim same and Sim spk are ahead in the same way (Sim same +0.054 and +0.057 over DDSP, +0.013 and +0.012 over Vocos; the small lead over Vocos in Sim same is not supported under T2 and T3). On test.other (seed 0) HiFi-GAN is at the ground-truth level (4.030 against 4.035) and ahead of DDSP by +0.334 and of Vocos by +0.233 (both 100 percent) [other]. **DDSP against Vocos on UTMOS is not robust:** Vocos is ahead by 0.065 (seed 0, not supported), 0.118 (seed 1, CI only, 69 percent) and 0.100 (test.other, CI only, 64 percent), with the same sign each time and never supported [paired, seeds, other]. After 800k steps (one run each, section 6.1.3) HiFi-GAN reaches the ground-truth UTMOS (4.207 against 4.208) and stays ahead of Vocos at 800k (+0.239, 97 percent), and Vocos at 800k is ahead of DDSP at 400k by 0.137 and 0.148 (supported against both DDSP seeds; one Vocos run) [long].
2. **Word and character error rates do not separate the vocoders (strong evidence of no resolvable difference; the same in both seeds and on test.other).** Corpus WER is 2.45, 2.52 and 2.55 percent (HiFi-GAN, DDSP, Vocos) in seed 0 and 2.49, 2.48 and 2.47 percent in seed 1, against 1.89 percent for the ground truth. No pair is supported in any condition in either seed (all-utterance cells; the only supported WER difference between two vocoders in any subset is Vocos - DDSP in the T2 different-chapter subset of seed 0, +0.17 percentage points, 77 percent); the largest difference is 0.14 percentage points in seed 0 (at most 66 percent of speakers agree) and 0.02 in seed 1 [paired, seeds]. The word errors are largely shared: the per-utterance excess over the ground truth correlates 0.55 to 0.63 between vocoders (seed 0) [failures]. On test.other the corpus WER is 3.37 percent for the ground truth and 5.03, 5.10 and 5.00 percent for the vocoders: the gap to the ground truth is 1.6 to 1.7 percentage points, 2.5 to 3 times the gap on test.clean, and the vocoders are still not separated (no pair supported in T1, T2 or T3) [other].
3. **The vocoders fail on different prosodic quantities (strong evidence for the Vocos F0 error and for the DDSP voicing and loudness errors; robust in both seeds and on test.other; the HiFi-GAN-DDSP F0 order is not robust).** Vocos has the largest F0 error in every evaluation (35.0 cents in seed 0, 33.5 in seed 1, 34.8 on test.other); it is 11.6 and 10.0 cents behind HiFi-GAN and 16.8 and 12.6 cents behind DDSP in the two seeds, robust. The error is caused by broad deviations of 50 to 150 cents and not by octave errors (6 of 797,816 voiced frames lie near an octave, seed 0), and Vocos is strongest on male voices (seed 0 only); the 100k ablations show that the F0 error is not caused by the multi-resolution discriminator (47 cents with the multi-scale one) and that Vocos option A lowers it by only 2 cents (section 8) [table, paired, failures, seeds]. DDSP has the largest VDE (0.066, 0.068, 0.067 on test.other) and loudness error (2.80, 2.89, 2.64 dB), robust against both other vocoders. DDSP follows F0 best in seed 0 (18.2 cents) and on test.other (17.5), but its F0 error rose by 2.7 cents in seed 1 (20.9, supported), so its lead over HiFi-GAN (23.5) shrinks to 2.6 cents (CI only, 69 percent), within the seed-to-seed change: the HiFi-GAN-DDSP F0 order is not robust under T1 (it is supported on test.other, +6.1 cents) [seeds, other]. HiFi-GAN is the best of the three on loudness (robust) and on VDE (robust against DDSP; against Vocos the difference is only 0.003 and 0.002 and is CI only on test.other) [table, paired, seeds, other].
4. **DDSP loses on male speakers (moderate evidence; correlational; the per-speaker pattern replicates in seed 1, the sex split was not repeated).** Its UTMOS is 3.69 for male and 3.93 for female speakers, against 4.15 and 4.17 for HiFi-GAN (seed 0). It marks 13.5 percent of ground-truth unvoiced frames as voiced, against 4.8 percent for each of the other two, and renders pauses too quietly. These properties correlate with the UTMOS drop within DDSP (VDE, r = -0.40), but a causal link was not tested [failures]. A DDSP with twice the parameters (448 channels) or with the multi-scale discriminator shows the same UTMOS drop on the two speakers with the largest drops, so capacity and the discriminator do not explain it (section 8.7). In seed 1 the speaker-level UTMOS drops against the ground truth are close to seed 0 (speaker 7021: 0.93 against 1.00; speaker 2300: 0.81 against 0.90; speaker 7729: 0.29 against 0.28; HiFi-GAN seed 1: 0.21, 0.12, 0.02), and the mean over speakers of the drop is 0.39 for DDSP in both seeds, against 0.04 for HiFi-GAN and 0.27 and 0.24 for Vocos [seeds, per_speaker.csv]. The sex split and the false-voicing share were not recomputed for seed 1 (the speaker sex is not in the copied tables). On test.other the mean speaker-level drop is 0.32 for DDSP, 0.27 for Vocos and 0.01 for HiFi-GAN, spread over more speakers (largest DDSP drop 0.56, against 1.00 on test.clean); the sex split was not analysed there [other, per_speaker.csv].
5. **Controllability: F0 and loudness edits are followed closely by all three vocoders; single-channel EMA edits are not (strong evidence for the finding, no explanation; replicates in both training seeds; test.clean only).** A constant plus or minus one-std edit of one EMA channel is reproduced to a mean gain of 0.12, 0.10 and 0.12 (HiFi-GAN, DDSP, Vocos), with a mean off-target movement of 0.24 std in all three, although the same loop reproduces natural EMA variation with a slope of 1.00 (gt floor, HiFi-GAN, Vocos) or 0.96 (DDSP). A constant edit of one natural standard deviation along the first four natural principal components also fails (gain 0.09 to 0.14, six utterances per vocoder). The data do not say why [control]. In seed 1 the same probes give a mean single-channel gain of 0.120, 0.102 and 0.115 and the same natural-variation slopes; HiFi-GAN and Vocos follow EMA edits slightly more than DDSP in both seeds (by 0.013 to 0.019, robust), and HiFi-GAN against Vocos is not robust [probes-seeds]. The principal-component probe was not repeated, the X-Y probes of section 5.3 are queued for seed 1, and test.other has no probes.
6. **Efficiency: Vocos is the cheapest and the most streaming-friendly (single timing session per device; a property of the architecture, timed on the main systems only).** Vocos has a median RTF of 0.0008 on an L40S GPU and 0.027 on one CPU thread, with a measured lookahead of 196 ms. HiFi-GAN is about 7 times slower on one CPU thread (0.18). DDSP has the fewest parameters (7.0 M against 14.9 M and 15.2 M) but a lookahead of about 0.9 s and a phase state that spans the whole utterance. The SPARC feature extractor dominates the cost of the full pipeline on every device [eff]. Timing does not depend on the training seed or the test split in the way the quality metrics do, and it was not repeated for seed 1 or test.other.
7. **T2 and T3 mostly expose the limits of the similarity metric and do not change the ranking (strong evidence; holds in both seeds and on test.other).** Sim same under T2 correlates 0.80 to 0.86 with the ECAPA cosine between the ground truth of the target and of its reference (seed 0). Under T3 HiFi-GAN and Vocos exceed the ground-truth Sim spk ceiling (0.787 and 0.783 against 0.749 for the ground truth; DDSP, at 0.731, stays below it), because the speaker-mean vector moves the output to the centre of the speaker's embedding cloud [failures, table]. The same holds in seed 1 (0.790, 0.785 and 0.735) and on test.other (0.753, 0.750 and 0.701 against a ceiling of 0.715) [seeds, other].
8. **Ablations at 100k steps (single runs, not repeated across seeds or on test.other; supported differences listed, section 8).** (a) *Training length*: the main runs gained 0.22 (HiFi-GAN), 0.31 (Vocos) and 0.07 (DDSP) UTMOS from 100k to 400k (seed 0; 0.22, 0.33 and 0.06 with the mean of both seeds), and DDSP's F0, voicing and loudness errors were worse at 400k; at 100k the UTMOS order of DDSP and Vocos is reversed (DDSP ahead by 0.18, CI only), so the 100k results cannot be used to rank the vocoders at 400k. (b) *Vocos option A against B*: B is better on UTMOS (+0.12) and speaker similarity (+0.014); A is only 0.6 percent lower in mel L1, so the pilot choice of B holds. (c) *`p_cross` = 0*: improves the self-conditioned T1 result (mel L1 -2.8 percent, Sim same +0.028) and worsens the T2 mel L1 (+3.3 percent) and PESQ (-0.026), but under T2 UTMOS, WER and speaker similarity do not change in a supported way (under T1 UTMOS rises by a supported +0.019). (d) *Native SPARC recipe*: higher UTMOS (+0.11 to +0.14) but worse mel L1 (+8.8 percent), PESQ (-0.14) and WER (+0.4 percentage points); the recipe changes several things at once. (e) *MSD instead of MRD*: lowers Vocos and DDSP on speaker similarity, F0 and voicing, raises HiFi-GAN UTMOS (+0.07) and leaves the ranking unchanged (the HiFi-GAN lead grows). (f) *Size-matched*: at about 6 M the HiFi-GAN lead over DDSP is not supported on UTMOS (+0.08, CI only) and Sim same is equal for all three, while the lead over Vocos (+0.36 UTMOS) is unchanged; doubling DDSP does not help it (UTMOS -0.05, F0 +4.4 cents). The seed-to-seed differences of the 400k runs (UTMOS up to 0.042, WER up to 0.08 percentage points, section 4.1) are of the same order as the smaller ablation effects (0.05 to 0.1 UTMOS), which were not repeated.
9. **Compute and training length (measurements, two seeds at 400k and one 800k run each of HiFi-GAN and Vocos).** Vocos needs 11.5 times fewer generator FLOPs per second of audio than HiFi-GAN (2.63 G against 30.3 G) with about equal parameters (13.78 M and 14.11 M), runs 7 to 10 times faster at inference and costs 1.5 times less to train (about 23 against 34 GPU-hours for 400k steps), but the end-to-end advantage in the speech-to-speech pipeline is about 10 percent (RTF 0.0471 against 0.0525 on an L40S), because the SPARC feature extractor dominates [conv, eff]. Whether 400k steps are enough differs by vocoder: DDSP has converged (the projected gain from 800k steps is below its seed difference, and its prosody errors are worse at 400k than at 100k), Vocos is about 2 percent above its fitted asymptote in dev mel L1, and HiFi-GAN is 3 to 8 percent above it, estimated from only two budgets with one assumed curve shape [conv]. The 800k runs (section 6.1.3) confirm the projection for HiFi-GAN: its dev mel L1 fell by 0.0075 (projected 0.0061 to 0.0095; 11 times its seed difference), and every test.clean metric except WER, CER and Sim spk improved robustly against both 400k seeds (UTMOS +0.042 and +0.046 to 4.207, the ground-truth level; F0 RMSE -2.0 and -2.1 cents; loudness RMSE -0.10 and -0.11 dB) [long, conv]. Vocos gained less than projected and mostly within its seed spread (dev mel L1 -0.0021 against the seed mean; UTMOS +0.072 and +0.030 against the two seeds, the second below its seed difference of 0.042; robust only for Sim same, EMA consistency and the 4 to 12 kHz mel bands), and is within about 1 percent of its fitted asymptote [long, conv]. The HiFi-GAN lead holds at equal steps (UTMOS +0.239 at 800k) and at about equal training compute (Vocos at 800k, 45.9 GPU-hours, against HiFi-GAN at 400k, about 34: UTMOS +0.196 and +0.192 for HiFi-GAN, F0 RMSE -9.5 and -9.4 cents) [long]. DDSP was not run for 800k steps, each 800k arm is one run, and only test.clean was evaluated at 800k.

## 2. Setup in brief

**Task and data.** The input is a 15-channel feature track at 50 Hz (12 EMA channels from the per-utterance MNGU0 refit of the linear inversion head, pitch, loudness, periodicity) and a 64-dimensional speaker vector, and the output is 24 kHz audio of length 480 times the number of frames [p1 sections 0 and 11; eval section 1]. Training data are the three filtered LibriTTS-R training subsets (`parler-tts/libritts_r_filtered`; 542 h, 2,299 speakers in the plan [p1 section 9]). The packed training split holds 339,007 utterances and 97.2 million frames [status, entry 2026-10-05 15:50]. All splits are speaker-disjoint [p1 section 9]. The test set is test.clean after the 171 not-encodable rows across all splits were removed: 4,687 utterances, 39 speakers [eval section 1; table]. The second split, test.other, has 4,709 utterances from 33 speakers (section 3.1) [other].

**Training.** One shared recipe for all three vocoders: multi-period plus multi-resolution discriminators, hinge loss, mel L1 loss at 24 kHz (weight 45), a 10,000-step mel-only warm-up, AdamW with learning rate 2e-4 and cosine decay, global batch 16 segments of 64 frames, 400,000 generator steps, fp32, seed 0, and speaker references taken from a different utterance of the same speaker with probability `p_cross` = 0.5 [p1 section 8; run configuration `runs/main/vocos/hydra/*/.hydra/config.yaml`: seed 0, lr 0.0002, `max_g_steps` 400000, `p_cross` 0.5, precision 32-true, batch size 16, crop 64]. The second runs (experiment `main_seed1`, section 4.1) use the same recipe with seed 1. Each speaker FFN is trained from scratch. The Vocos variant is option B (hop 240); it won the 25k-step A/B pilot on dev mel L1 (0.5382 against 0.5406, within the 2 percent rule) [status, entry 2026-10-06 01:55]. Parameter counts (generator plus speaker FFN): HiFi-GAN 15.22 M, DDSP 7.04 M, Vocos 14.90 M [eff]. The HiFi-GAN generator has 14.11 M, DDSP 5.92 M, Vocos 13.78 M [eff].

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

- Seed 0 only: the numbers in this section are for the first training run of each arm. Section 4.1 gives the second seed and the seed-to-seed differences, and section 3.1 gives test.other.
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

### 3.1 Second split: test.other

Question: do the findings of section 3 hold on a second evaluation split? test.other is the harder split by ground-truth word error rate (3.37 percent against 1.89 percent on test.clean) [other, table].

Setup. The same checkpoints (seed 0, step 400,000), the same pipeline, systems and conditions (T1, T2, T3) as in section 3 [other; `meta.json`, `fork_commit` `85c396283b41ed3d851b0ba2109fc7a837e4e0ee`]. The split has 4,709 utterances from 33 speakers (17 female, 16 male), and every utterance has a T2 reference, so T2 and T3 use all 33 speakers (T2 same chapter: 458 utterances, 5 speakers; different chapter: 4,251 utterances, 28 speakers) [other; `phase3/L_testother.md`]. Only the main systems were run: seed 1, the ablations, the probes and the efficiency measurements were not evaluated on test.other. The caveats of section 3 apply (one training run per arm in this subsection, en+ at 16 kHz, undocumented en+ training splits, which may include test.other). Four utterances are too short for the feature re-extraction, so the prosody and EMA cells use 4,705 utterances (the HiFi-GAN F0 cells 4,703), the PESQ cells of the vocoders use 4,705, and the en+ reference produced no audio for four utterances (its WER cells use 4,705); the `n` column of `table.md` still lists 4,709 for these cells [other, `meta.json`].

**Table 3.1. T1 on test.other**, mean (CI95, speaker-cluster bootstrap). n = 4,709 utterances, 33 speakers; WER and CER are corpus level, in percent; prosody cells use 4,703 to 4,705 utterances. Source: [other].

| system | UTMOS | PESQ | WER (%) | CER (%) | Sim same | Sim spk | F0 RMSE (cents) | VDE | Loudness RMSE (dB) | EMA r |
|---|---|---|---|---|---|---|---|---|---|---|
| ground truth | 4.035 (3.978, 4.092) | n/a | 3.37 (2.97, 3.76) | 1.28 (1.11, 1.46) | n/a | 0.715 (0.689, 0.741) | 5.8 (floor) | 0.000 | 0.00 | 0.997 (0.996, 0.997) |
| Vocos mel copy synthesis | 3.786 (3.733, 3.839) | 3.523 (3.458, 3.596) | 3.40 (2.98, 3.84) | 1.31 (1.12, 1.51) | 0.977 (0.974, 0.980) | 0.710 (0.685, 0.737) | 18.8 (17.5, 20.0) | 0.038 (0.034, 0.042) | 1.60 (1.52, 1.69) | 0.977 (0.975, 0.978) |
| SPARC en+ (16 kHz) | 3.899 (3.852, 3.947) | 1.841 (1.794, 1.889) | 5.48 (4.72, 6.25) | 2.52 (2.10, 2.97) | 0.804 (0.790, 0.818) | 0.673 (0.648, 0.698) | 27.9 (25.8, 29.8) | 0.070 (0.064, 0.077) | 2.54 (2.43, 2.66) | 0.959 (0.957, 0.961) |
| HiFi-GAN | 4.030 (3.985, 4.076) | 2.106 (2.058, 2.158) | 5.03 (4.37, 5.68) | 2.22 (1.89, 2.54) | 0.833 (0.820, 0.845) | 0.693 (0.668, 0.719) | 23.6 (21.8, 25.3) | 0.042 (0.037, 0.047) | 1.47 (1.41, 1.54) | 0.966 (0.964, 0.967) |
| DDSP | 3.696 (3.625, 3.768) | 1.955 (1.904, 2.012) | 5.10 (4.42, 5.79) | 2.29 (1.94, 2.64) | 0.781 (0.763, 0.800) | 0.662 (0.637, 0.690) | 17.5 (16.9, 18.2) | 0.067 (0.061, 0.074) | 2.64 (2.49, 2.81) | 0.953 (0.951, 0.956) |
| Vocos | 3.797 (3.746, 3.843) | 1.949 (1.910, 1.992) | 5.00 (4.31, 5.69) | 2.20 (1.87, 2.55) | 0.820 (0.808, 0.832) | 0.686 (0.662, 0.711) | 34.8 (31.5, 37.9) | 0.043 (0.037, 0.049) | 1.75 (1.66, 1.84) | 0.963 (0.962, 0.965) |


The ground-truth Sim spk ceiling on test.other is 0.715. Under T3 HiFi-GAN (0.753) and Vocos (0.750) are above it and DDSP (0.701) is below it, as on test.clean [other].

**Table 3.2. Paired differences of the vocoders on test.other, T1** (a - b, CI95, agreeing speakers; *CI only* and *n.s.* as in section 4; n = 4,709 utterances, 33 speakers, prosody rows 4,703 to 4,705). Source: [other] (`paired.csv`; the file stores the three pairs in the direction shown, so no sign is reversed here).

| metric | HiFi-GAN - DDSP | HiFi-GAN - Vocos | DDSP - Vocos |
|---|---|---|---|
| UTMOS | +0.334 (+0.287, +0.380), 100 % | +0.233 (+0.193, +0.279), 100 % | -0.100 (-0.158, -0.033), 64 %, CI only |
| PESQ | +0.150 (+0.132, +0.170), 100 % | +0.157 (+0.143, +0.172), 100 % | +0.006 (-0.014, +0.027), 64 %, n.s. |
| WER (pp) | -0.07 (-0.24, +0.09), 55 %, n.s. | +0.03 (-0.13, +0.19), 55 %, n.s. | +0.10 (-0.05, +0.25), 61 %, n.s. |
| CER (pp) | -0.08 (-0.17, +0.01), 61 %, n.s. | +0.01 (-0.06, +0.08), 58 %, n.s. | +0.09 (+0.02, +0.16), 73 %, CI only |
| Sim same | +0.0515 (+0.0431, +0.0599), 100 % | +0.0123 (+0.0102, +0.0144), 97 % | -0.0391 (-0.0480, -0.0300), 100 % |
| Sim spk | +0.0310 (+0.0226, +0.0398), 94 % | +0.0067 (+0.0035, +0.0100), 82 % | -0.0242 (-0.0321, -0.0164), 97 % |
| F0 RMSE (cents) | +6.08 (+4.10, +7.94), 100 % | -11.20 (-12.69, -9.67), 100 % | -17.28 (-20.49, -13.94), 100 % |
| VDE | -0.0255 (-0.0309, -0.0211), 100 % | -0.0012 (-0.0022, -0.0001), 70 %, CI only | +0.0243 (+0.0194, +0.0306), 100 % |
| Loudness RMSE (dB) | -1.18 (-1.28, -1.08), 100 % | -0.28 (-0.32, -0.24), 100 % | +0.90 (+0.81, +0.99), 100 % |
| EMA r | +0.0124 (+0.0109, +0.0139), 100 % | +0.0024 (+0.0021, +0.0027), 100 % | -0.0100 (-0.0115, -0.0085), 100 % |

Under T2 and T3 the signs are the same (UTMOS: HiFi-GAN - DDSP +0.332 and +0.329, HiFi-GAN - Vocos +0.243 and +0.253, DDSP - Vocos -0.090 and -0.076, all supported for the first two and CI only for the third); Sim same of HiFi-GAN against Vocos is not supported (+0.002 and +0.000), and no WER pair is supported in any condition [other].

**Comparison with the references (paired, T1)** [other]. HiFi-GAN UTMOS is above the Vocos mel copy synthesis by 0.244 (CI 0.224 to 0.265, 100 percent of speakers agree) and above en+ by 0.132 (CI 0.114 to 0.153, 100 percent); DDSP and Vocos are below en+ by 0.202 (88 percent) and 0.102 (94 percent). The corpus WER of the vocoders (5.00 to 5.10 percent) is lower than that of en+ (5.48 percent) by 0.45 (HiFi-GAN, supported, 82 percent), 0.48 (Vocos, supported, 85 percent) and 0.38 percentage points (DDSP, CI only, 70 percent). The Vocos mel copy synthesis again keeps the ground-truth WER (3.40 against 3.37 percent), and every articulatory vocoder is 1.6 to 1.7 percentage points above it.

**What replicates and what differs.**

- **Replicates.** The ranking and the sign of every supported difference of section 3: HiFi-GAN is ahead on UTMOS (+0.334 over DDSP, +0.233 over Vocos), PESQ, Sim same and Sim spk; DDSP has the lowest F0 error (17.5 cents) and the highest VDE and loudness error; Vocos has the highest F0 error (34.8 cents; 11.2 cents behind HiFi-GAN and 17.3 behind DDSP, against 11.6 and 16.8 on test.clean); no WER pair is supported; the DDSP-Vocos UTMOS difference is not supported (-0.100, CI only, 64 percent, against -0.065, not supported). The mean speaker-level UTMOS drop against the ground truth is 0.32 (DDSP), 0.27 (Vocos) and 0.01 (HiFi-GAN), against 0.39, 0.27 and 0.04 on test.clean (mean over speakers of the per-speaker difference) [other, per_speaker.csv; seeds, per_speaker.csv].
- **Differs: the gap to the ground truth in WER.** The vocoders lose 1.63 to 1.73 percentage points of WER against the ground truth on test.other, against 0.56 to 0.66 on test.clean, so the loss at the feature bottleneck is 2.5 to 3 times larger on the harder split, while the mel copy synthesis does not lose any (the equality of the two WER values is the only evidence for the bottleneck reading, as in section 3).
- **Differs: UTMOS against the ground truth.** HiFi-GAN is at the ground-truth level (4.030 against 4.035; on test.clean 4.165 against 4.208), and DDSP and Vocos are closer to it than on test.clean (-0.339 and -0.238, against -0.376 and -0.312), because the ground truth itself scores lower on test.other (4.035 against 4.208).
- **Differs: small differences.** HiFi-GAN against Vocos on VDE is -0.0012 (CI only, 70 percent) on test.other, against -0.0028 (supported) on test.clean, on mel L1 it is -0.0013 (-0.0032, +0.0008; 64 percent, not supported) against -0.0063 (supported), and the DDSP error pattern is less concentrated: the largest speaker-level UTMOS drop of DDSP is 0.56 (against 1.00), and the drop is spread over more speakers [other, per_speaker.csv]. Whether the male-speaker deficit replicates was not tested, because the speaker sex is not in the copied tables.

## 4. Paired comparisons of the vocoders

Method [paired]. Each row compares two systems on the same utterances. The difference is `a - b`. The interval is the speaker-cluster bootstrap CI95 (10,000 resamples, seed 0). A difference is called *supported* when the CI95 excludes 0 and at least 75 percent of the speakers have a mean difference with the sign of the overall difference. *CI only* marks a CI that excludes 0 with fewer than 75 percent agreeing speakers. No correction for multiple comparisons was applied; about 180 comparisons are listed in `paired.md`, and borderline results (CI edge within 0.001 of zero) should be read with that in mind. The criterion is a convention of this report and was not pre-registered. Because each arm in this section is one training run (seed 0), a supported difference shows that these two trained models differ on this test set; it does not show that the architecture is the cause. Section 4.1 repeats the comparison with a second seed.

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

### 4.1 Run-to-run variation (second seeds)

Design [seeds; `phase3/K_seeds.md`]. The experiment `main_seed1` repeats the main run of each vocoder with `seed: 1` and everything else unchanged (400,000 generator steps, the recipe of section 2, `p_cross` = 0.5). The seed sets the model initialization, the data order, the crop positions, the gain draws and the speaker-reference draws; the validation subsets, the feature cache and the evaluation seeds do not depend on it (seed audit in `K_seeds.md`). The three seed-1 checkpoints at step 400,000 were evaluated with the pipeline of section 2 on test.clean (T1, T2, T3; 4,687 utterances, 39 speakers). The seed-1 controllability probes are analysed in section 5.2. A seed against the other is compared with the method of section 4 (difference = seed 1 minus seed 0, same utterances, speaker-cluster CI95).

**Table 4.1. Seed 1 minus seed 0 for each vocoder, T1.** Each cell: seed 0 value, seed 1 value: difference (CI95), agreeing speakers. n = 4,687 utterances, 39 speakers (prosody rows 4,686). Source: [seeds].

| metric | HiFi-GAN | DDSP | Vocos |
|---|---|---|---|
| UTMOS | 4.165, 4.161: -0.004 (-0.010, +0.002), 59 %, n.s. | 3.832, 3.821: -0.011 (-0.026, +0.007), 62 %, n.s. | 3.896, 3.939: +0.042 (+0.030, +0.053), 79 % |
| PESQ | 2.091, 2.091: +0.000 (-0.006, +0.006), 59 %, n.s. | 1.913, 1.900: -0.013 (-0.025, +0.000), 49 %, n.s. | 1.903, 1.936: +0.033 (+0.026, +0.040), 87 % |
| WER (percent) | 2.45, 2.49: +0.03 (-0.04, +0.12), 46 %, n.s. | 2.52, 2.48: -0.04 (-0.10, +0.01), 51 %, n.s. | 2.55, 2.47: -0.08 (-0.15, -0.01), 49 %, CI only |
| Sim same | 0.8569, 0.8583: +0.0014 (+0.0006, +0.0023), 69 %, CI only | 0.8024, 0.8012: -0.0012 (-0.0032, +0.0009), 62 %, n.s. | 0.8438, 0.8466: +0.0028 (+0.0016, +0.0039), 77 % |
| Sim spk | 0.7307, 0.7318: +0.0011 (-0.0004, +0.0026), 58 %, n.s. | 0.6904, 0.6943: +0.0038 (+0.0015, +0.0062), 74 %, CI only | 0.7224, 0.7242: +0.0018 (+0.0000, +0.0035), 63 %, CI only |
| mel L1 | 0.4878, 0.4879: +0.0001 (-0.0002, +0.0004), 59 %, n.s. | 0.5146, 0.5191: +0.0045 (+0.0037, +0.0053), 95 % | 0.4940, 0.4913: -0.0028 (-0.0031, -0.0024), 92 % |
| F0 RMSE (cents) | 23.4, 23.5: +0.1 (-0.2, +0.3), 44 %, n.s. | 18.2, 20.9: +2.7 (+2.0, +3.3), 92 % | 35.0, 33.5: -1.5 (-2.0, -1.0), 77 % |
| VDE | 0.0388, 0.0393: +0.0005 (-0.0001, +0.0011), 59 %, n.s. | 0.0662, 0.0679: +0.0018 (+0.0013, +0.0023), 87 % | 0.0415, 0.0416: +0.0001 (-0.0006, +0.0009), 54 %, n.s. |
| loudness RMSE (dB) | 1.49, 1.49: +0.00 (-0.01, +0.02), 64 %, n.s. | 2.80, 2.89: +0.09 (+0.07, +0.11), 90 % | 1.83, 1.78: -0.05 (-0.06, -0.04), 92 % |
| EMA r | 0.9710, 0.9707: -0.0002 (-0.0004, -0.0001), 72 %, CI only | 0.9601, 0.9598: -0.0003 (-0.0005, -0.0000), 74 %, CI only | 0.9685, 0.9689: +0.0004 (+0.0002, +0.0005), 77 % |

Reading of Table 4.1:

- **HiFi-GAN reproduces closely.** No difference is supported; the largest are UTMOS -0.004 (not supported), Sim same +0.0014 (CI only) and F0 RMSE +0.1 cents (not supported).
- **DDSP reproduces on quality and not on prosody.** UTMOS (-0.011), PESQ and Sim same do not change in a supported way, but F0 RMSE rises by 2.7 cents (92 percent of speakers), VDE by 0.0018, loudness RMSE by 0.09 dB and mel L1 by 0.0045, and the share of frames within 50 cents of the ground-truth F0 falls by 0.015 (92 percent), so all of its prosody errors are worse in seed 1. This agrees with section 8.2, where its prosody errors were also worse after longer training.
- **Vocos is better in seed 1 on several metrics.** UTMOS +0.042 (79 percent), PESQ +0.033 (87 percent), mel L1 -0.0028, F0 RMSE -1.5 cents (77 percent) and loudness RMSE -0.05 dB are supported; the seed-to-seed UTMOS difference of Vocos is the largest of the three vocoders.
- **Corpus WER does not move.** The six values lie between 2.45 and 2.55 percent (seed 1: 2.49, 2.48 and 2.47 percent for HiFi-GAN, DDSP and Vocos); the largest seed difference is 0.08 percentage points (Vocos, CI only).
- **Scale.** The largest seed-to-seed differences are 0.042 UTMOS, 0.003 Sim same, 0.004 Sim spk (DDSP, CI only), 2.7 cents F0 RMSE, 0.0018 VDE and 0.09 dB loudness RMSE. A supported seed difference is a real difference between two trained models on this test set, of the same kind as a difference between two architectures; it is the reference scale for the rule below. The mean of the two seeds is 4.163, 3.83 and 3.92 UTMOS for HiFi-GAN, DDSP and Vocos [seeds].

**Which vocoder differences are robust.** A difference is robust only if it has the same sign and is supported in both seeds and its smaller absolute value over the two seeds exceeds the larger of the two vocoders' seed-to-seed differences of Table 4.1 (a point estimate, so the rule is crude). Table 4.2 applies the rule under T1. Lower values are better for WER, mel L1, F0 RMSE, VDE and loudness RMSE.

**Table 4.2. Vocoder differences in each seed, T1** (a - b, CI95, agreeing speakers; seed 0 is the main run of section 4, seed 1 the second run; n = 4,687 utterances, 39 speakers, prosody rows 4,686). Source: [seeds] (`paired.csv`, which stores DDSP - Vocos as Vocos - DDSP; the sign is reversed here; the rule was applied with the script reproduced in `phase3/O_results_followups.md`).

| comparison | metric | seed 0 | seed 1 | larger seed-to-seed difference of the two vocoders | verdict |
|---|---|---|---|---|---|
| HiFi-GAN - DDSP | UTMOS | +0.333 (+0.279, +0.398), 100 % | +0.340 (+0.295, +0.392), 100 % | 0.011 (DDSP) | robust |
| HiFi-GAN - DDSP | PESQ | +0.177 (+0.163, +0.191), 100 % | +0.190 (+0.174, +0.205), 100 % | 0.013 (DDSP) | robust |
| HiFi-GAN - DDSP | WER (percent) | -0.07 (-0.15, +0.01), 56 %, n.s. | +0.01 (-0.09, +0.12), 46 %, n.s. | 0.04 (DDSP) | not robust (seed 0 not supported, seed 1 not supported, sign changes) |
| HiFi-GAN - DDSP | Sim same | +0.0544 (+0.0483, +0.0612), 100 % | +0.0571 (+0.0511, +0.0636), 100 % | 0.0014 (HiFi-GAN) | robust |
| HiFi-GAN - DDSP | Sim spk | +0.0403 (+0.0326, +0.0486), 100 % | +0.0375 (+0.0290, +0.0465), 97 % | 0.0038 (DDSP) | robust |
| HiFi-GAN - DDSP | mel L1 | -0.0268 (-0.0282, -0.0251), 97 % | -0.0312 (-0.0331, -0.0291), 100 % | 0.0045 (DDSP) | robust |
| HiFi-GAN - DDSP | F0 RMSE (cents) | +5.2 (+4.1, +6.5), 100 % | +2.6 (+1.2, +4.3), 69 %, CI only | 2.7 (DDSP) | not robust (seed 1 not supported) |
| HiFi-GAN - DDSP | VDE | -0.0274 (-0.0304, -0.0244), 100 % | -0.0287 (-0.0319, -0.0255), 100 % | 0.0018 (DDSP) | robust |
| HiFi-GAN - DDSP | loudness RMSE (dB) | -1.31 (-1.43, -1.21), 100 % | -1.40 (-1.51, -1.30), 100 % | 0.09 (DDSP) | robust |
| HiFi-GAN - DDSP | EMA r | +0.0108 (+0.0102, +0.0115), 100 % | +0.0109 (+0.0104, +0.0115), 97 % | 0.0003 (DDSP) | robust |
| HiFi-GAN - Vocos | UTMOS | +0.268 (+0.239, +0.296), 97 % | +0.222 (+0.198, +0.245), 97 % | 0.042 (Vocos) | robust |
| HiFi-GAN - Vocos | PESQ | +0.188 (+0.168, +0.206), 100 % | +0.155 (+0.139, +0.170), 97 % | 0.033 (Vocos) | robust |
| HiFi-GAN - Vocos | WER (percent) | -0.10 (-0.16, -0.03), 56 %, CI only | +0.02 (-0.09, +0.13), 44 %, n.s. | 0.08 (Vocos) | not robust (seed 0 not supported, seed 1 not supported, sign changes) |
| HiFi-GAN - Vocos | Sim same | +0.0130 (+0.0108, +0.0152), 97 % | +0.0117 (+0.0102, +0.0131), 97 % | 0.0028 (Vocos) | robust |
| HiFi-GAN - Vocos | Sim spk | +0.0083 (+0.0058, +0.0107), 76 % | +0.0076 (+0.0047, +0.0101), 84 % | 0.0018 (Vocos) | robust |
| HiFi-GAN - Vocos | mel L1 | -0.0063 (-0.0078, -0.0047), 92 % | -0.0034 (-0.0047, -0.0020), 74 %, CI only | 0.0028 (Vocos) | not robust (seed 1 not supported) |
| HiFi-GAN - Vocos | F0 RMSE (cents) | -11.6 (-13.2, -10.3), 97 % | -10.0 (-12.0, -8.3), 100 % | 1.5 (Vocos) | robust |
| HiFi-GAN - Vocos | VDE | -0.0028 (-0.0036, -0.0020), 87 % | -0.0023 (-0.0035, -0.0014), 82 % | 0.0005 (HiFi-GAN) | robust |
| HiFi-GAN - Vocos | loudness RMSE (dB) | -0.34 (-0.36, -0.32), 97 % | -0.28 (-0.31, -0.26), 97 % | 0.05 (Vocos) | robust |
| HiFi-GAN - Vocos | EMA r | +0.0024 (+0.0021, +0.0027), 95 % | +0.0018 (+0.0016, +0.0021), 95 % | 0.0004 (Vocos) | robust |
| DDSP - Vocos | UTMOS | -0.065 (-0.146, +0.002), 64 %, n.s. | -0.118 (-0.177, -0.067), 69 %, CI only | 0.042 (Vocos) | not robust (seed 0 not supported, seed 1 not supported) |
| DDSP - Vocos | PESQ | +0.011 (-0.011, +0.031), 56 %, n.s. | -0.036 (-0.050, -0.020), 72 %, CI only | 0.033 (Vocos) | not robust (seed 0 not supported, seed 1 not supported, sign changes) |
| DDSP - Vocos | WER (percent) | -0.03 (-0.09, +0.04), 44 %, n.s. | +0.01 (-0.06, +0.08), 36 %, n.s. | 0.08 (Vocos) | not robust (seed 0 not supported, seed 1 not supported, sign changes) |
| DDSP - Vocos | Sim same | -0.0414 (-0.0488, -0.0350), 100 % | -0.0454 (-0.0517, -0.0396), 100 % | 0.0028 (Vocos) | robust |
| DDSP - Vocos | Sim spk | -0.0320 (-0.0403, -0.0246), 100 % | -0.0299 (-0.0374, -0.0231), 97 % | 0.0038 (DDSP) | robust |
| DDSP - Vocos | mel L1 | +0.0205 (+0.0192, +0.0220), 100 % | +0.0278 (+0.0261, +0.0296), 100 % | 0.0045 (DDSP) | robust |
| DDSP - Vocos | F0 RMSE (cents) | -16.8 (-19.5, -14.6), 100 % | -12.6 (-16.1, -9.7), 100 % | 2.7 (DDSP) | robust |
| DDSP - Vocos | VDE | +0.0246 (+0.0214, +0.0279), 100 % | +0.0263 (+0.0233, +0.0294), 100 % | 0.0018 (DDSP) | robust |
| DDSP - Vocos | loudness RMSE (dB) | +0.97 (+0.87, +1.09), 100 % | +1.12 (+1.02, +1.23), 100 % | 0.09 (DDSP) | robust |
| DDSP - Vocos | EMA r | -0.0084 (-0.0092, -0.0077), 100 % | -0.0091 (-0.0097, -0.0085), 97 % | 0.0004 (Vocos) | robust |

Result: 23 of the 30 comparisons of Table 4.2 are robust.

- **Robust: HiFi-GAN is ahead of DDSP and of Vocos** on UTMOS, PESQ, Sim same, Sim spk, VDE, loudness RMSE and EMA r, and ahead of Vocos on F0 RMSE (-11.6 and -10.0 cents), and ahead of DDSP on mel L1 and MCD. The margins of HiFi-GAN over Vocos on Sim spk (+0.008 and +0.008), VDE (-0.0028 and -0.0023) and EMA r (+0.0024 and +0.0018) are small, although each is more than four times the larger seed difference. Vocos has a lower MCD than HiFi-GAN in both seeds (0.025 and 0.030 dB, robust). The UTMOS lead of HiFi-GAN (+0.33 and +0.34 over DDSP, +0.27 and +0.22 over Vocos) is 5 or more times the largest seed difference.
- **Robust: the DDSP-Vocos pattern.** Vocos is ahead of DDSP on Sim same, Sim spk, mel L1, MCD, VDE, loudness RMSE and EMA r, and DDSP has the lower F0 RMSE (-16.8 and -12.6 cents); periodicity MAE and the share of frames within 50 cents of the ground-truth F0 are robust for all three pairs and are not in the table [seeds].
- **Not robust: every WER and CER difference.** No pair is supported in seed 1 under any condition, and none is supported in both seeds [seeds].
- **Not robust: the DDSP-Vocos UTMOS order.** Vocos is ahead in both seeds (0.065, not supported; 0.118, CI only, 69 percent) and on test.other (0.100, CI only, 64 percent), always with the same sign and never supported. The earlier preview in `STATUS.md` called the seed-1 difference supported; in the final `paired.csv` the agreement is 69 percent, so it is CI only. The PESQ difference between DDSP and Vocos changes sign (+0.011 and -0.036).
- **Not robust: the HiFi-GAN-DDSP F0 order.** DDSP is better by 5.2 cents in seed 0 and by 2.6 cents in seed 1 (CI only, 69 percent), because the DDSP F0 error rose by 2.7 cents between the seeds. The HiFi-GAN-Vocos mel L1 difference is also not robust (seed 1: CI only, 74 percent).
- **Cross-seed check.** For 22 of the 23 robust rows of Table 4.2, all four combinations of one run of a vocoder against one run of the other are supported with the same sign; the exception is HiFi-GAN - Vocos VDE (three of four) [seeds].

Other conditions and split [seeds, other]. Under T2 and T3 the same rows are robust, except that the small HiFi-GAN - Vocos differences in Sim same, VDE and mel L1 are not robust under T2 or T3 (Sim spk not under T3), and HiFi-GAN - DDSP F0 RMSE is not robust under T2 but passes under T3 (+5.4 and +3.2 cents; 76 percent in seed 1). On test.other (seed 0 only) 22 of the 23 robust differences are supported; the exception is HiFi-GAN - Vocos VDE (CI only, 70 percent). The HiFi-GAN-DDSP F0 difference, which is not robust, is supported on test.other (+6.1 cents, 100 percent), as in seed 0 on test.clean.

Consequence for the claims of this document: the HiFi-GAN lead on quality and speaker similarity, the Vocos F0 error, the DDSP voicing and loudness errors, and the absence of a WER difference are robust; the DDSP-Vocos UTMOS order and the HiFi-GAN-DDSP F0 order are not (section 6.1.3 adds single-run evidence on the first: Vocos at 800k is ahead of DDSP at 400k). The ablations of section 8 and the 800k runs of section 6.1.3 remain single runs.

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

**Seed 1 (follow-up, 2026-10-20).** The same probes were run on the three seed-1 runs: the same 50 utterances, the same 1,550 edits per system and the same configuration (`probes_meta.json` differs only in the checkpoint) [probes-seeds]. The analysis script reproduces every seed-0 cell above exactly. Vocoder differences are tested with the robustness rule of section 4.1, without the speaker-agreement criterion, because the probe statistics are medians over utterances.

| quantity [probes-seeds] | HiFi-GAN seed 0 / seed 1 | DDSP seed 0 / seed 1 | Vocos seed 0 / seed 1 |
|---|---|---|---|
| largest median F0 error over the four edits (cents) | 0.0 / 0.0 | 0.0 / 0.0 | 10.0 / 10.0 |
| loudness change for x0.5 and x2 (dB) | -5.89, +5.78 / -5.90, +5.78 | -6.00, +5.79 / -5.99, +5.80 | -6.01, +5.87 / -5.97, +5.82 |
| voiced fraction after setting periodicity to 0 (before: 0.56, 0.61, 0.56 / 0.56, 0.61, 0.55) | 0.17 / 0.16 | 0.28 / 0.33 | 0.08 / 0.08 |
| single EMA channel: gain (mean over 12 channels) | 0.117 / 0.120 | 0.095 / 0.102 | 0.117 / 0.115 |
| single EMA channel: leakage (std) | 0.244 / 0.241 | 0.243 / 0.248 | 0.239 / 0.235 |
| natural variation: centred slope / regression diagonal | 1.00, 0.98 / 1.00, 0.98 | 0.96, 0.93 / 0.96, 0.93 | 1.00, 0.97 / 1.00, 0.98 |

- **The findings above replicate in seed 1.**
  - F0 edits are followed to within 10 cents and loudness edits to within 0.24 dB.
  - Periodicity = 0 removes the most voicing in Vocos and the least in DDSP.
  - A single EMA channel is reproduced to a mean gain of 0.10 to 0.12 against a natural-variation slope near 1.
  - Tongue body X has a negative gain in all six runs (-0.12 to -0.17).
  - Every channel mean of the natural-variation measures changes by at most 0.007 between the seeds [probes-seeds].
- **Seed differences are small.** The largest supported change is the voiced fraction of DDSP after periodicity = 0 (+0.034, 72 percent of utterances, from 0.28 to 0.33), in line with its higher VDE in seed 1 (Table 4.1). The mean EMA gain moves by at most 0.004 per vocoder.
- **Robust vocoder differences are small.**
  - HiFi-GAN and Vocos follow single-channel EMA edits slightly more than DDSP: HiFi-GAN minus DDSP +0.019 in both seeds, Vocos minus DDSP +0.017 and +0.013. HiFi-GAN against Vocos is not robust.
  - Per channel, 10 of the 36 channel-pair differences are robust, nine of them with DDSP lower by 0.013 to 0.065 (largest for tongue body Y).
  - Vocos removes more voicing at periodicity = 0 than DDSP (-0.16 and -0.19) and HiFi-GAN (-0.06 and -0.04).
  - Vocos follows F0 edits 5 to 8 cents less exactly than the other two, below the 10 to 20 cent resolution of the pitch estimate.
  - Vocos follows loudness edits 0.05 to 0.07 dB more exactly than HiFi-GAN.
  - Vocos leaks 0.006 to 0.010 std less than both others.
  - The HiFi-GAN-DDSP F0-control difference passes the rule numerically (2.49 and 0.04 cents) but lies far below the pitch resolution and is not counted [probes-seeds].
- **Not repeated for seed 1:** the principal-component probe (6 utterances, CPU). The X-Y probes of section 5.3 are queued for seed 1 (section 5.3).

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

**Seed 1.** The same X-Y probes are queued for the three seed-1 runs (job 10865560, submitted 2026-10-20, tracking issue `nzxyin/Speech-Articulatory-Coding#30`); the comparison with seed 0 will be computed by `phase3/analysis/scripts/xy_probes_seeds.py`. The findings above are for seed 0 only.

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

### 6.1 Compute and training length

Source: [conv] (validation curves of every run from TensorBoard, generator FLOPs, [eff], the tables [seeds] and [abl-table]); the 800k results of section 6.1.3 from [long]. Two training seeds per main arm; the seed difference below is the absolute difference of the two seeds, a crude noise estimate. The FLOP count, timing and training-cost values are measured once on seed 0 checkpoints and shapes; the convergence analysis uses test.clean and the validation set only.

#### 6.1.1 Vocos against HiFi-GAN

| quantity | HiFi-GAN | Vocos | Vocos advantage |
|---|---|---|---|
| generator parameters | 14.11 M | 13.78 M | about equal |
| generator FLOPs per second of audio (torch FlopCounterMode, batch 1, includes front end and speaker FFN) | 30.3 G | 2.63 G | 11.5 times |
| RTF on an L40S, median of 50 utterances | 0.0061 | 0.0008 | 7.6 times |
| RTF on 1 CPU thread | 0.183 | 0.0266 | 6.9 times |
| RTF on 8 CPU threads | 0.0671 | 0.0070 | 9.6 times |
| training step, batch 16 x 1.28 s, fp32, L40S or 6000 Ada | 304 to 313 ms | 205 to 214 ms | 1.5 times |
| GPU-hours for 400k steps (measured, before preemption losses) | about 34 | about 23 | 1.5 times |
| end to end with the SPARC extractor, RTF on an L40S | 0.0525 | 0.0471 | 1.1 times |
| measured lookahead | 205 ms | 196 ms | about equal |

Vocos runs its ConvNeXt backbone at 100 frames per second and produces the waveform with an inverse STFT (n_fft 960, hop 240); HiFi-GAN runs its convolution stacks through transposed-convolution upsampling up to 24,000 samples per second, where most of its FLOPs are. Training cost is dominated by the shared discriminators (MPD and MRD, about 42 M parameters), so the training advantage (1.5 times) is much smaller than the inference advantage (7 to 11 times). In the articulatory pipeline the feature extractor (WavLM-large, CREPE; RTF 0.046 on the GPU, 3.5 on one CPU thread) dominates, so the end-to-end advantage is about 10 percent. The FLOP count of DDSP (0.93 G) excludes the sine evaluations of the oscillator bank, which the counter does not count; its measured CPU RTF (0.036 on one thread, section 6) is the better comparison [conv].

#### 6.1.2 Are 400k steps enough?

Each run uses a cosine schedule that decays to zero at its own end, so the flattening of a validation curve near the end is annealing and not convergence. Only the end points of fully annealed runs of different lengths are comparable: 25k (the Vocos pilots), 100k (the ablation baselines) and 400k (the main runs, two seeds). Dev mel L1 [conv]:

| vocoder | 25k | 100k | 400k (mean of 2 seeds) | seed difference at 400k |
|---|---|---|---|---|
| Vocos (option B) | 0.5382 | 0.5061 | 0.4920 | 0.0028 |
| HiFi-GAN | n/a | 0.5174 | 0.4883 | 0.0007 |
| DDSP | n/a | 0.5233 | 0.5168 | 0.0053 |

Fit `L(B) = L_inf + A B^-alpha`. Vocos has three budgets, each four times the previous, which fix the curve: the improvement shrinks from 0.0320 (25k to 100k) to 0.0141 (100k to 400k), a ratio of 0.44 per factor of four in steps, so alpha = 0.59. HiFi-GAN and DDSP have two budgets, so the table uses alpha from 0.4 to 0.8 around the Vocos value [conv].

| vocoder | alpha | asymptote L_inf | distance at 400k | gain from 800k steps | gain / seed difference |
|---|---|---|---|---|---|
| Vocos | 0.59 (fitted) | 0.481 | 0.011 (2.3 %) | 0.0037 | 1.3 |
| HiFi-GAN | 0.4 to 0.8 | 0.449 to 0.474 | 0.014 to 0.039 (2.9 to 8.0 %) | 0.0061 to 0.0095 | 9 to 14 |
| DDSP | 0.4 to 0.8 | 0.508 to 0.514 | 0.003 to 0.009 (0.6 to 1.7 %) | 0.0014 to 0.0021 | 0.3 to 0.4 |

Inside each 400k run the unannealed curve reaches the annealed 100k value only at step 170k to 175k (HiFi-GAN), 235k to 240k (Vocos) and 235k to 285k (DDSP), which is the usual cost of a longer schedule [conv].

Projected change of the test.clean metrics from 400k to 800k steps (T1), if each metric improves at the same rate per doubling as the Vocos dev mel L1 (0.265 times the 100k-to-400k change); the 100k values are the ablation baselines, the 400k values the mean of the two seeds [conv]:

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

WER changed by at most 0.2 percentage points from 100k to 400k for all three vocoders and is not limited by the training budget [conv].

Reading:

- **DDSP has converged** at 400k for this recipe: the remaining dev mel gain is below the seed difference, UTMOS gained only 0.06 from 100k to 400k, and its F0, voicing and loudness errors grew from 100k to 400k.
- **Vocos is close:** about 2 percent above its fitted asymptote in dev mel L1; doubling to 800k is projected to gain about 0.09 UTMOS (twice the seed difference of 0.042) and 0.016 Sim same.
- **HiFi-GAN is the furthest from convergence** in dev mel L1 (3 to 8 percent above its asymptote) and in prosody, but its UTMOS (4.16) is already close to the ground truth (4.21), so most of its remaining gain would be in the fidelity metrics.
- **Effect on the comparison:** longer training would probably widen the lead of HiFi-GAN on fidelity and prosody and move Vocos ahead of DDSP on UTMOS (Vocos is projected to gain about 0.07 more than DDSP), which would settle the seed-sensitive DDSP-Vocos UTMOS order of section 4.1. A fixed step budget also favours the slower model: at equal training GPU-hours Vocos could take 1.5 times as many steps as HiFi-GAN.

Limits [conv]: one curve shape (fitted on the Vocos dev mel L1) is used for every vocoder and every metric; HiFi-GAN and DDSP have only two budgets; two seeds give a crude noise estimate; UTMOS is bounded near the ground truth; the fit rests on annealed end points and says nothing about a different schedule. The projections are predictions and were not tested on test.other. The cheapest checks named in [conv] are the evaluation of the 25k Vocos pilots on test.clean (about 10 GPU-minutes) and one 200k annealed run per vocoder (about 12 to 17 GPU-hours each).

**Status.** The direct test of these projections is in section 6.1.3.

#### 6.1.3 Result of the 800k runs

Setup [long]. One run each of HiFi-GAN and Vocos with 800,000 generator steps (experiment `main_800k`, seed 0, cosine schedule over 800k steps, otherwise the recipe of section 2; commit `581dcb0`), evaluated on test.clean under T1, T2 and T3 with the pipeline of section 2 (`eval_systems=long`; `fork_commit` `34e3942` in `meta.json`; final checkpoints `runs/main_800k/<vocoder>/ckpt/step000800000.ckpt`, g_step 800,000 of 800,000). The 400k results of both seeds are reused from the seed evaluation. Training cost, as the wall time of all job segments including restarts: HiFi-GAN 69.7 GPU-hours (8 segments; 316 ms per step on an L40S in the longest), Vocos 45.9 GPU-hours (one segment on an L40S) [conv section 3; status]. Each 800k arm is a single run, so an 800k result is compared with both 400k seeds: an improvement is called *robust* if it is supported and of the same sign against each seed and the smaller of the two differences exceeds the seed-to-seed difference of Table 4.1 (the rule of section 4.1).

**Table 6.1.3a. Dev mel L1 at the annealed end points** [conv section 3, `curves_800k.json`, `convergence_data.json`].

| vocoder | 100k | 400k seed 0 | 400k seed 1 | 400k mean | 800k | gain from the 400k mean | projected gain (section 6.1.2) | seed difference |
|---|---|---|---|---|---|---|---|---|
| HiFi-GAN | 0.5174 | 0.4886 | 0.4880 | 0.4883 | 0.4809 | 0.0075 | 0.0061 to 0.0095 | 0.0007 |
| Vocos | 0.5061 | 0.4934 | 0.4906 | 0.4920 | 0.4899 | 0.0021 | 0.0037 | 0.0028 |

- **HiFi-GAN followed the projection.** It gained 0.0075, inside the projected range and 11 times its seed difference. With three budgets (100k, 400k, 800k) the curve is fixed: alpha 0.62, close to the 0.59 fitted on Vocos, and an asymptote of 0.467. HiFi-GAN was 4.6 percent above it at 400k and is 3.0 percent above it at 800k; a further doubling is projected to gain 0.0049 [conv section 3].
- **Vocos gained less than projected, within its seed spread.** The gain is 0.0021 against the seed mean and 0.0035 against the 400k run with the same seed, and the seed difference is 0.0028, so one 800k run cannot show whether the projection of 0.0037 was right. A fit to all four Vocos budgets gives alpha 0.65 and an asymptote of 0.484 (1.2 percent above it at 800k); the last three budgets alone give alpha 1.09 and 0.488 (0.4 percent). Vocos is within about 1 percent of its asymptote at 800k [conv section 3].

**Table 6.1.3b. 800k against 400k, test.clean T1** (paired, a - b, CI95, agreeing speakers; *CI only* and *n.s.* as in section 4; n = 4,687 utterances, 39 speakers, prosody rows 4,686). Source: [long] (`paired.csv`; all rows in `phase3/analysis/long_report_T1.md`).

| vocoder | metric | 400k seed 0, seed 1 | 800k | 800k - seed 0 | 800k - seed 1 | seed difference | projected (section 6.1.2) | robust |
|---|---|---|---|---|---|---|---|---|
| HiFi-GAN | UTMOS | 4.165, 4.161 | 4.207 | +0.042 (+0.035, +0.050), 97 % | +0.046 (+0.037, +0.055), 92 % | 0.004 | +0.06 (capped) | yes |
| HiFi-GAN | PESQ | 2.091, 2.091 | 2.165 | +0.075 (+0.069, +0.081), 97 % | +0.075 (+0.068, +0.082), 97 % | 0.000 | | yes |
| HiFi-GAN | WER (percent) | 2.45, 2.49 | 2.42 | -0.03 (-0.09, +0.03), 51 %, n.s. | -0.07 (-0.15, +0.01), 44 %, n.s. | 0.03 | none | no |
| HiFi-GAN | Sim same | 0.857, 0.858 | 0.867 | +0.0098 (+0.0086, +0.0111), 97 % | +0.0084 (+0.0074, +0.0095), 95 % | 0.0014 | +0.015 | yes |
| HiFi-GAN | Sim spk | 0.731, 0.732 | 0.733 | +0.0025 (+0.0010, +0.0042), 66 %, CI only | +0.0014 (-0.0003, +0.0032), 61 %, n.s. | 0.0011 | | no |
| HiFi-GAN | mel L1 | 0.4878, 0.4879 | 0.4802 | -0.0076 (-0.0080, -0.0071), 100 % | -0.0077 (-0.0080, -0.0073), 97 % | 0.0001 | | yes |
| HiFi-GAN | F0 RMSE (cents) | 23.4, 23.5 | 21.4 | -1.98 (-2.26, -1.70), 92 % | -2.07 (-2.33, -1.78), 90 % | 0.09 | -1.9 | yes |
| HiFi-GAN | VDE | 0.0388, 0.0393 | 0.0370 | -0.0018 (-0.0023, -0.0013), 87 % | -0.0023 (-0.0029, -0.0016), 85 % | 0.0005 | | yes |
| HiFi-GAN | loudness RMSE (dB) | 1.49, 1.49 | 1.39 | -0.103 (-0.121, -0.085), 97 % | -0.108 (-0.124, -0.092), 97 % | 0.004 | -0.09 | yes |
| HiFi-GAN | EMA r | 0.9710, 0.9707 | 0.9724 | +0.0015 (+0.0013, +0.0016), 97 % | +0.0017 (+0.0016, +0.0018), 97 % | 0.0002 | | yes |
| Vocos | UTMOS | 3.896, 3.939 | 3.969 | +0.072 (+0.058, +0.085), 90 % | +0.030 (+0.020, +0.040), 90 % | 0.042 | +0.09 | no (0.030 is below 0.042) |
| Vocos | PESQ | 1.903, 1.936 | 1.947 | +0.044 (+0.037, +0.052), 95 % | +0.011 (+0.005, +0.017), 77 % | 0.033 | | no |
| Vocos | WER (percent) | 2.55, 2.47 | 2.44 | -0.11 (-0.18, -0.05), 67 %, CI only | -0.03 (-0.10, +0.04), 54 %, n.s. | 0.08 | none | no |
| Vocos | Sim same | 0.844, 0.847 | 0.852 | +0.0085 (+0.0072, +0.0099), 92 % | +0.0058 (+0.0045, +0.0071), 90 % | 0.0028 | +0.016 | yes |
| Vocos | Sim spk | 0.722, 0.724 | 0.724 | +0.0021 (+0.0004, +0.0039), 68 %, CI only | +0.0002 (-0.0013, +0.0018), 53 %, n.s. | 0.0018 | | no |
| Vocos | mel L1 | 0.4940, 0.4913 | 0.4902 | -0.0038 (-0.0042, -0.0034), 97 % | -0.0011 (-0.0015, -0.0006), 77 % | 0.0028 | | no |
| Vocos | F0 RMSE (cents) | 35.0, 33.5 | 32.9 | -2.08 (-2.65, -1.66), 97 % | -0.55 (-1.48, +0.10), 49 %, n.s. | 1.53 | -1.4 | no |
| Vocos | VDE | 0.0415, 0.0416 | 0.0422 | +0.0006 (-0.0001, +0.0015), 64 %, n.s. | +0.0006 (+0.0000, +0.0010), 67 %, CI only | 0.0001 | | no (worse) |
| Vocos | loudness RMSE (dB) | 1.83, 1.78 | 1.82 | -0.009 (-0.020, +0.002), 49 %, n.s. | +0.046 (+0.033, +0.058), 87 % (worse) | 0.054 | -0.07 | no |
| Vocos | EMA r | 0.9685, 0.9689 | 0.9695 | +0.0010 (+0.0008, +0.0012), 90 % | +0.0006 (+0.0004, +0.0008), 79 % | 0.0004 | | yes |

Reading of Table 6.1.3b [long, conv section 3]:

- **HiFi-GAN improved robustly on every metric except WER, CER and Sim spk** (also PESQ, MCD, MR-STFT, all mel bands, F0 median error, periodicity and EMA RMSE). Its UTMOS (4.207) equals the ground truth (4.208). The F0 and loudness gains match the projection; UTMOS (+0.044 against the seed mean) and Sim same (+0.009) reached about 74 and 61 percent of it. Under T2 and T3 the same rows are robust, and Sim spk is robust under T3 (`long_report_T2.md`, `long_report_T3.md`).
- **Vocos improved less, and mostly within its seed spread.** Against the seed mean it gained 0.051 UTMOS (57 percent of the projection), 0.007 Sim same (45 percent) and 1.3 cents of F0 RMSE (94 percent of the projection, but less than its seed difference of 1.5 cents); loudness and voicing did not improve. Only Sim same, the EMA consistency (EMA r and EMA RMSE) and mel L1 in the 4 to 8 and 8 to 12 kHz bands are robust; under T2 and T3 the same rows pass except mel L1 4 to 8 kHz under T2.
- **WER does not depend on the training length** for either vocoder (2.42 to 2.55 percent; no 800k difference is supported against both seeds).

**Table 6.1.3c. HiFi-GAN - Vocos at 400k and at 800k, T1** (400k: Table 4.2; 800k: [long] `paired.csv`).

| metric | 400k seed 0 | 400k seed 1 | 800k |
|---|---|---|---|
| UTMOS | +0.268, 97 % | +0.222, 97 % | +0.239 (+0.210, +0.267), 97 % |
| PESQ | +0.188, 100 % | +0.155, 97 % | +0.219 (+0.197, +0.237), 97 % |
| Sim same | +0.0130, 97 % | +0.0117, 97 % | +0.0144 (+0.0126, +0.0160), 97 % |
| mel L1 | -0.0063, 92 % | -0.0034, 74 %, CI only | -0.0100 (-0.0117, -0.0083), 97 % |
| F0 RMSE (cents) | -11.6, 97 % | -10.0, 100 % | -11.5 (-13.0, -10.3), 97 % |
| VDE | -0.0028, 87 % | -0.0023, 82 % | -0.0052 (-0.0063, -0.0042), 97 % |
| loudness RMSE (dB) | -0.34, 97 % | -0.28, 97 % | -0.44 (-0.47, -0.40), 97 % |

- **At equal steps the HiFi-GAN lead holds or grows.** The UTMOS gap at 800k (+0.239) lies between the two 400k values; the gaps in PESQ, mel L1, VDE and loudness widen, and the lower MCD of Vocos at 400k (0.025 and 0.030 dB) is gone at 800k (-0.010, n.s.). This is the direction projected in section 6.1.2.
- **At about equal training compute the HiFi-GAN lead also holds.** Vocos at 800k (45.9 GPU-hours) against HiFi-GAN at 400k (about 34 GPU-hours), recomputed from `per_speaker.csv` with the method of section 8 [long, `phase3/analysis/long_extra_pairs.md`]: UTMOS +0.196 (+0.172, +0.220) and +0.192 (+0.169, +0.215) for HiFi-GAN seed 0 and seed 1, 97 percent each; F0 RMSE -9.5 and -9.4 cents (100 percent); loudness RMSE -0.33 dB (100 percent); Sim same +0.005 and +0.006 (77 and 82 percent); mel L1 -0.0024 and -0.0023 (CI only). The cheaper training step of Vocos does not buy back the gap.
- **Vocos at 800k is ahead of DDSP at 400k on UTMOS**, by +0.137 (+0.075, +0.209), 77 percent, against DDSP seed 0 and +0.148 (+0.098, +0.206), 79 percent, against seed 1, both supported and more than three times the larger seed difference (0.042, Vocos) [`long_extra_pairs.md`]. At 400k this order was not robust (section 4.1). DDSP had converged by projection and was not run for 800k steps, and the Vocos 800k arm is one run, so this is single-run evidence that longer training settles the order in favour of Vocos.
- **The HiFi-GAN-DDSP F0 order stays unresolved.** HiFi-GAN at 800k has an F0 RMSE of 21.4 cents; DDSP at 400k is better by 3.2 cents against seed 0 (supported, 97 percent) and by 0.5 cents (not supported) against seed 1 [`long_extra_pairs.md`].

Limits: each 800k arm is a single run; the HiFi-GAN fit uses three points and has no residual; DDSP was not run for 800k steps; test.other, the probes and the efficiency measurements were not repeated at 800k.

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

**Status: complete.** All twelve ablation runs trained to 100,000 generator steps and were evaluated with the same pipeline and conditions as the main set (T1, T2, T3). Every group of every system and condition is complete; there is no failed or missing run [abl-table, Data completeness]. Each ablation is a single training run, so every difference below is a statement about two trained models and includes run-to-run variance, which was not estimated at 100k steps (the two seeds of the 400k main runs, section 4.1, give a reference scale: at most 0.042 UTMOS and 0.08 percentage points of WER).

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
- Not shown: which component of the native recipe produces its UTMOS gain and its losses in fidelity; whether the 400k ranking of Vocos and DDSP can be inferred from 100k; run-to-run variance at 100k steps. Every effect of 0.1 UTMOS or smaller, and every WER difference of 0.2 percentage points or smaller, could be produced by run variance, which was not estimated for the 100k runs. The seed-to-seed differences of the 400k main runs (at most 0.042 UTMOS and 0.08 percentage points of WER, section 4.1) suggest that this is conservative, but the 100k runs were not repeated.

## 9. Open questions and next steps

Open questions, each tied to what was and was not tested:

1. **Architecture or run variance.** Partly answered. The three main arms now have two training seeds (section 4.1): 23 of the 30 vocoder comparisons of Table 4.2 are robust under the rule of section 1, and the DDSP-Vocos UTMOS order, the HiFi-GAN-DDSP F0 order and every WER difference are not. Two runs per arm bound the variance only crudely (a seed difference is itself a noisy estimate, and the rule is a convention). The ablations at 100k steps are still single runs, and several of their supported effects (UTMOS 0.05 to 0.1, WER 0.1 to 0.2 percentage points) are of the order of the largest seed difference at 400k (0.042 UTMOS); the 100k ranking also differs from the 400k ranking. A third seed of the main arms, or a second seed of selected ablations, would be the next step.
2. **Why isolated EMA offsets are not reproduced.** The candidate explanations in section 5.2 are untested. A probe that scales the dynamics of an utterance around its mean, or shifts only speech frames, would separate them. A probe with real articulatory change (a re-inversion of audio that is known to differ in tongue position) is not available.
3. **Vocos F0 precision.** Whether the 50 to 150 cent deviations and the error below 70 Hz come from the iSTFT head, the 50 Hz frame rate or the amount of low-pitched training speech is untested. The ablations rule out two causes: the multi-resolution discriminator (the F0 error is 47 cents without it) and the FFT window of option A (37 cents against 39). The HiFi-GAN comparison (half the error rate with the same dependence on pitch) remains partial evidence.
4. **DDSP voicing and the male-speaker deficit.** The false voicing and the quiet pauses correlate with the UTMOS drop but a causal link is untested; a voicing gate or a different loudness path would test it. The size-matched DDSP (448 channels) and MSD ablations show that neither capacity nor the discriminator removes the UTMOS drop on the speakers with the largest drops (section 8.7); the voicing error is worse with MSD (VDE 0.071 against 0.063; the false-voicing share itself was not measured for the MSD arm) and the 448-channel model has a slightly lower VDE (-0.004) and a larger F0 error (+4.4 cents). A causal test of a voicing gate is still open.
5. **Shared word errors.** A common cause in the shared input is plausible (three vocoders fail on the same content) and untested. The Vocos mel copy synthesis has WER equal to the ground truth, which points to the feature bottleneck, but the WER of the features themselves was not measured by a separate probe.
6. **Metrics.** Sim spk is not a fidelity measure under T3, and Sim same under T2 depends on the target-reference similarity. A speaker-similarity measure that corrects for this (for example, using several references per speaker) or a listening test would be needed. No human listening was done in this phase; UTMOS is a predictor, not a listener.
7. **Statistics.** The paired criterion (CI95 excludes 0 and at least 75 percent of speakers agree) is a convention; roughly 180 comparisons were made in the main set and about 1,700 paired rows were produced in the ablation set, without correction.
8. **Streaming.** Only offline timings and a single-utterance lookahead probe exist. A chunked implementation would be needed to measure real latency; for DDSP the oscillator phase state is the open item.
9. **Second split and the reference.** test.other was evaluated for the main systems (section 3.1, seed 0 only), and the findings replicate with the differences listed there. It was not evaluated for seed 1, the ablations, the probes or the efficiency measurements. The training splits of en+ are undocumented, so en+ may have seen test.clean or test.other.
10. **Unexplained timing difference** in the en+ pipeline on 8 CPU threads (section 6).
11. **Confounded ablations.** The native recipe changes the discriminators, loss form and weights, optimizer and warm-up together, and the MSD arm also changes the adversarial weight; the source of the native recipe's UTMOS gain and fidelity loss is not identified. Single-factor runs (the shared recipe with only the optimizer settings or only the loss form changed) would separate them.
12. **`p_cross` between 0 and 0.5, and the T3 effect.** Only 0 and 0.5 were run, so the trade between T1 fidelity and T2 fidelity is known at two points. Why `p_cross` = 0 raises Sim spk under T3 by 0.023 was not tested.
13. **DDSP at 400k.** Its F0, voicing and loudness errors are worse at 400k than at 100k although its mel L1 is lower; whether this is a property of the run, of the loss or of longer training was not tested. The second seed adds a data point: its F0 error rose by a further 2.7 cents (section 4.1), so the F0 error of DDSP is also seed-dependent.
14. **Training length.** Partly answered by the 800k runs (section 6.1.3, issue `nzxyin/Speech-Articulatory-Coding#28`): HiFi-GAN followed the projected curve (alpha 0.62 from three budgets) and is still about 3 percent above its fitted dev mel L1 asymptote at 800k, so it would gain further from longer training; Vocos gained less than projected, within its seed spread, and is within about 1 percent of its asymptote. Each 800k arm is one run; DDSP was not run for 800k steps (it had converged by projection); test.other was not evaluated at 800k. A second 800k seed would bound the run variance at that budget, and a 200k run per vocoder would add a budget point for DDSP.
15. **Seed-sensitive findings and analyses done for seed 0 only.** The DDSP-Vocos UTMOS order and the HiFi-GAN-DDSP F0 order are not robust (section 4.1). Longer training adds single-run evidence on the first: Vocos at 800k is ahead of DDSP at 400k by about 0.14 UTMOS, supported against both DDSP seeds; the second stays unresolved at 800k (DDSP better by 3.2 cents against seed 0, supported, and by 0.5 cents, not supported, against seed 1) (section 6.1.3). The controllability probes and the natural-variation baseline (section 5.2) replicate in seed 1; the X-Y probes (section 5.3) are queued for seed 1; the principal-component probe, the failure analysis, the sex split of the DDSP deficit and the efficiency measurements were done for seed 0 on test.clean only.

Next steps: (a) and (b) are done: the ablation support (config group `eval_systems`, `table_name` in `io.py`, copied experiment configs, `tests/eval/test_ablations.py`) was committed as `ea550ad` and `phase3-eval` was merged into `vocoder-comparison` as `4af6f33` [status, 2026-10-12 entries]; (c) is done for the main systems: `test.other` was evaluated with the same code (`eval.split=test.other`, section 3.1); (d) was done for the three main arms at 400k steps instead of one 100k arm (section 4.1); (e) single-factor runs that separate the components of the native recipe (open question 11) remain; (f) is done: the 800k-step runs of HiFi-GAN and Vocos were evaluated and section 6.1.3 and the summary were updated (open question 14); (g) the seed-1 controllability probes are analysed (section 5.2) and the seed-1 X-Y probes are queued; a third seed of the main arms or a second seed of selected ablations remains, if the budget allows (open questions 1 and 15).

## 10. Reproducibility

**Code.**

- Training branch `vocoder-comparison` of the fork `nzxyin/Speech-Articulatory-Coding` (base `9777c30`). Commits: `603c58b` scaffolding, `85a65c1` pipeline, `a966a75` review fixes (`cudnn.benchmark` off), `c91ce98` documentation, `a11d1c4` ablation_100k configuration, `05bf159` D10 ablation configurations, `8f114d2` experiment `main_seed1` (seed 1; file `src/sparc/conf/experiment/main_seed1.yaml`), `581dcb0` experiment `main_800k` (`git log` of `~/Speech-Articulatory-Coding`). The main runs started on 2026-10-05 16:02 to 2026-10-06 01:48 (hydra directories), after `c91ce98`. The commit that each run used is not recorded in the run directories that I read; it was one of `c91ce98` or the later configuration-only commits, which do not change training code, but this was not verified.
- Evaluation branch `phase3-eval`, worktree `~/sparc-eval-wt`. The tables record `fork_commit` `374142c96bcd5039b799ac188f62dca864162da6` (`Changelog: evaluation fixes`) [meta.json]. Earlier commits: `da436ab` contract, `fb66135` pipeline, `848840c` efficiency markers, `5e77425` WER normalization. The ablation tables record `374142c96bcd5039b799ac188f62dca864162da6+dirty`, because the ablation support was uncommitted when they were built; it was committed afterwards as `ea550ad` and merged into `vocoder-comparison` as `4af6f33` [meta.json; status].
- Issues: tracking issue `nzxyin/Speech-Articulatory-Coding#19`; Phase 1 bugs #9 to #15; `pysptk`/`pkg_resources` issue #20 [p1 section 13; status].

**Data and features.**

- Feature cache: `$SPARC_VOC_CACHE` = `$SV_ROOT/cache` (358,332 valid utterances, 171 not encodable, 0 invalid), packed splits in `cache/packed/` (train 339,007 utterances, 97.2 million frames), training statistics `cache/stats/train_stats.json`, manifest `cache/manifest.parquet` [status, 2026-10-05 15:50].
- T2 reference manifests: `cache/sources/ref_manifest_test.clean.jsonl` (and `ref_manifest_dev.clean.jsonl`); the manifest generator is deterministic and was validated [p1 section 4; `ls cache/sources`].
- Inversion head: `/data/user_data/xoy/mngu0_linear_aai_refit/linear_aai_mngu0_refit_peralign.pkl`, sha256 `f6514ec405d311a9...` as recorded in [status] (`f6514ec4…` in [p1 section 13]; full value in `STATUS.md`).
- SPARC en+ Hugging Face snapshot `2e6a07d4b35022d4366a2896a6293fe16aaed78a`; `charactr/vocos-mel-24khz` revision `0feb3fdd…`; filtered list `parler-tts/libritts_r_filtered` snapshot `8980d653…`; torch 2.9.0, torchaudio 2.9.0, Lightning 2.6.5, transformers 4.57.1 [p1 section 13]. Evaluation models: UTMOS22 strong (`tarepan/SpeechMOS:v1.2.0`), `openai/whisper-large-v3`, `speechbrain/spkrec-ecapa-voxceleb`, `microsoft/wavlm-base-plus-sv` as secondary [eval section 4]. Model revisions and package versions are recorded in the `meta.json` sidecars of each stage.

**Runs and checkpoints.** Run directories `$SV_ROOT/runs/<experiment>/<vocoder>/` hold `ckpt/`, `hydra/`, `predictions/`, `tb/`, `status.json`. Seed 0 for every run in `runs/main/` and in the ablation experiments (`seed: 0` in the run configuration and in `train_config.yaml`), seed 1 for `runs/main_seed1/`. Checkpoints used in the main evaluation [table, Checkpoints; efficiency JSON]:

| system | checkpoint | g_step |
|---|---|---|
| HiFi-GAN | `$SV_ROOT/runs/main/hifigan/ckpt/step000400000.ckpt` | 400,000 of 400,000 |
| DDSP | `$SV_ROOT/runs/main/ddsp/ckpt/step000400000.ckpt` | 400,000 of 400,000 |
| Vocos | `$SV_ROOT/runs/main/vocos/ckpt/step000400000.ckpt` | 400,000 of 400,000 |
| HiFi-GAN, seed 1 | `$SV_ROOT/runs/main_seed1/hifigan/ckpt/step000400000.ckpt` | 400,000 of 400,000 |
| DDSP, seed 1 | `$SV_ROOT/runs/main_seed1/ddsp/ckpt/step000400000.ckpt` | 400,000 of 400,000 |
| Vocos, seed 1 | `$SV_ROOT/runs/main_seed1/vocos/ckpt/step000400000.ckpt` | 400,000 of 400,000 |
| HiFi-GAN, 800k | `$SV_ROOT/runs/main_800k/hifigan/ckpt/step000800000.ckpt` | 800,000 of 800,000 |
| Vocos, 800k | `$SV_ROOT/runs/main_800k/vocos/ckpt/step000800000.ckpt` | 800,000 of 800,000 |

Preemption history: HiFi-GAN was requeued at least twice and DDSP at least once (several hydra start directories in `runs/main/<vocoder>/hydra/`); training resumes at a batch boundary, and the final checkpoint is the one evaluated [status]. No checkpoint was selected on test metrics. The runs of 800,000 steps (`runs/main_800k/hifigan` and `runs/main_800k/vocos`, seed 0, cosine schedule over 800k steps, checkpoints every 50k steps) finished at step 800,000 (`status.json`): HiFi-GAN in 8 job segments (requeued after preemption on `preempt` and at the 2-day limit of `general`; the last segment only found the finished marker), Vocos in one. The submission record is `phase3/N_submit_800k.sh` and `N_submit_800k.out` (training jobs 10799205 and 10799206, evaluation jobs 10799207 to 10799215 chained after them) [status, entry 2026-10-14 10:45]. Their evaluation set is `eval_systems=long`; the aggregate job 10799215 wrote `tables/test.clean/long/` on 2026-10-17 [long].

**Evaluation outputs** under `$SV_ROOT/eval/`: `audio/`, `results/test.clean/<system>/<cond>/<group>/part-*.parquet`, `embeddings/`, `arrays/`, `probes/<system>/`, `efficiency/<system>.json`, `tables/test.clean/{table.md, table.csv, paired.csv, per_speaker.csv, meta.json}`, `samples/`, `failures/<case>/`. Copies of the final tables, efficiency JSON files and sample index: `~/sparc-vocoders-work/phase3/final/`. Analysis files and scripts: `~/sparc-vocoders-work/phase3/analysis/` (`paired_report.py`, `build_master.py`, `add_sims.py`, `build_cases.py`, `cases.py`, `failure_stats.py`, `efficiency.py`, `probes_table.py`, `natural_variation.py`, `pc_probe.py`, `controllability_report.py` under `scripts/`). Per-utterance results of all systems with Sim same and Sim spk: `analysis/master.parquet`.

**Commands.** `python -m sparc.cli.eval_vocoder stage=<gt|refs|signal|utmos|asr|spk|reextract|prosody|probes|efficiency|aggregate|samples> system=<name|all> condition=<T1|T2|T3|all> eval.split=test.clean`, run as `~/sparc-eval-wt/.venv/bin/python` on a compute node after `source ~/sparc-vocoders-work/env.sh`. Slurm wrappers: `scripts/slurm/eval.sh` and `submit_eval.sh`. Second split: `eval.split=test.other` (tables in `$SV_ROOT/eval/tables/test.other/`, copy `phase3/final/tables/test.other/`, `fork_commit` `85c396283b41ed3d851b0ba2109fc7a837e4e0ee` [meta.json]; submission script `phase3/L_submit.sh`). Second seed: `eval_systems=seeds eval.table_name=seeds` (tables in `$SV_ROOT/eval/tables/test.clean/seeds/`, copy `phase3/final/tables/seeds/`, same `fork_commit`; submission script `phase3/K_submit.sh`). Training length: `eval_systems=long` (tables in `$SV_ROOT/eval/tables/test.clean/long/`, copy `phase3/final/tables/long/`, `fork_commit` `34e3942cf06662c3858c06ec054148833c271fb0`; submission script `phase3/N_submit_800k.sh`). Ablation evaluation: add `eval_systems=ablations`; its tables are in `$SV_ROOT/eval/tables/test.clean/ablations/` (copy: `phase3/final/tables/ablations/`), the checkpoints are those of experiments `ablation_100k`, `abl_pcross0`, `abl_native` and `abl_msd` under `$SV_ROOT/runs/` (names in section 8.1; I did not inspect `/data`). The extra paired differences of section 8 are recomputed with the script reproduced in `phase3/I_ablation_section.md`.

## Appendix: inconsistencies found between sources

1. **GPU RTF values.** `STATUS.md` (2026-10-08 03:40) quotes 0.0058, 0.0007 and 0.0046 for HiFi-GAN, Vocos and DDSP; `efficiency.md` uses the medians 0.0061, 0.0008 and 0.0050. Both are correct for their definition: the JSON files hold the pooled value (total time over total audio: 0.00575, 0.00070, 0.00464) and the median. This document uses the medians and states the pooled values.
2. **Ceiling wording in the failures summary.** The hand-over summary says that under T3 the vocoders exceed the ceiling "0.783 to 0.787 against 0.749 for HiFi-GAN and Vocos". The value 0.749 is the ground-truth ceiling (table row "ground truth", Sim spk); DDSP under T3 (0.731) is below it. This document uses the table.
3. **Dev mel L1 of the Vocos B ablation.** `STATUS.md` gives 0.5061 at 100k (2026-10-07 03:00 entry) and 0.506 at 95k (2026-10-06 12:30 entry); the two are consistent.
4. **Efficiency timing mismatch** for the en+ pipeline on 8 CPU threads (31 percent) is stated in the source and left unexplained.
5. **Statement on the evaluation of the lookahead for DDSP.** `efficiency.md` gives a measured lookahead of 929 ms for all channels and 717 to 1,063 ms per channel group; the hand-over summary says "about 0.9 to 1.1 s". The per-group maximum is 1,063 ms and the all-channel value is 929 ms, so "about 0.9 to 1.1 s" is a fair rounding of the range.
6. **Paired-table cell detail.** Several cells of Table 3 give only the verdict ("supported") because `paired.md` summarizes them in words; the numeric values are in `paired_vocoders.csv`.
7. **Run commit.** The training commit used by each main run is not stored in the run directory (section 10).
8. **DDSP-Vocos UTMOS in seed 1.** The preview in `STATUS.md` (entry 2026-10-13 13:20) lists the seed-1 difference as supported; the final `seeds/paired.csv` gives -0.118 (CI -0.177 to -0.067) with 69 percent of speakers agreeing, which is CI only under the 75 percent criterion. This document uses the final table (section 4.1).
9. **Earlier summary wording on F0.** The first version of this document said that every vocoder pair is supported on F0 RMSE. That holds for seed 0 and for test.other and not for HiFi-GAN against DDSP in seed 1 (+2.6 cents, CI only, 69 percent); the summary was corrected (section 1, item 3).
10. **Cell counts of the test.other table.** The `n` column of `test.other/table.md` lists 4,709 utterances for the prosody, EMA and PESQ cells, while the metric counts in `table.csv` and `paired.csv` are 4,705 (4,703 for the HiFi-GAN F0 cells, 4,701 for en+ PESQ). Four utterances are too short for the re-extraction. This document states the real counts (section 3.1).
11. **Statements about run variance.** Sections 1, 3, 4, 8 and 9 of the previous version said that run-to-run variance was not estimated and that test.other was not evaluated. Both statements were updated (second seeds exist for the main arms; test.other exists for the main systems) and the ablation statements were limited to the 100k runs, which still have a single seed.
12. **Rounding in the convergence analysis.** `convergence.md` lists the ratio of the projected 800k gain of HiFi-GAN to its dev mel L1 seed difference as 9 to 14; with the seed difference of 0.0006 printed in `convergence.md` the ratio would be 10 to 16, but the unrounded seed difference is 0.00067 (0.4886 against 0.4880, `convergence_data.json`), which rounds to 0.0007 and gives 9 to 14; the table of section 6.1.2 shows 0.0007. For DDSP the printed 0.0052 is likewise 0.0053 unrounded. The conclusion (the gain is about ten times the seed difference) is unchanged.
13. **Baselines of the training-length gains.** Section 8.2 and summary item 8(a) give the 100k-to-400k gains of seed 0 (UTMOS +0.22, +0.31, +0.07 for HiFi-GAN, Vocos, DDSP); `convergence.md` and section 6.1 use the mean of the two seeds (+0.22, +0.33, +0.06). Both are correct for their baseline.
14. **Interim reading of the Vocos 800k result.** The `STATUS.md` entry of 2026-10-16 and `phase3/analysis/800k_interim.md` (written before the HiFi-GAN 800k evaluation) read the Vocos gains against the mean of the two 400k seeds; `800k_interim.md` listed MR-STFT and MCD among the gains larger than the seed spread. Under the rule of section 4.1 (the smaller of the two differences must exceed the seed difference) they are not robust (MR-STFT: 0.0055 against a seed difference of 0.0066). This document uses the rule (section 6.1.3).
