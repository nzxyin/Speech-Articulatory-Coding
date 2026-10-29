# What does the periodicity input bring to the articulatory vocoder?

Question (2026-10-29): the original SPARC vocoder takes 12 EMA channels, pitch and loudness (14 channels) and uses
periodicity only to weight the speaker-embedding pooling. This fork's vocoders take periodicity as a 15th frame input. What
does adding periodicity to the vocoder input bring over the original SPARC inputs, for voicing, fricatives, vowels and the
rest?

Two systems and three inference-time probes, all HiFi-GAN, trained on LibriTTS-R only, 100k generator steps, one seed:

| name | what it is |
|---|---|
| fork input set (+ periodicity) | the vocoder comparison's reference run `ablation_100k` (read only), 15 input channels |
| original SPARC input set (EMA + pitch + loudness) | the same recipe without the periodicity channel in the generator input: experiment `periodicity_none_100k`, 14 channels in the original order (`ema[12], pitch, loudness`, as `sparc.sparc`) |
| probe: training mean | the fork model with its periodicity input replaced at inference by the training mean (0.437) |
| probe: voiced/unvoiced mask | replaced by a mask: 0 where periodicity is 0 (unvoiced), 0.802 where it is above 0 (mean periodicity of the voiced dev-clean frames) |
| probe: smoothed | replaced by a 9-frame (180 ms) moving average |

The probes show how much the trained model relies on the channel. They do not show what training with the channel gains;
the ablation run does that.

## Method

**Ablation.** `FeatureFrontend(drop_periodicity=True)` leaves the periodicity channel out of the network input
(`models/frontend.py`); `HiFiGANOriginalInputsVocoder` (`models/hifigan_inputs.py`, built on the unchanged `HiFiGANVocoder`)
uses it with a 14-channel input convolution. The default path is unchanged (`tests/test_model_hifigan_inputs.py`: default
frontend output unchanged, the ablated input equals the first 14 default channels, the ablated generator output does not
depend on the periodicity column, the default one does). Periodicity is used nowhere else in the HiFi-GAN generator, the losses or the
discriminators (searched); the speaker-embedding pooling keeps reading the cached periodicity weights, outside the model.
Checkpoint configs differ only in run name and paths, the generator class and `in_channels` (14), plus keys of later code
that sit at their defaults and are absent from the older reference config (`python -m sparc.vocoders.eval.periodicity_analysis
cfgdiff`; among them the conditioning switches `cond_type: film`, `cond_sites: residual`, which equal the reference
behavior by default). Run: `/data/user_data/xoy/spkemb/phase3/runs/periodicity_none_100k/hifigan` (seed 0, step 100000,
smoke run of 300 steps first). Both runs were evaluated at their final checkpoints (step 100000).

**Items.** Resynthesis of an utterance from its own features and its own speaker vector (condition T1 of the vocoder
comparison; checked: my synthesis of the reference reproduces its stored test-clean predictions exactly, maximum sample
difference 0). Fixed subsets, up to 15 utterances of 2-12 s per speaker (seeded by speaker), no selection among variants:
LibriTTS-R dev-clean (validation, 537 utterances, 39 speakers), test-clean (test, 542, 38), VCTK validation half (300, 20
speakers) and VCTK test half (300, 20), the halves of `docs/spkemb_validation_splits.json`. Every number below is a mean over
speakers of per-utterance values unless stated; differences are system minus the fork model, with a speaker-level paired
bootstrap (10,000 resamples) 95% interval; `*` marks an interval that excludes 0. No multiplicity correction (many metrics).

**Global metrics** (existing evaluation code, `docs/vocoders/EVALUATION.md`): UTMOS, Whisper large-v3 WER and CER (corpus
ratio), training-mel L1, multi-resolution STFT distance, PESQ, ECAPA and CAM++ cosine to the recording of the same utterance.
**Voicing:** both the output and the recording go through the same feature extractor (CREPE with SPARC's periodicity
threshold, no dither); voiced = periodicity above 0. Voiced/unvoiced decision error (VDE, fraction of frames), F0 error in
cents on frames voiced in both, periodicity MAE. **HNR** is my own normalized-autocorrelation estimator (50 ms Hann window,
lags 70-545 Hz, window-corrected, clipped to +-60 dB); the error is output minus recording per 20 ms frame, reported as
the mean absolute error and the signed mean (bias). **High band** is the 4-12 kHz log energy (Hann 1024 frames at 50 Hz),
same error definitions. All on the 50 Hz frame grid; mel L1 is the training mel pooled per 20 ms frame.

**Sound classes** come from forced alignment with the Montreal Forced Aligner (`english_mfa` dictionary and acoustic model,
`scripts/spkemb/periodicity_mfa.sh`) of the recordings. Coverage: all 1679 utterances aligned; 94.8% (dev-clean), 95.0%
(test-clean), 99.5% (both VCTK halves) of frames fall in a class; the rest are `spn` (spoken noise / out-of-vocabulary)
labels. Classes by IPA label: vowels (including diphthongs), voiced fricatives (z ʒ v ð), voiceless fricatives (s ʃ f θ h ç),
stops (p b t d k g and variants), affricates (tʃ dʒ), nasals, approximants / liquids (l ɫ ɹ j w ʋ ɾ ...; the flap and
syllabic consonants included), silence (empty labels). A frame belongs to the phone interval that contains its centre
(`(t + 0.5) * 20 ms`). Stop closure and burst are one MFA interval, so I split it by time: the last 40 ms are "burst", the
rest "closure" (a stop under 60 ms is all burst); with 20 ms frames this is a coarse proxy. "Voicing change" frames are those
within 30 ms of a voiced/unvoiced change of the recording's voicing track (4 frames around the change; voiced or unvoiced runs
shorter than 3 frames are first absorbed to avoid tracker flicker); onsets and offsets are also reported separately in the
full tables. Test-clean frames per class: vowels 58,342, silence 21,494, voiceless fricatives 15,792, nasals 13,380, stop
burst 13,361, approximants 12,186, stop closure 11,864, voiced fricatives 8,324, affricates 1,918.

Code and tests: `src/sparc/vocoders/eval/periodicity_analysis.py`, `tests/test_periodicity_analysis.py`. Results, per-frame
arrays, alignments, audio and every table: `/data/user_data/xoy/sparc-vocoders-periodicity/` (`summary/summary.md` holds all
tables for all four splits, `summary/results.json` every estimate with its interval).

## Results

### Global metrics, original SPARC input set minus the fork model

Fork model value, then the difference with its interval. Test splits first (the reportable numbers), validation after.

| metric | test-clean: fork | original minus fork | VCTK test: fork | original minus fork |
|---|---|---|---|---|
| UTMOS | 4.013 | -0.050 [-0.076, -0.023]* | 3.848 | -0.024 [-0.056, +0.008] |
| WER (%) | 2.38 | +0.18 [-0.04, +0.42] | 2.92 | +0.36 [-0.40, +1.21] |
| CER (%) | 0.88 | +0.09 [-0.05, +0.24] | 1.44 | +0.17 [-0.25, +0.59] |
| mel L1 | 0.5090 | +0.0087 [+0.0078, +0.0097]* | 0.6800 | +0.0009 [-0.0035, +0.0054] |
| multi-res. STFT distance | 1.207 | +0.0187 [+0.0164, +0.0211]* | 1.487 | +0.0137 [+0.0059, +0.0242]* |
| PESQ (WB) | 1.872 | -0.096 [-0.111, -0.081]* | 1.712 | -0.034 [-0.051, -0.016]* |
| ECAPA similarity to recording | 0.8118 | -0.0035 [-0.0065, -0.0007]* | 0.5818 | -0.0010 [-0.0078, +0.0065] |
| CAM++ similarity to recording | 0.8053 | -0.0102 [-0.0144, -0.0064]* | 0.5444 | +0.0031 [-0.0055, +0.0118] |
| voiced/unvoiced decision error | 0.0455 | +0.0290 [+0.0258, +0.0322]* | 0.0354 | +0.0301 [+0.0243, +0.0363]* |
| F0 error on both-voiced frames, RMSE (cents) | 32.3 | +1.7 [+0.2, +2.8]* | 31.4 | +1.2 [-1.4, +3.2] |
| periodicity MAE | 0.0526 | +0.0246 [+0.0225, +0.0269]* | 0.0374 | +0.0258 [+0.0213, +0.0307]* |
| HNR error, mean absolute dB | 2.944 | +0.171 [+0.143, +0.199]* | 3.379 | +0.179 [+0.137, +0.226]* |
| HNR error, mean dB (bias) | -0.111 | -0.056 [-0.105, -0.004]* | +1.079 | +0.189 [+0.085, +0.295]* |
| 4-12 kHz error, mean absolute dB | 3.343 | +0.049 [+0.033, +0.065]* | 4.912 | +0.098 [+0.036, +0.163]* |
| 4-12 kHz error, mean dB (bias) | -1.407 | -0.079 [-0.109, -0.048]* | -2.598 | -0.278 [-0.381, -0.170]* |

The recording itself scores UTMOS 4.246 (test-clean) and 3.960 (VCTK test), WER 1.87% and 1.26%. Validation agrees in sign
and size: dev-clean UTMOS -0.036 [-0.059, -0.014]*, WER +0.24 pts [+0.02, +0.47]*, mel L1 +0.0084*, PESQ -0.112*, VDE
+0.029*, HNR error +0.115 dB*; VCTK validation half UTMOS -0.006 [-0.041, +0.025], mel L1 -0.0011 [-0.0062, +0.0040], PESQ
-0.027*, VDE +0.028*, HNR error +0.197 dB*.

### Per sound class, original SPARC input set minus the fork model (LibriTTS-R test-clean)

| class | mel L1 (fork) | mel L1 diff | VDE (fork) | VDE diff | HNR abs. error dB (fork) | HNR diff | 4-12 kHz abs. error dB (fork) | 4-12 kHz diff |
|---|---|---|---|---|---|---|---|---|
| all frames | 0.509 | +0.0087* | 0.045 | +0.029* | 2.94 | +0.17* | 3.34 | +0.049* |
| vowels | 0.493 | +0.0119* | 0.045 | +0.035* | 2.70 | +0.27* | 3.32 | +0.047* |
| voiced fricatives | 0.504 | +0.0094* | 0.088 | +0.048* | 3.29 | +0.09 | 3.28 | +0.081* |
| voiceless fricatives | 0.432 | +0.0041* | 0.043 | +0.014 | 1.75 | +0.07* | 2.07 | +0.006 |
| stops, closure | 0.574 | +0.0062* | 0.058 | +0.038* | 2.90 | +0.14* | 3.93 | +0.023 |
| stops, burst | 0.582 | +0.0074* | 0.067 | +0.035* | 3.19 | +0.06 | 4.00 | +0.017 |
| affricates | 0.533 | +0.0136* | 0.050 | +0.042* | 2.10 | +0.21* | 3.41 | +0.103 |
| nasals | 0.545 | +0.0098* | 0.056 | +0.019* | 4.71 | +0.23* | 3.59 | +0.114* |
| approximants / liquids | 0.506 | +0.0096* | 0.052 | +0.032* | 3.66 | +0.24* | 3.46 | +0.066* |
| silence | 0.521 | +0.0029 | 0.007 | +0.002* | 2.91 | +0.08* | 3.67 | +0.041 |
| within 30 ms of a voicing change | 0.518 | +0.0093* | 0.082 | +0.044* | 2.47 | +0.26* | 3.40 | +0.054* |

Intervals for every cell are in `summary/summary.md`. On VCTK test the same pattern holds for the voicing measures (VDE
+0.062 vowels, +0.066 voiced fricatives, +0.063 approximants, +0.063 at voicing changes, all `*`; HNR error +0.39 dB vowels,
+0.56 voiced fricatives, +0.35 voicing changes, +0.49 approximants, `*`), and for mel L1 per class (+0.006 to +0.014 for
every sound class `*`, except affricates), but silence frames get *better* without periodicity (mel L1 -0.011 [-0.020,
-0.003]*), which cancels the sound classes in the all-frames mean (+0.0009, n.s.).

### Reliance probes on the fork model (no training), test-clean differences from the unmodified fork model

| metric | periodicity = training mean | voiced/unvoiced mask | smoothed over 9 frames |
|---|---|---|---|
| UTMOS | -0.430* | -0.051* | -0.098* |
| WER (%) | +0.80* | +0.13 | +0.15 |
| mel L1 | +0.1496* (+29%) | +0.0144* | +0.0275* |
| PESQ (WB) | -0.546* | -0.090* | -0.202* |
| ECAPA similarity | -0.0631* | -0.0024* | -0.0160* |
| voiced/unvoiced decision error | +0.120* | -0.003* | +0.033* |
| F0 RMSE (cents) | +51.8* | +1.8* | +8.6* |
| HNR error, mean absolute dB | +1.92* | +0.22* | +0.43* |
| 4-12 kHz error, signed dB | -1.27* | -0.12* | -0.29* |

Per class (test-clean, mel L1 difference): the training-mean probe hurts silence most (+0.291, from 0.521; the output loses
3.6 dB of 4-12 kHz energy and 2.2 dB of HNR bias there), then voiceless fricatives (+0.147, 4-12 kHz energy -2.1 dB), vowels
(+0.128), stop closure (+0.122), approximants (+0.120) and nasals (+0.108); voicing decisions go wrong on 11-16% more frames
in every sound class (vowels +0.137, voiced fricatives +0.162; silence +0.045) and vowel HNR falls by 3.1 dB. The mask probe is nearly harmless
(largest mel L1 loss +0.021 for vowels and nasals; voicing decisions slightly better than the fork's own, -0.5 pt on vowels;
HNR of voiced fricatives +0.6 dB too high, stops and affricates +0.2 to +0.4 dB). Smoothing mostly hurts at voicing changes
(mel L1 +0.038, HNR error +0.59 dB, VDE +0.070, high-band error +0.28 dB) and at vowels, nasals and approximants (mel L1
+0.029 to +0.037), and removes 0.4 to 0.8 dB of 4-12 kHz energy at stops and fricatives. The probes agree in direction and
size on dev-clean and on both VCTK halves (UTMOS: mean -0.37 to -0.43, mask -0.02 to -0.05, smoothed -0.06 to -0.10; WER on
VCTK test +2.2 pts for the mean, -0.2 for the mask, +0.6 for smoothing).

## What periodicity helps and hurts

1. **It gives the generator an explicit, frame-precise voiced/unvoiced (and silence) gate.** Without it the model still
   renders speech, but it makes more voicing mistakes: the output's voiced/unvoiced decision disagrees with the recording on
   7.4% of test-clean frames instead of 4.5% (VCTK test 6.6% instead of 3.5%), a 60-90% relative increase. The biggest
   increases are where voicing is hard to infer from the other inputs: voiced fricatives (8.8% to 13.6% test-clean),
   vowels (+3.5 pts, VCTK +6.2), stops and affricates, and frames at voicing onsets and offsets (8.2% to 12.6%). Voiceless
   fricatives, nasals and silence change least.
2. **The cost of removing it is small in size.** In domain: UTMOS -0.05 (4.01 to 3.96), mel L1 +1.7%, PESQ -0.10, HNR error
   +0.17 dB, F0 RMSE +1.7 cents; ECAPA similarity -0.004 and CAM++ -0.010 (small, significant on test-clean). WER/CER rise by
   0.1-0.2 points, not distinguishable from 0 on test-clean (+0.24 points, `*`, on dev-clean). Per class mel L1 rises by
   0.6-2.5% (vowels and affricates most, silence not significantly). For scale: the two seeds of the 400k HiFi-GAN run differ by
   0.0001 in test-clean mel L1, 0.006 in UTMOS and 0.0005 in VDE, far below these differences; no seed pair exists at 100k.
3. **Vowels, approximants and nasals (voiced, harmonic sounds) lose the most harmonicity accuracy** (HNR error +0.23 to
   +0.27 dB; VCTK +0.39 dB for vowels, +0.56 for voiced fricatives); **voiceless fricatives, stops and silence lose least in
   spectral terms**: their mel L1 rises by 0.6-1.3% and their high-band error does not change significantly (stops still
   lose 3.5-3.8 points of voicing agreement, voiceless fricatives 1.4, not significant). Periodicity is not what makes
   fricative noise; it separates voiced from voiceless and keeps voiced sounds periodic.
4. **Out of domain (VCTK, never seen in training) the benefit does not carry to global quality**: the original input set is
   not worse in UTMOS, mel L1, WER, or speaker similarity (all intervals include 0; PESQ and the STFT distance are still
   slightly worse), and is better on silence frames. The
   voicing metrics (VDE, HNR error) are still worse by the same absolute amount as in domain. So periodicity helps voicing
   consistently, quality only for matched recording conditions (and quality measured on silence frames can favor the
   model without it).
5. **The graded value matters much less than the voicing decision.** Replacing periodicity by a voiced/unvoiced mask
   (keeping only where it is above 0) costs almost nothing (UTMOS -0.05, VDE slightly better), the constant is catastrophic
   (UTMOS -0.43, mel L1 +29%, WER +0.8 pts, voicing wrong on 17% of frames), and smoothing in time (which blurs voicing
   edges) costs about twice the mask. The trained model reads the voicing gate and its onset/offset timing, not the
   periodicity strength. The pitch, loudness and EMA channels carry much of the rest: a model trained without periodicity
   loses about as much as the mask probe on mel L1 (0.009 against 0.014) and far less than the training-mean probe (0.150),
   and it keeps most voicing agreement (7.4% against 4.5% decision error, where the mean probe has 16.6%).
6. **What it hurts.** In this single run: nothing measurable in domain. Out of domain, the model with periodicity renders
   silence frames slightly worse (mel L1 +0.011 on VCTK silence), possibly because its silence gate meets recording noise
   it did not see in training (not tested).

## Caveats

- One seed per system and 100k steps (a quarter of the main recipe), LibriTTS-R training only; the 400k seed pair is the only
  indication of run-to-run noise, and it is for a different training length. A second seed of the ablation would remove the
  largest doubt about the small differences (WER, speaker similarity, UTMOS on VCTK).
- Subsets: 1679 utterances (about 40 speakers per LibriTTS split, 20 per VCTK half), not the full test sets; intervals are
  over those speakers and do not cover the choice of subset. Many metrics, no multiplicity correction.
- Voicing is judged by the same extractor on output and recording (CREPE periodicity threshold), so VDE measures agreement
  of this tracker, not phonetic voicing; the extractor's periodicity also drives the model's input, so the fork model's VDE
  advantage partly reflects the input and the judge sharing the same tracker.
- The mask probe could not use "F0 > 0": the cached F0 channel is never 0 (unvoiced frames carry filled values), so voicing
  is periodicity > 0 (the extractor's own decision), with 0.802 (mean of voiced dev-clean frames) for voiced frames. The
  probes are out-of-distribution inputs for the trained model; they show reliance, not benefit.
- HNR, the high-band energy and the stop closure/burst split are my own estimators on a 20 ms grid (HNR windows are 50 ms), so
  burst and onset timing are not resolved finer than that; class frames can include coarticulation. Forced alignments come from
  MFA on the recordings (not hand-checked); about 5% of LibriTTS frames are unlabelled noise labels.
- The ablation was trained with the repository state of 2026-10-29, which contains other, off-by-default conditioning switches
  (FiLM sites); the stored config shows their defaults equal to the original recipe. The original SPARC also used its own
  pitch scaling; here only the periodicity channel is removed and nothing else changed.

## Reproduction

```
python -m sparc.vocoders.eval.periodicity_analysis prepare                  # subsets and MFA corpus
scripts/spkemb/periodicity_mfa.sh                                           # alignments
python -m sparc.vocoders.eval.periodicity_analysis synth --systems gt,ref,probe_const,probe_binary,probe_smooth,none --check
python -m sparc.vocoders.eval.periodicity_analysis metrics --systems gt,ref,none,probe_const,probe_binary,probe_smooth
python -m sparc.vocoders.eval.periodicity_analysis summarize                # tables, bootstrap
python -m sparc.vocoders.eval.periodicity_analysis cfgdiff                  # training configs of reference and ablation
```

Training: `scripts/spkemb/p3_train.sh hifigan periodicity_none_100k` through `slurm-alloc` (`TIME=16:00:00`, restartable),
log `/data/user_data/xoy/spkemb/phase3/logs/p3t_periodicity_none_100k_10976729.out`.
