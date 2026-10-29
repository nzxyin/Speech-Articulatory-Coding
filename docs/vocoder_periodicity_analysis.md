# What does the periodicity input bring to the articulatory vocoder?

Question (2026-10-29): the original SPARC vocoder takes 12 EMA channels, pitch and loudness (14 channels) and uses
periodicity only to weight the speaker-embedding pooling. This fork's vocoders take periodicity as a 15th frame input. What
does adding periodicity to the vocoder input bring over the original SPARC input set, for voicing, fricatives, vowels and the
rest?

Revised 2026-10-29 after an adversarial review: voicing is now also judged by a tracker that is not the vocoder's input
(pYIN) and by phone labels, a second seed of the fork recipe gives a noise reference, and the two runs' configs and training
code are compared.

All systems are HiFi-GAN, trained on LibriTTS-R only, 100k generator steps:

| name | what it is |
|---|---|
| fork input set (+ periodicity), seed 0 | the vocoder comparison's reference run `ablation_100k` (read only), 15 input channels |
| fork input set, seed 1 | the same recipe with training seed 1, experiment `p4_b0_s1` (read only): the noise reference |
| original SPARC input set (EMA + pitch + loudness) | the same recipe as seed 0 without the periodicity channel in the generator input: experiment `periodicity_none_100k`, 14 channels in the original order (`ema[12], pitch, loudness`, as `sparc.sparc`) |
| probe: training mean | the seed-0 fork model with its periodicity input replaced at inference by the training mean (0.437) |
| probe: voiced/unvoiced mask | replaced by a mask: 0 where periodicity is 0 (unvoiced), 0.802 where it is above 0 (mean periodicity of the voiced dev-clean frames) |
| probe: smoothed | replaced by a 9-frame (180 ms) moving average |

The probes show how much a model trained with the channel relies on it. They do not show what training with the channel
gains; the ablation run does that.

## Method

**Ablation.** `FeatureFrontend(drop_periodicity=True)` leaves the periodicity channel out of the network input
(`models/frontend.py`); `HiFiGANOriginalInputsVocoder` (`models/hifigan_inputs.py`, built on the unchanged `HiFiGANVocoder`)
uses it with a 14-channel input convolution. The default path is unchanged (`tests/test_model_hifigan_inputs.py`: default
frontend output unchanged, the ablated input equals the first 14 default channels, the ablated generator output does not
depend on the periodicity column, the default one does). Periodicity is used nowhere else in the HiFi-GAN generator, the losses or the
discriminators (searched); the speaker-embedding pooling keeps reading the cached periodicity weights, outside the model.
Run: `/data/user_data/xoy/spkemb/phase3/runs/periodicity_none_100k/hifigan` (seed 0, step 100000, smoke run of 300 steps
first). All three runs were evaluated at their final checkpoints (step 100000). Comparability of the runs: see the section
after the caveats.

**Items.** Resynthesis of an utterance from its own features and its own speaker vector (condition T1 of the vocoder
comparison; checked: my synthesis of the seed-0 reference reproduces its stored test-clean predictions exactly, maximum sample
difference 0, so the current generator code reproduces the code the reference was trained with). Fixed subsets, up to 15
utterances of 2-12 s per speaker (seeded by speaker), no selection among variants: LibriTTS-R dev-clean (validation, 537
utterances, 39 speakers), test-clean (test, 542, 38), VCTK validation half (300, 20 speakers) and VCTK test half (300, 20), the
halves of `docs/spkemb_validation_splits.json`. Every number below is a mean over speakers of per-utterance values unless
stated; differences are system minus the fork model, with a speaker-level paired bootstrap (10,000 resamples) 95% interval; `*`
marks an interval that excludes 0. No multiplicity correction anywhere (many metrics, 17 frame classes, four splits).

**Seed spread.** The seed-1 fork model gives the run-to-run noise at this recipe: spread = seed 1 minus seed 0, same items,
same paired bootstrap. A difference "exceeds the seed spread" when (original input set minus seed 0) and (original minus
seed 1) both have intervals excluding 0 on the same side and the smaller of the two is larger in size than the spread. This is
a rule of thumb from one seed pair, not a test: the spread is itself noisy (its interval is reported), and the ablation is
one run, so run-to-run variation of the ablation is not covered at all.

**Global metrics** (existing evaluation code, `docs/vocoders/EVALUATION.md`): UTMOS, Whisper large-v3 WER and CER (corpus
ratio), training-mel L1, multi-resolution STFT distance, PESQ, ECAPA and CAM++ cosine to the recording of the same utterance.

**Voicing measures.** Three kinds, because the first is circular:
1. *Voicing agreement with the input tracker* (the metric previously called voiced/unvoiced decision error, now as agreement =
   1 - error): output and recording both go through the vocoder feature extractor (CREPE with SPARC's periodicity threshold,
   no dither); voiced = periodicity above 0; fraction of frames on which output and recording agree. The same tracker's
   periodicity is the input of the fork model, so this measure favors it.
2. *Voicing agreement with pYIN*: librosa pYIN (fmin 65 Hz, fmax 600 Hz, 64 ms frames, audio resampled to 16 kHz, frames
   centred on the 20 ms grid) run on the recording and on each output; fraction of frames on which the two voiced flags
   agree. pYIN is a different algorithm from CREPE and sees neither model's input. It is also a noisy judge: it calls
   only 87% of the frames of voiced phones voiced in the recordings.
3. *Phone-based voicing*, from the forced alignment (below): the fraction of frames of voiced phones that a tracker calls
   voiced, and of voiceless phones that it calls unvoiced, for each system (the reference being the phone labels, not the
   recording). Voiced phones: vowels, nasals, approximants, voiced fricatives, voiced stops, dʒ. Voiceless phones: voiceless
   stops, voiceless fricatives except h, tʃ. Sonorants (vowels, nasals, approximants) are also reported alone.

F0 error in cents on frames voiced in both, periodicity MAE: input-tracker quantities. **HNR** is my own normalized-autocorrelation
estimator (50 ms Hann window, lags 70-545 Hz, window-corrected, clipped to +-60 dB); the error is output minus recording per 20 ms
frame, reported as the mean absolute error and the signed mean (bias). **High band** is the 4-12 kHz log energy (Hann 1024
frames at 50 Hz), same error definitions. All on the 50 Hz frame grid; mel L1 is the training mel pooled per 20 ms frame.

**Sound classes** come from forced alignment with the Montreal Forced Aligner (`english_mfa` dictionary and acoustic model,
`scripts/spkemb/periodicity_mfa.sh`) of the recordings. Coverage: all 1679 utterances aligned; 94.8% (dev-clean), 95.0%
(test-clean), 99.5% (both VCTK halves) of frames fall in a class; the rest are `spn` (spoken noise / out-of-vocabulary)
labels. Classes by IPA label: vowels (including diphthongs), voiced fricatives (z ʒ v ð), voiceless fricatives (s ʃ f θ ç ...),
**h as its own class** (changed from the first version, which counted it as a voiceless fricative: /h/ is often breathy-voiced
between vowels, so it belongs in neither voicing set; 1126 test-clean frames), stops (p b t d k g and variants), affricates (tʃ,
dʒ), nasals, approximants / liquids (l ɫ ɹ j w ʋ ɾ ...; the flap and syllabic consonants included), silence (empty labels). The
glottal stop has no voicing value and is in no phone-voicing set. A frame belongs to the phone interval that contains its centre
(`(t + 0.5) * 20 ms`). Stop closure and burst are one MFA interval, so I split it by time: the last 40 ms are "burst", the rest
"closure" (a stop under 60 ms is all burst); with 20 ms frames this is a coarse proxy.

**Voicing-transition classes** (4 frames, all within 30 ms of a change), three definitions, all reported:
(a) *input tracker*: a change of the recording's CREPE-periodicity voicing track (as in the first version; runs shorter than 3
frames are absorbed first); (b) *pYIN*: the same on the recording's pYIN track; (c) *phone boundary*: a boundary between two
adjacent phone intervals whose voicing differs (voiced vs voiceless, silence counted as voiceless; an interval without a voicing
value, such as h or spoken noise, breaks the adjacency). Test-clean frames per class: vowels 58,342, silence 21,494, voiceless
fricatives 14,666, nasals 13,380, stop burst 13,361, approximants 12,186, stop closure 11,864, voiced fricatives 8,324,
affricates 1,918, h 1,126; transitions: input tracker 54,647, pYIN 36,778, phone boundary 45,468; voiced phones 100,893, sonorants
83,908, voiceless phones 33,148.

Code and tests: `src/sparc/vocoders/eval/periodicity_analysis.py`, `tests/test_periodicity_analysis.py`. Results, per-frame
arrays, pYIN tracks, alignments, audio and every table: `/data/user_data/xoy/sparc-vocoders-periodicity/` (`summary/summary.md` holds all
tables for all four splits, including the seed-spread tables for every metric and class; `summary/results.json` every estimate with its interval).

## Results

### Global metrics, original SPARC input set minus the fork model, against the seed spread

Fork model (seed 0) value, difference with its interval, the seed spread (seed 1 minus seed 0) with its interval, and whether the
difference exceeds the spread by the rule above. Test splits (validation splits agree in sign and size; see `summary.md`).
The recording itself scores UTMOS 4.246 (test-clean) and 3.960 (VCTK test), WER 1.87% and 1.26%.

| metric | test-clean: fork | original minus fork | seed spread | exceeds | VCTK test: fork | original minus fork | seed spread | exceeds |
|---|---|---|---|---|---|---|---|---|
| UTMOS | 3.997 | -0.050 [-0.076, -0.023]* | -0.016 [-0.036, +0.005] | yes | 3.848 | -0.024 [-0.056, +0.008] | +0.000 [-0.014, +0.015] | no |
| WER (%) | 2.50 | +0.18 [-0.04, +0.42] | +0.12 [-0.14, +0.39] | no | 3.59 | +0.36 [-0.40, +1.20] | +0.67 [-0.10, +1.46] | no |
| CER (%) | 0.93 | +0.09 [-0.05, +0.24] | +0.06 [-0.09, +0.20] | no | 1.79 | +0.17 [-0.25, +0.59] | +0.35 [-0.28, +0.90] | no |
| mel L1 | 0.5086 | +0.0087 [+0.0078, +0.0097]* | -0.0005 [-0.0011, +0.0002] | yes | 0.6736 | +0.0009 [-0.0035, +0.0054] | -0.0064 [-0.0117, -0.0019]* | no |
| multi-res. STFT distance | 1.2044 | +0.0187 [+0.0164, +0.0211]* | -0.0025 [-0.0044, -0.0007]* | yes | 1.4830 | +0.0137 [+0.0059, +0.0242]* | -0.0037 [-0.0076, -0.0001]* | yes |
| PESQ (WB) | 1.887 | -0.096 [-0.111, -0.081]* | +0.015 [-0.000, +0.028] | yes | 1.725 | -0.034 [-0.051, -0.016]* | +0.013 [+0.001, +0.026]* | yes |
| ECAPA similarity to recording | 0.8249 | -0.0035 [-0.0065, -0.0007]* | +0.0132 [+0.0102, +0.0162]* | **no** | 0.6016 | -0.0010 [-0.0078, +0.0065] | +0.0198 [+0.0122, +0.0273]* | no |
| CAM++ similarity to recording | 0.8198 | -0.0102 [-0.0144, -0.0064]* | +0.0145 [+0.0101, +0.0189]* | **no** | 0.5832 | +0.0031 [-0.0055, +0.0118] | +0.0388 [+0.0295, +0.0478]* | no |
| voicing agreement, input tracker (CREPE) | 0.9549 | -0.0290 [-0.0322, -0.0258]* | +0.0003 [-0.0009, +0.0015] | yes | 0.9637 | -0.0301 [-0.0363, -0.0243]* | -0.0008 [-0.0034, +0.0018] | yes |
| voicing agreement, pYIN (independent) | 0.9197 | -0.0085 [-0.0125, -0.0049]* | +0.0010 [-0.0013, +0.0033] | yes | 0.6497 | -0.0267 [-0.0376, -0.0171]* | -0.0062 [-0.0107, -0.0018]* | yes |
| F0 RMSE, both-voiced frames (cents) | 30.49 | +1.65 [+0.19, +2.83]* | -1.84 [-3.15, -0.84]* | no | 29.60 | +1.23 [-1.45, +3.21] | -1.76 [-4.31, -0.11]* | no |
| periodicity MAE | 0.0522 | +0.0246 [+0.0225, +0.0269]* | -0.0004 [-0.0011, +0.0004] | yes | 0.0380 | +0.0258 [+0.0213, +0.0307]* | +0.0007 [-0.0009, +0.0021] | yes |
| HNR error, mean absolute dB | 2.948 | +0.171 [+0.143, +0.199]* | +0.005 [-0.020, +0.029] | yes | 3.398 | +0.179 [+0.137, +0.225]* | +0.019 [-0.013, +0.049] | yes |
| 4-12 kHz error, mean absolute dB | 3.356 | +0.049 [+0.033, +0.065]* | +0.013 [-0.003, +0.029] | yes | 4.932 | +0.098 [+0.036, +0.163]* | +0.020 [-0.040, +0.078] | yes |
| 4-12 kHz error, signed dB | -1.448 | -0.079 [-0.109, -0.048]* | -0.041 [-0.079, -0.003]* | no | -2.818 | -0.278 [-0.381, -0.170]* | -0.220 [-0.316, -0.128]* | no |

Reading it:
- Differences that exceed the seed spread on both test sets: STFT distance, PESQ, voicing agreement (both trackers),
  periodicity MAE, HNR error, 4-12 kHz absolute error. Differences that exceed it on test-clean only: UTMOS, mel L1 (VCTK:
  difference ~0, spread -0.006). Within the seed spread: WER, CER, F0 error, signed 4-12 kHz bias, and **speaker similarity**.
  The earlier statements that ECAPA/CAM++ similarity is slightly lower without periodicity are withdrawn: the second seed of
  the fork model is 0.013 (ECAPA) and 0.015 (CAM++) higher than the first on test-clean (0.02 and 0.04 on VCTK), larger than
  the ablation difference.
- Speaker similarity to the recording varies more between seeds of the same recipe than with the input set; any
  seed-0-only comparison of the similarity metrics at this recipe is not interpretable.
- The VCTK pYIN number (0.65 agreement) is low for all systems because pYIN on the noisy VCTK recordings calls hum and
  background voiced (agreement in silence frames 0.40); the pYIN comparison is only meaningful inside speech frames there (see
  the phone-based rows).

### Independent voicing: with and without the periodicity input

The first version judged voicing with the tracker whose periodicity is the fork model's input. With pYIN the effect of removing
periodicity on test-clean shrinks from -0.029 to -0.0085 [-0.0125, -0.0049] agreement (about 30%; it still exceeds the seed
spread, +0.001), and the phone-based measure shows no loss at all:

| system (test-clean) | pYIN voiced, in voiced phones | pYIN unvoiced, in voiceless phones | input tracker voiced, in voiced phones | input tracker unvoiced, in voiceless phones |
|---|---|---|---|---|
| recording | 0.866 | 0.614 | 0.774 | 0.859 |
| fork, seed 0 | 0.875 | 0.600 | 0.776 | 0.854 |
| original input set | 0.883 | 0.594 | 0.790 | 0.850 |
| original minus fork (interval) | +0.0083 [+0.0022, +0.0139]* | -0.0058 [-0.0131, +0.0014] | +0.0142 [+0.0094, +0.0196]* | -0.0033 [-0.0080, +0.0015] (as unvoiced rate) |
| seed 1 minus seed 0 | -0.0041 [-0.0078, -0.0006]* | -0.0016 [-0.0081, +0.0040] | -0.0002 [-0.0031, +0.0022] | -0.0006 [-0.0037, +0.0024] |

- By phone labels the original-input model is not worse at voicing: voiced phones are called voiced slightly more often
  (+0.8 points pYIN, +1.4 points input tracker; exceeds the seed spread) and voiceless phones unvoiced about as often
  (-0.6 and -0.3 points, intervals span 0). Both systems voice voiced phones more than the recordings' own tracks do (clean
  synthetic voicing against tracker misses on the recording), so a higher voiced rate here is not a defect.
- What remains is a small drop in matching the recording's own voicing track, mostly at voicing changes: pYIN agreement at
  vowels -0.012 [-0.017, -0.008], approximants -0.009, phone boundaries -0.011 [-0.018, -0.004], pYIN-defined changes -0.011
  (exceeds the seed spread against seed 0 but not seed 1), input-tracker-defined changes -0.011; at voiced fricatives, stops and
  nasals the pYIN agreement does not change significantly. Most of the first version's voicing effect (-0.029 vs the input
  tracker, -0.044 at its transitions) was agreement with the tracker that supplies the fork model's input.
- Out of domain (VCTK test) the phone-based rates move the same way and further: voiced phones called voiced +4.7 points
  (pYIN) and +4.5 points (input tracker), voiceless phones unvoiced -1.1 (n.s.) and -1.6 points; the recordings themselves
  are called voiced in only 73% (pYIN) and 67% (input tracker) of voiced-phone frames, so VCTK voicing judgments are the least
  reliable numbers here.
- The mask probe and the other probes are reported next to these (below).

### Per sound class: original SPARC input set minus the fork model, LibriTTS-R test-clean

Cell: difference (original minus fork seed 0); `*` interval excludes 0; `m` marginal (the interval's nearer end is within 25% of
the estimate of 0); `Y` exceeds the seed spread by the rule above (against both seeds), `.` does not. **No multiplicity
correction: 17 classes x 5 metrics here, more than 600 cells over all splits and metrics in `summary.md`; at 5% about one in twenty cells without an effect would
carry a `*`.** Seed-0 fork values per class and all intervals are in `summary/summary.md`.

| class | mel L1 | HNR error, dB | 4-12 kHz error, dB | agreement, input tracker | agreement, pYIN |
|---|---|---|---|---|---|
| all frames | +0.0087* Y | +0.171* Y | +0.049* Y | -0.0290* Y | -0.0085* Y |
| vowels | +0.0119* Y | +0.272* Y | +0.047* . | -0.0346* Y | -0.0124* Y |
| voiced fricatives | +0.0094* Y | +0.087 . | +0.081*m . | -0.0483* Y | -0.0038 . |
| voiceless fricatives (without h) | +0.0036* Y | +0.087* Y | +0.016 . | -0.0139 . | -0.0149* . |
| h | +0.0074 . | +0.052 . | -0.201*m Y | -0.0285* Y | -0.0056 . |
| stops, closure | +0.0062* Y | +0.144* . | +0.023 . | -0.0384* Y | -0.0041 . |
| stops, burst | +0.0074* Y | +0.057 . | +0.017 . | -0.0351* Y | -0.0079 . |
| affricates | +0.0136* Y | +0.208* . | +0.103 . | -0.0417* Y | -0.0215*m . |
| nasals | +0.0098* Y | +0.231*m Y | +0.114* . | -0.0190* Y | +0.0004 . |
| approximants / liquids | +0.0096* Y | +0.236* Y | +0.066* . | -0.0318* Y | -0.0088*m Y |
| silence | +0.0029 . | +0.075*m . | +0.041 . | -0.0017*m Y | -0.0129 . |
| voiced phones (all) | +0.0112* Y | +0.243* Y | +0.061* Y | -0.0358* Y | -0.0089* Y |
| sonorants | +0.0113* Y | +0.263* Y | +0.058* Y | -0.0322* Y | -0.0097* Y |
| voiceless phones | +0.0057* Y | +0.068* Y | +0.031*m Y | -0.0207* Y | -0.0106* Y |
| change, input tracker | +0.0093* Y | +0.259* Y | +0.054* Y | -0.0442* Y | -0.0114* Y |
| change, pYIN | +0.0080* Y | +0.153* Y | +0.042*m Y | -0.0392* Y | -0.0110* . |
| change, phone boundary | +0.0087* Y | +0.156* Y | +0.059* Y | -0.0401* Y | -0.0109* Y |

- mel L1 rises in every sound class (+0.004 to +0.014; silence and h not distinguishable from 0), and every one of these
  exceeds the seed spread (spreads of at most 0.0014 in size, 0.003 in silence). It is the most robust per-class result.
- HNR error rises for the harmonic classes (vowels, nasals, approximants, and the voiced / sonorant / transition groups) by
  0.2-0.3 dB, above the seed spread; the stop, affricate and voiced-fricative cells are within the larger per-class seed spread
  (up to 0.14 dB) or marginal.
- The 4-12 kHz absolute error changes little (+0.02 to +0.11 dB, per class mostly within the seed spread); the one `Y` for h
  (-0.20 dB, marginal, 1126 frames) is probably chance given the number of cells.
- Agreement with the input tracker drops in every class (the shared-tracker effect, above); with pYIN it drops 0.9-1.2
  points in vowels, approximants and the transition classes, 1.5 in voiceless fricatives (within the seed spread) and 2.2 in
  affricates (marginal), and not detectably elsewhere.
- VCTK test (summary.md): mel L1 rises in the same sound classes (vowels +0.0125, voiced fricatives +0.0114, approximants
  +0.0142, voiceless phones +0.0089, all `*`, above the seed spread) but not over all frames (+0.0007), because silence
  frames get better without periodicity (-0.0113*, but seed 1 is 0.018 better than seed 0 there, so silence is within
  noise). HNR error rises in vowels (+0.39 dB) and voiced fricatives (+0.56), above the seed spread; approximants
  (+0.49) do not exceed that cell's spread.

### Reliance probes on the seed-0 fork model (no training), test-clean differences from the unmodified fork model

| metric | periodicity = training mean | voiced/unvoiced mask | smoothed over 9 frames |
|---|---|---|---|
| UTMOS | -0.430* | -0.051* | -0.098* |
| WER (%) | +0.80* | +0.13 | +0.15 |
| mel L1 | +0.1496* (+29%) | +0.0144* | +0.0275* |
| PESQ (WB) | -0.546* | -0.090* | -0.202* |
| ECAPA similarity | -0.0631* | -0.0024* | -0.0160* |
| voicing agreement, input tracker | -0.1202* | +0.0027* | -0.0325* |
| voicing agreement, pYIN | -0.1777* | -0.0018 | -0.0210* |
| F0 RMSE (cents) | +51.8* | +1.8* | +8.6* |
| HNR error, mean absolute dB | +1.92* | +0.22* | +0.43* |
| 4-12 kHz error, signed dB | -1.27* | -0.12* | -0.29* |

Per class (test-clean, mel L1 difference): the training-mean probe hurts silence most (+0.291, from 0.518; the output loses
3.6 dB of 4-12 kHz energy and 2.2 dB of HNR bias there), then voiceless fricatives without h (+0.150, 4-12 kHz energy -2.2 dB),
vowels (+0.128), stop closure (+0.122), approximants (+0.120) and nasals (+0.108); agreement with the input tracker falls by
0.11-0.16 in every sound class except silence (-0.045) (pYIN: by 0.07-0.33, most in silence and voiceless fricatives) and vowel HNR error rises by 2.6 dB.
The mask probe is nearly harmless (largest mel L1 loss +0.021 for vowels and nasals; pYIN agreement unchanged except
voiceless fricatives -0.015 and voiced fricatives -0.010; HNR of voiced fricatives +0.6 dB too high, stops and affricates +0.2
to +0.4 dB). Smoothing mostly hurts at voicing changes (mel L1 +0.038, HNR error +0.59 dB, input-tracker agreement -0.071,
pYIN agreement -0.040) and at vowels, nasals and approximants (mel L1 +0.029 to +0.037), and removes 0.4 to 0.8 dB of 4-12 kHz
energy at stops and fricatives. The probes agree in direction and size on dev-clean and on both VCTK halves (UTMOS: mean
-0.37 to -0.43, mask -0.02 to -0.05, smoothed -0.06 to -0.10; WER on VCTK test +2.2 pts for the mean, -0.2 for the mask, +0.6
for smoothing). Probe differences are far above the seed spread (UTMOS spread 0.016, mel L1 0.0005), except the mask probe's ECAPA
similarity (-0.0024 against a spread of +0.013) and WER.

## What periodicity helps and hurts (revised)

1. **The input tracker's voicing is reproduced better with the periodicity input; phone-level voicing is not.** Against the
   recording's own CREPE-based voicing track the original-input model disagrees on 2.9 points more frames (test-clean,
   exceeding the seed spread), but that tracker is the fork model's input. With pYIN the drop is 0.85 points
   [0.49, 1.25] (still above the seed spread of 0.1), concentrated at vowels, approximants and voicing changes, and against the phone
   labels the original-input model voices voiced phones slightly more often (+0.8 points) and leaves voiceless phones unvoiced
   about as often. So the first version's "frame-precise voiced/unvoiced gate" claim is not supported as stated: what is
   supported is a modest (about 1 point) loss of agreement with an independent tracker around voicing changes.
2. **The cost of removing it is small and mostly in harmonic quality.** Test-clean, exceeding the seed spread: mel L1 +1.7%
   (+0.0087), STFT distance +0.019, PESQ -0.10, HNR error +0.17 dB (vowels, nasals, approximants +0.23 to +0.27 dB), periodicity
   MAE +0.025, UTMOS -0.05 (the spread is -0.016, the difference to seed 1 is -0.034). Not distinguishable from seed noise:
   WER, CER, F0 error, signed high-band bias, ECAPA and CAM++ similarity.
3. **Fricatives and stops are changed least.** Voiceless fricatives (without h): mel L1 +0.0036 (+0.7%), HNR error +0.09 dB, 4-12 kHz
   error not significant; stops +0.006 to +0.007 mel L1, 4-12 kHz error not significant. The training-mean probe lowers the high band of
   voiceless fricatives by 2.2 dB, but a probe changes the input of a model trained with the channel, so it shows reliance, not
   what periodicity contributes to fricative noise; the model trained without periodicity renders them within 0.016 dB of the
   fork model's high-band error (not significant).
4. **Out of domain (VCTK, never seen in training) the benefit does not reach global quality metrics**: UTMOS, mel L1 (all
   frames), WER and speaker similarity show no difference beyond the noise (PESQ and the STFT distance do). Per sound class
   the harmonic and voicing differences remain (above). One VCTK subset, 40 speakers.
5. **What the mask probe does and does not show.** For the model trained with periodicity, replacing the channel by a
   binary voiced/unvoiced mask costs little (UTMOS -0.05, mel L1 +0.014), while the training mean is catastrophic and
   smoothing costs about twice the mask. So that model gets almost everything it takes from the channel from its voicing
   decision and onset/offset timing, not from the graded value. This says nothing about the mechanism inside the network
   (it may just as well read the channel as a loudness-like or silence cue), and nothing about what a model trained with a
   different input would learn. The model trained without periodicity loses about as much as the mask probe on mel L1
   (0.009 against 0.014) and far less than the training-mean probe (0.150).
6. **What it hurts.** Nothing measurable in this experiment, which rests on one ablation run (one seed), two fork runs, and
   one VCTK subset of 40 speakers; silence frames on VCTK were better without periodicity in the seed-0 comparison but that
   difference is within the seed spread there (seed 1 differs from seed 0 by 0.018 in the same cell), so no hurt is shown and
   none is excluded.

## Caveats

- One ablation run and two fork runs (seed 0, seed 1), 100k steps (a quarter of the main recipe), LibriTTS-R training only.
  The seed spread comes from one pair, so it is itself uncertain; the ablation's own run-to-run variation is not measured. Speaker
  similarity varies more between the two fork seeds (0.013-0.039) than with the input set.
- Subsets: 1679 utterances (about 40 speakers per LibriTTS split, 20 per VCTK half), not the full test sets; intervals are
  over those speakers and do not cover the choice of subset. Many metrics, 17 classes, no multiplicity correction; per-class `*`
  cells and marginal ones (`m`) should be read as leads, not findings.
- Voicing: the input-tracker measure is circular (the same tracker's periodicity is the fork model's input) and is kept only as
  "agreement with the input tracker". pYIN is independent of both models but noisy (87% of voiced-phone frames in LibriTTS
  recordings, 73% on VCTK, where it also calls noise voiced), and "voiced rate in voiced phones" is limited by forced-alignment
  quality and by phone-internal devoicing; neither tracker is ground truth. The phone-based sets contain phones with partial
  voicing (voiced stops, voiced fricatives, dʒ).
- The mask probe could not use "F0 > 0": the cached F0 channel is never 0 (unvoiced frames carry filled values), so voicing
  is periodicity > 0 (the extractor's own decision), with 0.802 (mean of voiced dev-clean frames) for voiced frames. The
  probes are out-of-distribution inputs for the trained model; they show reliance, not benefit.
- HNR, the high-band energy and the stop closure/burst split are my own estimators on a 20 ms grid (HNR windows are 50 ms), so
  burst and onset timing are not resolved finer than that; class frames can include coarticulation. Forced alignments come from
  MFA on the recordings (not hand-checked); about 5% of LibriTTS frames are unlabelled noise labels. The class `h` has few
  frames (1126 test-clean, 155 VCTK test).
- The original SPARC also used its own pitch scaling; here only the periodicity channel is removed and nothing else changed.

## Comparability of the runs

Training code at launch (the runs were trained on three different code states, none of which changes the training path at
default settings as far as a read of the diffs shows; no run was repeated on identical code):

| run | launched | code state |
|---|---|---|
| fork, seed 0 (`ablation_100k`) | 2026-10-06 12:21 | commit `fb66135` (12:13); its only change to training-side code is prediction-time skipping in `datamodule.py`, the training/model/loss code is that of `a966a75` |
| fork, seed 1 (`p4_b0_s1`) | 2026-10-21 08:59 | working tree that was committed as `cd29534` (09:45, Phase 4); inferred from its config, which holds `speaker.swap/identity/rel_f0` keys that `aaf1aa0` (10-20) lacks |
| original input set (`periodicity_none_100k`) | 2026-10-29 02:51 | working tree committed as `a8d0d07` (02:51:49, Phase 8 conditioning switches) plus the ablation code |

Composed-config differences (Hydra `config.yaml` of each run; stored checkpoint configs agree):

```
seed 0 -> seed 1 (p4_b0_s1; python -m sparc.vocoders.eval.periodicity_analysis cfgdiff --pair ref,ref_s1):
  experiment_name: ablation_100k -> p4_b0_s1        seed: 0 -> 1
  paths.runs_root, run_dir, predict.output_dir: run location only
  + speaker.{type: ffn, n_refs: null, layer_stats_root: null, hidden_dim: 1024, use_std: true, vae.enabled: false (+3 vae keys),
    tc.weight: 0.0 (+2 keys), mix.p: 0.0 (+2 keys), swap.p: 0.0 (+f0_stats), identity.weight: 0.0 (+grad_probe_every: 0, sv_dir),
    rel_f0.enabled: false}      (keys of later code at their off / default values; absent from the older config)
  + predict.skip_existing: false

seed 0 -> original input set (periodicity_none_100k; ... cfgdiff --pair ref,none):
  vocoder.generator._target_: HiFiGANVocoder -> HiFiGANOriginalInputsVocoder        vocoder.generator.in_channels: 15 -> 14
  experiment_name: ablation_100k -> periodicity_none_100k        (seed 0 in both)
  + vocoder.generator.cond_type: film, cond_sites: residual     + speaker.cond_stats.{f0_range: false, loudness: false}
  + the same speaker.* keys at their off values as above        + predict.skip_existing: false
```

No loss weight, optimizer, schedule, batch size (16), crop (64 frames), discriminator, speaker-network (type ffn, layer l6,
dropout 0.2, p_cross 0.5) or data key differs. Training-path code, `fb66135` against `cd29534` and `a8d0d07`
(`training/module.py`, `training/speaker_terms.py`, `data/dataset.py`, `data/datamodule.py`, `models/speaker.py`,
`models/frontend.py`, `models/hifigan.py`): the later code adds off-by-default options (multi-reference speaker encoder, VAE,
beta-TC, code mixing, swap-in-training, identity loss, speaker-relative F0, conditioning statistics, AdaLN / site options, the
`pitch_mode` log_relative). At defaults: the speaker network is built by `build_speaker` as the same `SpeakerFFN` (same
hyperparameters; `embed` is `forward` for one reference); the generator update adds a zero tensor from `_speaker_terms` and uses the
same mel, adversarial, feature-matching and multi-resolution STFT terms and weights; the swap / mix generators are created only
when their probability is above 0 (no extra random draws); the dataset draws the reference with the same single `rng.random()` +
`_reference` order as before; extra batch keys appear only when a switch is on; the generator's FiLM modules and state-dict keys
are unchanged (`films` empty ModuleList for the other conditioning type, modulation sites created last). The synthesis check
(max sample difference 0 against stored predictions) confirms that the current generator reproduces the old one in the forward
direction; I did not retrain on identical code, so a behavioural difference in the training path that the diff reading missed
cannot be excluded. Environment: the packages (torch and others) and GPU types at each launch were not recorded in the
checkpoints; the three runs ran on different nodes, and training is not bit-reproducible (`trainer.deterministic: false`).

**RNG caveat.** The seed-0 reference and the ablation share seed 0, so they see identical crops, gains and references
(`default_rng([seed, epoch, position])` in `CropDataset`) and the same speaker-network initialization draws, but the
generator's first convolution has a different shape (15 vs 14 input channels), which shifts every later draw of the global
torch RNG (initialization of later modules, dropout), and nondeterministic kernels add more; so the ablation differs from the
reference in the input channel and in initialization / dropout noise, not only in the channel, and the seed-0 pairing gives no
variance reduction that can be assumed. The seed-1 run addresses this: it changes the whole stream (data order, crops, gains,
references, initialization, dropout) with the input set fixed, which makes its difference from seed 0 an estimate of the noise
a run-to-run change introduces; it is one pair, and it is on a third code state.

## Reproduction

```
python -m sparc.vocoders.eval.periodicity_analysis prepare                  # subsets and MFA corpus
scripts/spkemb/periodicity_mfa.sh                                           # alignments
python -m sparc.vocoders.eval.periodicity_analysis synth --systems gt,ref,ref_s1,probe_const,probe_binary,probe_smooth,none --check
python -m sparc.vocoders.eval.periodicity_analysis metrics --systems gt,ref,ref_s1,none,probe_const,probe_binary,probe_smooth
python -m sparc.vocoders.eval.periodicity_analysis pyin                     # pYIN tracks of every system (CPU, librosa)
python -m sparc.vocoders.eval.periodicity_analysis summarize                # classes, bootstrap, seed spread, tables
python -m sparc.vocoders.eval.periodicity_analysis cfgdiff --pair ref,none  # training configs (also ref,ref_s1)
```

Training: `scripts/spkemb/p3_train.sh hifigan periodicity_none_100k` through `slurm-alloc` (`TIME=16:00:00`, restartable),
log `/data/user_data/xoy/spkemb/phase3/logs/p3t_periodicity_none_100k_10976729.out`; the seed-1 run's log is
`p3t_p4_b0_s1_10876882.out` in the same directory.
