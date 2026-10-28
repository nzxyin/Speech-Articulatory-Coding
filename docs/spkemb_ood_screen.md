# Out-of-domain proxy screen of speaker-encoder ideas: pre-registration (2026-10-28)

Written and committed before any extraction, training or scoring of this screen. It extends the harness of
`docs/spkemb_encoder_screen.md` (same training data, trainer, 6000 steps, objective, 64-d output, seeds 0, 1, 2); read
that document for the protocol. Code: `src/sparc/vocoders/spkscreen/{augment,ood_sv,ood_embed,ood_report}.py` and
`variants/ood.py`. Outputs: `/data/user_data/xoy/spkemb/encoder_screen/{runs,ood}/`.

Caveat that governs use: proxy gains did not transfer to the vocoder in Phases 2, 3 and 6. A variant that passes here is a
candidate for a vocoder run, not a result about the vocoder.

## Question

The out-of-domain gap of the cloning evaluation (own-speaker resynthesis reaches 0.6 of the ground-truth similarity on VCTK
and GLOBE against 0.9 on LibriTTS-R) is partly a property of the speaker code. Which representation changes make the code
better on speakers and channels unlike LibriTTS-R?

## Evaluation sets

All scored with the existing suite on the same trained encoders. No evaluation speaker is in encoder training.

| corpus | set | speakers | utterances | use |
|---|---|---|---|---|
| LibriTTS-R (in domain) | held-out test speakers of the screen, `libri_*` | the Libri subset of the 62 test speakers | up to 30 each | reported |
| VCTK | held-out test speakers of the screen, `vctk:*` | the VCTK subset of the 62 | 30 each | primary |
| GLOBE real speakers | `globe_spk`: the Phase 7 pool of 400 Common Voice contributors from GLOBE dev + test | 400 | 3 each (1,200) | primary |
| GLOBE pseudo-speakers | `globe`: the Phase 4 pool of 400 ECAPA-clustered pseudo-speakers | 400 | 3 each | reported only |

`globe_spk` contributors share nothing with any GLOBE training speaker (Phase 7 build report: overlap 0 with the training
lists). Frame features are extracted with the harness (`cache extract --set globe_spk`, same extractor settings). The
ridge probes of M1 and M3 are fitted on the 126 non-test held-out speakers (LibriTTS-R and VCTK) exactly as in the
screen, and applied to every set. M1 uses the cyclic same-speaker partner; with 3 utterances per GLOBE speaker each
utterance predicts the next one's targets. All metrics per corpus carry a leave-one-speaker-out jackknife SE.

Primary metrics: M1 (RMSE in target-std units, lower is better) and EER (cosine, all same- against all different-speaker
pairs) on VCTK and on GLOBE real speakers. The out-of-domain M1 (OOD M1) is the mean of the VCTK and the GLOBE-real M1.
Secondary: LibriTTS-R M1 / EER, 1 s and 2 s M1, 5-shot identification, F0 predictability, M3 content leakage (all held-out
speakers, as in the screen).

## Variants (hyperparameters fixed here)

All read WavLM Large layer 6 frames and use the shipped `SpeakerFFN` (1024 hidden, dropout 0.2, input z-score fitted in
`setup`) for the final 64-d map unless stated.

1. `ood_gruconv`: frame head, then the baseline periodicity-weighted mean. Frames are z-scored with training-frame
   statistics, then LayerNorm, Linear(1024, 256), GELU, dropout 0.1, bidirectional GRU (128 per direction, packed
   sequences), conv1d 256 to 1024 with kernel 9 (centered, zero-initialised), added to the z-scored frames. Starts from the
   baseline's pooled vector.
2. `ood_gruconv_asp`: the same head, then ECAPA-style channel-wise attentive statistics pooling (bottleneck 128; mean and
   std, 2048-d) instead of the weighted mean.
3. `ood_sv`: frozen WeSpeaker ResNet34 (VoxCeleb2, 256-d; not ECAPA, WavLM-SV or CAM++), computed on the same waveform crop
   as the training batch (one cached frame = 320 samples at 16 kHz), alone into the FFN (256-d input).
4. `ood_sv_l6`: layer-6 periodicity-weighted mean (1024-d) concatenated with the ResNet34 embedding (1280-d input).
5. `ood_center`: the trained baseline encoders (the existing `baseline` runs, seeds 0 to 2), with per-corpus centering of the
   pooled layer-6 vector at embedding time. Exactly: for an evaluation set, each utterance's pooled 1024-d vector (before
   the FFN's z-score) has the mean of the pooled full-utterance vectors of all utterances of its corpus in that set (no
   labels, no speaker information; corpora = LibriTTS-R, VCTK, `globe`, `globe_spk`) subtracted and the training-set mean
   added back; 1 s and 2 s embeddings use the same full-utterance corpus mean. The ridge probe is fitted on centered
   non-test LibriTTS-R and VCTK embeddings in the same way. This is a test-time (transductive, label-free) procedure that
   needs unlabeled reference audio of the target corpus; the training is the baseline's.
6. `ood_aug`: the baseline architecture trained with channel augmentation by re-extraction: one augmented copy of every
   training utterance (`train_aug`, 6,344 utterances; effect drawn per utterance from its seed among additive coloured
   noise at 5 to 25 dB SNR, synthetic reverberation with RT60 0.2 to 0.9 s, band-limiting to 6 / 8 / 12 kHz, and an
   8 kHz mu-law 8-bit telephone channel; random gain -12 to +6 dB; `spkscreen/augment.py`). WavLM, CREPE and periodicity
   are re-extracted from the augmented audio. Clean and augmented utterances of a speaker are one pool (training samples
   either with equal chance); the auxiliary targets of an augmented copy are the clean utterance's. Feature-space
   augmentation was not used.
7. Controls. `ood_width_ctl` for variant 4 (the layer-6 mean concatenated with its own first 256 dimensions: same
   1280-d input width and weight count, no new information). `ood_framemlp` for variants 1 and 2 (the same front end and
   a per-frame MLP, no temporal context, same residual zero-initialisation). Variants 3, 5 and 6 are not wider than the
   baseline, so their control is the baseline.

Training of every trained variant: seeds 0, 1, 2, nothing tuned on any evaluation speaker. The baseline is the existing
`baseline` runs (same code, same seeds).

## Selection rule (fixed)

A variant is **promising** if all three hold:
1. **Beats the baseline on OOD M1:** the seed-paired mean difference (baseline minus variant, positive = variant better)
   of OOD M1 has a speaker-level paired 95% interval excluding 0 and a Holm-adjusted p < 0.05 across the six variants 1 to
   6. The interval comes from the leave-one-speaker-out jackknife SE of the seed-averaged difference in each corpus;
   SE(OOD) = sqrt(SE_VCTK^2 + SE_GLOBE^2) / 2 (disjoint speakers), t interval with (smaller corpus speaker count - 1)
   degrees of freedom.
2. **Beats its width-matched control:** the same OOD M1 difference of control minus variant has an interval excluding 0
   (no Holm).
3. **Does not raise content leakage:** variant M3 minus baseline M3 (PCC, all held-out speakers) is at most 0.05.

The OOD EER difference, per-corpus M1 and EER, LibriTTS-R values and the number of speakers improved are reported but do
not enter the rule. With three seeds and one speaker pool per corpus, "promising" means worth a vocoder run.

## Amendment 1 (2026-10-28): selection on validation speakers only

Written before any score of this screen was computed (no `ood_report` run exists; the trained variants and embeddings were
produced under the pre-registration above and are not changed). Reason: user rule of 2026-10-28, every choice among
variants is made on validation speakers (`docs/spkemb_validation_protocol.md`). This amendment replaces the evaluation
sets' "primary" use and the place where the selection rule is applied; the rule itself (three conditions, Holm over the six
variants, 95% intervals, M3 limit 0.05) is unchanged.

* **Selection split.** The selection rule is computed on the VALIDATION speakers of the screen (`ood_report --split val`):
  LibriTTS-R dev-clean (13), VCTK validation half (19), GLOBE real speakers (400, all validation: no GLOBE test contributor
  has three utterances), GLOBE pseudo-speakers (370). The ridge probes of M1 and M3 are fitted exactly as before on the
  126 non-test held-out speakers (the split restricts what is scored, not the probe's fitting rows). M3 and the other
  pooled statistics are restricted to validation speakers. SE and degrees of freedom follow the same formulas with the
  validation speaker counts (smaller corpus: VCTK, 19 speakers).
* **Test speakers** (LibriTTS-R test-clean 10, VCTK test half 19, GLOBE pseudo-speakers 30; no GLOBE real-speaker test
  set exists) are used only to report (a) the baseline and (b) the variants selected on validation, with the same
  statistics. Because GLOBE real speakers are absent from the test split, the test out-of-domain statistic is VCTK alone
  (the test table says so). A selected variant's test row is descriptive, not a second selection step; a variant that is
  not selected is not promoted by test numbers.
* **Transparency table.** A separate table, labelled "all variants on test speakers, not used for selection", reports every
  variant on the test split. It does not enter any decision.
* The all-speaker report (`ood.md`, no `--split`) is not produced as a decision table.
* The caveat above stands: a promising variant is a candidate for a vocoder run only.

## Results of the Amendment 1 scoring (superseded)

The first validation scoring (probes fitted on every non-test held-out row, centering means over the whole corpus, 19 + 19
screen VCTK speakers) selected channel augmentation only (OOD M1 gain +0.006, Holm p 0.035). It is superseded by the clean
re-scoring below; its tables are kept in `/data/user_data/xoy/spkemb/encoder_screen/ood/summary.md`.

## Amendment 2 (2026-10-28): clean validation scoring (probe-fit speakers, centering statistics, VCTK speakers)

Written and committed before any new scoring job. A review of the Amendment 1 scoring found three places where validation
selection was not clean; the numbers in "Results" above are superseded by the re-scoring below and must not be used for
selection. Nothing is retrained: the trained encoders are unchanged; only `ood_center` is re-embedded (it needs new centering
statistics) and every variant is re-scored. The selection rule (three conditions, Holm over six variants, 95% intervals, M3
limit 0.05) is unchanged.

1. **Probe-fit speakers are disjoint from scored speakers.** The ridge probes of M1, M3 and F0 were fitted on every held-out
   row that was not a Phase 1 test speaker, and that set contained VCTK validation speakers that are then scored. Now the
   screen defines its scored speakers as the validation and test speakers (item 3) and fits every probe on the held-out
   speakers in **neither** split (LibriTTS-R dev/test-clean speakers outside the 13 + 10 scored ones, and VCTK speakers
   outside the 40 cloning speakers, including p280). The probe's target scale (per-channel std) comes from the same fit
   rows. GLOBE: no GLOBE speaker has cached screen features other than the scored pools (`globe_spk`, `globe`), the screen's
   training speakers are LibriTTS-R train-clean-100 only and their embeddings are not stored (re-embedding them would also
   give a probe fitted on the encoder's own training speakers). GLOBE sets are therefore scored with the same probe, fitted on
   the held-out LibriTTS-R / VCTK fit speakers, which contains no GLOBE speaker; the probe is never fitted on a scored speaker.
   A unit test checks that the fit set is disjoint from the validation and the test speakers.
2. **Centering statistics (`ood_center`) never mix splits.** The per-corpus mean used to be taken over every utterance of the
   corpus in the cache set, test speakers included. Now it is taken per (corpus, group) with group in {probe-fit, validation,
   test}: validation speakers are centered with the mean over validation utterances of that corpus only, test speakers with
   the mean over test utterances only, probe-fit speakers with the mean over fit utterances. The design still needs the
   target corpus's own mean (unlabeled audio of the evaluation split); it never crosses splits. GLOBE real speakers are all
   validation; `globe` pseudo-speakers are split by the validation/test lists.
3. **VCTK scored speakers are the 40 cloning-evaluation speakers** (20 validation / 20 test, the cloning protocol's
   gender-balanced split in `docs/spkemb_validation_splits.json`). The previous screen split used 19 + 19 VCTK speakers, 19
   of which are in the vocoder's VCTK training data. The 21 cloning speakers that were Phase 1 probe-fit speakers now move
   into the scored sets and the 20 non-cloning Phase 1 test speakers become probe-fit speakers. Split-file contents are
   unchanged (`screen_scored` in `eval/splits.py` derives the screen's scored speakers from `cloning.vctk` and the
   LibriTTS-R screen lists). Validation SE and degrees of freedom use 20 VCTK speakers (df 19).

Selection on validation (`ood_report --split val`) and the baseline plus selected variants on test (`--split test`) follow
Amendment 1. The all-variants test table is labelled "not used for selection".


## Results of the clean re-scoring (Amendment 2; scored after commit 44abf5e, 2026-10-28 16:48)

Outputs `/data/user_data/xoy/spkemb/encoder_screen/ood_clean/ood_{val,test}.{md,json}` (written 16:51-16:53, after the
amendment commit). `ood_center` re-embedded with split-wise centering means; no encoder retrained.

Validation speakers (LibriTTS-R dev-clean 13, VCTK cloning-validation 20, GLOBE real 400, GLOBE pseudo 370). OOD M1 gain
over the baseline (positive = better), Holm over six variants, gain over the control, M3 change:

| variant | OOD M1 gain [95% CI] | p (Holm) | gain vs control [95% CI] | M3 change | promising |
|---|---|---|---|---|---|
| BiGRU + conv head, mean pooling | -0.0277 [-0.0346, -0.0208] | 0.000 | -0.0132 [-0.0178, -0.0086] | -0.090 | no |
| BiGRU + conv head, attentive pooling | -0.0276 [-0.0342, -0.0210] | 0.000 | -0.0131 [-0.0201, -0.0062] | -0.107 | no |
| frozen speaker-verification embedding alone | -0.0699 [-0.0862, -0.0535] | 0.000 | (baseline) | -0.181 | no |
| layer-6 mean + verification embedding | -0.0004 [-0.0053, +0.0044] | 0.860 | -0.0058 [-0.0119, +0.0004] | -0.015 | no |
| per-corpus centering | -0.0328 [-0.0432, -0.0225] | 0.000 | (baseline) | +0.004 | no |
| channel augmentation | +0.0061 [+0.0019, +0.0104] | 0.014 | (baseline) | +0.016 | **yes** |

Selected on validation: channel augmentation only (same conclusion as Amendment 1, now with clean fitting and statistics).

Test speakers (LibriTTS-R test-clean 10, VCTK cloning-test 20, GLOBE pseudo 30; no GLOBE real-speaker test set, so the test
OOD statistic is VCTK alone). Baseline and the selected variant:

| system | LibriTTS-R M1 | VCTK M1 | GLOBE pseudo M1 (descriptive) | VCTK EER % | M3 |
|---|---|---|---|---|---|
| baseline | 0.877 | 0.909 | 1.515 | 4.8 | 0.381 |
| channel augmentation | 0.876 | 0.904 | 1.494 | 4.6 | 0.396 |

Confirmation of augmentation on test VCTK: +0.0044 [-0.0052, +0.0140], not confirmed.

All variants on test speakers, **not used for selection** (VCTK M1 / GLOBE pseudo M1 / M3; OOD gain = VCTK gain, 95% CI):

| variant | VCTK M1 | GLOBE pseudo M1 | M3 | OOD M1 gain [95% CI] |
|---|---|---|---|---|
| BiGRU + conv head, mean pooling | 0.951 | 1.540 | 0.285 | -0.0424 [-0.0587, -0.0261] |
| BiGRU + conv head, attentive pooling | 0.944 | 1.526 | 0.263 | -0.0356 [-0.0486, -0.0225] |
| frozen speaker-verification embedding alone | 0.950 | 1.557 | 0.188 | -0.0415 [-0.0672, -0.0157] |
| layer-6 mean + verification embedding | 0.911 | 1.507 | 0.364 | -0.0024 [-0.0097, +0.0049] |
| per-corpus centering | 0.966 | 1.521 | 0.383 | -0.0571 [-0.0829, -0.0314] |
| channel augmentation | 0.904 | 1.494 | 0.396 | +0.0044 [-0.0052, +0.0140] |

Change from Amendment 1: the augmentation gain on validation is +0.0061 (before +0.0060); per-corpus centering is worse
(-0.033, was -0.045) with split-wise means. The proxy caveat stands: at most a candidate for a vocoder run.
