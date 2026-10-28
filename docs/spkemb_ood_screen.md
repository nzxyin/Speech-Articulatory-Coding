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
