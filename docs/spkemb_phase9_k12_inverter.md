# Phase 9: vocoder on WavLM k12 inverter features (pre-registered 2026-10-29)

Written before any Phase 9 feature, checkpoint or number exists. User direction (2026-10-29): register a vocoder arm on
the k12 inverters from the speech-to-EMA work, both the single-speaker and the multi-speaker one; the speaker
de-normalization of the multi-speaker inverter is inferred from the speaker embedding.

## Motivation

Every vocoder so far (Phases 3-8) reads articulatory features from SPARC's own path: WavLM Large layer 9 and the
per-utterance MNGU0-refit linear head, trained on one speaker. The speech-to-EMA work recommends WavLM Large truncated to
12 layers with a BiGRU -> conv head for unseen speakers. Phase 8.1 found that a target's own articulatory and pitch
inputs carry part of its identity (another speaker's content lowers similarity by 0.45 of the gap), so the inverter that
produces those inputs may matter for out-of-domain speakers. The shipped vocoder cannot take another inverter's
outputs as they are (38.8-94.7 % WER), so each arm retrains the vocoder on its own inverter's features.

## Arms (100k steps, seed 0, HiFi-GAN, the `p5_fv` recipe: filtered LibriTTS-R + GLOBE + VCTK, up to 8 references,
speaker-relative pitch). Only the 12 articulatory input channels change; pitch, loudness and periodicity stay the
cached ones.

1. **Single-speaker k12 inverter.** Frozen WavLM Large k12 + BiGRU -> conv 9, trained on MNGU0
   (`/data/user_data/xoy/xlsr_ema/final_models/mngu0/frozen_k12_gruconv9_s0.pt`), with its stored layer, shift and
   feature setting; outputs de-normalized to MNGU0 units with its stored target statistics, then normalized for the
   vocoder with statistics from the LibriTTS-R training splits, as the existing features are.
2. **Multi-speaker k12 inverter.** The same architecture trained on all seven EMA speakers
   (`/data/user_data/xoy/xlsr_ema/final_models/all7/frozen_k12_gruconv9_s0.pt`). Its outputs are in each speaker's
   own z units; no EMA statistics exist for a new speaker, so the outputs stay in the normalized space and the
   vocoder learns the de-normalization from the speaker code: a per-channel scale and offset on the 12 articulatory
   input channels predicted from the speaker code, zero-initialized so it starts as the identity. This arm therefore
   bundles the inverter and the learned de-normalization (by design).

Reference: the `p5_fv` run (layer-9 MNGU0-refit linear head features, same data and recipe).

## Selection and decision (validation-only rule)

- Each arm vs `p5_fv` with the Phase 4/5 rule on the validation pools (LibriTTS-R dev-clean, VCTK validation half,
  GLOBE dev real speakers): ECAPA gain >= 0.0266 with the CI excluding 0 on LibriTTS-R, the same sign on VCTK, GLOBE
  and CAM++, and every guard (own-speaker similarity drop <= 0.01 on LibriTTS-R dev and the VCTK validation half;
  UTMOS drop <= 0.10 in every pool; similarity to the source rise <= 0.02; WER <= +0.5 points). Variant `k5_f0_loud`.
- The best advancing arm by mean ECAPA gain runs once on the test pools against `p5_fv`. If no arm passes every guard
  the phase stops; results are reported descriptively.
- Reported, not decisive: own-speaker resynthesis similarity on VCTK and GLOBE (the out-of-domain gap measure of
  `docs/spkemb_ood_gap.md`), WER and UTMOS per pool, and arm 1 vs arm 2 directly.

## Caveats fixed in advance

- Single seed per arm; the margin comes from test-pool seed differences.
- Arm 2 changes two things (inverter and learned de-normalization); a positive result is credited to the pair.
- Both inverters are bidirectional (offline); the features are low-passed and shifted as each inverter was trained.
- The inverters' own accuracy on new speakers is known only from the EMA corpora (leave-one-speaker-out), not on
  LibriTTS-R, VCTK or GLOBE voices.
