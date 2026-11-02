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

## Seed-1 replication of arm 1 (pre-registered 2026-11-01, before training)

Arm 1 (MNGU0 k12 inverter) tied the reference on cloning but lowered WER on every validation pool with one seed
(LibriTTS-R dev -0.47, VCTK validation half -0.59, GLOBE dev real speakers -2.80 points; variant `k5_f0_loud`), and
failed only the VCTK own-speaker guard (ECAPA -0.0116, limit 0.01). A second seed checks whether the WER gain is real.

- Run: `p9_mngu0_seed1` (identical to `p9_mngu0` except seed 1), paired with the reference's second seed `p5_fv_seed1`
  (already scored on the validation pools). Same pools, variant, scoring and inverter path as seed 0.
- Primary: WER difference (arm - reference) per validation pool. The WER gain **replicates** if the seed-1 difference is
  negative in all three pools **and** the mean over the two seed pairs is <= -0.5 points (the WER guard size) in at
  least two pools.
- Also reported, not decisive: the full Phase 4/5 and out-of-domain rules for seed 1 (margin 0.0266); whether the VCTK
  own-speaker guard failure recurs; two-seed means of the ECAPA gains and guards; UTMOS.
- No test pools are run. If the WER gain replicates, adopting the k12 inverter for the vocoder input is a separate
  decision (cloning is a tie and the own-speaker guard result decides whether it is safe).
