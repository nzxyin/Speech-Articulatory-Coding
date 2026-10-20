# Speaker embedding for resynthesis: layer selection and a resynthesis-oriented encoder (plan, 2026-10-20)

## Current state (from the code)

- `SpeakerEncoder` (`src/sparc/spk_encoder.py`): one WavLM Large hidden state, pooled over time with the CREPE
  periodicity as weights, then `SpeakerEncodingLayer` (Linear 1024 -> 1024, GELU, Linear -> 64). The 64-d vector
  conditions every generator block.
- Which layer is pooled is inconsistent: `load_model("en")` / `feature_extraction` pool `hidden_states[0]` (CNN +
  positional conv output, before any transformer layer); `en+` pools `hidden_states[6]` with thresholded periodicity
  (issue #13, item 4). No layer choice is documented as tested.
- The FFN is trained jointly with the vocoder (`training/lightning_module.py`) from cached pooled features
  (`training/prepare_spk_raw.py`), with the reference embedding taken from the **same utterance** that is
  reconstructed. Nothing in that objective stops the embedding from carrying utterance content (it does not need
  to generalize across utterances), and nothing ties it to speaker verification either.
- Speaker similarity of resynthesis so far is judged with ECAPA cosine (`sparc_multihead.py eval`): 0.45 (EMA
  speakers) / 0.81 (LibriTTS-R) for the shipped model.

## Goal

An embedding that makes the vocoder reproduce the speaker's voice (timbre, vocal-tract length, voice quality) from
any reference utterance of that speaker, including short ones and unseen speakers, without carrying the content
(the vocoder already gets EMA, pitch and loudness). Speaker verification (discriminating speakers) is a means at
most, not the target: an SV embedding can discard exactly the spectral detail a vocoder needs and keep cues it
does not need.

## Phase 1 -- layer screen without vocoder training (cheap; started 2026-10-20)

Data: VCTK (109 speakers, 30 utterances each) and LibriTTS-R dev-clean + test-clean (all speakers, up to 30
utterances each). Per utterance: all 25 WavLM Large hidden states pooled (a) with SPARC's periodicity weights and
(b) uniformly over speech frames, plus utterance-level acoustic targets over voiced frames: mean and std of the
80-bin log-mel spectrum, spectral tilt, mean MFCC 1-24 (timbre / vocal-tract proxies), and median F0 (diagnostic
only: pitch reaches the vocoder separately).

Metrics per layer and pooling, probes fitted on training speakers, scored on disjoint speakers:
- **M1 cross-utterance timbre (primary).** Ridge from the embedding of utterance A to the acoustic targets of a
  *different* utterance B of the same speaker; RMSE (in target-std units) and PCC averaged over target
  dimensions. This is what a resynthesis speaker embedding must supply: speaker-level spectral information that
  transfers across content.
- M2 speaker separability: few-shot nearest-centroid speaker identification and Fisher ratio on held-out speakers
  (reported, not used to select: verification-oriented).
- M3 content leakage: predicting an utterance's own deviation from its speaker mean (targets of A minus speaker
  mean) from its embedding minus the speaker's mean embedding. Higher = more content in the embedding.
- M4 short references: M1 and cosine-to-full-utterance embedding with the reference cut to its first 1 s and 2 s.
- M5 pitch confound: how well the embedding predicts median F0 (pitch is given to the vocoder separately).

Decision rule (fixed before running): shortlist the layers within one standard error (over speakers) of the best
M1 RMSE, plus layers 0 and 6 (the two shipped choices); prefer lower M3 and better M4 among ties.

## Phase 2 -- resynthesis test of the shortlisted layers (needs GPUs, no full vocoder training)

Freeze an existing generator and refit only the speaker FFN for each shortlisted layer (mel + multi-resolution STFT
loss; reference from a different utterance of the same speaker than the target segment). Score on unseen speakers
(LibriTTS-R test-clean, held-out VCTK speakers): speaker similarity with at least two SV models that are not used
anywhere in training (ECAPA, WavLM-base-plus-SV, Resemblyzer) so that no single model is optimized for; mel /
MCD distance; WER (content preserved); UTMOS. Plus a voice-conversion swap: speaker embedding from speaker X with
EMA / pitch / loudness from Y; target-speaker similarity vs source-speaker similarity, and WER.
Caveat: the frozen generator was trained with one specific layer's embeddings (layer 0 for `en`), so this ranks
layers by how well they can drive that generator, which favours its own layer. Phase 3 removes that bias.

## Phase 3 -- a speaker encoder trained for resynthesis (vocoder training; scope with the vocoder work first)

- Input: frozen WavLM Large, learned softmax weights over all layers (or the Phase-2 winner), attentive statistics
  pooling (mean + std, initialized from periodicity weights), MLP to a small embedding (64-128 d).
- Objective: train with the vocoder end to end, with the embedding computed from **other utterances** of the same
  speaker (1-3 references, random lengths 1-10 s) than the segment being reconstructed, so only speaker-level
  information is useful. Reconstruction (mel, MR-STFT), adversarial and feature-matching losses as in the current
  recipe. Optional auxiliaries, ablated: a speaker-consistency term (embeddings of one speaker close), a
  content-leakage penalty (adversary predicting phones / EMA from the embedding), embedding dropout.
- Baselines in the same vocoder recipe: (a) shipped definition (layer 0, same-utterance reference), (b) best single
  layer (Phase 2), (c) frozen SV embedding (ECAPA / WavLM-SV) + FFN, (d) the proposed encoder.
- Data: LibriTTS-R train (2000+ speakers) + VCTK for speaker diversity; GLOBE_V2 (accents) and ESD (emotion:
  the embedding should not change with emotion) as robustness tests; unseen-speaker evaluation as in Phase 2.
- Evaluation as Phase 2, plus a small listening test (speaker-similarity MOS / MUSHRA) if the objective metrics
  disagree; that needs the user's decision on listeners.

## Coordination and compute

Phase 1: ~2-4 GPU-h extraction + CPU analysis. Phase 2: ~4 GPU-h per layer (5 layers). Phase 3: several vocoder
trainings (the earlier vocoder runs took days each); to be scoped with the vocoder comparison work
(`vocoder-compare`, issue #19) before starting. All jobs through `slurm-alloc --agent sparc-xlsr-ema`.
