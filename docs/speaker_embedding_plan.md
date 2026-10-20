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

### Phase 3 scoping (2026-10-20, read-only review)
- Build on branch `vocoder-comparison` (24 kHz HiFi-GAN / DDSP / Vocos; `src/sparc/vocoders/`), not the legacy
  `training/` loop: it already has a packed LibriTTS-R cache (339k utterances, ~2,299 speakers, 542 h) with pooled
  `spk_l0`, `spk_l6`, `spk_enplus64` per utterance, a cross-utterance reference sampler (`CropDataset`,
  `training_reference`, `p_cross` default 0.5, prefers another chapter and >= 3 s), T1 / T2 / T3 evaluation (own
  utterance / other utterance / speaker mean) and a preemption-safe module (`VocoderGANModule`, `SpeakerFFN` in
  `models/speaker.py`, called in `synthesize()`).
- The problem is already visible there: HiFi-GAN test-clean ECAPA similarity 0.857 with the own-utterance vector
  (T1) vs 0.724 with another utterance's (T2) and 0.756 with the speaker mean (T3); T1 is optimistic because the
  embedding carries content. A `p_cross = 0` ablation (100k steps, 1 run) improved T1 and worsened T2.
- Integration: a config switch for the speaker encoder (`ffn` | `layerpool` | `asp`), `p_cross = 1` with 1-3
  references, cached all-layer pooled mean+std (~34 GB fp16 for 339k utterances) for the cheap arms, frozen WavLM
  inside the module for attentive pooling. Judge only on T2, T3 and a speaker-swap condition, with >= 2 SV models.
- Compute (estimates): 5 arms x 100k steps ~60 GPU-h; 2-3 finalists x 400k ~150-250 GPU-h (+ a second seed); one-time
  caches ~20-40 GPU-h. `vocoder-comparison` and `$SV_ROOT` belong to the vocoder work: changes need agreement first.

### Phase 3 design: an interpolatable speaker space (2026-10-20; fixed before any Phase 3 run)

Requirement: for two speakers A and B, decoding the path z(a) = (1 - a) z_A + a z_B must give natural speech whose
voice moves steadily from A to B. Three things are needed. (1) The region between speakers must be densely used
in training; otherwise the generator sees codes there for the first time at inference. (2) The generator must be
smooth in z. (3) Something must check that a code between speakers decodes to an in-between voice. Reconstruction
losses cannot do (3): no audio exists for an in-between speaker.

How the options compare:
- A deterministic encoder (the current FFN) satisfies none of the three: nothing fills the space between training
  speakers, and the vocoder is free to be arbitrary there.
- VAE / beta-VAE (KL to N(0, I)) gives (1) and (2). The KL packs the codes of all speakers into one dense
  Gaussian, and posterior noise trains the generator on a neighbourhood around each code. The KL also limits how
  much information the code carries, which removes utterance content (Phase 1: content leakage is high in the
  shallow layers that carry timbre best). Risk: posterior collapse, since the generator gets EMA, pitch and
  loudness anyway. We use a capacity-controlled KL (Burgess et al. 2018: gamma |KL - C| with C ramped up)
  rather than a fixed beta, so the information budget is set explicitly.
- beta-TC-VAE adds a total-correlation penalty that makes the latent dimensions statistically independent. That
  helps if separate controls are wanted (e.g. one axis for vocal-tract length, one for breathiness). It does not
  by itself make the space interpolatable, and unsupervised disentanglement is not identifiable without inductive
  biases (Locatello et al. 2019). Run as an ablation only.
- Interpolation training addresses (3) directly:
  - Decode mixed codes z_mix = (1 - a) z_A + a z_B (two speakers in the batch, a ~ U(0, 1)) and prior samples
    z ~ N(0, I).
  - These have no target audio, so they get only the adversarial loss: realism, which removes quality dips
    along the path.
  - A latent-cycle loss ties the voice to the code: the encoding of the output must move by the same vector as
    the code, E(G(x, z_mix)) - E(x_real) = z_mix - z_A, computed with frozen WavLM on the crop.
- Pitch: the Phase 1 embedding predicts median F0 almost perfectly (PCC 0.98), while pitch reaches the vocoder
  separately. Along interpolation paths the source F0 contour is therefore scaled geometrically toward the target,
  f0 x (F_B / F_A)^a. Paths are also scored without the pitch scaling, as a check.

Encoder (all arms see the same inputs):
- Per reference utterance, the cached periodicity-weighted mean and std of all 25 WavLM Large layers.
- Learned softmax layer weights (separately for mean and std), then a fixed z-score, then an MLP 2048 -> 1024 -> 64.
- K = 1-3 references, all from other utterances of the speaker (p_cross = 1).
  - Deterministic arms average the K outputs.
  - VAE arms output (mu, log var) per reference and combine them as a product of Gaussians that includes the
    N(0, I) prior. The posterior then gets sharper with more references, which turns K into an explicit
    "enrolment" knob.
- Evaluation uses the posterior mean.
- The VAE terms involve only the encoder, which runs on cached statistics. They are therefore computed on a side
  batch of 256 random training references per step, much larger than the 16-crop vocoder batch, so the KL and the
  total-correlation estimate are not limited by the vocoder batch size.

Arms (HiFi-GAN, experiment ablation_100k recipe of `vocoder-comparison`: 100k generator steps, 16 x 64-frame
crops, one seed each; generator, losses and data unchanged):
| arm | speaker input | references | latent | interpolation training |
|---|---|---|---|---|
| B0 | l6 FFN (existing ablation_100k run, p_cross 0.5) | 1, own or other | deterministic | - |
| B1 | l6 FFN | 1-3 other | deterministic | - |
| B2 | layer pool, mean + std | 1-3 other | deterministic | - |
| B3 | layer pool | 1-3 other | VAE, capacity C -> 32 nats, PoG | - |
| B4 | layer pool | 1-3 other | as B3 | mixup + prior samples (25 % of the crops), adversarial only |
| B5 | layer pool | 1-3 other | as B3 | as B4 + latent cycle |
| B6 | layer pool | 1-3 other | as B3 + beta-TC (beta_TC = 4) | - |

Evaluation (unseen speakers: LibriTTS-R test-clean; held-out VCTK speakers):
- Speaker similarity with ECAPA and WavLM-base-plus-SV. Neither is used in training.
- Conditions T2 (another utterance) and T3 (speaker mean of the other utterances), plus a speaker swap.
- WER (whisper-large-v3), UTMOS, mel distance.
- Interpolation, on 40 unseen-speaker pairs (20 same-gender, 20 cross-gender):
  - 9 points a = 0, 1/8, ..., 1, on the content of a third speaker's utterance.
  - Each output is scored against both endpoint speakers' reference means, giving d(a) = SIM_B - SIM_A per SV model.
  - Monotonicity: per-pair Spearman of d with a, and the fraction of increasing steps.
  - Evenness: the largest step |d(a+1/8) - d(a)| divided by the range d(1) - d(0); 1/8 is ideal.
  - Range: d(1) - d(0).
  - Crossing: |a* - 0.5|, where d(a*) = 0.
  - Quality dip: the minimum UTMOS along the path minus the mean of its endpoints, plus the midpoint WER.
  - Deterministic arms are interpolated linearly in their embedding, in the same way.
- Prior samples (VAE arms), 200 draws:
  - Naturalness (UTMOS), WER, diversity (mean pairwise 1 - SIM, compared with that of real test speakers) and
    novelty (the highest SIM to any training speaker).

Pre-registered contrasts and decisions (paired over the test speakers, with a speaker-level bootstrap; one seed,
so these contrasts screen arms rather than establish effects):
- S1, B1 vs B0 (training on other-utterance references): T2 SIM.
- S2, B2 vs B1 (layer pool + std vs l6 FFN): T2 SIM.
- S3, B3 vs B2 (VAE): T2 SIM may drop by at most 0.01 (mean of both SV models), and the interpolation score must
  improve. The interpolation score is the mean of monotonicity Spearman, 1 - evenness excess and 1 + quality dip.
- S4, B4 / B5 vs B3 (interpolation training): interpolation score and prior-sample naturalness.
- B6 is reported descriptively.
- Finalists (400k steps x 2 seeds), chosen on these screens:
  - the best deterministic arm by T2 SIM;
  - the best interpolatable arm, i.e. the best interpolation score among arms within 0.01 T2 SIM of the best arm.

## Coordination and compute

Phase 1: ~2-4 GPU-h extraction + CPU analysis. Phase 2: ~4 GPU-h per layer (5 layers). Phase 3: several vocoder
trainings (the earlier vocoder runs took days each); to be scoped with the vocoder comparison work
(`vocoder-compare`, issue #19) before starting. All jobs through `slurm-alloc --agent sparc-xlsr-ema`.
