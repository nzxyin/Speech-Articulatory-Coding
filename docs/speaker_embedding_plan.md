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

## Phase 4 -- zero-shot cloning of unseen voices (goal set by the user 2026-10-21; fixed before any Phase 4 number)

Evidence from existing outputs (B0, 100k, LibriTTS-R test.clean; design workflow 2026-10-21):
- **Resynthesis.** Identity is largely kept in resynthesis.
  - Nearest-speaker ID among 39 speakers (ECAPA): ground truth 0.966, T2 0.956, T3 0.974.
  - T3 sim_spk is within 0.01 of the ground-truth leave-one-out level.
  - T3 - T2 = +0.029. The single-reference gap is reference noise, not encoder capacity.
- **Conversion is the gap.**
  - Swap target similarity is 0.648 with F0 scaled, against 0.724 for same-speaker T2.
  - It falls from 0.686 to 0.488 as |ln F0 ratio| grows from < 0.15 to > 0.5.
  - The source keeps 0.34-0.38.
  - The generator never sees a content / speaker mismatch in training.
- **Source features still identify the speaker.** LDA on per-utterance statistics, 246 speakers, chance 0.4 %:
  - EMA mean + std 27.8 %; all features 39 %.
  - Between-speaker spread: EMA means 0.05-0.10 of the global sd, scales 3-8 %; F0 4.7 semitones.
- **Long utterances lose more identity.** The synth - ground-truth gap goes from -0.03 at < 2 s to +0.07 at > 12 s.
  The sim_spk metric is itself duration-confounded: ground truth reaches 0.53 at < 2 s and 0.88 at > 12 s.

Protocol (`eval/clone.py`, to be built):
- **Pools.** Unseen-speaker pools are LibriTTS-R test.clean, VCTK and GLOBE. The last two are required for any
  unseen-voice claim.
- **Swap pairs.** 400 per pool: half same-gender and half cross-gender, stratified by target duration.
  Each target speaker gets 10 references of at least 3 s, all disjoint from the target utterance.
- **Inference variants.** These need no training:
  - K-reference code averaging;
  - target F0 median plus semitone-range matching;
  - loudness matching;
  - EMA mean / std matching to the reference audio's inverter EMA, run as an ablation only.
- **Controls.**
  - F0 / loudness matching with the source speaker's own code. This separates pitch from embedding.
  - A wrong-speaker code.
- **Primary metric.** Swap ECAPA cosine to the target's reference-mean embedding, averaged per target speaker
  first, then over speakers. Reported per pool and per duration stratum.
- **Secondary metrics.**
  - Margin over both controls.
  - Top-1 speaker ID among the pool's speakers.
  - Verification acceptance at the ground-truth EER threshold.
  - WavLM-SV similarity.
  - A held-out third SV model, used in no training loss and no selection. It needs a download; see the
    decisions below.
- **Guards.**
  - T2 / T3 sim_spk, overall and for > 4 s targets.
  - WER: at most +0.5 points over the baseline.
  - UTMOS: at most -0.10.
  - Similarity to the source: at most +0.02.
  - A reference-length curve over 1, 3, 10 and 30 s of references.
  - Noisy references (white noise, 20 dB SNR).
- **Baseline for every trained arm.** B0 plus the best inference variant, chosen on the LibriTTS dev split, never
  on test.

Screen arms (100k HiFi-GAN recipe, one seed; plus a second B0 seed for the noise floor, about 8 GPU-h):
- **S+V.** Swap-in-training plus an SV identity loss.
  - Swap-in-training pairs content j with speaker i's code, scales F0 to i's median and uses the adversarial loss
    only.
  - The identity loss is 1 - cos(WavLM-SV(output crop), WavLM-SV reference embedding).
  - Arm V is trained against WavLM-SV, so its decision uses ECAPA and the held-out SV model only.
- **S.** Swap-in-training without the identity loss, as the ablation of S+V.
- **V.** The identity loss alone, on own-speaker crops.
- **R-F0.** Source F0 as semitones relative to the utterance median. The target speaker's median goes into the
  speaker code input, so pitch level comes only from the target.
- **M.** Training with up to 8 references, evaluated with K = 5.
- Not run: R-EMA (only if the EMA-matching inference variant helps), input perturbation, VAE / beta-TC, reference
  cross-attention, a bigger encoder, new training data.

Decision rule:
- **Noise floor.** The margin m = max(0.02, 2 x the B0 seed-to-seed difference in the primary metric), fixed once
  the second B0 seed is scored, before any arm is looked at.
- **Advancing.** An arm advances if all of these hold:
  - its primary gain over the baseline is >= m, with the speaker-level bootstrap CI excluding 0 in LibriTTS
    test.clean;
  - the gain has the same sign in VCTK and GLOBE, and in the held-out SV model;
  - all guards pass.
- **After the screen.** At most one combination of passing arms, then the winner at 400k with 2 seeds.
- **Success target at 400k.** On duration-matched pairs, swap target similarity of at least 0.93 of the
  ground-truth same-speaker level, and top-1 ID of at least 0.95. Similarity to the source must not exceed the
  ground-truth level between different speakers.

### Phase 4 addendum: training-data diversity arm G (2026-10-21; fixed before any G number)
The B0 day-0 evaluation (100k steps, ECAPA) shows that most of the out-of-domain gap exists without any speaker
swap. Own-speaker resynthesis reaches this fraction of the ground-truth same-speaker level:

| pool | own-speaker | ground truth | fraction |
|---|---|---|---|
| LibriTTS | 0.73 | 0.80 | 0.90 |
| VCTK | 0.52 | 0.80 | 0.65 |
| GLOBE | 0.42 | 0.72 | 0.58 |

Swap verification acceptance is 54-84 % on LibriTTS against 0.3-2 % on VCTK and GLOBE. The lever is training-data
diversity, which the conversion arms above do not address. The user proposed adding GLOBE v3 (2026-10-21).
- Data:
  - LibriTTS-R train (the comparison's filtered 339k) plus the filtered GLOBE v3 train list of the articulatory-tts
    accent work (`/data/user_data/xoy/globe_v3_accent_splits/v1/train.tsv`; 228,606 utterances).
    - Common Voice 21 English, 13 accents, SNR > 25, train capped at 40k per accent.
  - Features re-extracted with this cache's `SparcFeatureExtractor` (refit head, soxr_hq, batch size 1, per-utterance
    CREPE seed).
  - GLOBE has no speaker ids in train, so each GLOBE utterance is its own speaker. Its speaker reference is always
    itself, which is p_cross = 0 for GLOBE.
  - Train utterances whose ECAPA embedding has cosine >= 0.70 to any utterance of the evaluation GLOBE pool are
    dropped, to reduce speaker leakage. Utterance overlap is already 0.
  - Normalization statistics stay the LibriTTS ones.
- Arm G: B0 recipe (`ablation_100k`, l6 FFN, p_cross 0.5, 100k steps), trained on the combined data, uniform over
  utterances.
- Decision rule for G:
  - G advances if its primary swap gain over B0 is >= m with the bootstrap CI excluding 0 in both VCTK (fully out of
    domain) and GLOBE (unseen speakers of a training corpus).
  - LibriTTS must not drop by more than m.
  - The held-out CAM++ model must agree in sign.
  - All guards apply.
  - Variant: the same inference variant for both arms, chosen on the LibriTTS dev pool.
  - VCTK is the cleaner test of the two, because GLOBE becomes partly in-domain. If they disagree, VCTK decides and
    the result is reported as such.
- Compute: GLOBE extraction ~10-25 GPU-h, G ~9 GPU-h. A 400k confirmation of the winner, possibly G combined with a
  passing conversion arm, comes after the screen.

### Phase 4 decisions and amendments (recorded 2026-10-22)

**Settings fixed before any arm was scored:**
- **Held-out SV model:** CAM++ (WeSpeaker VoxCeleb2, pinned, `eval/metrics/campplus.py`). Ground-truth EER 4.2 % on
  LibriTTS-R test.clean.
- **Identity-loss weight:** 2.0, from the gradient probe. At weight 20 the identity / mel gradient ratio was 0.92-1.26;
  the target was 0.05-0.2. No evaluation metric was used.
- **Baseline variant:** `k5_f0_loud`, chosen on the LibriTTS dev pool. The held-out-reference and primary rankings
  agree.
- **Margin:** m = 0.0266, twice the B0 seed-to-seed difference of 0.0133 on the primary metric.

**Implementation deviations:**
- The identity target is the real crop of the speaker whose code was used, not that speaker's reference utterance.
- GLOBE pseudo-speakers have 2-9 references.
- In arm G, GLOBE utterances serve as their own references, so about 70 % of crops use their own utterance as
  reference, against 50 % in B0.
- The T2/T3 sim_spk guard was set at a drop of at most 0.01 per SV model. The builder chose this value; the user
  confirmed it on 2026-10-22, after the screen results.

**Screen outcome (100k):**
- **M:** advances.
- **R-F0:** gains +0.035 to +0.045 in every pool, and the held-out CAM++ model agrees. It fails only the T2/T3 guard
  on GLOBE (-0.011), so it does not advance.
- **G:** the VCTK gain of +0.024 is below m. GLOBE improves by +0.080, with guard failures: GLOBE UTMOS -0.50 and
  LibriTTS CAM++ own-speaker resynthesis -0.030. G does not advance.
- **S:** worse on every primary metric, with WER up 8-34 points.

**Amendment by the user (2026-10-22, after the screen):** train the combination M + R-F0 + G at 400k steps with two
seeds (`p4_mrg_main`, `p4_mrg_main_seed1`). The baselines are the B0 400k runs `main` / `main_seed1`, compared
seed by seed with the same rule, variant and margin.
- This departs from "at most one combination of passing arms": R-F0 and G did not pass. The combination's result
  must therefore be reported as a post-screen choice, not as a confirmed screen winner.
- M alone at 400k was not run.

**Outcome of the amendment** (2026-10-23; `phase4/eval/decision_mrg_400k_s{0,1}.md`):
- **Primary gains:** M + R-F0 + G clears the margin in every pool, seed and SV model. The gains in ECAPA similarity
  to the target, seed 0 / seed 1, are:

  | pool | seed 0 | seed 1 |
  |---|---|---|
  | LibriTTS | +0.061 | +0.056 |
  | VCTK | +0.101 | +0.097 |
  | GLOBE | +0.135 | +0.134 |

  CAM++ gains are of the same size. All CIs exclude 0.
- **Guards:** it fails three in both seeds, so by the rule it does **not** advance.
  - LibriTTS own-speaker T2 drops by 0.016-0.017 (ECAPA) and 0.025-0.027 (CAM++), against a limit of 0.01.
  - GLOBE UTMOS drops by 0.14-0.15, against a limit of 0.10.
  - GLOBE similarity to the source rises by 0.03 (ECAPA) and 0.05 (CAM++), against a limit of 0.02.
- **VCTK:** every guard passes, and UTMOS rises by 0.38-0.42.

Decisions for the user:
- Downloading a held-out SV model (e.g. WeSpeaker ResNet34 or CAM++, ~0.1-0.3 GB).
- The screen compute, about 55 GPU-h, then about 70 GPU-h to confirm the winner.

## Phase 5: three-corpus training data (LibriTTS-R + filtered VCTK + filtered GLOBE)

Fixed 2026-10-24, before any Phase 5 data, feature cache, filter statistic or number exists. Everything not stated here
is the Phase 4 protocol unchanged.

### Motivation
The 400k combination (up to 8 references, speaker-relative pitch, GLOBE training data) gave large cloning gains in
every pool, seed and speaker-verification model (ECAPA gain +0.056 to +0.061 LibriTTS, +0.097 to +0.101 VCTK, +0.134
to +0.135 GLOBE), but failed three guards in both seeds: LibriTTS own-speaker resynthesis similarity, GLOBE UTMOS and
GLOBE similarity to the source. Question: does adding filtered VCTK (real speakers with many utterances each) and/or a
quality filter on GLOBE keep the gains and pass the guards?

### Arms [SUPERSEDED by Amendment 1]
All arms use the combination recipe (up to 8 references, speaker-relative pitch), the 100k-step HiFi-GAN recipe and
seed 0. Training data is sampled in natural corpus proportions, uniform over utterances, with no reweighting.
- **R** (LibriTTS-R plus GLOBE): the Phase 4 combination data (LibriTTS-R train plus the filtered GLOBE v3 train list,
  228,391 utterances). Its 100k run does not exist yet (Phase 4 ran this combination only at 400k) and is trained here.
- **V** (R plus filtered VCTK): R plus the VCTK training set defined below.
- **VQ** (LibriTTS-R plus filtered VCTK plus quality-filtered GLOBE): as V, but GLOBE restricted by the quality filter
  below.
The baseline for every arm is the existing speaker network (the comparison's ablation recipe) at 100k steps, seed 0,
from Phase 4.

### VCTK training set
- **Speakers.** Drop the 40 speakers of the VCTK evaluation pool (`phase4/pools/vctk/table.parquet`). All other VCTK
  speakers are kept. A speaker without transcripts is kept (transcripts are unused).
- **Audio.** Source `/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/wav48`, with the same trimming and resampling
  as the evaluation pool build (`clone_pools.py`), to 24 kHz, and the same feature extraction as the other corpora
  (`SparcFeatureExtractor`, soxr_hq, batch size 1, per-utterance CREPE seed). Normalization statistics stay the
  LibriTTS ones.
- **Duration.** Keep utterances of 1 to 15 s after trimming.
- **Noise.** [SUPERSEDED by Amendment 1] Keep utterances with SNR > 25 dB. The GLOBE filter used the dataset's own precomputed `snr` field
  (`build_globe_v3_manifest.py`, `snr > 25`), which cannot be computed for VCTK, and no estimator code exists in the
  GLOBE work. The estimator is therefore WADA-SNR, applied to the trimmed 24 kHz audio. This differs from the GLOBE
  field, and the two thresholds are not calibrated against each other.
- **Speaker identity.** Each kept VCTK speaker is one training speaker (real speaker ids; references are other
  utterances of the same speaker, p_cross as for LibriTTS-R).
- **Leak check.** Drop every training utterance whose ECAPA embedding has cosine >= 0.70 to any utterance of any of the
  three evaluation pools (LibriTTS-R test.clean, VCTK, GLOBE), as in Phase 4. Counts dropped by each filter are
  reported.

### GLOBE quality filter (arm VQ only) [SUPERSEDED by Amendment 1]
- Score every utterance of the GLOBE training list with the UTMOS predictor used in the Phase 4 evaluation, and a fixed
  random sample of 5000 LibriTTS-R training utterances (numpy seed 0, drawn from the packed train list).
- Threshold: the 25th percentile of the LibriTTS-R sample's UTMOS. Keep GLOBE utterances with UTMOS >= this threshold.
- The ECAPA leak filter against the GLOBE pool still applies. Arm R uses the unfiltered list.
- The threshold and kept count are recorded before any arm is trained or scored.

### Decision rule
- **Per arm vs baseline:** the Phase 4 rule, unchanged. Primary metric: swap similarity to the target's reference-mean
  embedding (ECAPA, WavLM-SV, held-out CAM++), speaker-level bootstrap, inference variant `k5_f0_loud`, margin
  m = 0.0266. An arm advances if its primary gain is >= m with the CI excluding 0 on LibriTTS-R test.clean, the gain
  has the same sign in VCTK and GLOBE and in CAM++, and all guards pass.
- **Guards (Phase 4):** own-speaker T2/T3 similarity drop at most 0.01 per SV model; UTMOS drop at most 0.10 in every
  pool; similarity to the source rise at most 0.02; WER at most +0.5 points; reference-length curve and noisy-reference
  checks reported as before.
- **Caveat for VCTK:** the evaluation pool has no training speaker in any arm, but V and VQ train on other VCTK
  speakers, so VCTK is in-domain for them in recording conditions. It is no longer a fully out-of-domain test for V and
  VQ, and LibriTTS-R and GLOBE results carry that weight; this is reported next to the VCTK numbers.
- **Descriptive contrasts (no decision):** V vs R and VQ vs V, paired by speaker within each pool, for the primary
  metrics and every guard quantity.
- **Advancing:** the best arm that passes every guard (best by mean ECAPA gain over the three pools) goes to 400k with
  two seeds, each paired with the existing network at 400k of the same seed (Phase 4 `main`, `main_seed1`), with the
  same rule. If no arm passes every guard, the results are reported and the phase stops, with no combination or
  rescue arm.
- Outputs: `/data/user_data/xoy/spkemb/phase5/`.

### Phase 5 Amendment 1 (2026-10-24, user direction: stricter filtering for a vocoder)

Written before any Phase 5 data, statistic or number exists. It REPLACES the arms (R / V / VQ), the VCTK "Noise" bullet
and the "GLOBE quality filter" section above (marked superseded there; kept for the record). Everything else in Phase 5
(decision rule, margin m = 0.0266, guards, pools, inference variant `k5_f0_loud`, leak check, outputs) is unchanged,
except where stated here. Reason: the earlier filters were a single SNR threshold (GLOBE's shipped `snr > 25`, WADA-SNR
for VCTK, not calibrated against each other) and one UTMOS cut on GLOBE only. A vocoder learns from the audio itself, so
noise, reverberation, narrow bandwidth, clipping and unnatural or mis-transcribed speech all matter.

**Measures.** One implementation, computed by us on every corpus: LibriTTS-R training splits, the GLOBE training list
(228,391 utterances) and VCTK non-eval speakers; 16 kHz input where the model needs it.
- (a) Noise: WADA-SNR and the Squim SI-SDR estimate (`torchaudio.pipelines.SQUIM_OBJECTIVE`).
- (b) Reverberation: Brouhaha C50 if it installs in the venv without network access at run time; otherwise Squim STOI
  as a proxy. The report states which was used.
- (c) Naturalness: UTMOS (the Phase 4 predictor) and the Squim reference-free PESQ estimate. True PESQ is intrusive and
  needs a clean reference, which corpus audio lacks.
- (d) Bandwidth: effective bandwidth = the highest frequency below which 99% of the spectral energy lies, on the
  native-rate audio before resampling. Audio upsampled from a lower native rate is flagged.
- (e) Clipping: fraction of samples within 0.1% of full scale.
- (f) WER and CER with Whisper large-v3 against the reference text for LibriTTS-R and VCTK. GLOBE already passed an
  exact-match Whisper check upstream; it is recomputed on a fixed random 5,000-utterance GLOBE sample only, to confirm.

**Threshold procedure (values fixed later).** A calibration report gives each measure's distribution per corpus and per
GLOBE accent, and the retained hours per corpus and accent at candidate thresholds anchored on the 5th, 10th and 25th
percentiles of LibriTTS-R test.clean and of the VCTK eval pool. The user reviews the report. The final thresholds are
written into this section BEFORE any Phase 5 training starts, and are the same for all corpora. Accent guard: any
accent losing more than 50% of its utterances is reported.

**VCTK set.** All VCTK speakers except the 40 eval-pool speakers; same trimming and 24 kHz resampling as the eval pool
build; 1 to 15 s after trimming; ECAPA cosine >= 0.70 leak check against all three eval pools as in Phase 4. The
strict filters above replace the earlier VCTK SNR rule.

**Arms** (100k steps, seed 0, combination recipe: up to 8 references plus speaker-relative pitch, natural corpus
proportions, uniform over utterances):
1. LibriTTS-R plus GLOBE as used so far (reference; no 100k run of this combination exists).
2. LibriTTS-R plus GLOBE, both strictly filtered (isolates the effect of filtering).
3. LibriTTS-R plus GLOBE plus VCTK, all strictly filtered.
The baseline is still the existing speaker network at 100k, seed 0. Descriptive contrasts: arm 2 vs 1 and arm 3 vs 2,
paired by speaker within each pool. "Advancing" applies to the best arm passing every guard, as above.

## Coordination and compute

Phase 1: ~2-4 GPU-h extraction + CPU analysis. Phase 2: ~4 GPU-h per layer (5 layers). Phase 3: several vocoder
trainings (the earlier vocoder runs took days each); to be scoped with the vocoder comparison work
(`vocoder-compare`, issue #19) before starting. All jobs through `slurm-alloc --agent sparc-xlsr-ema`.
