# Out-of-domain speaker similarity for the articulatory vocoder: literature and design analysis (2026-10-28)

Scope: nine ideas for raising speaker similarity of the 24 kHz HiFi-GAN articulatory vocoder on out-of-domain speakers
(accented, unseen corpora: VCTK, GLOBE). No experiments were run. Verification: a handful of web searches on 2026-10-28
confirmed the items marked (checked); everything marked (uncertain) is from memory and must be looked up before it is
quoted. Models are described in plain words.

## 1. What our own results constrain

- Gap exists without any swap: own-speaker resynthesis reaches only 0.65 (VCTK) and 0.58 (GLOBE) of the ground-truth
  same-speaker similarity, against 0.90 on LibriTTS. So the embedding (or the vocoder's use of it) fails to represent
  these voices at all; swap-specific problems (pitch, leakage) are secondary.
- Training data is the largest lever so far: adding GLOBE gives +0.08 to +0.13 ECAPA on GLOBE and +0.02 to +0.10 on VCTK;
  up to 8 references and speaker-relative pitch give about +0.03 to +0.04. Filtering raises naturalness but not similarity.
  The remaining failures are guards (GLOBE naturalness, LibriTTS own-speaker drop), not similarity.
- Added capacity or information in the encoder has not transferred: learned all-layer pooling (worse at 400k), VAE
  (worse), and in the encoder screen every frame network, attention, cluster or gate variant was worse than the plain
  mean, because the encoder memorises its roughly 212 proxy training speakers (screen results, "What the screen
  supports", point 3). Proxy gains did not transfer in Phases 2, 3 and 6.
- Identity loss on generated audio (speaker-verification model on the output) failed: no gain alone, and it hurt WER
  (up to +34 points) when combined with swapped pairs.
- GLOBE has no speaker labels (each utterance is its own speaker), so the speaker network has never seen a real
  multi-utterance reference for 114,609 GLOBE training utterances. Phase 7 (`p7_gspk`) addresses this.
- Evaluation models (ECAPA, WavLM-SV, CAM++) are all trained on speaker-verification corpora. Any feature taken from a
  model with similar training data can raise these scores without changing what listeners hear (see idea 2).

Scoring: prior 1-5 for improving out-of-domain similarity at acceptable risk, given the above (5 = most likely).

## 2. Idea by idea

### 1. Temporal BiGRU -> conv head on WavLM frames before pooling (prior 1)
- Prior work: x-vector / ECAPA-style systems use frame-level temporal networks before pooling, but they are trained on
  1-6k speakers with discriminative losses (uncertain on exact corpus sizes). In our setting the AAI result
  (BiGRU -> conv best head) concerned trajectories, a different task.
- Our evidence: the encoder screen already tested frame-level networks before pooling (dilated convolution and a
  two-layer transformer): M1 0.908 and 0.897 against 0.859 plain mean (worse), 1 s references worse still. WavLM layer 6
  is already contextual over neighbouring frames.
- OOD effect: expected negative; more capacity memorises the training speakers.
- Risk: content leakage through temporal structure; cost low (frame cache exists) but not worth a vocoder run.

### 2. Frozen pretrained speaker-verification embeddings as input (prior 3)
- Prior work: YourTTS conditions on a frozen H/ASP speaker encoder; many zero-shot TTS systems use frozen verification
  embeddings (checked: a 2025 study compares H/ASP, x-vector and ECAPA inside YourTTS; no numbers retrieved). Counter-view
  (checked): a 2022 VC paper argues verification embeddings adapt poorly to zero-shot conversion and give weak similarity.
  Mega-TTS and others avoid them partly because they are the metric (checked: Mega-TTS scores with a WavLM-SV model).
- OOD effect: verification networks trained on in-the-wild corpora (VoxCeleb, CN-Celeb, multi-corpus) saw many accents and
  channels, which our LibriTTS-heavy WavLM-layer-6 mean did not. This is the most direct source of out-of-domain speaker
  knowledge we do not have. No accent-labelled head-to-head of WeSpeaker ResNet34/293, ERes2Net, TitaNet was found
  (checked, nothing found); WeSpeaker VoxSRC2023 numbers show ResNet293 best in-domain with CAM++ close (checked).
- Risks: (a) metric contamination: any VoxCeleb-trained embedding correlates with ECAPA, WavLM-SV (VoxCeleb1) and
  probably CAM++, so similarity could rise without listeners hearing it. Prefer a model trained on different data from all
  three (check the model cards: ERes2Net / 3D-Speaker variants are Mandarin- or multi-corpus-trained in some releases,
  uncertain; TitaNet training sets uncertain) and add a listening or MOS check or a metric from another family.
  (b) Verification embeddings carry F0 and channel; swaps may copy source pitch less, or copy reference channel.
  (c) Phase 3 / 6 showed more input information was not helpful, so concatenation alone may be absorbed or ignored.
- Cost: moderate. Embeddings for about 700 h of training data (one forward pass of a small model per utterance), a new
  FFN input width, one 100k vocoder run per candidate model. Resemblyzer (GE2E, LibriSpeech/VoxCeleb, older) is a weak
  candidate; WeSpeaker ResNet34 is a cheap first choice for a test and ResNet293 the strongest.
- Interaction: gains here are the type most likely to be measured by our own metrics, so use the CAM++-held-out rule
  only if CAM++ training data differs from the candidate's.

### 3. Set encoder / attention across reference utterances (prior 2)
- Prior work: neural-process style set encoders and attention over reference sets (Attentron, Choi et al. 2020, fine-grained
  attention over reference frames; uncertain), mean aggregation of per-utterance embeddings as in our Phase 4 (up to 8).
- Our evidence: mean of 3 reference embeddings gave +0.016 M1 at inference in the screen (H13, no further gain), and
  Phase 4 already trains with up to 8. Learned attention is extra capacity before the memorisation bottleneck.
- OOD effect: references of one speaker are near-exchangeable, so weighting adds little; its value would be handling mixed
  or mismatched references (for GLOBE pseudo-speakers) rather than similarity.
- Cost: small. Risk: low. Not worth a run until real GLOBE labels exist (idea 5) and references vary in quality.
  A cheaper variant is duration-weighted frame pooling over all references instead of mean of means.

### 4. Domain robustness: centering/whitening and channel augmentation (prior 3 for each, separately)
- Per-corpus centering: (checked) WeSpeaker ResNet34 cross-domain EER fell from 14.57 % to 11.40 % with mean
  normalization on the CN-Celeb validation setting of VoxSRC2023 track 3; standard back-ends use centering, LDA,
  length normalization and PLDA for domain adaptation (checked). Whitening beyond centering for accented speech: nothing
  found.
- Mechanism here: our FFN z-scores with training-set statistics. An out-of-domain corpus has a shifted mean in this
  space (channel, accent, recording style); subtracting the pool mean (unlabelled, transductive) removes the shift but
  also removes real between-corpus voice differences, and a single unseen reference has no pool. In evaluation the
  pools are whole corpora, so this is testable at inference on existing checkpoints (no retraining). State it as
  transductive when reported.
- Channel/codec augmentation of references only (noise, reverberation, MP3/Opus, bandwidth limits; target stays clean):
  standard in verification (MUSAN and RIR augmentation; uncertain on which systems) and the cure for the embedding
  carrying channel. It matches our own finding that references from the same recording as the target let the vocoder
  copy channel, and it may help the GLOBE case where reference channel differs from vocoder training audio. It does not
  fix a missing accent or voice-type coverage.
- Risk: heavy augmentation reduces timbre cues (high-frequency tilt) that the encoder uses; calibrate to mild.
  Cost: low to moderate (augment the reference cache; 100k run).

### 5. Real speaker labels for GLOBE (prior 4)
- Prior work: multi-utterance references per speaker with other-utterance sampling is standard (YourTTS, VCTK/LibriTTS
  based systems). Our own Phase 3 result: training only on other-utterance references did not improve LibriTTS
  transfer (-0.004 / -0.009 T2), but this was in-domain where labels already existed; GLOBE is the case where the
  reference is currently the target itself, so the vocoder can learn to copy content/channel from it.
- OOD effect: the best-motivated idea, because the largest out-of-domain gain so far came from GLOBE data and its
  references are the weakest part. Phase 7 (`p7_gspk`) tests it, with 13,362 contributors, median 2 utterances,
  5.0 % singletons.
- Risks: the contributor hash is not a person identifier (shared accounts, multiple devices or microphones per contributor,
  uncertain); the largest contributor has 6.9 % of the utterances. Wrong labels add noise but do not leak.
- Cost: already in progress. It also unlocks ideas 3 and 6.

### 6. Speaker loss on the embedding over diverse speakers (GE2E / AAM) (prior 3)
- Prior work: GE2E (Wan et al. 2018) trains the encoder used by SV2TTS; angular-margin softmax / AAM is the verification
  standard (ECAPA, ResNet). Auxiliary speaker-classification or contrastive losses on the conditioning embedding appear in
  VC/TTS systems (uncertain on specific references).
- Difference from our failed loss: the failed loss ran a verification network on generated audio, which trains the
  generator through a model with its own artefacts and interacts with content (WER +8 to +34 points). A discriminative
  loss on the 64-d embedding of the real reference acts on the encoder only, no gradient through the output, no WER
  mechanism. It can also use labels that never touch the vocoder.
- OOD effect: the encoder screen found the encoder memorises roughly 212 speakers. With real labels the speaker count rises
  to about 13k GLOBE contributors plus LibriTTS-R and VCTK speakers (about 2-3k), which is the regime where discriminative
  speaker training works (verification systems use 5-6k+ speakers). The loss pushes the embedding to separate speakers
  across accents and recording conditions.
- Risks: verification embeddings discard cues that synthesis needs (voice quality, breathiness) since only
  between-speaker separability is rewarded; a margin loss can concentrate embeddings on a few directions; it is again
  "metric-like" (trains the property our evaluation models measure), so confirm with listening or a different-family
  metric. Needs idea 5 first; keep weight small and the vocoder reconstruction loss primary.
- Cost: moderate (labels, a classification head, one 100k run).

### 7. Richer conditioning: more FiLM/AdaLN sites, cross-attention to reference frames (prior 3 for cross-attention, 2 for sites)
- Prior work (checked where noted): Seed-VC conditions a diffusion transformer on the whole reference clip instead of one
  vector, with an external timbre shifter on source audio during training to reduce leakage; it reports better similarity
  and lower WER than OpenVoice and CosyVoice (checked, self-reported, single preprint). kNN-VC matches source frames to
  reference frames in WavLM layer-6 space (checked). NaturalSpeech 3 takes timbre from the prompt through a factorized
  codec, Sim-O 0.67 against 0.64 best baseline on LibriSpeech (checked); NaturalSpeech 2, F5-TTS, StyleTTS 2 and
  VALL-E-type systems use prompt context or style vectors (not re-checked; uncertain on details). AdaSpeech uses
  conditional layer norm driven by the speaker embedding (checked); AdaSpeech 4 argues CLN generalises better to new
  speakers than direct embedding input (checked, as summarised).
- Mechanism: timbre depends on the sound (an /s/ differs per speaker in ways a global vector cannot give); attention over
  reference frames, conditioned on the current articulatory/content state, gives phone-conditioned timbre. The Phase 6
  voiced / unvoiced / silence pooling is a coarse version of the same idea (about 12 % better on 1 s proxy M1).
- Our evidence: more information into a pooled vector did not help; cross-attention is far more capacity and exactly the
  kind of change that memorises training speakers and copies reference content or pitch. Those systems train on 10^4-10^5
  hours; we have about 700 h, with the OOD data the weakest part.
- OOD effect: possible large upside on accents (phone-specific timbre), but only with enough diverse speakers; real GLOBE
  labels first.
- Risks: leakage (copying reference prosody, noise), stability of a GAN vocoder with attention, inference cost. More
  FiLM sites alone (e.g., at several vocoder depths) is cheap but just a different use of the same 64-d vector; if the
  vector lacks the voice, extra sites cannot help. Cost: high (architecture change, retrain, new eval variant) for
  cross-attention; low for extra FiLM sites.

### 8. Test-time embedding fitting / speaker adaptation (prior 4)
- Prior work: AdaSpeech fine-tunes conditional-layer-norm parameters and the speaker embedding (about 5k speaker-specific
  parameters) with a few sentences (checked; degrades gracefully, noticeable loss below about 10 sentences). Speaker-embedding
  fitting for new speakers from short untranscribed samples, e.g., Nachmani et al. 2018 (uncertain). No paper on direct
  test-time optimisation of the conditioning vector against a frozen zero-shot model was found (checked, nothing found),
  though it is a standard analysis-by-synthesis / inversion idea.
- Method here: invert the reference audio with the existing articulatory front end (the inverter gives articulation, pitch,
  loudness), resynthesise with a candidate 64-d embedding through the frozen vocoder, and minimise the mel / multi-resolution
  STFT loss to the reference audio (several references, held-out reference for early stopping). Initialise from the
  encoder output. This asks the vocoder which embedding reproduces this voice, instead of asking an encoder that never
  saw such voices.
- OOD effect: addresses the exact failure (own-speaker resynthesis is poor out of domain) without retraining; could
  recover a large part of the gap if the vocoder's embedding space contains the voice. It is testable on the existing
  `p5_fv` and `p7_gspk` checkpoints with the existing cloning evaluation, no training.
- Risks: overfits channel/noise of the references (embedding absorbs it; mitigate with clean-reference choice and
  regularisation towards the encoder output); inversion error in the inverter on accented speech; fit on the target
  reference text/prosody leaks content when evaluating same-utterance; always fit on other utterances than the test ones;
  if the embedding space does not contain the voice, fitting finds a degenerate point (cheap check: fit on training
  speakers and on a held-out LibriTTS voice first). Optimised embeddings may leave the distribution the vocoder saw
  (naturalness drop); constrain with a norm / distance-to-encoder penalty.
- Full fine-tuning of FiLM weights on a few minutes (few-shot): bigger gain potentially, but not zero-shot; separate
  deliverable.
- Cost: low (a few hundred gradient steps per speaker; evaluation 40-400 pairs per pool).

### 9. Nearest-speaker retrieval / kNN-VC-style matching (prior 1-2)
- Prior work (checked): kNN-VC replaces each source frame with the mean of its k=4 nearest reference frames in WavLM-Large
  layer 6 and a vocoder trained to invert those features; it needs a matching set of enough reference audio and degrades
  with short targets (checked; Phoneme Hallucinator addresses it by set expansion). A multi-layer variant (layers 6 and 12)
  exists (checked, from a VoicePrivacy report).
- Applicability: our vocoder takes articulatory features, pitch and loudness plus a speaker vector, not WavLM frames, so
  kNN-VC does not transfer directly. Retrieval variants that do apply: (a) retrieve the nearest training speakers in
  embedding space and blend their embeddings: this shrinks toward the training distribution and can only hurt OOD
  similarity; (b) frame-level reference matching to build phone-conditioned speaker statistics (a data-driven form of
  idea 7).
- Risk: (a) is counter-productive; (b) is the same as cross-attention in disguise. Cost low for (a), high for (b).

## 3. Ranked list

| Rank | Idea | Prior | Why | Cost |
|---|---|---|---|---|
| 1 | 8 Test-time embedding fitting | 4 | no retraining, targets exactly the own-speaker gap, testable on existing checkpoints | low |
| 2 | 5 Real GLOBE speaker labels | 4 | in progress; best-motivated data change; unlocks 3, 6, 7 | in progress |
| 3 | 6 Speaker loss (AAM/GE2E) on the embedding with real labels | 3 | enough speakers after idea 5; avoids our output-side loss failure; metric-contamination risk | moderate |
| 4 | 2 Frozen SV embedding from a non-evaluation model | 3 | external in-the-wild speaker knowledge; contamination of metrics | moderate |
| 5 | 4a Reference channel / codec augmentation | 3 | cheap, targets channel; does not add voice coverage | low |
| 6 | 4b Per-corpus centering at inference | 3 | cheap inference-only test on existing checkpoints; transductive | very low |
| 7 | 7 Cross-attention to reference frames | 3 | highest upside, highest cost and memorisation risk | high |
| 8 | 3 Set encoder across references | 2 | exchangeable references; no gain from mean of 3 | low |
| 9 | 7 Extra FiLM sites alone | 2 | same vector, new use | low |
| 10 | 9 Retrieval / kNN | 1-2 | shrinks toward training voices | low-high |
| 11 | 1 BiGRU -> conv before pooling | 1 | frame networks tested and worse | low |

Suggested order: run 4b and 8 first (inference-only on existing checkpoints), read Phase 7, then pick between 6, 2 and 5-based
combinations for one 100k run each. Any proxy-screen result is a filter only.

## 4. Uncertainties

- Which pretrained SV models share training data with ECAPA / WavLM-SV / CAM++: not verified; read the model cards.
- No head-to-head accent benchmark for WeSpeaker / ERes2Net / TitaNet was found by search.
- The Attentron, Nachmani 2018, MUSAN/RIR-augmentation attributions and the 1-6k speaker training sizes of verification
  systems are from memory.
- Seed-VC and NaturalSpeech 3 numbers are self-reported on other test sets, with no accent evaluation.
- Contributor hashes of Common Voice may merge or split real speakers.
