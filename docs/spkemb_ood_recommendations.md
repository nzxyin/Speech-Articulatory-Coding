# Out-of-domain speaker similarity: what is known, ranked recommendations, next experiments (2026-10-29)

Synthesis of `spkemb_ood_lit.md`, `spkemb_ood_screen.md`, `spkemb_ood_gap.md`, `spkemb_phase7_globe_speakers.md`,
`spkemb_validation_protocol.md` and the Phase 1-6 sections of the project notes. No new numbers. Rule throughout: every choice
among variants is made on validation splits; test splits report the chosen configuration and the baseline only.
Cost units: a 100k HiFi-GAN run is 6-9 GPU-hours plus 2-4 for cloning evaluation (about 10-13 per arm); 400k is 24-34.

## 1. What is now known

| Finding | Evidence | Strength |
|---|---|---|
| The gap exists without any swap. Own-speaker resynthesis with the standard code reaches 0.69-0.70 of ground-truth similarity on VCTK and GLOBE (ECAPA; CAM++ 0.63-0.71), against 1.00 on LibriTTS-R. | gap diagnosis, standard code | solid (20 speakers per pool; 40 GLOBE items) |
| Fitting the code with a reconstruction loss does nothing, even on the target utterance. That loss is nearly flat in the code. | gap diagnosis, first part | solid, but uninformative about headroom |
| Fitting the code to maximize verifier similarity through the frozen generator raises similarity on verifiers that were not in the loss (CAM++, WeSpeaker): VCTK 0.66 -> 0.78-0.81 of ground truth (closure 0.36-0.44), GLOBE pseudo-speakers 0.75 -> 0.89-0.91 (closure 0.57-0.64, 8 items). The code space has headroom. | speaker-verification fit, test run once (settings chosen on validation) | moderate: all verifiers are VoxCeleb-trained, so part of the transfer may be shared sensitivity, not voice |
| That fit costs naturalness on test: UTMOS -0.12 (LibriTTS-R) and -0.17 (GLOBE) for the input-vector variant, -0.18 (GLOBE) for the 64-d code, -0.08 to -0.10 on VCTK. WER does not rise. The validation selection used a 0.1 UTMOS limit, and both chosen settings sit at a grid edge. | same | solid |
| 56-64 % of VCTK's gap remains even with a verifier-fitted code. The generator or its other inputs (content features, pitch path) limit it. Another speaker's code lowers similarity by 0.37-0.44, so the generator is strongly code-sensitive. | same | moderate |
| Encoder changes did not help: learned all-layer pooling and the VAE were worse; a frame network before pooling -0.028; frozen verification embedding alone -0.070; layer 6 plus verification embedding 0.000; per-corpus centering -0.033 (validation, proxy). Only channel augmentation passed (+0.006 [+0.002, +0.010], all from GLOBE real speakers; VCTK validation -0.0004; VCTK test +0.004, CI spans 0, not confirmed). | encoder screens, clean re-score | solid that nothing large exists in this family |
| Real GLOBE speaker labels (Phase 7) had no effect: gains within +-0.009, all CIs span 0 (VCTK held-out verifier +0.009 [-0.0005, +0.018]); guards pass on the existing pools, the own-speaker guard fails on the real-speaker pool. | Phase 7 decision files | solid (one seed, margin 0.0266) |
| Proxy gains have not transferred to vocoder results in Phases 2, 3 and 6. Data (adding GLOBE: +0.08 to +0.13 ECAPA on GLOBE) and the combination recipe (up to 8 references, speaker-relative pitch) are the only things that moved cloning similarity, at a guard cost (GLOBE UTMOS, LibriTTS-R own-speaker). | phases 2-6 | solid |

Reading: the encoder is a bottleneck (a better code exists in the existing 64-d space), but the encoder-side changes tried so far
do not find it, and the code alone cannot reach ground truth on VCTK. Nothing so far shows that raised verifier similarity is
heard by listeners.

## 2. Ranked recommendations

Ranked by expected value per GPU-hour given the evidence, not by size of possible gain.

| Rank | Recommendation | For | Against | Cost | Selection under the validation rule |
|---|---|---|---|---|---|
| 1 | **Listening test of verifier-fitted vs standard codes** (blind comparison to a real recording, VCTK and GLOBE items) | Every other recommendation depends on whether the verifier gain is voice or shared verifier sensitivity; the UTMOS drop is consistent with either | Needs listeners and a protocol; few items | 0 GPU (synthesis exists, or about 1 GPU-hour); human time | Items and instructions fixed in advance; choose the fitted variant (input vector vs 64-d code) on validation items only; run test items once |
| 2 | **Inference-time speaker-code fitting as a cloning method with a stronger quality constraint** (fit on the speaker's other utterances, never the target) | Largest measured similarity gain with no training; closure 0.36-0.64 on held-out verifiers | UTMOS drop above 0.1 on test; verifier overfit; needs several reference utterances and a differentiable verifier; settings at grid edges | 2-4 GPU-hours for a validation grid with a quality term, plus the test run | Grid over steps, learning rate, distance-to-standard-code weight and a naturalness term; pick by held-out verifier similarity with UTMOS drop <= 0.05 and WER <= +0.5 on validation pools |
| 3 | **Train the encoder to produce such codes**: (a) distill fitted codes (regress onto per-speaker fitted codes, regularized toward the existing code), or (b) a speaker loss through the frozen generator on the code path only | Removes inference cost; a stronger target than the proxy screens; the failed identity loss trained the generator and hurt WER, this does not touch it | The encoder memorised its ~200 training speakers in every earlier screen; (b) trains toward the metric; fitted codes cost naturalness | (a) fitting for a few hundred training speakers 10-20 GPU-hours plus encoder training 3-6 (generator frozen); (b) about 10-15 | Regression weight, distance penalty or loss weight chosen on validation speakers; gain confirmed on held-out verifiers, not the one in the loss; UTMOS and WER guards |
| 4 | **Input-path test**: articulatory features from an inverter not trained on the evaluated speaker in the own-speaker resynthesis condition | Separates the generator's content and pitch inputs from conditioning: the VCTK gap that fitting does not close may come from an input path that carries the original speaker | Needs an inverter variant; diagnostic only; the answer may be "inputs are fine" | 3-5 GPU-hours (extraction and resynthesis, no training); more if an inverter is retrained | Contrast fixed in advance on validation pools; decision rule in 3.3 |
| 5 | **Conditioning changes**: more FiLM sites, then cross-attention to reference frames | Motivated by the remaining VCTK gap and phone-dependent timbre; the site variant is cheap | Same 64-d vector used more, so little if the vector is the limit; cross-attention memorises at ~700 h; proxies never transferred | extra sites 10-13 per arm; cross-attention 30-60 including evaluation | Only after 1-4 show the code and inputs are not the limit; two seeds, validation pools, paired gain plus guards |
| 6 | **Frozen verification embedding from a model whose training data differs from all evaluation verifiers**, as an extra input | External accent and channel knowledge | Alone -0.070 and combined 0.000 in the screen; metric contamination; untested in a vocoder | 10-13 per arm | Held-out similarity on validation; contamination checked by listening (item 1) |
| 7 | **Channel augmentation** (low priority) | Only passing proxy, +0.006 | Small; VCTK validation about 0; test CI spans 0; the gain is from GLOBE | 10-13 per arm | Already selected on validation; a vocoder run would be a confirmation |

Not recommended: more variants of real GLOBE labels (no effect), per-corpus centering, frame networks, pooling or VAE changes (all
worse), retrieval toward training speakers (shrinks toward training voices), another proxy screen used as a filter for vocoder runs.

## 3. Proposed next experiments (at most three)

### 3.1 Listening test of fitted vs standard codes
- **Arms:** standard code; verifier-fitted code (input-vector variant); verifier-fitted code (64-d variant). Same content and
  speaker-relative pitch in all; a real recording of the target speaker (different utterance) is the reference in every trial.
- **Selection (validation):** the fitted variant listeners prefer on validation items (VCTK validation half, `globe_spk` dev,
  LibriTTS-R dev-clean); if neither is preferred, the fitted codes are dropped.
- **Test reporting:** the chosen variant against standard on the test items of the fitting run, once; items fixed before listening.
- **Measure:** blind ratings of similarity to the reference and of naturalness; preference rate with a speaker-level bootstrap.
- **Decision rule:** the verifier gain counts as real only if the fitted preference rate for similarity is above 0.5 with the
  interval excluding 0.5 and naturalness is not rated lower by more than a margin set before listening. Otherwise treat it as
  verifier-specific and stop recommendations 2 and 3.
- **Guards:** UTMOS drop <= 0.1 and WER <= +0.5 points on the same items; LibriTTS-R items as an in-domain control (expect no difference).

### 3.2 Inference-time code fitting with a quality constraint
- **Arms:** standard code; fit with a distance-to-standard-code penalty plus a naturalness term (ECAPA in the loss, generator frozen);
  the earlier unconstrained fit as reference. CAM++ and WeSpeaker are held out for scoring.
- **Selection (validation):** penalty weight, naturalness weight, steps and learning rate on LibriTTS-R dev, the VCTK validation half
  and `globe_spk`, by the mean of CAM++ and WeSpeaker similarity subject to UTMOS drop <= 0.05 and WER <= +0.5. Fit only on the
  speaker's other utterances.
- **Test reporting:** standard against the chosen setting on LibriTTS-R test-clean, the VCTK test half and the GLOBE pseudo-speaker
  test half, once; closure reported as before.
- **Decision rule:** advances if VCTK closure is at least 0.25 with the interval above 0, CAM++ and WeSpeaker agree in sign, and the 3.1
  listening result is positive. Without the listening result the gain is reported as descriptive.
- **Guards:** UTMOS drop <= 0.1 in every pool (the earlier fit exceeded it on GLOBE and LibriTTS-R), WER <= +0.5 points,
  LibriTTS-R own-speaker drop <= 0.01, source similarity rise <= 0.02 on swaps.
- **Cost:** about 2-4 GPU-hours.

### 3.3 Input-path test for the remaining VCTK gap
- **Arms (own-speaker resynthesis, no training):** standard code with standard features; standard code with features from an inverter
  not trained on any evaluated speaker; the verifier-fitted code with those features; the existing fitted-code result as reference.
  Same items as the gap diagnosis.
- **Selection (validation):** which alternative feature source is used, by held-out similarity on the validation halves; the test half
  reports the chosen source.
- **Decision rule:** if the fitted code with alternative features closes at least 0.25 of the gap left by the fitted code alone on VCTK
  validation, with the interval above 0, the input path limits the gap and training on such features replaces recommendation 5.
  Otherwise the generator or conditioning is the limit and recommendation 5 is next.
- **Guards:** UTMOS and WER as above; LibriTTS-R own-speaker as control.
- **Cost:** about 3-5 GPU-hours.

Order: 3.1 first (no GPU; gates 3.2 and recommendation 3), 3.2 and 3.3 in parallel. Training the encoder to produce fitted codes
(recommendation 3) is pre-registered only after 3.1 and 3.2 are positive.

## 4. Open caveats

- All three verifiers are VoxCeleb-trained speaker models, and UTMOS is one naturalness model; no listener evidence yet that any
  similarity gain is audible.
- The fitted GLOBE row is 8 own-speaker items from ECAPA-clustered pseudo-speakers, with references inside the scoring set. VCTK has
  18-20 speakers per half, so validation margins are noisy (the 0.0266 margin was measured on LibriTTS-R test-clean).
- The fitting settings sit at grid edges (200 steps for the code, learning rate 0.03 for the input vector); longer or faster fits were not examined.
- Ground-truth similarity is not a ceiling for listeners; closure toward it is a convenience measure.
- Earlier arm decisions (Phases 2-6) were taken on test pools and carry selection optimism; only Phase 7's single comparison and the
  code-fitting settings followed the validation-only rule.
- One seed per vocoder arm; the margin comes from seed differences on the Phase 4 recipe.
- GLOBE contributor hashes may merge or split real speakers; the real-speaker test pool has one reference per speaker and no
  own-speaker items, so the own-speaker guard cannot be evaluated there.
- The generator's content input may carry the original speaker; untested until 3.3.
