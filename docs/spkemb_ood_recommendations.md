# Out-of-domain speaker similarity: what is known, ranked recommendations, next experiments (2026-10-29, revised after review)

Synthesis of `spkemb_ood_lit.md`, `spkemb_ood_screen.md`, `spkemb_ood_gap.md`, `spkemb_phase7_globe_speakers.md`,
`spkemb_validation_protocol.md`, the Phase 6 decision files (`/data/user_data/xoy/spkemb/phase6/ev*/decision_e1.md`, `decision_e2.md`,
`contrast_e2_vs_e3.md`) and the Phase 1-6 sections of the project notes. No new numbers except the GPU-hour totals read from `sacct`.
Rule throughout: every choice among variants is made on validation splits; test splits report the chosen configuration and the
baseline only (one pre-registered confirmatory comparison may use a test pool, per the protocol).

Cost units: a 100k HiFi-GAN run is 6-9 GPU-hours plus 2-4 for cloning evaluation (8-13 per arm); 400k is 24-34. The speaker-verification
code-fitting study took about 5 GPU-hours in total (validation sweep about 4.0, test runs about 1.0, from `sacct` run times of the 14
sweep and 11 test jobs, one GPU each).

Evidence labels used below: **V** = choice made on validation splits; **T** = judged on test pools (allowed only as a single confirmatory
comparison; earlier phases used test pools for decisions and carry selection optimism).

## 1. What is now known

| Finding | Evidence | Label | Strength |
|---|---|---|---|
| The gap exists without any swap. Own-speaker resynthesis with the standard code reaches 0.69-0.70 of ground-truth similarity on VCTK and GLOBE (ECAPA; CAM++ 0.63-0.71), against 1.00 on LibriTTS-R. | gap diagnosis, standard code | T (diagnostic pools) | solid as a measurement (20 speakers per pool; 40 GLOBE items). What causes it is open: voice, or channel (ground truth is a recording on its own microphone and room, the output goes through a vocoder trained on filtered audio). No control separates the two yet. |
| Fitting the code with a reconstruction loss does nothing, even on the target utterance. That loss is nearly flat in the code. | gap diagnosis, first part | V for steps and rate | solid, but uninformative about headroom |
| Fitting the code to maximize ECAPA similarity through the frozen generator raises similarity on two other verifiers (CAM++, WeSpeaker): VCTK 0.66 -> 0.78-0.81 of ground truth (closure 0.36-0.44). On ECAPA itself, the fitted objective, VCTK reaches 0.97 of ground truth (0.748 against 0.768) while the other two reach only 0.78-0.81. So the code can be moved toward higher verifier scores; whether that is the voice is not shown. **Verifier-measured, not perceptual.** CAM++ and WeSpeaker were the selection criteria (settings were chosen by their mean on validation), so they are not independent judges; all three verifiers are VoxCeleb-trained. The ECAPA-versus-others difference is the signature of partial overfit to the fitted model. | speaker-verification fit | V for settings, T for the one test run | moderate that the code space can move verifier scores; weak as evidence of headroom in voice |
| GLOBE pseudo-speakers, same fit: 0.75 -> 0.89-0.91, closure 0.57-0.64. **Descriptive only**: 8 items, a speaker bootstrap over 8 items, and the scoring references include the references the code was fitted on ("similarity to all references, none held out"), which inflates this row in particular. Not used as support for headroom. | same | T | weak |
| That fit costs naturalness on test: UTMOS -0.12 (LibriTTS-R) and -0.17 (GLOBE) for the speaker-network-input fit, -0.18 (GLOBE) for the 64-d code, -0.08 to -0.10 on VCTK. WER does not rise. The chosen settings have validation UTMOS drops of 0.097 and 0.071 (limit 0.1), so validation understates test (0.097 became 0.17 on GLOBE), and both settings sit at a grid edge. The unconstrained fit therefore already fails a 0.1 UTMOS guard on test before any listening. | same | V and T | solid |
| Part of VCTK's gap remains after the fit (closure 0.36-0.44), but nothing here isolates the generator as the reason. The fit is capped by the 0.1 UTMOS rule and sits at grid edges; fitting toward the target utterance's own embedding is worse on the other verifiers (closure 0.27-0.29) than fitting toward other utterances (0.36-0.44), so the optimizer is not an upper bound; and ground truth is on a different channel than anything the vocoder was trained to produce. Another speaker's code lowers similarity by 0.37-0.44, so the generator is strongly code-sensitive. | same | T | the code-sensitivity result is solid; the "generator limits half the gap" reading is not supported |
| Encoder-side changes tried in the proxy screen did not help on validation: a frame network before pooling -0.028, frozen verification embedding alone -0.070, layer 6 plus verification embedding 0.000, per-corpus centering -0.033; learned all-layer pooling and the VAE were worse in the vocoder (400k). Only channel augmentation passed the proxy rule (+0.006 [+0.002, +0.010], all from GLOBE real speakers; VCTK validation -0.0004; VCTK test +0.004, CI spans 0). | proxy screen (clean re-score, Amendment 2); Phase 3 | V (screen), T (Phase 3) | solid that no large effect exists in this family |
| Encoder-side changes in the vocoder (Phase 6, single seed, 100k, test pools): small gains on VCTK only, below the 0.0266 margin, none on GLOBE or LibriTTS-R. Layer 1 plus layer 6: VCTK held-out verifier +0.0205 [+0.0101, +0.0305], ECAPA +0.0130 [+0.0029, +0.0235], WavLM-SV +0.003 (CI spans 0). Voiced / unvoiced / silence pooling: VCTK held-out verifier +0.0252 [+0.0146, +0.0354], ECAPA +0.0106 [+0.0015, +0.0192], WavLM-SV +0.0051 [+0.0013, +0.0089]; it fails a guard, and against its width-matched control only the held-out verifier gain remains (+0.0064 [+0.0001, +0.0128]). Neither advances. | Phase 6 decision files | T | the gains are real in sign and VCTK-specific, and small: about 0.02 against about 0.12 for the code fit |
| Real GLOBE speaker labels (Phase 7) had no effect: gains within +-0.009, all CIs span 0 (VCTK held-out verifier +0.009 [-0.0005, +0.018]); guards pass on the existing pools, the own-speaker guard fails on the real-speaker pool. | Phase 7 decision files | T (single pre-registered comparison) | solid for this use of labels. Labels were used only to choose reference utterances; a labelled speaker loss is untested. |
| Training data and the recipe are the only changes that moved cloning similarity by more than the margin, each at a guard cost. 100k, against the existing network: up to 8 references +0.028 / +0.044 / +0.034 (LibriTTS-R / VCTK / GLOBE); adding the GLOBE data to that +0.080 on GLOBE (an increment of +0.046 over references alone) but only +0.024 on VCTK and guards fail; the three Phase 5 data variants +0.092 to +0.099 on GLOBE. 400k combination (references, speaker-relative pitch and GLOBE data together, two seeds): +0.056-0.061 / +0.097-0.101 / +0.134-0.135. 100k to 400k alone adds +0.02 to +0.05 per pool for the existing network. | Phases 4-5 | T | solid; each figure belongs to the arm named |

Reading: the code space can move the output toward higher similarity on the verifiers used, and the fitted ECAPA score reaches
ground-truth level on VCTK, but because the other verifiers were selection criteria and ECAPA overshoots them, this is a measured
verifier effect and not yet evidence of audible headroom. Encoder-side feature and pooling changes give small VCTK-only gains; the
larger measured effects came from training scale and diversity. The gap itself is not yet decomposed into voice and channel, so
"the generator limits half of it" and "the encoder is the bottleneck" are both open until the copy-synthesis control below is run.

## 2. Ranked recommendations

Ranked by expected value per GPU-hour given the evidence, not by size of possible gain.

| Rank | Recommendation | For | Against | Cost | Selection under the validation rule |
|---|---|---|---|---|---|
| 1 | **Channel ceiling and input-path controls on the gap** (experiment 3.1): score the real recording passed through a separate mel-to-wave vocoder and through the vocoder's own channel (bandwidth, loudness, filtered-data processing), and an input-path test with a matched control | Reinterprets the 0.66 -> 1.0 gap: how much is channel and not voice. No training; cheap | The answer may be "gap is mostly channel", which lowers the value of everything below; the input-path part needs an alternative feature source | about 1-2 GPU-hours (by analogy to the 1 GPU-hour test run; not measured) | Which channel variant and feature source on validation halves; test halves report the chosen ones |
| 2 | **Constrained code fitting, validation first** (experiment 3.2): a naturalness term added to the existing sweep rows, a third verifier trained on different data, held out from loss and selection | Largest measured similarity gain with no training; constrained rows already exist on validation (below) | Gain is verifier-measured; UTMOS drop above 0.1 on test for the unconstrained fit; needs several reference utterances and a differentiable verifier | about 5 GPU-hours (the previous sweep plus test run, measured); a naturalness term may add backward cost | Grid on validation pools; select on one verifier (CAM++), report on the others |
| 3 | **Listening test of the constrained fit chosen in 3.2 against standard** (experiment 3.3), single confirmatory comparison on test items | Only direct evidence on whether the gain is heard | Needs listeners; few speakers; synthesis for the constrained variant does not yet exist | under 1 GPU-hour synthesis (estimate); human time | Items and instructions fixed before listening; the variant comes from 3.2's validation choice |
| 4 | **Training scale and diversity**: more or more diverse accented VCTK and GLOBE speakers, and how much of the 100k-to-400k gain (+0.02 to +0.05) remains with the combination recipe | The only interventions with test-pool-sized effects (+0.097 to +0.135 at 400k combination); not yet decomposed | Guards fail with the combination (GLOBE UTMOS, LibriTTS-R own-speaker); data availability; 400k costs 24-34 per arm | 100k arm 8-13; 400k 24-34 per arm and seed | Arms and data mixtures chosen on validation pools; test pools report the chosen one against the existing network |
| 5 | **Speaker loss on the embedding with real GLOBE labels (AAM or GE2E)**, encoder only, no gradient through the output | Ranked 3rd in the literature file, which says real labels "unlock" it; Phase 7 used labels only for reference sampling, so this is untested; avoids the earlier output-side loss failure | The encoder memorised its ~212 proxy speakers; verification-style losses discard synthesis cues; contaminates the verifier metrics; Phase 4's identity loss alone gave +0.009 on VCTK | one 100k arm, 8-13 | Loss weight on validation pools; verifier trained on different data plus listening |
| 6 | **Few-shot adaptation of the conditioning layers (AdaSpeech-style)**, non-zero-shot baseline | Shows how much code fitting is worth against a standard few-sentence adaptation of conditioning weights | Not zero-shot; separate deliverable; fine-tuning cost per speaker | not estimated | Number of steps and layers on validation speakers |
| 7 | **Train the encoder to produce fitted codes** (distillation, or a speaker loss through the frozen generator on the code path) | Removes inference cost | Fitted codes for training speakers are not out-of-domain codes: the generator was trained on those speakers and LibriTTS-R (seen) has no gap (standard 0.95 of ground truth, closure undefined). The targets would have to come from unseen-like speakers, for which no data is reserved. The encoder-memorisation failure and the weak identity-loss result apply to the loss variant. | not estimated (earlier figures were unsourced and are withdrawn) | Only after 3.2 and 3.3 are positive and a pool of unseen-like training speakers exists |
| 8 | Conditioning changes (more FiLM sites, cross-attention); frozen verification embedding from a model with different training data; channel augmentation | Phone-dependent timbre; external accent and channel knowledge; the only passing proxy | Same 64-d vector; cross-attention memorises at ~700 h; the embedding was -0.070 alone and 0.000 combined; augmentation gain +0.006, test CI spans 0 | 8-13 per arm; cross-attention 30-60 | Only after 1-3; two seeds, validation pools, paired gain plus guards |

Not recommended: more variants of real-label reference sampling (no effect), per-corpus centering, frame networks, pooling or VAE
changes (all worse), retrieval toward training speakers (shrinks toward training voices), another proxy screen used as a filter for
vocoder runs. Phase 6 shows small VCTK-only encoder gains exist; at about 0.02 they are not worth a 400k run on their own.

Validation rows of the constrained fit that already exist (`sweep_table.md`, 28 validation items; standard code 0.599, chosen
unconstrained 0.681, UTMOS drop limit 0.1): speaker-network input, learning rate 0.03, distance penalty 0.3, reconstruction weight 0.1,
200 steps: 0.660, UTMOS drop 0.040, keeping about 74 % of the validation gain; learning rate 0.03, penalty 0.3, no reconstruction term, 25
steps: 0.661, drop 0.045. So a constrained fit is not new; what is missing is a naturalness term that acts on the output, a test run of
a constrained row, and a verifier not used for selection. "Closure 0.36-0.64" describes the unconstrained fit only.

## 3. Proposed next experiments (at most three)

### 3.1 Channel ceiling and input-path controls on the gap
- **Conditions (own-speaker items of the gap diagnosis; no training):** (i) the real recording passed through a separate mel-input
  vocoder from the vocoder comparison (check the checkpoint is available; otherwise bandwidth limiting and loudness matching to the
  training data) and (ii) the real recording processed to the vocoder's channel. Scored against the same reference sets as the gap
  diagnosis. These give the similarity a perfect voice would reach through such a channel.
- **Input-path part:** own-speaker resynthesis with content features from a different source than the shipped inverter (which is trained
  on MNGU0 only). The alternative inverters tested earlier gave 38.8-94.7 % WER through the shipped vocoder, so any alternative changes
  the vocoder's behaviour for reasons unrelated to speaker leakage. Control: the same shipped inverter with the original speaker's F0
  and loudness replaced by those of another utterance of the same speaker, and with content taken from a different-speaker utterance
  (that isolates how much speaker information the content path carries). A difference is attributed to the source speaker only if the
  matched control does not move.
- **Selection (validation):** which channel variant and which feature source, by similarity on the validation halves; test halves report
  the chosen ones.
- **Decision rule (V, VCTK validation half, 10 speakers):** the gap is mainly channel if the channel-ceiling similarity is at or below
  0.85 of ground truth; mainly voice if it is at or above 0.95; in between the gap is mixed. The input path limits the gap only if the
  alternative features raise similarity by at least 0.25 of the remaining gap with the interval's lower bound above 0 **and** the
  matched control does not.
- **Cost:** about 1-2 GPU-hours (estimate).

### 3.2 Constrained code fitting, validation first
- **Arms:** standard code; the existing constrained rows (listed above) re-run; the same fit with a naturalness term acting on the output
  (design: penalize the distance of the fitted output's pooled WavLM features or its mel statistics to the standard output on the same
  content; weight on the validation grid); the unconstrained fit as reference. Fit on the speaker's other utterances only.
- **Verifiers:** ECAPA in the loss; selection on CAM++ alone; WeSpeaker and a third verifier trained on non-VoxCeleb data (model card to
  be checked; the lit file lists this as unverified) held out from loss and selection and used only for test reporting.
- **Selection (validation):** penalty, naturalness weight, steps and learning rate on LibriTTS-R dev, the VCTK validation half and `globe_spk`
  by CAM++ similarity subject to UTMOS drop <= 0.05 and WER <= +0.5. Report validation and test side by side: validation UTMOS drops
  understate test (0.097 became 0.17 on GLOBE), and the validation GLOBE pool is real speakers (10 items) while the test pool is
  pseudo-speakers (8 items), a different population with a different reference count.
- **Test reporting:** standard against the chosen setting, once, on LibriTTS-R test-clean, the VCTK test half and the GLOBE pseudo-speaker
  half; closure with speaker-bootstrap intervals; GLOBE reported as descriptive.
- **If the guard fails on test:** the result is recorded as a failure; no re-tuning on test. The setting is then judged by the listening
  test (3.3) only as "similarity gain at a naturalness cost", not as a candidate method.
- **Decision rule:** advances if (VCTK closure lower bound >= 0.25 on WeSpeaker and on the non-VoxCeleb verifier) and UTMOS drop <= 0.1 in every
  pool and WER <= +0.5 points and LibriTTS-R own-speaker drop <= 0.01 and swap source-similarity rise <= 0.02. Without the 3.3 listening
  result the gain is reported as verifier-measured, not perceptual.
- **Cost:** about 5 GPU-hours (the previous sweep and test run took about 4.0 and 1.0 measured; the naturalness term may add to it).

### 3.3 Listening test of the constrained fit against standard
- **Arms:** standard code; the constrained fit chosen in 3.2 on validation; a hidden real recording of a different utterance of the
  target speaker as the reference in every trial; the unconstrained fit is not run because it already fails the UTMOS guard on test.
  Synthesis is needed for the constrained variant (the validation sweep stored only parquet rows, no audio, and the existing test
  audio is the unconstrained fit), so the listening items are generated after 3.2's choice, on the test items of the test run.
- **Items and listeners:** items fixed before listening (VCTK test half, 18 speakers; GLOBE pseudo-speaker items as descriptive; LibriTTS-R
  as in-domain control). A pilot with about 5 listeners gives the variance for a power calculation; the main test uses the listener
  number that gives 80 % power for a preference rate of 0.60 against 0.50, with listener and speaker as random effects. This number
  and the margins are written before the main listening starts; none is set here because no variance estimate exists yet.
- **Measure:** blind ratings of similarity to the reference and of naturalness, plus preference.
- **Decision rule (set before listening):** the verifier gain counts as audible only if the similarity preference rate's interval
  excludes 0.5 and naturalness is not rated lower by more than a pre-set margin. Otherwise treat it as verifier-specific and stop
  recommendations 2 and 7.
- **Order:** 3.1 and 3.2 run in parallel (3.2's validation sweep first); 3.3 starts when 3.2 has fixed its validation choice. 3.3 does
  not gate 3.1. Training the encoder to produce fitted codes (recommendation 7) is pre-registered only after 3.2 and 3.3 are positive.

## 4. Open caveats

- All three verifiers are VoxCeleb-trained speaker models, CAM++ and WeSpeaker were used for selecting the fit settings, and UTMOS is one
  naturalness model; no listener evidence yet that any similarity gain is audible.
- The fitted GLOBE row is 8 own-speaker items from ECAPA-clustered pseudo-speakers, with the fitted references inside the scoring set;
  it is descriptive. VCTK has 18-20 speakers per half, so validation margins are noisy (the 0.0266 margin was measured on LibriTTS-R
  test-clean).
- The fitting settings sit at grid edges (200 steps for the code, learning rate 0.03 for the input vector); longer or faster fits were
  not examined.
- Ground-truth similarity is not a ceiling for listeners, and it is on a different channel than anything the vocoder produces; closure
  toward it mixes voice and channel until 3.1 is run.
- Which results were validation-selected and which were test-judged: the proxy screen (Amendments 1-2) and the code-fitting settings were
  selected on validation; Phase 7 was a single pre-registered comparison on test pools; Phases 2-6 arm decisions, including Phase 6, were
  taken on test pools and carry selection optimism; the GLOBE data and the combination figures in section 1 are test pools. Section 1
  labels each row.
- One seed per vocoder arm (Phase 4 combination two seeds); the margin comes from seed differences on the Phase 4 recipe.
- GLOBE contributor hashes may merge or split real speakers; the real-speaker test pool has one reference per speaker and no
  own-speaker items, so the own-speaker guard cannot be evaluated there.
- The generator's content input may carry the original speaker; untested until 3.1.
