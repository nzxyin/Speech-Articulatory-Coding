# Plan: Lin's concordance correlation coefficient (CCC) as the main AAI metric (2026-11-05)

Fixed before any CCC number was computed. No CCC value on any cached prediction has been looked at; this file defines the
variants, the questions, the tests and the recommendation rule. Anything computed later that is not listed here is labelled
exploratory. Code will go in `src/sparc/compression/ccc.py` (reusing `rigor.load_arm`, `utt_stats`); outputs in
`/data/user_data/xoy/xlsr_ema/analysis_ccc/` (`summary.md`, `results.json`).

## Why the question is not trivial

- CCC = 2 cov(p, y) / (var p + var y + (mean p - mean y)^2) = PCC x C_b, where C_b = 2 s_p s_y / (s_p^2 + s_y^2 + (m_p - m_y)^2)
  is one factor that mixes scale and location agreement. CCC penalizes amplitude shrinkage and offsets; rmse_z rewards some
  shrinkage when predictions are uncertain (MSE-optimal damping); PCC ignores both.
- Held-out prediction / target std ratios are 0.72-0.83. The LoRA + probe blend's rmse_z gain matched one gain on LoRA's
  output (P1); only its PCC gain is beyond shrinkage. A metric that charges for shrinkage may reverse that reading.
- Held-out rmse_z is dominated by per-utterance offsets (bias share 0.39-0.49); rmse_z and centered rmse_z rank arms very
  differently (Kendall tau 0.30). CCC contains the offset term, so it will sit between PCC and rmse_z.
- Downstream evidence is thin (vocoder controls: amplitude shrinkage barely changed WER; error size mattered a lot).

## Data and status

Same 7 leave-one-speaker-out folds, `test_unseen` sentences, z units of the held-out speaker's training-split statistics, 3
seeds for trained arms, as in `docs/rigor_prereg.md`. All arms were already seen on these folds: results are labelled
**selected on the test folds**; no untouched data exist. Arms: every cached arm that `rigor.py analyze` and `tier2` score
(probes per encoder and depth, frozen heads per encoder, LoRA arms, blends, matched LoRA, shrunk-LoRA arms). The arm list is
taken from `results.json` / `tier2.json` so no arm is added or dropped after seeing CCC. Arm- or fold-level selection is never
done on CCC in this study (validation-only rule); CCC is reported for the held-out folds only.

## 1. CCC variants (all per channel, mean over the 12 channels)

Degenerate cases (zero variance of p or y in a unit) are dropped as NaN, as for the project's PCC.

- **CCC_utt (primary candidate):** CCC per utterance and channel from that utterance's frames, mean over channels and
  utterances (the PCC convention, so CCC_utt <= PCC_utt elementwise and C_b_utt = CCC_utt / PCC_utt per element). Includes the
  utterance's own mean offset and amplitude mismatch. Short utterances make the variance terms noisy; the frame-count
  distribution is reported and a sensitivity run drops utterances under 1 s.
- **CCC_spk:** frames of all of a speaker's test utterances concatenated, one CCC per channel, mean over channels. Offsets
  that are constant over the speaker count; per-utterance offsets enter as extra variance. Compared with `pcc_concat`.
- **CCC_spk_centered (companion):** CCC_spk after removing each utterance's mean from p and y. Isolates scale and correlation
  from location.
- **Decomposition, reported for every CCC variant:** CCC = PCC x C_b, with C_b split into
  scale factor C_s = 2 s_p s_y / (s_p^2 + s_y^2) and location factor C_l,
  defined exactly by C_b = C_s x C_l with C_l = (s_p^2 + s_y^2) / (s_p^2 + s_y^2 + (m_p - m_y)^2). The three factors (PCC, C_s,
  C_l) are reported per arm; CCC differences between arms are expressed in log space as log CCC = log PCC + log C_s + log C_l
  so each factor's share of an arm difference adds up.
- Existing standing metrics stay: rmse_z, PCC (per utterance), centered rmse_z, std ratio, bias share.

Unit tests (before any real number): CCC of identical series = 1; CCC(p, y) with p = a y + b matches the closed form and
equals PCC when a = 1, b = 0 is not required but C_b = 1 exactly at a = 1, b = 0; log decomposition sums to log CCC; CCC is
symmetric in (p, y); NaN handling matches `utt_stats`; agreement with a reference implementation on random data.

## 2. Evaluation questions

All fold values are the mean over seeds; fold = held-out speaker; tests as in the rigor pre-registration (exact two-sided
sign-flip over 7 fold differences, hierarchical bootstrap 95% interval, 80%-power detectable effect
(t_0.975,6 + t_0.8,6) x sd(fold differences) / sqrt(7)). No new p-value family is claimed: the contrasts are re-tests.

**Q1. Ranking agreement.** For all cached arms (n arms), compute the arm means (over 7 folds) of rmse_z, centered rmse_z, PCC,
PCC_concat, CCC_utt, CCC_spk, CCC_spk_centered. Kendall tau (tau-b) between CCC_utt and each of rmse_z (sign flipped),
centered rmse_z (sign flipped), PCC; the same for CCC_spk. Also tau within subgroups: probes only, frozen heads only, LoRA
only, arms of a single encoder. Bootstrap CI over folds (resample folds, recompute tau). Reading rule fixed now: tau >= 0.8
= same ranking, 0.5-0.8 = partly, < 0.5 = different ranking. Report the arm pairs whose order flips between CCC_utt and
PCC, and between CCC_utt and rmse_z.

**Q2. Discriminability.** For each pre-registered contrast (P1, P2, P3, T1, T2, N1) and the main contrasts (encoder: WavLM
k12 vs XLS-R 300M k18 / 1B / wav2vec2 large under probe and frozen BiGRU->conv; head: temporal heads vs MLP, BiGRU->conv vs
BiMamba vs causal conv, frozen vs ridge probe; LoRA vs frozen under matched and original pipelines; blend vs LoRA; blend vs
shrunk LoRA): effect, bootstrap 95% interval, folds won, sign-flip p, seed sd and fold sd of the fold differences, and the
effect detectable at 80% power at 7 folds, under rmse_z, PCC, CCC_utt, CCC_spk. Primary discriminability summary: the
signal-to-noise ratio |mean diff| / sd(fold diffs) and the detectable effect expressed in units of the metric's own across-arm
spread (detectable effect / sd of arm means), so metrics with different scales are comparable. Counted: how many of the
contrasts are significant at p < 0.05 (unadjusted) and with a clear sign (interval excludes 0) under each metric. Because
sign-flip p has a floor of 2/128, ties at 7/7 are compared by the standardized effect, not p. Seed-vs-fold variance
components per metric (ratio of seed variance to fold variance) show whether CCC adds seed noise.

**Q3. Which component drives CCC differences.** For each pair of arms in Q2 and for the arm means, report the share of the
CCC difference attributable to correlation, scale (C_s) and location (C_l) in log space. Also report each arm's mean
std ratio and mean utterance offset in units of target sd, the share of between-arm CCC variance explained by PCC alone
(R^2 across arms), and by PCC + std ratio. Question answered: is CCC just PCC with a penalty that is nearly constant across
arms (then it adds nothing), or does scale/location vary enough across arms to change decisions?

**Q4. Shrinkage sweep.** For LoRA arms (original and matched pipelines, each encoder and head cached) and frozen heads and
probes at WavLM k12: multiply predictions (in the speaker's z units, around the speaker's zero = training mean, as in the
rigor shrinkage control) by a gain g on a grid 0.5-1.5 in steps of 0.02, per arm, fold and seed. Report for each metric the
curve and the optimal g: rmse_z optimum (closed form g = sum(p y)/sum(p p)), CCC_utt optimum, CCC_spk optimum, and PCC (flat by
construction, used as a check). Gain choices for the held-out speaker use gains fitted on the other six folds (leave-fold-out,
as in P1); oracle gains are labelled oracle. Reported: (a) disagreement between the rmse_z and CCC optimal gains and the size
of the CCC loss at the rmse_z-optimal gain, and of the rmse_z loss at the CCC-optimal gain; (b) the rule's implication for the
blend: blend vs LoRA x gain under CCC with each metric's own leave-fold-out gain, and under a common gain; (c) whether the
blend's CCC gain over LoRA is beyond what any single gain can produce (the maximum over g of CCC for LoRA vs the blend at
g = 1 and at its own optimum). A gain changes PCC not at all, so any CCC change comes only through C_s and C_l.

**Q5. Re-read of key conclusions under CCC** (each row: direction and size under rmse_z, PCC, CCC_utt, CCC_spk; "conclusion
unchanged / weakened / reversed"): (i) encoder choice, WavLM k12 vs XLS-R 300M k18, 1B, wav2vec2-large (P3 and the encoder
table); (ii) LoRA vs frozen BiGRU->conv at k12 (original pipeline, matched, T1); (iii) sequence heads vs ridge probe and vs
MLP (P2); (iv) blend vs shrunk LoRA (P1); (v) nested depth selection (N1) and the depth sweep optimum if per-depth
predictions are cached; (vi) gender conditioning (A0-A4): per-utterance predictions are needed. The cached
`analysis_gender/*/*.json` hold fold-level summaries; at the start of the analysis, check whether per-utterance predictions
exist for those arms. If they do not, say so and do not recompute them in this study (a rerun would be a separate, costed
decision); report gender conditioning as not assessable under CCC. The same statement applies to any other arm whose per-
utterance predictions are not cached (for instance the speaker-adaptation and data-curve arms in `extra/`): listed as not
assessable, not approximated from summary numbers.

**Q6. Downstream validity for SPARC.** Existing evidence only (no new resynthesis in this study): (a) `sparc_multihead`
outputs: per-system articulatory agreement (rmse_z, PCC, CCC_utt of the multi-speaker heads' outputs mapped to the shipped
space against the shipped head's own output, where aligned arrays are available in `heads.json`/`head_*.npz`) against
USC-TIMIT and LibriTTS-R WER, UTMOSv2 and ECAPA similarity for the nine listed systems; Spearman rank correlation of each
metric with each downstream measure (n is 5-9 systems, so descriptive only, with the caveat stated in the report). (b) The
vocoder controls: shrunk shipped outputs (CCC lower, WER unchanged, 9.9 vs 8.6 %), circularly shifted residual (same error
size, not aligned), oracle MNGU0 mapping: state which metric ordered these systems consistently with WER. A metric that
penalizes shrinkage but cannot predict WER when amplitude is shrunk is a worse fit to the vocoder than the plain error. (c)
What is missing and would be needed to decide on downstream grounds: retraining a vocoder on a multi-speaker inverter's
outputs and comparing arms that differ in scale (shrunk vs unshrunk) and in offset; a perceptual/WER test of
gain-calibrated vs uncalibrated outputs from the same head; more than nine systems. This is listed as an open item, not run.

## 3. Reporting rules

- Report the primary variant as fixed above (CCC_utt) with the decomposition; CCC_spk and the centered variant are
  secondary. No variant is chosen after seeing results; if variants disagree, both are shown.
- Seven folds, one corpus family (4 USC, 3 EMA_5EMO), selected on the test folds: every statement is labelled as such.
- Plain names in the write-up: "concordance correlation (per utterance)", "correlation factor", "scale factor",
  "location factor", not the abbreviations B0/S/MRG.
- Results go in `analysis_ccc/summary.md`, the findings in CLAUDE.md under a new heading; the changes in CHANGELOG.md.

## 4. Recommendation rule (fixed now)

Adopt CCC_utt as the **primary** AAI metric (replacing the separate rmse_z and PCC headline pair; rmse_z, PCC, centered
rmse_z and std ratio stay as companions, and the decomposition is always reported) only if **all** hold:

1. **Discriminability:** on the pre-registered contrasts and the main contrasts of Q2, CCC_utt's standardized effect
   (|mean diff| / sd of fold differences) is at least that of PCC on at least as many contrasts as it is lower (ties within
   10% count for neither), and its detectable effect at 7 folds, in units of the across-arm spread, is no larger than PCC's.
2. **Decomposition reported and informative:** the correlation, scale and location factors are reported for every arm; C_b
   varies across arms by more than its fold noise (else CCC adds nothing over PCC and PCC stays primary for simplicity).
3. **No reversal of a downstream-supported conclusion:** CCC_utt does not reverse the direction of a conclusion that has
   downstream support (the vocoder controls and the multi-speaker WER table), and it is not worse than rmse_z at ordering the
   nine `sparc_multihead` systems against WER (Spearman). The blend vs shrunk-LoRA reading may change under CCC; that is
   acceptable only if the Q4 sweep shows the CCC-optimal gain is the better calibrated output for downstream use, which the
   current evidence cannot show. If CCC reverses P1 (blend beats shrunk LoRA under CCC), the statement in the report is that the
   conclusion depends on whether amplitude matters downstream, not that the blend is better.
4. **Stability:** CCC_utt arm ranking Kendall tau with CCC_spk >= 0.8 (the computation choice does not decide the answer), and
   the sensitivity run dropping utterances under 1 s does not change a conclusion.

Otherwise report CCC as a **companion** metric (with the decomposition) next to rmse_z and PCC and keep the current pair as
primary. A mixed outcome (e.g. passes 1, 2, 4 but fails 3 because downstream evidence is missing) resolves to companion, with
the downstream experiment of Q6(c) named as the step that would allow promotion. Whatever the outcome, the user decides
whether the project's metric convention changes; this study only supplies the evidence. Seed sd, fold sd, and the "selected
on the test folds" label accompany every table.

## 5. Execution notes

- Per-utterance statistics come from `rigor.utt_stats` extended with per-utterance mean/variance terms already stored
  (psum, psq, ysum, ysq give p and y means and variances; the cross term needs a new `pysum`). CCC needs no new model runs.
- CPU only; the full re-score of cached predictions (232 MB) is expected to take minutes and may run on a compute node; if
  over about 15 minutes of CPU, submit through `slurm-alloc submit --agent sparc-xlsr-ema --restartable --class eval`.
- Tests: `run_ms.sh -m pytest tests/<ccc test file> -q` (decomposition identities, closed forms, NaN handling).

## 6. Operationalizations and deviations (2026-11-05, compute stage; written before any CCC number on the cached predictions was computed)

Only unit tests on synthetic data had run when this was written.

- Code layout: the CCC functions are in `src/sparc/compression/ccc.py`, the analysis (Q1-Q6) in
  `src/sparc/compression/ccc_analysis.py`, tests in `tests/test_ccc.py`. `rigor.utt_stats` gained the `pysum` key (sum of p*y).
- Arm-level decomposition: the three factors are defined so that their product is exactly the arm's CCC: correlation factor
  A = mean PCC, scale factor B = mean(PCC C_s) / A, location factor C = mean(PCC C_s C_l) / mean(PCC C_s), over the same
  units (utterance x channel for the per-utterance variant, channel for the speaker variants). Plain means of C_s and C_l are
  reported beside them.
- Gain sweep: gains are applied exactly through the sufficient statistics (verified against rescoring the scaled frames in the
  unit tests). The rmse_z gain is the closed form sum(p y) / sum(p p) over all frames and channels of the fitting folds; the CCC_utt and
  CCC_spk gains maximize the fold-mean of that metric on the grid 0.5-1.5 step 0.02 over the six fitting folds (same seed);
  oracle = the grid optimum on the evaluated fold itself.
- Q1 uses every arm of `results.json` (including the gain-derived arms, which have the same correlation by construction) plus
  the matched-pipeline LoRA of `tier2.json`; the other three tier-2 arms duplicate arms of `results.json`. A second Q1 table
  without the gain-derived arms is reported as exploratory (rigor.py's own metric-definition check used that set).
  Subgroups: probes (arm name contains "probe"), frozen heads ("frozen"), LoRA (name contains "LoRA", no gain, no blend, plus the matched
  arm), single encoder (WavLM k12: names starting "W12").
- Q2 bootstrap: hierarchical (folds, seeds, utterances) with 1000 resamples, the same resample used for all four metrics.
  N1 is not assessable under CCC unless per-depth predictions are cached; the check is made at the start of the analysis
  and reported.
- Rule 2 ("C_b varies across arms by more than its fold noise") is operationalized as: the across-arm standard deviation of the
  mean per-utterance C_b exceeds the median paired fold standard error (sd of the fold differences / sqrt 7) of C_b over the Q2
  contrasts, and the paired t interval of the C_b difference excludes zero on at least half of the Q2 contrasts.
- Rule 3: "not worse than rmse_z at ordering the systems against WER" is the Spearman correlation (sign chosen so that a better
  agreement is positive) of each metric with USC-TIMIT corpus WER over the systems that have a computable per-system
  articulatory agreement; a "reversal" is a Q5 row whose direction under CCC_utt is opposite to both rmse_z and PCC and whose
  interval excludes zero, for a conclusion with downstream support.
- Q6 data: the shipped head's weights are read from the cached `wavlm_large-9_cut-10_mngu_linear.pkl` (the "en" checkpoint
  cannot be loaded offline); the per-channel gain/offset maps of the cached heads reproduce the pickle's output std on the training states (checked in
  the run). Per-system articulatory agreement is computed on the USC-TIMIT test utterances for which states are cached (the
  utterances of the WER table), z units of the shipped head's output std per channel. LibriTTS-R states are not cached, so the
  LibriTTS-R WER, UTMOSv2 and ECAPA numbers are related to the agreement measured on USC-TIMIT items (a system property)
  and labelled so.
