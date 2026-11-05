# Concordance correlation as the AAI metric: analysis

Plan and fixed decision rule: `docs/ccc_metric_plan.md` (section 6 holds the operationalizations written before any real CCC
number was computed). Code: `src/sparc/compression/ccc.py`, `ccc_analysis.py`, `ccc_report.py`; tests: `tests/test_ccc.py`.
Full tables (about 840 lines) and `results.json`: `/data/user_data/xoy/xlsr_ema/analysis_ccc/`.

## Answer

**No: keep rmse_z and PCC as the headline pair and report CCC beside them as a companion, always with its three factors.**
The fixed rule gives "companion". The user decides whether the project's metric convention changes.

Why, in plain words:

- CCC (Lin's concordance correlation) is PCC times a scale factor (does the prediction have the target's amplitude?) times a
  location factor (does it have the target's mean?). It is the only one of the three scores that punishes amplitude shrinkage
  and per-utterance offsets together with correlation. That is its value: it shows something neither rmse_z nor PCC shows alone.
- It is not a better ranking tool for the comparisons this project makes. On the contrasts, PCC separates arms more cleanly
  (CCC adds offset noise from the 7 folds), and rmse_z separates them least.
- On the real arms (not the gain-edited ones) CCC ranks arms only partly like rmse_z (Kendall tau 0.39) and more like PCC (0.63).
- The one question where CCC would change a conclusion in a way that matters (is the LoRA + probe blend better than LoRA?) is a
  question about whether output amplitude matters downstream. The existing vocoder evidence is too weak to say, so CCC cannot
  settle it. The experiment that could (calibrated against uncalibrated outputs of one head through a vocoder or WER test) does
  not exist yet.

## Setup

7 leave-one-speaker-out folds (4 USC-TIMIT, 3 EMA_5EMO), `test_unseen` sentences, z units of the held-out speaker's training
statistics, mean over seeds within a fold, then mean over folds. 46 arms were re-scored from the 232 MB of cached predictions
(25 real arms, 21 arms made by applying a gain to LoRA output). The rebuilt arms reproduce `analysis_rigor/results.json` rmse_z
and PCC exactly (maximum difference 0), and an independent re-implementation matched the arm-level numbers to 1e-6.
**Every arm was already selected on these folds; no untouched data exist, and the "oracle" gain rows are fitted on the
evaluated folds (descriptive only).** CCC variants: per utterance and channel, averaged like the project's PCC (CCC utt);
concatenated per speaker (CCC spk); speaker-centered (CCC spk centered). Decomposition: correlation x scale x location, exact
product at arm level.

Not assessable under CCC, and so not covered by this study (no per-utterance predictions cached; nothing was rerun): N1 (nested
depth selection), the gender-conditioning study, and the `extra/` arms (speaker adaptation, data curves, alignment). Their
conclusions are neither confirmed nor changed here.

## Q1. Do the rankings agree?

Kendall tau-b over arm means (bootstrap intervals in the full tables).

| arms | CCC utt vs rmse_z | vs PCC | vs PCC concatenated | CCC utt vs CCC spk |
|---|---|---|---|---|
| 25 real arms (primary) | 0.39 [0.18, 0.72] | 0.63 | 0.79 | 0.81 |
| all 46 arms | 0.08 [-0.30, 0.55] | 0.24 | about 0.45 | 0.92 |
| probes (5) | 1.00 | 0.80 | 0.60 | 1.00 |
| frozen heads (5) | 0.80 | 0.60 | 0.60 | 0.60 |
| LoRA (8) | 0.71 | 0.57 | 0.57 | 0.79 |

Read the 25-arm row as the result. The 46-arm values are pulled down by the 21 gain-edited arms, which share PCC with their
parent LoRA arm by construction and differ only in scale; they are a stress test, not a ranking of models. So for real
models the agreement with rmse_z is partial to weak (0.39), and within the small groups it is partial to strong (0.6-1.0,
but n is 5-8 and the intervals are wide). The concatenated-PCC value for 46 arms is tie-sensitive (0.447-0.458 by float
tie handling); read it as about 0.45. Order flips between CCC utt and PCC: 375 of 1035 arm pairs (all arms). The two CCC
computations (per utterance, per speaker) agree, so that choice does not decide the answer.

## Q2/Q3. What CCC measures beyond PCC

Over the 25 real arms CCC utt is about PCC times a nearly constant factor: CCC/PCC 0.78-0.84 (sd 0.014); PCC alone explains
61 percent of the CCC variance, and adding the std ratio adds almost nothing (0.618). Over all 46 arms the ratio spans
0.67-0.84; there R^2 on PCC alone is 0.04 and rises to 0.92 with the std ratio, but that jump comes from the gain-edited arms.

The loss from PCC to CCC for unshrunk arms:

- scale factor about 0.94-0.95 (plain mean 0.942 over the real arms);
- location factor about 0.83 (plain mean 0.829, range 0.80-0.85; the PCC-weighted factor in the table is 0.83-0.88).
  The location factor is the larger loss. It is the per-utterance offset (about 0.6 target sd) that is the same quantity as
  rmse_z's bias share (39-49 percent). Part of it is not model error: an utterance whose target mean differs from the speaker's
  training mean is penalized for any regressing predictor, as in rmse_z. So CCC utt = PCC x scale x offset, and the offset part
  is what rmse_z already weighs heavily. Per-speaker CCC has a location factor of 0.95-0.98, confirming that most of the loss
  is utterance-level offset, not speaker-level.
- Arms with better correlation (sequence heads, LoRA) tend to have worse location factors (P2 location -0.014, LoRA vs
  frozen -0.025), so the location term partly cancels PCC gains rather than adding independent information. The C_b
  (scale x location) difference between arms is larger than its paired fold noise on 52 percent of contrasts (rule 2 passes).

For the metric variant that isolates amplitude and correlation without the offset component, use CCC spk centered (it is in
every table).

## Q4. CCC and rmse_z want opposite gains

Applying a gain g to the prediction (around the speaker's zero) leaves PCC unchanged by definition. Held-out, over the 13
LoRA, frozen and probe arms:

| | rmse_z-optimal gain | CCC utt-optimal gain | CCC spk-optimal gain |
|---|---|---|---|
| closed form, pooled frames | 0.62-0.80 (mean 0.69) | | |
| grid on the reported metric | 0.68-0.86 (mean 0.77) | 1.10-1.24 | 1.14-1.30 |

Per-arm optima are the numbers above (probe about 0.8 / 1.24; frozen heads 0.72-0.73 / 1.16-1.18; LoRA 0.62-0.69 / 1.10-1.16;
the extremes are per-arm, individual folds range wider). The closed form is the frame-pooled optimum; the grid optimum of the
metric as scored is higher, and the CCC loss column in the full tables is computed at the closed-form gain, so it overstates the
CCC cost of the grid rmse_z gain. With that caveat: moving to the rmse_z gain costs CCC utt about 0.035-0.06; moving to the CCC
gain costs rmse_z about 0.05-0.10. Example (W12 LoRA, causal conv, held out): rmse_z 0.8032 at gain 1; 0.7769 with the
leave-fold-out rmse_z gain (0.647); 0.8412 with the CCC gain; oracle 0.7589. CCC utt: 0.5736 at gain 1; 0.5011 with the rmse_z
gain; 0.5756 with the CCC gain.

So the two metrics disagree about what a calibrated output is, and no metric-only analysis decides which is right. The
held-out std ratio of unshrunk arms is 0.73-0.80; the rmse_z gain brings it to about 0.53.

## Q5. Key conclusions re-read under CCC

Labels are descriptive and relative to each contrast's pre-registered metric (rmse_z or PCC; both when none). Several
decisions about these labels were made after seeing the first results (see "Changes made after seeing numbers").

| contrast | rmse_z | PCC | CCC utt | reading |
|---|---|---|---|---|
| P1 blend vs LoRA x cross-fold rmse_z gain | +0.009 (4/7, p 0.59) | +0.009 (7/7) | +0.081 (7/7, p 0.016) | reading changes (see below) |
| P2 frozen BiGRU->conv vs probe | -0.002 (4/7) | +0.033 (7/7) | +0.017 (6/7, p 0.031) | weakened but still significant |
| P3 W12 vs X18 frozen BiGRU->conv | +0.026 (4/7) | +0.003 (4/7) | +0.014 (3/7, p 0.25) | unchanged, not significant |
| T1 matched LoRA vs frozen | -0.021 (3/7) | +0.010 (6/7, p 0.078) | -0.004 (2/7, p 0.44) | not significant |
| T2 matched vs W12 LoRA | +0.005 (3/7) | +0.005 (6/7, p 0.031) | -0.001 (5/7, p 0.88) | not significant under CCC; PCC alone was significant |

Head against ridge probe, which rmse_z called a tie: the BiMamba and BiGRU->conv heads become significant under both CCC (p
0.031) and PCC (p 0.016), at about half PCC's size under CCC. The causal-conv head is borderline (CCC utt p 0.25, PCC p 0.109;
only CCC spk p 0.031) and the MLP head is null under every metric. Nothing among the non-gain contrasts is reversed outright.

**P1 in detail.** Under CCC the blend beats the cross-fold rmse_z-gain LoRA by +0.081 (7/7 folds; rmse_z showed a tie). Read
this as a consequence of the comparison, not as support for the blend: the shrunk LoRA has a std ratio of about 0.53 and 77
percent of the log difference is the scale factor, so CCC is penalizing a deliberately damped comparator. The downstream
vocoder control covers std ratio 0.82 only, not 0.53. The same-protocol comparisons are small:

| blend (gain 1) against | CCC utt difference | folds | selection |
|---|---|---|---|
| LoRA at its leave-fold-out CCC gain (W12 causal conv) | +0.009 (X18: +0.010) | 5/7 (p 0.094) | fitted on other folds |
| LoRA at its best single gain, shared across folds | -0.001 | 3/7 | fitted on test folds, favors LoRA |
| blend at its best gain against LoRA at its best gain | +0.012 (+0.013 in the table) | 7/7 | both fitted on test folds |

The blend's remaining gain over a calibrated LoRA is small and positive, about +0.009 to +0.013 depending on protocol, which
is its PCC gain. The plan's required statement applies: the conclusion depends on whether amplitude matters downstream; it is
not that the blend is better.

## Q6. Downstream validity (existing evidence only)

Eight of nine resynthesis systems (the MNGU0 oracle is not computable from cached arrays), agreement with the shipped
inverter's own output on 182 USC-TIMIT items; LibriTTS-R numbers are related to that agreement and are labelled so.

| metric | Spearman vs USC WER (n 8) | vs Libri WER (n 5) | vs ECAPA USC (n 8) |
|---|---|---|---|
| rmse_z | +0.99 | +0.87 | +0.99 |
| PCC | +0.99 | +0.97 | +0.99 |
| CCC utt | +1.00 | +0.90 | +1.00 |

All five metrics order the five head systems identically. With n of 5-8 systems whose WER spans 8.6-146 percent this is
saturated: it shows CCC is not worse than rmse_z at ordering gross quality differences, and nothing more (no intervals).

The amplitude-only shrink of the shipped output (std ratio 0.82; CCC utt 0.971, PCC 1.000, rmse_z 0.206) changed USC WER
from 8.6 to 9.9 percent, ECAPA similarity from 0.460 to 0.432 (USC) and 0.813 to 0.778 (Libri), and left Libri WER at 3.0.
That is one mild shrink with no uncertainty: it is consistent with a small amplitude cost and with none. It does not show that
amplitude matters little, and it does not show that it matters. The shifted-residual control (same rmse_z 0.514 as `all_lin`,
CCC 0.875 against 0.825, WER 30.4 against 38.8 percent) shows only that larger error degrades resynthesis.

## Recommendation by the fixed rule (plan section 4)

1. Discriminability: **fail.** CCC utt's standardized effect is higher / lower / tie against PCC on 4 / 33 / 2 contrasts (7
   gain-only contrasts set aside, where PCC cannot differ); detectable effect in units of the across-arm spread 0.55 (CCC utt),
   0.49 (PCC), 2.05 (rmse_z). Median standardized effect: PCC 2.18, CCC utt 1.59, rmse_z 0.50. A lower effect for CCC is
   expected, since it adds offset noise; it is not itself evidence that CCC is a worse metric, and discriminability is not
   validity (a metric blind to calibration is trivially more discriminable). The 33 against 4 are not independent: many
   contrasts repeat one comparison across systems. The counts of significant contrasts (28 PCC, 31 CCC utt, 9 rmse_z, of 46)
   include the 7 gain-only ones, where PCC cannot be significant by construction.
2. Decomposition informative: **pass.**
3. No reversal of a downstream-supported conclusion: **fail only through a clause added after the numbers.** By the
   operationalization written in plan section 6 (a reversal is a direction opposite to both rmse_z and PCC with an interval
   excluding zero), no contrast is reversed, so rule 3 would pass; P1 changed under CCC, and the plan's own sentence treats that
   as acceptable only if downstream evidence supports the CCC-optimal gain, which does not exist, so the run counted it as a
   failure. Spearman against WER is not worse than rmse_z.
4. Stability: **pass on the first clause** (tau 0.92 all arms, 0.81 real arms, between per-utterance and per-speaker CCC). The
   second clause is untested: the "drop utterances under 1 s" run removes nothing (shortest test utterance 59 frames, 1.18 s).
   An exploratory 3 s cut (145 of 397 utterances remain under 3 s, so it is a real subset) changes the significance reading of 9
   of 46 contrasts, mostly head-against-probe, but it was applied to CCC only; many p values sit at 0.03-0.1 with 7 folds, so
   much of that is threshold crossing.

Outcome: **companion.** Rule 1 fails whatever the other rules do, and the plan's mixed-outcome clause resolves to companion.
Report CCC utt with its correlation, scale and location factors beside rmse_z, PCC, centered rmse_z and the std ratio, and
state the calibration used when a gain is applied. The step that would allow promotion is Q6(c): retrain a vocoder (or run a
WER and perceptual test) on gain-calibrated against uncalibrated and offset-corrected against uncorrected outputs of one head.

## Changes made after seeing numbers

- The Q5 label function was rewritten after its first version mislabelled contrasts; it now compares exact sign-flip
  significance on 7 folds with the standardized effect and has "now significant" and "reference cannot see" categories. Counts
  of "weakened" or "unchanged" across rows compare against different reference metrics and are not comparable between rows.
  The raw contrast numbers are unaffected.
- Rule 1 sets aside the 7 gain-only contrasts (conservative against CCC: including them gives 11 higher against 33 lower,
  still a fail).
- Rule 3 was extended to count a P1 change as failing (see above).
- The 1 s sensitivity run is vacuous; an exploratory 3 s cut was added instead.

## Caveats

- Seven folds, one corpus family (USC-TIMIT and EMA_5EMO), all selected on the test folds; the gain rows labelled oracle are
  fitted on the evaluated folds. Only the cross-fold gain arms are an honest shrinkage control.
- Short-utterance behavior of per-utterance CCC is untested rather than absent: frames at 50 Hz are autocorrelated, so the
  squared mean difference carries upward sampling noise that lowers the location factor, and that is not arm-neutral. A
  stratified check by length, the 3 s cut on PCC and rmse_z as controls, and a trimmed or median per-unit CCC were not run.
  Utterance-channels with degenerate variance are dropped; their number per arm is not reported.
- All significance statements use the same 7 folds, so exact p values are coarse (smallest 0.016).
- Q6 rests on 5-8 systems with a large dynamic range and no intervals; the MNGU0 oracle is missing; the shipped head was read
  from the cached pickle because `load_model("en")` cannot run offline (its output scale is reproduced to 0.986-1.011).
- The unit tests detected all 8 deliberately injected bugs in an independent check; gaps: no single-frame test, and the gain test
  asserts only an ordering of optima on synthetic data.
