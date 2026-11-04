# Pre-registration: does speaker-gender conditioning help the multi-speaker articulatory inverter? (2026-11-04)

Fixed before any result of this study. Everything below (arms, draws, thresholds, decision rule) is fixed here; the only
numbers already known are the earlier head-screen LOSO results, which are not used to set any threshold except where a
quantity is explicitly computed from the new A0 runs. Format follows `docs/rigor_prereg.md`.
Code: `src/sparc/compression/gender_cond.py`, tests `tests/test_gender_cond.py`. Outputs:
`/data/user_data/xoy/xlsr_ema/analysis_gender/`. Draws: `docs/gender_conditioning_draws.json` (identical copy in the output
directory), committed before the first training job (the timing pilot included).

## Question

Does the multi-speaker speech-to-EMA inverter predict a speaker's articulation better, in held-out-speaker rmse_z and PCC,
when it is told the speaker's gender (male and female articulation under separate conditions) than when it is not?
Asked separately for male and for female held-out speakers.

## Gender assignment and its evidence

| speaker | gender | evidence |
|---|---|---|
| usc_M1, usc_M3 | male | USC-TIMIT corpus speaker IDs (median F0 114 Hz, 150 Hz) |
| usc_F1, usc_F5 | female | USC-TIMIT corpus speaker IDs (F0 221 Hz, 190 Hz) |
| 5emo_jn | female | EMA_5EMO README: jn is F1 (not aligned to the occlusal plane, matching our 5emo_jn rotation); F0 169 Hz |
| 5emo_jr | female | README: two females and one male; jr and kf separated by median F0 (pYIN, 12 utterances each): jr 218 Hz |
| 5emo_kf | male | same: kf 112 Hz |

The split 3 male / 4 female matches the README count (1 male, 2 female in 5EMO). The 5EMO kf/jr assignment rests on F0
only (the README gives no per-ID gender); it is the one inferred label. Sensitivity: the ridge analogues are also run with
kf and jr swapped (male group {usc_M1, usc_M3, 5emo_jr}); if the ridge conclusion about conditioning flips under the swap
(definition under "Ridge analogues"), the primary verdict is reported as fragile. The trained arms are not rerun under the
swap.

## Fixed reference system (all arms)

Frozen WavLM Large, layer 12 (k12), cached zero-phase 10 Hz low-passed features, head `gruconv9` (LayerNorm -> Linear
1024->256 -> GELU -> BiGRU -> centered conv 9 -> Linear 256->12), no output smoother. LOSO folds over the seven speakers
(`datasets.EMAMulti(held_out=(h,), norm_by="norm_group")`; targets z-scored per normalization group from that group's
training split, usc_F5 has two groups). Per fold: alignment shift = the fold's shift from `depth_sweep` (not re-chosen; it
was chosen with training speakers only; asserted equal to the stored head-screen shift); same shift for every arm. AdamW (wd
0.01; none on biases, norms, FiLM tables), cosine schedule with one-epoch warm-up, <= 40 epochs, patience 8, clipping 1.0,
batch 8, early stopping on the validation sentences of the arm's training speakers. Learning rate: per (arm, fold, draw)
from {1e-4, 3e-4, 1e-3, 3e-3} with the arm's first seed on the arm's training-speaker validation rmse_z, then the other two
seeds at that rate. Seeds 0, 1, 2 (A0b: 10, 11, 12). The held-out speaker's data are never used for any choice. Training
loop = a copy of `head_screen.train_one` taking an optional per-utterance group label; with no labels it must reproduce
`head_screen.train_one` (unit test, CPU, same seed, identical weights). Data are loaded once per fold with the stems kept
after `probe.aligned` drops short utterances, and every subset (A2, A3 training and validation) is taken by speaker on those
stems (unit tests check label/feature alignment).

Hardware: all jobs are constrained to the GPU family of the head screen (RTX 6000 Ada, L40S, L40). A6000 nodes are about 4
times slower in fp32 and use other GRU kernels.

**Pipeline acceptance check (before any arm is read).** The new A0 must reproduce the stored head-screen result
(`analysis_heads/wavlm-large/k12_gruconv9.json`, rmse_z 0.765, PCC 0.709): 7-fold mean within 0.010 rmse_z and 0.005 PCC,
and every fold's shift identical to the stored one. The per-fold chosen learning rates are reported next to the stored ones
(a flip between near-tied rates moves a fold value by about one seed sd). If the check fails, `analyze` withholds every
verdict; the pipeline is debugged, never the reference re-tuned, and the stored numbers are never substituted for A0.

## Arms

Notation: for held-out h, T(h) = the six training speakers, G(h) = h's gender. Male folds: usc_M1, usc_M3, 5emo_kf
(T = 2 male + 4 female). Female folds: usc_F1, usc_F5, 5emo_jn, 5emo_jr (T = 3 female + 3 male).

- **A0 pooled, unconditioned.** All of T(h), the plain `CascadeHead` (no tables), unchanged recipe, rerun through the new
  code path.
- **A0b second pooled rerun.** Identical to A0 with independent seeds 10-12 (learning rate chosen with seed 10). Purpose: the
  empirical noise floor of a fold difference, and a direct check of the threshold tau below.
- **A1 pooled, gender-conditioned (oracle label at train and test).** Same data as A0. A learned table of K = 2 (gender)
  vectors (gamma, beta), each 256-d, **zero-initialized**, applied as FiLM `x -> x * (1 + gamma[g]) + beta[g]` at two sites:
  after the front end (GELU output) and after the BiGRU (before the conv). The tables are created after every other
  parameter, so the random initialization of the head is that of A0 and the function is identical to A0 at step 0 (unit
  test); after the first update the tables learn and the runs diverge, so A0 and A1 are not paired run by run. The tables add
  2,048 parameters (2 sites x 2 labels x gamma/beta x 256); the A1-versus-A0 contrast therefore includes that extra
  capacity, which is exactly what A4 controls. Tables are never weight-decayed (the parameter-group function in the module
  puts every parameter named `film*` in the no-decay group). Why FiLM on hidden activations rather than input concatenation:
  the input is a 1024-d frozen feature and a constant 1-d label would be dwarfed or absorbed by the LayerNorm; a hidden-site
  gain/bias can rescale and shift the temporal dynamics (lag, amplitude, cross-channel mixing) per gender, which is the
  hypothesis. One mechanism is fixed; input concatenation and single-site variants are not tried. The held-out speaker
  receives its own true gender label (oracle).
- **A2 separate per-gender model.** Train only on the speakers of T(h) with gender G(h) (male folds: the other 2 males;
  female folds: the other 3 females); validation sentences of the same speakers; unconditioned head. Target statistics
  `ym, ys` from the subset.
- **A3 count-, gender-mixture- and corpus-matched control for A2.** Unconditioned pooled head trained on a subset of T(h)
  with A2's speaker count that (i) contains at least one speaker of each gender (a truly mixed subset: an all-opposite-gender
  subset is cross-gender training, not a control for same-gender training) and (ii) contains the same number of speakers
  from h's corpus (USC or 5EMO) as A2's set, because same-gender training sets overlap with the held-out speaker's corpus
  and an A2 advantage could otherwise be a recording-setup effect. A2's own set is excluded by (i). Candidate counts: usc_M1
  4, usc_F1 8, usc_M3 4, usc_F5 8, 5emo_jn 10, 5emo_jr 10, 5emo_kf 4. **D = 6 draws per fold**: the candidates are shuffled
  once (fixed stream) and draw j takes position j mod (number of candidates), so a fold with fewer than 6 candidates uses
  every candidate and repeats the first ones; repeats are one training run counted in several replicate slots (unique runs:
  4, 6, 4, 6, 6, 6, 4). The replicate-rank null chance 1/(D+1) is therefore approximate and the effective number of distinct
  replicates is reported. The A3 control value is the mean over the distinct subsets with equal weight (a repeated slot does
  not re-weight it; amended before any result, the earlier wording averaged over slots); the replicate slots, repeats included,
  feed the replicate-rank check only. The learning-rate grid is re-run for every unique subset.
- **A4 pseudo-gender control for A1.** Identical to A1 (same zero-init FiLM, K = 2, same data as A0) but with labels from a
  random partition of all seven speakers into two groups with the same sizes as the real gender partition (male folds: 3 + 4;
  female folds: 4 + 3), the held-out speaker h placed in the group of the size of its real gender group (3 for male folds,
  4 for female folds), so h shares its label with the same number of training speakers as in A1. The partition differs from the
  real one: male folds 14 candidates, female folds 19. **D = 8 draws per fold** without replacement (fixed stream). The
  partition of a draw may share training speakers with the true gender group and with the corpus split; both overlaps are
  recorded per draw in the draw file, the corpus-equal draws are flagged, and the replicate gains are reported against the
  overlaps (descriptive). Because a partition can contain true same-gender speakers, A4 is a control for "any two-group
  label with this size structure", not a gender-free control.
- **A4c corpus-label control for A1.** Same mechanism and data as A1 with the label = corpus (USC vs 5EMO) for all seven
  speakers, h in its own corpus group, in all seven folds. Group sizes are not matched to gender here on purpose (a corpus
  split exists in every fold only in this form). Reported as a qualifier of the primary verdict (below), not as a gate.
- **A1-wrong (diagnostic, no training).** The trained A1 heads scored with the opposite gender label on the held-out speaker.
- **Ridge analogues (cheap, same cached features, same shift; ridge strength from `probe.ALPHAS` on the validation
  sentences of the arm's training speakers; sufficient-statistics implementation equal to `probe.Ridge`, unit-tested):**
  R0 pooled on T(h) (checked against the depth-sweep k12 probe, 0.764 / 0.676 over the folds); R2 on the same-gender T(h);
  R3 on every A3 candidate (exhaustive, mean); R1 interaction ridge on `[x, g x]` with g = 1 for h's gender group, shared
  and gender-specific parts penalized through the usual column-wise standardization (the `g x` columns are standardized
  like all others, which rescales their penalty; the interaction is therefore not an exactly "equal penalty" model); R4 the
  same with each of the A4 partitions (exhaustive, mean). A gender *intercept* is meaningless here because every speaker's
  targets are z-scored (each speaker's mean is 0 by construction), so the interaction form is the only ridge conditioning
  with content. **Ridge indicator for a gender:** the G-mean gain of R1 over R4 is positive on both rmse_z and PCC. The same
  analogues are run with kf/jr swapped; the primary verdict is flagged fragile if the indicator differs between the real and
  the swapped assignment for either gender.

### Draws

All draws are generated once by `gender_cond.make_draws()` with `numpy.random.default_rng`, one child stream per (arm, fold)
via `SeedSequence([20261104, arm_id, fold_index])`, sampled before any training and written to
`analysis_gender/draws.json` and `docs/gender_conditioning_draws.json`; the files are committed before the first training job
(the pilot included). All draws are always run; none is dropped for compute reasons. Runs refuse to start if the stored draw
file differs from the regenerated one. The same draw index j in A3 (A4) is used for all seeds.

## Metrics

Per held-out speaker (`test_unseen`, the speaker's own test sentences; z units of the speaker's full training-split
statistics, as in the head screen): **rmse_z** (frames pooled per channel, RMSE, mean over 12 channels) and **PCC**
(per utterance and channel, mean) are the primary pair; **centered rmse_z** (per-utterance mean error removed;
`rigor.utt_stats` / `metrics_from`) is reported alongside and is a gate for a "helps" verdict (below), because held-out
rmse_z uses the speaker's own EMA statistics and is dominated by per-utterance offsets, and a label-dependent beta can
change that offset alone. Not r^2. Fold value = mean over the 3 seeds; for A3/A4 the fold value is the mean over seeds, then
over the D draws (the control value; for A3 over distinct subsets, see A3), and the D draw values are kept as replicate values. Gender aggregate = mean over that
gender's folds (3 male, 4 female). Seen speakers: each training speaker's text-disjoint `test` sentences, per speaker and per
gender of the speaker, descriptive only; for A2/A3 only the speakers in the arm's own training set count, compared with A0 on
the same (fold, speaker) pairs.

## Decision rule (fixed)

Sign convention: gain = (reference rmse_z - arm rmse_z) and (arm PCC - reference PCC); positive = better.

**tau is a noise scale, not an effect-size threshold.** tau_rmse = root-mean-square over the 7 folds of the within-fold seed sd
of A0 rmse_z (ddof = 1, 3 seeds), tau_pcc likewise, each floored at 0.005 rmse_z and 0.003 PCC (earlier screen: about 0.015
and 0.006). It comes from held-out scores of the new A0 runs, i.e. it estimates noise and does not select anything. The sd of a
difference of two 3-seed means is about 0.8 x tau; a fold therefore exceeds tau on one metric by chance in roughly 10 percent
of cases and on both (the two metrics are correlated) in perhaps 5-8 percent, and lr-selection noise is not in the seed sd. For
these reasons tau is never the whole rule: majority, controls, replicate ranks and the centered gate are required too, and the
independent A0b rerun gives the empirical false-win rate of the fold criterion (reported), the paired seed-difference sd of
A1 - A0 is reported, and the primary rule is applied to every random-grouping replicate as a fake treatment (below).

A fold **wins** a contrast when its difference exceeds tau on **both** rmse_z and PCC (loses: below -tau on both). G is male
(3 folds, majority = at least 2) or female (4 folds, majority = at least 3).

**Primary (conditioning), per gender G: gender conditioning helps G iff all of**
1. A1 vs A0: wins in a majority of G's folds, and the G-mean difference is positive on both rmse_z and PCC;
2. A1 vs A4 (control value): wins in a majority of G's folds, and the G-mean difference is positive on both metrics;
3. the G-mean gain of A1 over A0 exceeds the G-mean gain of every one of the 8 pseudo-gender replicates over A0 (replicate j =
   the draw j of each of G's folds), on both metrics (rank 8/8; under the null the one-sided chance per metric is 1/9, and
   the two metrics are strongly correlated, so the joint null chance is nearer 1/9 than 1/81);
4. the G-mean centered rmse_z gain is positive for A1 vs A0 and for A1 vs A4 (a gain only in the per-utterance offset fails).

**Hurts:** the mirror (loses in a majority against both, negative G-means on both metrics against both, A1's gain below every
replicate on both metrics, negative centered gains).
**No detectable effect:** neither of the above, and for both contrasts (A1 vs A0, A1 vs A4) the G-mean difference is within
+/- tau on both rmse_z and PCC and no single fold's difference exceeds 2 tau on either metric.
**Inconclusive:** everything else. "Inconclusive" is not reported as "no effect", and a failed conservative rule is not
evidence of no effect.

**Qualifier.** A "helps" verdict whose A1 vs A4c contrast does not win a majority of G's folds (A1 not better than a corpus
label by the same criterion) is reported as "helps, not distinguished from a corpus label".

**Secondary (separate models), per gender G:** the same four-valued verdict for A2 against its control: A2 vs A3 (control value)
wins in a majority of G's folds with positive G-mean differences on both metrics and a positive centered gain, and A2's G-mean
gain over A0 exceeds every one of the 6 count-matched replicate gains over A0 on both metrics. A2 vs A0 is reported but is not
the criterion: A2 trains on fewer speakers, and speaker count is the largest known effect (1 -> 6 training speakers: rmse_z
-0.11 to -0.13).

**Overall verdict** (per question, primary and secondary separately): both genders "helps"; "helps one gender only" (stated with
the reminder that 3-4 folds support no per-gender test); otherwise the two per-gender verdicts are written out side by side
(never collapsed to "not supported" or "no effect"). Descriptive only, never changing a verdict: seen-speaker results,
A1-wrong, ridge analogues (except the fragility flag), the runs with corpus-equal partitions excluded, A4 gains against
overlaps, and the exact sign-flip p over all 7 folds (minimum 2/128).

**Operating characteristics (reported with the verdict).** The primary rule is applied to each of the 8 pseudo-gender replicates
as a fake treatment (against A0, against the mean of the other 7 replicates, ranked against the other 7), and the number
passing is reported per gender, together with the A0 vs A0b fold differences against tau. The rule is a conservative
descriptive screen, not a calibrated test: with 3-4 folds and 8 replicates the best attainable replicate-rank evidence is one
in nine per metric.

## Power and caveats

- Detectable effect with 7 folds is ~0.06 rmse_z / ~0.017 PCC (80% power, fold sd 0.06-0.13). Per gender with 3 or 4
  folds, an exact sign-flip test has minimum two-sided p = 0.25 (3 folds) and 0.125 (4 folds): no per-gender significance is
  attainable. The rule above is a descriptive consistency rule, not an inference; a small effect (< ~0.03 rmse_z) cannot be
  told from fold heterogeneity. Two genders are examined, so "helps one gender only" arises by luck about twice as often as
  "helps" for a named gender.
- Targets are z-scored per speaker, so any overall gender difference in articulator size or mean position is normalized
  away; a gain can only come from shape, dynamics or cross-channel structure.
- Oracle gender at test time; no gender classifier is involved.
- Gender is confounded with corpus and speaker identity: 5EMO has 1 male and 2 females, USC has 2 and 2; the male
  folds train on only 2 males (e.g. the usc_M1 fold: usc_M3 and 5emo_kf), so a "male" embedding is learned from 2 speakers. A1
  can also exploit the label as a speaker-group (corpus/session) code. A4 partitions and the A4c corpus control address this
  partly.
- A2 and A3 train on 2-3 speakers: the epoch has 40-50 percent fewer optimizer steps than A0, while the epoch limit, the
  one-epoch warm-up, the patience and the cosine schedule are fixed, and early stopping uses only about 80-130 validation
  utterances. A2 vs A0 is confounded by this step budget and by speaker count; A2 vs A3 shares both. Epochs and steps used are
  stored per run.
- Folds share five of six training speakers and are not independent; all selection already seen on these seven folds
  (k12, head type) was made without any gender arm. Everything here is selected-on-test-fold in the same sense as the
  earlier screen and is labelled so.

## Compute

762 head trainings: A0 42, A0b 42, A1 42, A2 42, A3 216 (36 unique runs), A4 336 (56 runs), A4c 42 (each run = 4 learning-rate
trials + 2 extra seeds), plus cheap ridge fits. Measured reference: the head screen's gruconv9 job (49 trainings including
evaluation) took 1 h 19 min on one RTX 6000 Ada, about 97 s per training; so about 20 GPU-hours at one run per GPU, less for
A2/A3 (40-50 percent of the data). The per-training time of this code path is measured by a one-run timing pilot (A0, one
fold, 6 trainings) whose output is deleted unread; the same run is later executed through the normal resumable path.
Jobs are under 60 minutes each (batched by fold and arm), restartable, submitted through `slurm-alloc`.

## Implementation notes

New module `src/sparc/compression/gender_cond.py`, reusing `head_screen` (`make_head`, `batches`, `zm`, `fold_shift`,
`LRS`, `SEEDS`, `load_features`), `probe.aligned` / `load_targets` / `ALPHAS`, `depth_sweep.load_all`,
`datasets.EMAMulti`, `rigor.utt_stats` / `metrics_from`. `predict`, `train_one`, `evaluate` and the data loading are copies
with a label path (the originals have none). Per-run resumable JSON under `analysis_gender/<arm>/<key>.json` with learning
rate trials, per-seed held-out and per-speaker seen scores (rmse_z, PCC, centered rmse_z), epochs/steps, wall time and, for A1,
the wrong-label score. `analyze` applies the rule exactly as written and writes `summary.md` and `results.json`.
Order: draws committed -> timing pilot -> all arms (no reading of A1/A2 before every control has finished; `analyze` reads
everything at once and requires every run to be complete).

## Reporting

Plain-word arm names in the write-up ("pooled model", "pooled model told the gender", "per-gender models",
"count-matched pooled models", "pooled models told a random grouping", "pooled model told the corpus"). A results section is
added to `CLAUDE.md` and the change to `CHANGELOG.md` only after the summary is produced.
