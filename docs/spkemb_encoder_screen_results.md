# Speaker-encoder screen: results, ranking and recommendation

Pre-registration: `docs/spkemb_encoder_screen.md`. Literature priors: `docs/spkemb_encoder_screen_lit.md`.
Per-family tables: `/data/user_data/xoy/spkemb/encoder_screen/results_{segment,pooling,multiscale}.md`.
All numbers are the proxy screen only (212 training speakers, 62 held-out speakers, 3 seeds, 6000 steps). No vocoder was trained.

## Metrics

- M1 RMSE: prediction of another same-speaker utterance's spectral targets (lower is better). Speaker-level SE is about 0.015.
- EER, 5-shot identification, M3 (content leakage PCC, lower is better), F0 PCC (pitch is given to the vocoder separately).
- 1 s and 2 s: M1 when the reference is cut to 1 or 2 seconds.
- Seed SE has 2 degrees of freedom and ignores the choice of test speakers. Review found that every "promising" flag
  rests on it, and that none of the gains exceeds the speaker-level SE.

## Baseline

Layer 6 periodicity-weighted mean plus FFN (1.115M parameters): M1 0.8589, EER 0.0739, ID 0.944, M3 0.375, F0 PCC 0.937,
M1 1 s 1.110, 2 s 0.892. Shipped en+ 64-d: M1 0.8361, EER 0.1169.

## Results (dM1 is baseline minus variant, positive is better)

### Voiced / unvoiced use

| Variant (plain words) | Hyp | Params | M1 | dM1 | EER | M3 | 1 s M1 | Review |
|---|---|---|---|---|---|---|---|---|
| Voiced, unvoiced speech and silence means side by side | H1 | 3.2M | 0.8498 | +0.0091 | 0.0715 | 0.364 | 0.983 | weakly holds; capacity confound |
| Voiced and non-voiced means side by side | H1 | 2.2M | 0.8562 | +0.0028 | 0.0744 | 0.361 | 0.959 | not promising |
| Periodicity-weighted voiced mean plus unvoiced-speech mean | H1 | 2.2M | 0.8586 | +0.0003 | 0.0725 | 0.357 | 0.987 | no gain |
| Learned soft three-way gates (two forms) | H2 | 3.2M | 0.884 / 0.891 | -0.025 / -0.032 | 0.100 / 0.088 | 0.28-0.29 | 1.04 | worse |
| Fixed k-means codebook, 6 or 16 clusters, per-cluster means | H3 | 1.0-2.3M | 0.870 / 0.876 | -0.012 / -0.017 | 0.077 / 0.083 | 0.30 / 0.29 | 0.99 / 0.97 | worse, least leakage |
| Baseline plus smoothed log-mel statistics (voiced, unvoiced) | H11 | 1.2M | 0.8549 | +0.0040 | 0.0763 | 0.372 | 1.074 | noise |
| Two-way split plus the log-mel statistics | H11 | 2.3M | 0.8539 | +0.0051 | 0.0727 | 0.377 | 0.962 | not separable from the two-way split |

### Pooling operators

| Variant | Hyp | M1 | dM1 | EER | M3 |
|---|---|---|---|---|---|
| Attentive statistics (with / without periodicity prior) | H4 | 0.878 / 0.882 | -0.019 / -0.023 | 0.090 / 0.094 | 0.31 / 0.28 |
| 4-head attentive; 4-query attentive | H5 | 0.880 / 0.893 | -0.021 / -0.034 | 0.098 / 0.093 | 0.30 / 0.27 |
| NetVLAD; GhostVLAD (8 clusters, 2 ghost); k-means 8 | H6 | 0.901 / 0.891 / 0.900 | -0.042 / -0.032 / -0.041 | 0.088 / 0.085 / 0.092 | 0.22-0.25 |
| Mean plus std; plus skewness; low-rank covariance | H7 | 0.859 / 0.857 / 0.879 | 0.000 / +0.002 / -0.020 | 0.077 / 0.084 / 0.090 | 0.38 / 0.37 / 0.31 |
| Trimmed mean; quantiles 0.1, 0.5, 0.9 | H8 | 0.8545 / 0.8527 | +0.0044 / +0.0062 | 0.074 / 0.079 | 0.376 / 0.371 |

### Multi-layer, multi-scale, frame networks, adversaries, multi-reference

| Variant | Hyp | Params | M1 | dM1 | EER | M3 | 1 s M1 | Review |
|---|---|---|---|---|---|---|---|---|
| Layer 1 and layer 6 means side by side | H9 | 2.2M | 0.8443 | +0.0147 | 0.0749 | 0.419 | 1.127 | gain about 1 speaker SE; M3 up; capacity confound |
| Learned layer weights over layers 1, 3, 6, 12 | H9 | 1.1M | 0.8567 | +0.0022 | 0.0848 | 0.395 | 1.126 | EER worse |
| Utterance mean plus 0.5 s segment statistics | H10 | 1.9M | 0.8955 | -0.037 | 0.089 | 0.255 | 1.136 | worse |
| Multi-layer aggregation per frame (MFA style) | H10 | 2.7M | 0.9033 | -0.044 | 0.097 | 0.298 | 5.31 | worse, fails short references |
| Dilated-convolution frame network; two-layer transformer | H12 | 4.8M / 2.4M | 0.908 / 0.897 | -0.049 / -0.038 | 0.095 / 0.078 | 0.25 / 0.29 | 1.90 / 1.18 | worse |
| Gradient reversal against content; against F0; both | H14 | 1.1M | 0.858 / 0.869 / 0.866 | +0.001 / -0.010 / -0.007 | 0.076 / 0.079 / 0.077 | 0.366 / 0.359 / 0.348 | 1.06 / 1.11 / 1.12 | no gain |
| Mean of embeddings of 3 references (inference only) | H13 | 1.1M | 0.8435 | +0.0155 | not computed | not computed | no gain | 3x the audio; Phase 4 already uses up to 8 references |

## What the screen supports

1. The only consistent effect across variants is on short references. Every voiced/unvoiced split and the fixed codebooks
   cut 1 s M1 from 1.110 to 0.96-0.99 (about 12 percent), against a seed SD of 0.03 to 0.10. This is the largest and
   most repeated signal, although it is on a proxy and 1 s references are not how the vocoder is used (it uses up to 8
   references at inference).
2. Full-length M1 gains are 0.3 to 1.7 percent, below the 0.015 speaker-level SE. No variant is established better.
3. Anything that adds learned capacity before or in pooling (attention, clusters, frame networks, gates) is worse: the
   encoder memorises the 212 training speakers, and the extra capacity reduces generalisation. Learned gates collapse.
4. Content-leakage (M3) reduction does not buy identity: the least-leaking variants (clusters, segment pooling, frame
   networks) have the worst M1, EER and F0 PCC.
5. Layer 1 raises M3 to 0.419 (Phase 1 also found shallow layers leak content). Higher M1 here may partly be leaked content.

## Ranking (screen outcome after review, prior, cost, history of non-transfer)

| Rank | Idea | Screen | Prior | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | Voiced / unvoiced speech / silence split pooling (H1) | weak positive, short references | 4 | cache exists | select |
| 2 | Layer 1 plus layer 6 (H9) | largest full-length gain, leakage risk | 2 | cache exists | select, as the raw-information test |
| 3 | Two-way voiced / non-voiced split (H1) | short-reference gain, no full-length gain | 4 | cache exists | select as the ablation of rank 1 (no silence branch) |
| 4 | Spectral side branch (H11) | not separable from the split | 4 | low | fold into rank 1 only if rank 1 wins |
| 5 | Trimmed mean (H8) | noise-level, zero parameters | 2 | 2x training time | optional free add-on |
| 6 | Multi-reference fusion (H13) | needs 3x audio | 3 | inference | not an architecture; Phase 4's 5-reference mean covers it |
| 7 | Fixed codebook, K=6 (H3) | lowest leakage, worse M1 and EER | 3 | low | drop unless leakage is seen in the vocoder |
| 8 | Layer weights (H9), mean+std, skewness (H7) | no gain, EER worse | 2 | low | drop |
| 9 | Attentive, multi-head, covariance (H4, H5, H7) | clearly worse | 2-3 | medium | drop |
| 10 | NetVLAD, GhostVLAD, learned codebook (H6) | worst M1 | 3 | frame cache | drop |
| 11 | Soft gating (H2) | worse; gates collapse | 2 | frame cache | drop |
| 12 | Segment, MFA and frame networks (H10, H12) | worse; fail short references | 1-2 | high | drop |
| 13 | Gradient reversal (H14) | F0 PCC falls and M1 worsens; content reversal no gain | 2 | medium | drop |

Combination worth trying only after a winner exists: three-way split with the log-mel statistics (H1 plus H11) and,
if layer 1 survives the leakage check, layer 1 and layer 6 inside each segment. Trimmed mean is a free add-on.
Do not combine before at least one arm clears the margin alone: with gains this small, a combination cannot be attributed.

## Recommended vocoder-in-the-loop test

Preconditions (cheap, before any vocoder run):
1. Paired speaker-level jackknife of dM1 for the candidates, from the saved `embeddings.npz` files. The seed SE cannot
   decide this. If no candidate is above zero at the 62-speaker level, run the arms for information only and expect no advance.
2. Capacity-matched control in the proxy screen: the baseline pooled vector repeated three (and two) times into the same
   head width, to separate "more parameters" from "voicing information". Submit with `scripts/spkscreen/submit_train.sh`.
3. Out-of-domain check of the silence branch (VCTK and GLOBE) on the proxy: if its gain vanishes, drop the silence
   branch (arm B replaces arm A).

Vocoder arms (HiFi-GAN, 24 kHz, 100k steps, seed 0, Phase 4 combination recipe: up to 8 references in training,
speaker-relative pitch, natural LibriTTS-R data, so that the encoder is the only change):

| Arm | What changes | Head params |
|---|---|---|
| 0 reference | existing layer 6 periodicity-mean FFN network (reuse the Phase 4/5 100k combination-recipe run if identical; otherwise retrain) | 1.1M |
| A | three-way pooling: separate layer 6 means over voiced, unvoiced speech and silence frames (periodicity at least 0.5; silence by low loudness; at least 3 frames, else back-off), concatenated, same FFN to 64-d | 3.2M |
| B | two-way pooling: voiced mean and everything-else mean | 2.2M |
| C | layer 1 mean and layer 6 mean side by side | 2.2M |
| D (control, only if the proxy capacity control is not clearly worse than A) | pooled layer 6 vector repeated into the A head width | 3.2M |

Each speaker encoder is trained as in the screen (AAM-softmax plus M1 head, 6000 steps) and frozen; the vocoder is then
trained with the encoder's 64-d output through FiLM, as for the existing network.

Judgement: Phase 4 rule unchanged. An arm advances when its mean ECAPA gain to the target over the three pools (LibriTTS,
VCTK, GLOBE) exceeds the margin 0.0266, with the same sign in VCTK and GLOBE and in held-out CAM++, and every guard
passes (own-speaker similarity drop at most 0.01 per SV model, UTMOS drop at most 0.10 in every pool, no WER rise).
Add own-utterance (T1) against other-utterance (T2) references to expose content leakage, and 1-2 s reference slices to
test the one robust proxy signal. Arm C also reports the T1 minus T2 gap, because of the M3 rise.

Cost (fp32 on L40S / 6000 Ada or better, using the `SBATCH_CONSTRAINT` in CLAUDE.md): a 100k-step HiFi-GAN run is about
6-9 GPU-hours (400k is 24-34), cloning evaluation about 2-4 GPU-hours per arm. Roughly 10-13 GPU-hours per arm: A, B, C
about 30-40, plus the reference (0 or about 10) and optional control D (about 12). Total 30-65 GPU-hours, one seed. A
second seed (10-13 GPU-hours each) is only warranted for an arm that advances on seed 0, since the margin is twice the
seed difference. These are estimates from the 400k timings, not measurements.

Expectation, stated plainly: proxy gains are 0.3 to 1.7 percent of M1 and Phase 2 and 3 showed proxy gains do not
transfer, so the most likely outcome is that no arm clears 0.0266. The out-of-domain gap in Phase 4 (own-speaker
resynthesis reaches only 0.58-0.65 of ground truth on GLOBE and VCTK before any swap) looks more like a data and
frame-level problem than a pooling one. Use the test to falsify "unvoiced frames carry the missing identity", not as an
expected win.

## Ideas to drop

Attentive, multi-head and multi-query pooling; NetVLAD and GhostVLAD; learned gates; k-means and phone-class codebooks;
segment-level and MFA pooling; dilated-convolution and transformer frame networks; covariance and skewness; quantiles;
gradient reversal against content or F0; learned layer weights. In each case the variant is worse than the baseline on
M1 and EER, not merely unproven, and capacity-hungry in a setting where the encoder already memorises the training
speakers. Multi-reference fusion is not dropped but is already part of Phase 4's inference.

## Precondition checks (2026-10-26)

Code: `src/sparc/vocoders/spkscreen/{precond,precond_report,globe_embed}.py`, `variants/capacity.py`. Numbers and full tables
(all metrics, every comparison, per corpus): `/data/user_data/xoy/spkemb/encoder_screen/precond/{precond.json,tables.md}`.

### Method

- Saved `embeddings.npz` of the 3 seeds, no retraining for check 1. The ridge probe is fitted once per run exactly as in
  `metrics.evaluate` and kept fixed; absolute values reproduce the table above (baseline M1 0.8589, EER 0.0739).
- Paired difference per seed (same seed), averaged over the 3 seeds. For pooled statistics (M1, 1 s and 2 s M1, EER, M3,
  F0 PCC) the SE is the leave-one-speaker-out jackknife of the seed-averaged difference over the 62 held-out speakers;
  for 5-shot identification it is SD / sqrt(n) over speakers. 95 percent CI = estimate +- t(61) SE. "Improved" counts
  speakers with a better seed-averaged per-speaker value (M1, 1 s, 2 s, EER of the speaker's utterances against all others).
- Holm correction over the three shortlisted variants (M1 p-values from the t statistic). Trimmed mean and the log-mel
  branch are outside that family; their p-values are raw.
- Positive differences mean the variant is better (lower M1 / EER / M3 / 1 s / 2 s, higher ID / F0 PCC).

### 1. Paired speaker-level inference against the baseline (62 speakers)

| Variant | dM1 (SE) [95% CI] | Holm p | speakers improved | dEER | d1 s M1 | d2 s M1 | dM3 | dF0 PCC |
|---|---|---|---|---|---|---|---|---|
| Voiced / unvoiced / silence split | +0.0091 (0.0031) [+0.0029, +0.0153] | 0.009 | 40/62 | +0.0024 (0.0022) | +0.126 (0.027) | +0.022 (0.009) | +0.011 (0.009) | -0.004 (0.003) |
| Voiced / non-voiced split | +0.0028 (0.0033) [-0.0038, +0.0094] | 0.405 | 35/62 | -0.0004 (0.0024) | +0.151 (0.024) | +0.022 (0.009) | +0.014 (0.011) | -0.001 (0.003) |
| Layer 1 + layer 6 | +0.0147 (0.0030) [+0.0087, +0.0206] | <0.001 | 48/62 | -0.0009 (0.0031) | -0.017 (0.008) | +0.011 (0.005) | -0.043 (0.005) | -0.001 (0.003) |
| Trimmed mean (raw p) | +0.0044 (0.0014) [+0.0015, +0.0073] | 0.003 raw | 37/62 | -0.0004 (0.0011) | +0.006 (0.003) | +0.002 (0.003) | -0.000 (0.004) | +0.002 (0.001) |
| Two-way split + log-mel branch (raw p) | +0.0051 (0.0031) [-0.0011, +0.0113] | 0.106 raw | 39/62 | +0.0012 (0.0022) | +0.148 (0.023) | +0.024 (0.009) | -0.001 (0.010) | -0.007 (0.004) |

dM3 is variant minus baseline sign-flipped like the others (negative = more content leakage in the variant).
5-shot identification: no difference excludes zero except the log-mel branch (+0.0096, SE 0.0036); at most 25 of 60
speakers improve, so identification does not move. EER: no difference excludes zero.

### 2. Capacity-matched control

Control: the baseline pooled layer 6 vector (same z-score) repeated k times into the head of the variant (`cap_rep3`,
3072-d input, 3.2M parameters, matches the three-way split; `cap_rep2`, 2048-d, 2.2M, matches layer 1 + layer 6 and the
two-way split). Repetition keeps the baseline's information and function class and gives the same weights, initialisation
scale and weight decay; zero padding was rejected because the padded weights get no gradient (it would be the baseline).
Same objective, 6000 steps, seeds 0-2 (`scripts/spkscreen/submit_train.sh cap_rep3`, `cap_rep2`).

| Comparison | dM1 (SE) [95% CI] | d1 s M1 (SE) | speakers improved (M1) |
|---|---|---|---|
| Repeated x3 vs baseline | +0.0058 (0.0020) [+0.0019, +0.0098] | -0.070 (0.011), worse | 39/62 |
| Repeated x2 vs baseline | -0.0013 (0.0020) [-0.0053, +0.0027] | -0.226 (0.023), worse | 25/62 |
| Three-way split vs repeated x3 | +0.0033 (0.0025) [-0.0018, +0.0083] | +0.197 (0.034) | 35/62 |
| Two-way split vs repeated x2 | +0.0041 (0.0032) [-0.0023, +0.0105] | +0.377 (0.044) | 40/62 |
| Layer 1 + layer 6 vs repeated x2 | +0.0160 (0.0024) [+0.0112, +0.0208] | +0.210 (0.019) | 50/62 |

About two thirds of the three-way split's full-length M1 gain (0.0058 of 0.0091) is reproduced by extra head width alone;
what the split adds beyond width is not distinguishable from zero on M1. Its short-reference gain is real against both
the baseline and the control, and wider repeated heads are worse than the baseline on 1 s references (width itself hurts
short references). Layer 1 + layer 6 gains nothing from width (control -0.0013) and beats its control by +0.0160.

### 3. Silence branch out of domain (per corpus)

Held-out test speakers by corpus: 23 LibriTTS-R (13 dev-clean, 10 test-clean) and 39 VCTK. GLOBE is not in the held-out
set; its Phase 4 pool (1240 utterances, 400 speakers, 3 each, about 2 h) was extracted with the same code (`cache
extract --set globe`, about 10 GPU-minutes), embedded with the saved encoders (`globe_embed`) and scored with each run's
own ridge probe (fitted on the held-out non-test rows). GLOBE has no 5-shot identification; only M1 (full, 1 s, 2 s) and EER.

dM1 against the baseline (SE), seed-averaged, positive = better:

| Variant | LibriTTS-R (23) | VCTK (39) | GLOBE (400) |
|---|---|---|---|
| Three-way split | +0.0020 (0.0042) | +0.0128 (0.0041) | +0.0082 (0.0028) |
| Two-way split | -0.0112 (0.0046) | +0.0099 (0.0041) | -0.0153 (0.0034) |
| Layer 1 + layer 6 | +0.0082 (0.0024) | +0.0179 (0.0042) | +0.0135 (0.0039) |
| Three-way minus two-way (silence branch) | +0.0132 (0.0028) | +0.0028 (0.0027) | +0.0235 (0.0026) |

d1 s M1 against the baseline:

| Variant | LibriTTS-R | VCTK | GLOBE |
|---|---|---|---|
| Three-way split | -0.152 (0.022), 1/23 improved | +0.255 (0.014), 39/39 | +0.101 (0.008) |
| Two-way split | -0.085 (0.013), 2/23 | +0.254 (0.017), 39/39 | +0.047 (0.009) |
| Layer 1 + layer 6 | +0.010 (0.003) | -0.023 (0.011) | +0.045 (0.006) |

EER differences are within about 1.5 SE everywhere except the two-way split on VCTK (+0.0059, SE 0.0027) and layer 1 +
layer 6 on VCTK (+0.0089, SE 0.0031). The silence branch is not what produces the VCTK gain (three-way minus two-way
+0.0028, not different from zero on VCTK, and no 1 s difference on VCTK); its value is robustness: the two-way split
loses M1 on LibriTTS-R and GLOBE while the three-way split does not. Both splits make 1 s references on LibriTTS-R much
worse (baseline 1 s M1 0.838, splits 0.92-0.99), so the short-reference gain is a VCTK and GLOBE effect, not a general one.

### Verdict per shortlisted variant

| Variant | CI of dM1 excludes 0 after Holm | Beats capacity control | Silence branch / corpora |
|---|---|---|---|
| Three-way split | yes (p 0.009) | no on M1 (+0.0033, CI includes 0); yes on 1 s | holds against the baseline on VCTK and GLOBE; not different from the two-way split on VCTK; LibriTTS-R 1 s worse |
| Two-way split | no (p 0.405) | no | worse than the baseline on M1 in LibriTTS-R and GLOBE |
| Layer 1 + layer 6 | yes (p < 0.001) | yes (+0.0160) | better on all three corpora; M3 +0.043 (leakage), 1 s worse |

### Recommendation for the 100k HiFi-GAN test

Run the reference, arm C (layer 1 + layer 6: the only arm clearing the speaker-level CI, its capacity control and all
three corpora; report own-utterance against other-utterance references because of the M3 rise), arm A (three-way split:
the only arm whose short-reference gain survives the control, and the more robust of the two splits) and its control D
(repeated x3, since most of A's M1 gain is width). Drop arm B (two-way split), the trimmed mean and the log-mel branch.
Estimated cost 36-50 GPU-hours at one seed (the earlier estimate minus arm B). The proxy still does not predict the
vocoder; the judgement rule is unchanged.
