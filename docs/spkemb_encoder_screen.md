# Speaker-encoder architecture screen: pre-registration (2026-10-26)

Written and committed before any extraction or training of this screen. Code: `src/sparc/vocoders/spkscreen/`.
Outputs: `/data/user_data/xoy/spkemb/encoder_screen/` (`cache/`, `runs/`).

## Question

The speaker encoder conditions a 24 kHz HiFi-GAN articulatory vocoder through FiLM. The shipped network (baseline
below) reads WavLM Large layer 6, pools it over time with SPARC en+'s periodicity weights (so unvoiced and silent frames
get little weight) and maps the mean through an FFN to 64 numbers. Which changes to the pooling and to the frame-level
network give a better speaker code for resynthesis? This screen ranks ideas cheaply, on frozen cached features and
proxy metrics; it does not train a vocoder.

Caveat that governs how the result is used: in earlier phases proxy gains did not reliably transfer to the vocoder
(Phase 1 layer ranking by M1 was not reproduced by the frozen-generator refit of Phase 2; learned pooling over all 25
layers was worse than the layer 6 FFN at 400k steps in Phase 3). A "promising" variant here is a candidate for a
vocoder run, not a result about the vocoder.

## Hypotheses (ids used in reports)

Segment-aware: H1 voiced / unvoiced / silence split pooling; H2 learned soft segment gating; H3 sound-class-conditional
pooling (unsupervised frame codebook); H11 spectral side branch (pooled log-mel per segment class).
Pooling operators: H4 attentive statistics pooling; H5 multi-head / multi-query attentive pooling; H6 clustering-based
pooling (NetVLAD / GhostVLAD, or a fixed k-means codebook with per-cluster means); H7 higher-order statistics (std,
skewness, low-rank covariance); H8 robust pooling (trimmed mean, median, quantiles).
Multi-scale and architecture: H9 multi-layer (shallow layer plus layer 6, or learned layer weights); H10 multi-scale
temporal statistics and multi-layer aggregation before pooling; H12 frame-level network before pooling (TDNN / ECAPA
blocks, small transformer); H13 multi-reference aggregation at inference; H14 disentanglement (gradient reversal
against content and F0).

This commit implements the harness and the baseline only. Variants are added later, each as one file under
`spkscreen/variants/`, with hyperparameters fixed in that file before its first run. A variant is run with one fixed
configuration; nothing is tuned on the held-out speakers.

## Data (fixed)

- Encoder training set: LibriTTS-R train.clean.100 (246 speakers in the packed cache index
  `/data/user_data/xoy/sparc-vocoders/cache/packed/train.clean.100/index.parquet`). Per speaker 30 utterances drawn
  with `numpy.random.default_rng(0)` among those with 100 <= T <= 800 feature frames (2 to 16 s) and no periodicity
  fallback; speakers with fewer than 20 such utterances are dropped. Audio and per-utterance CREPE seeds are the packed
  index's (`wav_path`, `seed`).
- Evaluation set: exactly the Phase 1 data of `/data/user_data/xoy/spkemb/layers/*.npz` (5470 utterances: VCTK 109
  speakers x 30, LibriTTS-R dev-clean and test-clean up to 30 per speaker; 188 speakers), in Phase 1's row order, with
  Phase 1's split: 62 held-out test speakers drawn by `default_rng(0).choice(sorted(speakers), n // 3, replace=False)`
  and 126 remaining speakers. The 126 speakers are used only to fit the M1 / M3 / M5 ridge probes (and the shipped
  en+ reference); the 62 test speakers are used only for scoring. No evaluation speaker is in encoder training
  (asserted: LibriTTS-R dev/test speaker ids are disjoint from train.clean.100, VCTK is a different corpus).
- Phase 1's utterance targets (log-mel mean and std, MFCC 1-24 mean, tilt, median F0) are read from the npz files, not
  recomputed. Targets for the training utterances are computed with the same function (Phase 1 `acoustic_targets`).
- Frame-level cache (50 Hz, T = floor(L24/480) - 1 + [L24 mod 480 >= 119] frames), batch size 1, same extractor
  settings as the vocoder cache (`SparcFeatureExtractor`, refit head, fp32, TF32 off, CREPE dither on with the
  utterance seed): WavLM Large hidden states at layers {0, 1, 3, 6, 12, 24} (fp16), SPARC periodicity, z-scored
  loudness, raw loudness and F0 (Hz), 80-bin log-mel (24 kHz, n_fft 1024, hop 480). Layer 24 is dropped or frames
  subsampled if the cache exceeds 150 GB (the report states what was done).

## Training objective (identical for every variant)

- Encoder output 64-d (the size of the shipped embedding); parameter count of the encoder is reported.
- Loss = AAM-softmax speaker loss over the training speakers (additive angular margin 0.2, scale 30, cosine
  classifier on the 64-d output) + 1.0 x auxiliary M1-style head: Linear(64, 256), GELU, Linear(256, 185) predicting the
  standardized targets of a different utterance of the same speaker (mean squared error).
- Optimizer AdamW (lr 1e-3, weight decay 0.01, betas 0.9 / 0.999), 300 warmup steps then cosine decay to 0, 6000
  steps, gradient-norm clip 5. Batch 64 utterances, one per distinct speaker, each a random crop of uniform length
  50 to 300 frames (1 to 6 s; shorter utterances whole). A variant may add its own auxiliary losses only if the
  hypothesis requires them (H14), in which case this is stated in its report and it is compared against the same
  baseline.
- Seeds 0, 1, 2 for every variant (initialization, speaker / crop sampling).
- Baseline: layer 6, periodicity-weighted mean over frames (weights = cached periodicity, uniform if they sum to 0),
  per-dimension z-score of the pooled vector with training-set statistics, Linear(1024, 1024), GELU, dropout 0.2,
  Linear(1024, 64) (the shipped `SpeakerFFN`), trained as above.
- Reference only (not trained): the shipped en+ 64-d embedding stored in the Phase 1 files.

## Metrics (held-out speakers; every number has a speaker-level SE)

Embeddings are computed from each full evaluation utterance (and from its first 1 s and first 2 s for robustness).
- Primary, M1 (Phase 1 definition): ridge from utterance A's embedding to a different same-speaker utterance B's
  targets (cyclic partner within speaker in Phase 1 order), ridge fitted on the 126 non-test speakers' embeddings
  (alpha by 20 percent holdout, as Phase 1), RMSE in target-std units averaged over targets, with the PCC; SE = SD
  over test speakers of the per-speaker RMSE / sqrt(62).
- Secondary: 5-shot speaker identification (first 5 utterances enrol, cosine to centroids, accuracy; SE over
  speakers) and equal error rate (cosine, all same-speaker against all different-speaker test pairs; SE by
  leave-one-speaker-out jackknife); M3 content leakage (Phase 1 definition, PCC, jackknife SE); median-F0
  predictability M5 (PCC, jackknife SE); short-reference robustness (M1 RMSE with 1 s and 2 s references, probe fitted on
  full-utterance embeddings as in Phase 1).
- Cost: encoder parameters, training seconds per run, forward milliseconds per utterance.

## Selection rule (fixed)

For each variant, take the three seed-paired differences to the baseline (same seed index). Let d_M1 = baseline M1 -
variant M1 (positive = variant better), d_EER likewise (baseline EER - variant EER), d_M3 = variant M3 - baseline M3.
SE of a paired difference = SD of the three differences / sqrt(3). A variant is **promising** if (1) mean d_M1 > 2 SE,
(2) mean d_EER > -2 SE (it does not lose on EER by more than 2 SE), and (3) mean d_M3 <= 0.05. Promising variants
are ranked by mean M1, then mean EER; cost is reported beside the ranking. Variants that fail are reported with the
same table. The speaker-level SEs of the individual runs are reported but do not enter the rule. With three seeds
the rule is permissive about noise in the seed SD (two degrees of freedom) and 14 hypotheses are screened, so promising
means "worth a vocoder run", with the multiplicity stated in the report; no variant is called better for the vocoder
on this screen alone.
