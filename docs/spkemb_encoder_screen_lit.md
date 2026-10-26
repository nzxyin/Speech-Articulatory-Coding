# Speaker encoder architecture screen: literature and design analysis (2026-10-26)

Scope: fourteen hypotheses (H1-H14) for improving the speaker encoder that conditions the 24 kHz HiFi-GAN articulatory
vocoder through FiLM. No experiments were run for this note. Citations are from memory; anything marked (uncertain) was
not re-checked against the source and should be verified before it is quoted. Variants are described in plain words.

## 1. Baseline and constraints from Phases 1-4

Baseline ("l6 FFN", `SpeakerFFN` in `src/sparc/vocoders/models/speaker.py`): one pooled 1024-d vector per reference
(WavLM Large layer 6, periodicity-weighted mean only), z-scored, then Linear 1024-1024, GELU, dropout 0.2, Linear to 64-d.
With several references the output embeddings are averaged. Training is joint with the vocoder from cached pooled vectors.

What the earlier phases say, and how it bears on this screen:

- Phase 1: timbre (M1 probe) is best in layers 0-3 and decays with depth (L0 0.819, L6 0.845, L12 0.882, L18-20 0.916);
  content leakage runs the other way (M3 0.78-0.81 at L0-L3, 0.72 at L6); the trained FFN lowers leakage to 0.505 while
  keeping L6's timbre. Median F0 is almost fully predictable from the embedding (PCC 0.95-0.98). References of 1 s fail
  at every layer, 2 s are marginal. Consequence: any extra statistic must work with short references, and any
  shallow-layer addition raises the content-leakage risk that the FFN must then remove.
- Phase 2: with a frozen generator, layer choice among 0-6 is unresolvable (<= 0.021 ECAPA SIM, about 1-1.7 SE); the
  training procedure dominates. Consequence: frozen-generator refits are a poor screen for architecture changes too.
- Phase 3 (100k screening, 400k finalists): learned pooling over all 25 layers (mean and std, softmax layer weights) tied
  the l6 FFN at 100k and was worse at 400k in both seeds (T2 similarity -0.012 paired, ECAPA -0.022; swap similarity
  0.63 against 0.67). The VAE variants lost 0.04 or more of similarity. Consequence: more input information and learned
  mixing did not help; the std statistic was included in that arm, so H7 starts with negative evidence. The prior for
  "add more features, let the network sort it out" is low.
- Phase 4: multiple references (1-8) and speaker-relative pitch help cloning; matching target F0 median and range at
  inference gives about +0.06 everywhere; out-of-domain pools (VCTK, GLOBE) lose most of the gap without any swap
  (own-speaker resynthesis reaches only 0.65 / 0.58 of the ground-truth level, against 0.90 on LibriTTS). Consequence:
  domain and channel mismatch, not only timbre capacity, limits cloning; an encoder that carries channel cues can hurt
  here, and one that is robust to them can help.
- Proxy gains did not transfer reliably (Phase 1 layer order against Phase 2 order; Phase 3 pooling). Treat any
  probe-only result as a filter, not as a decision.

Margin conventions already in use: Phase 3 screening margin 0.01 SIM; Phase 4 margin 0.0266 (twice the seed spread).

## 2. Evidence on unvoiced speech carrying speaker information

- Source-filter and vocal tract length: formant scaling with vocal tract length is classical (Fant 1960; Fitch and Giedd
  1999 for the length-formant relation in humans). Voiced vowels expose formants well; fricatives expose the length of
  the front cavity.
- Fricative acoustics differ by speaker and sex: spectral peak and moments of /s/ and /sh/ are speaker-dependent
  (Jongman, Wayland, Wong 2000, sex and spectral moment differences; Newman et al. 2001 on /s/ variation; details
  uncertain). Spectral moments of sibilants are a standard speaker cue in phonetics.
- Phone-level speaker discrimination: Eatock and Mason 1994 ranked phoneme classes by speaker discrimination: nasals and
  vowels best, fricatives and stops lower but not useless (recalled, ranking uncertain). Earlier: Sambur 1975, Su et al.
  1974 on nasals and vowels.
- Spectral tilt, breathiness, and noise component are voice-quality cues present in both voiced and unvoiced portions
  (Hillenbrand, Klatt and Klatt 1990 for breathiness; uncertain).
- Speaker verification practice: modern systems (x-vector, ECAPA) use all speech frames after voice activity detection,
  with no voicing split; i-vector UBM statistics implicitly stratify by sound class through GMM posteriors. There is no
  widely cited result that a voiced / unvoiced split improves verification EER (uncertain; I do not know of one).
- Conclusion: the physics supports unvoiced frames as a source of vocal tract and spectral-tilt cues that periodicity
  weights discard. The open question is whether the l6 FFN already recovers them from voiced frames (voiced frames
  are 40-60 percent of speech time, and WavLM layer 6 has context spanning neighbouring unvoiced sounds).

## 3. Hypothesis by hypothesis

Prior score 1-5 (5 = most likely to improve vocoder identity at equal or lower risk). Cost notes: a "frame cache" means
storing per-frame WavLM features (about 211 GB in fp16 for 572 h at 50 Hz and 1024 dims for one layer; feasible on
/data but large); a "stats cache" means a few pooled vectors per utterance (megabytes; recomputation needs a WavLM pass,
roughly the cost of the existing `spk_layers` cache).

### H1 Voiced / unvoiced / silence split pooling (score 4)

- Closest prior work: none for the three-way split exactly. Related: i-vector Baum-Welch statistics per UBM component
  (Dehak et al. 2011), which stratify by sound class softly; VAD-gated pooling in x-vectors (Snyder et al. 2018) drops
  silence; GhostVLAD ghost clusters (Zhong et al. 2018; Xie et al. 2019 for speaker use) absorb noise frames.
- Known effect: unknown for voicing splits. VAD removal helps verification in the wild, which supports separating
  silence rather than mixing it in.
- Expected for vocoder conditioning: moderate gain. Unvoiced speech supplies vocal tract length and tilt cues (section
  2); the silence branch supplies noise floor and channel. The silence branch is a double-edged tool: for a vocoder
  trained on restored LibriTTS-R and filtered GLOBE, reproducing a noise floor is not wanted, and Phase 4 shows
  domain shift already hurts VCTK and GLOBE. Recommended: voiced and unvoiced branches, silence branch off by default
  and tested as an ablation (or silence statistics used only as a nuisance input with dropout).
- Content risk: unvoiced frames are strongly phone-dependent (which fricative or burst), so their mean varies with what
  was said. Class means over a short reference are noisy and content-dependent. Mitigate with per-branch z-scoring,
  mean-only for the unvoiced branch first, and cross-utterance training (Phase 4 recipe already uses other-utterance
  references).
- Channel risk: moderate for silence, low for the others. F0 risk: voiced branch unchanged; unvoiced branch carries none.
- Cost: input grows from 1024 to 2048-3072 dims; parameters about +1-2 M; stats cache only. Voicing from the same
  periodicity signal the pooling already uses; thresholds for voiced / unvoiced / silence need loudness and periodicity.
- Implementation: low-medium. New cache module beside `features/spk_layers.py` (pooled mean per class for layer 6,
  optionally std); an encoder variant taking `[B, K, C, 1024]`; frame-to-class assignment must use the same 50 Hz frame
  grid as the periodicity weights (see CLAUDE.md note on feature length and frame centres).
- Interaction with phases: the Phase 3 result (more inputs, no gain) is a caution, but B2 added layers, not a new
  information source that the baseline discards by design; this is the clearest case of discarded information.

### H2 Soft segment gating (score 2)

- Prior work: attention weights over frames (Okabe et al. 2018 attentive statistics; ECAPA channel-and-context
  attention, Desplanques et al. 2020). Gates conditioned on periodicity and loudness are a learned version of the
  existing fixed weights.
- Effect: in verification, attentive pooling gives a modest relative EER gain over plain statistics pooling (order 5-10
  percent in published ablations; figures uncertain).
- Vocoder: attention trained only by the resynthesis loss tends to focus on frames that make the reconstruction easy,
  i.e. content-rich frames, because the cheapest path to a lower loss is information about this utterance. Phase 4
  trains on other-utterance references, which mitigates this, but gates can still collapse on vowels.
- Cost: needs a frame cache (211 GB fp16 per layer) and a frame-level forward pass during vocoder training, which slows
  data loading. Implementation: medium-high. Largely subsumed by H1 plus H4.

### H3 Phone-class-conditional pooling (score 3)

- Prior work: phonetically aware i-vectors with DNN posteriors (Lei et al. 2014; Kenny et al. 2014), phonetic attention
  (Zhou et al. 2019 "phonetic-attention scoring", uncertain), and phone-aware pooling for text-dependent verification.
  Unsupervised variants: k-means on SSL features (HuBERT units, Hsu et al. 2021). TTS: AdaSpeech (Chen et al. 2021)
  uses phoneme-level acoustic conditioning.
- Effect: strongest for text-dependent or short-utterance verification where content mismatch dominates; modest
  otherwise (uncertain).
- Vocoder: attractive because it removes content dependence of the mean, which is exactly the failure Phase 1 measured
  (M3). Weakness: with 2-3 s references most classes have few frames; need back-off (empty classes filled with the
  utterance mean or a learned default) and class dropout, otherwise short references fail as in Phase 1 M4.
- Risk: content leakage drops (good); parameter growth with number of classes. Use 6-10 broad classes from a fixed
  k-means over WavLM L6 frames grouped by an articulatory-phone-centroid heuristic, or a pretrained phone recognizer
  (heavier, adds an external dependency).
- Cost: class assignment pass over the corpus; stats cache; encoder input grows by number of classes times 1024.
  Implementation: medium. Combines naturally with H1 (voicing as one more partition axis).

### H4 Attentive statistics pooling (score 3)

- Prior work: Okabe, Koshinaka, Shinoda 2018; ECAPA-TDNN (Desplanques, Thienpondt, Demuynck 2020); WavLM weighted-layer
  features with ECAPA head (Chen et al. 2022, WavLM paper SV results).
- Effect: robust small gain in verification. For vocoder identity the benefit is unproven; Phase 3 found even all-layer
  mean-plus-std no better than the baseline.
- Risk: frame attention can be content-biased (see H2). Cost: frame cache; parameters small (a 128-d attention bottleneck).
  Implementation: medium (needs frame-level encoder path). Prior 3 only if run on top of H1 gating (voiced / unvoiced
  attention heads), otherwise 2.

### H5 Multi-head / multi-query attentive pooling (score 2)

- Prior work: MQMHA (Zhao et al. 2022, VoxSRC / Interspeech; uncertain on exact title), self-attentive pooling heads
  (Zhu et al. 2018; India et al. 2019 multi-head attention for speaker embedding).
- Effect: small verification gains over ASP, needing large training data (VoxCeleb scale). Our encoder trains on about
  572 h of LibriTTS-R plus GLOBE speakers inside a vocoder loss, with a weak speaker-discriminative signal.
- Vocoder: no direct evidence; more heads mostly raise capacity, which Phase 3 suggests is not the limit. Cost medium,
  frame cache. Dominated by H4 and H6.

### H6 Clustering-based pooling: NetVLAD / GhostVLAD, or fixed codebook (score 3)

- Prior work: NetVLAD (Arandjelovic et al. 2016); GhostVLAD for speaker recognition (Zhong et al. 2018; Xie et al. 2019,
  "Utterance-level aggregation for speaker recognition in the wild"); i-vector UBM stats as hard/soft VLAD
  (Dehak 2011); learnable dictionary encoding (Cai et al. 2018).
- Effect: competitive with or better than average pooling on VoxCeleb in those papers; unclear against ASP-ECAPA.
- Vocoder: ghost clusters are the principled way to dump silence and noise without a hard silence branch, which targets
  the Phase 4 domain gap. Fixed k-means with per-cluster means (no learned assignment) is the no-training-risk form and
  equals H3 without class labels. Residual aggregation removes the global mean, which strips channel offset in the mean
  but keeps cluster-wise differences, helping robustness.
- Risk: output dimension K times D (K=16, D=1024 gives 16 k inputs; needs a projection before the FFN, which is what the
  hypothesis states). Cluster collapse; assignment instabilities with joint vocoder gradients. Cost: frame cache for the
  learned version, stats cache for the fixed codebook. Implementation: medium (learned) / low-medium (fixed).
- Suggested form to test: fixed k-means (K about 32 on L6) with ghost-like "garbage" clusters identified by low
  periodicity and loudness, per-cluster mean with a count-based gate, concatenated and projected. This is H3, H1 and H6
  in one cache.

### H7 Higher-order statistics (score 2)

- Prior work: standard deviation in x-vector statistics pooling; covariance and second-order pooling (Li et al. on
  global covariance pooling in vision; speaker-verification second-order pooling papers exist, e.g. Wang et al., exact
  citation uncertain); low-rank bilinear pooling; skewness in some voice-quality analyses.
- Effect: std gives measurable gains in x-vector over mean-only. Skewness and kurtosis are rarely used and noisy on
  short references. Covariance is expensive and needs low-rank factorization.
- Vocoder: Phase 3 B2 used mean plus std with no gain; the evidence against is direct. Std also carries utterance
  content variability. Prior 2 and mostly for the voiced branch of H1 as a cheap add-on.
- Cost: std free; low-rank covariance rank 16 adds about 16 k dims; frame cache needed for covariance.

### H8 Robust pooling: trimmed mean, median, quantiles (score 2)

- Prior work: median / trimmed statistics are standard robust estimators; in speaker embeddings, robust pooling has
  limited literature (uncertain). Outlier frame removal is usually done by VAD and attention.
- Effect: probable small gain on noisy references (GLOBE), none on clean LibriTTS-R. Periodicity weighting already
  downweights bursts and silence.
- Vocoder: a cheap stats-cache variant: per-dimension median or 10-90 percent trimmed mean over voiced frames next to
  the mean. Quantile features (10/50/90) also capture dispersion without std's outlier sensitivity.
- Risk low, gain probably small; useful mostly as a cheap addition in the first batch of stats-cache variants.

### H9 Multi-layer concatenation (score 2)

- Prior work: weighted layer sum in SUPERB (Yang et al. 2021) and WavLM SV (Chen et al. 2022) is the norm; MFA
  concatenates layer outputs (see H10). Shallow layers hold more speaker/timbre information in SSL models (Chen et al.
  2022 layer analysis; Phase 1 here).
- Effect on this project: Phase 1 favours L0-L3 for M1, Phase 2 cannot resolve 0-6, Phase 3 B2 (learned mixing of all
  layers) lost at 400k. Concatenating L1 and L6 is different from B2's softmax mixing (mixing within each dimension
  versus separate inputs), but B2's result lowers the expected value.
- Risk: content leakage rises with shallow layers (M3 0.78-0.81); F0 predictable. Cost: input 2048, stats cache already
  exists (`spk_layers` has all layers). This is the cheapest arm to run (no new cache) and is a good control for
  H1/H11: if concatenating L1 gives nothing, then raw feature information is not the limit.

### H10 Multi-scale temporal pooling (score 2)

- Prior work: MFA-Conformer (Zhang et al. 2022) aggregates multi-layer outputs before pooling; multi-scale aggregation
  (Gao et al. 2019 Res2Net; Tang et al. 2019); segment-level embeddings pooled again.
- Effect: gains on long utterances in verification. For a vocoder, a single fixed embedding per reference means
  time-varying speaker traits are not modelled anyway.
- Risk: low; benefit unclear for 2-10 s references; frame cache needed. Implementation: medium-high. Could instead be
  approximated by pooling stats over 0.5 s windows and computing their across-window std (a stats-cache change), which
  measures within-speaker variability (speaking style), not identity.

### H11 Spectral side branch (score 4)

- Prior work: long-term average spectrum (LTAS) is a classical speaker and voice-quality descriptor (Pittam 1987 on LTAS
  voice quality, uncertain; Nolan 1983); mean MFCC/CMS-like statistics in GMM-UBM systems (Reynolds 2000); mel-based
  speaker encoders in TTS: GE2E / SV2TTS (Wan et al. 2018; Jia et al. 2018), StyleTTS 2 (Li et al. 2023) style encoder on
  mel with average pooling, Meta-StyleSpeech (Min et al. 2021), reference encoders (Skerry-Ryan et al. 2018; Wang et al.
  2018 GST).
- Effect: not tested here beyond Phase 1, whose M1 targets are exactly log-mel mean/std, tilt and MFCC; these targets are
  easy to compute and to feed. In reference-encoder TTS, mel-based encoders clone timbre well but leak content and
  prosody.
- Vocoder: the vocoder is trained with a mel loss and cares about the spectral envelope, so spectral statistics are in
  the right space; WavLM L6 holds less of it than L0-L3. The side branch is cheap: per-class mean of log-mel (voiced,
  unvoiced), smoothed across frequency (cepstral lifter or 24 MFCC) to remove harmonic structure, which otherwise
  leaks F0.
- Risk: F0 and loudness leakage (mitigate: cepstral smoothing, drop c0, normalise by utterance level); channel leakage in
  unvoiced and silence classes (mean log-mel includes the recording channel; this is the same concern as H1). This is
  also the arm with the most direct handle on spectral tilt.
- Cost: negligible; mel from the audio during cache building (the vocoder already computes 80-bin mel at 24 kHz).
  Implementation: low. Pair with H1 (statistics per voicing class), as stated in the hypothesis.

### H12 Frame-level network before pooling (score 1)

- Prior work: x-vector TDNN (Snyder 2018), ECAPA-TDNN, ResNet-based, Conformer; on SSL inputs a small TDNN or
  transformer before pooling is common (SUPERB SV downstream uses x-vector head).
- Effect: needed when features are not already contextual; WavLM layer 6 is already contextualised. A frame-level
  network trained from only a resynthesis loss risks learning content features.
- Cost high: frame cache, forward pass over thousands of frames per reference per step, throughput loss; implementation
  high. Not recommended unless a cheaper arm shows that frame-level information matters.

### H13 Multi-reference aggregation (score 3)

- Prior work: neural voice cloning with a few samples (Arik et al. 2018), SV2TTS averages embeddings (Jia et al. 2018),
  VALL-E style prompts concatenate acoustic tokens (Wang et al. 2023), NaturalSpeech 2 and 3 prompting; attention over
  references (several zero-shot TTS papers; uncertain). Precision-weighted fusion is what the Phase 3 VAE did, with poor
  results caused by the capacity cap, not by the fusion.
- Effect: Phase 4 already shows multi-reference helps (training with 1-8 references). The remaining questions are how to
  fuse: mean of per-reference outputs (current), mean of pooled statistics before the FFN (cheaper, weights by
  periodicity mass of each reference), duration-weighted averaging, or a learned scalar attention per reference.
- Risk low; the first two are inference-only and need no training. Gain bounded (diminishing past about 5 references).
  Implementation: low for the inference-only variants, medium for learned weights.

### H14 Disentanglement with gradient reversal (score 2)

- Prior work: gradient reversal (Ganin and Lempitsky 2015; Ganin et al. 2016), adversarial speaker/content factoring in
  voice conversion (Chou et al. 2018; AutoVC bottleneck, Qian et al. 2019; SpeechSplit, Qian et al. 2020), adversarial
  noise-robust speaker embeddings (Meng et al. 2019, uncertain).
- Effect: typically reduces content or channel at some cost in speaker information and with training instability; in
  joint vocoder training the adversary has to be tuned against a changing generator.
- Vocoder: removing F0 from the embedding is appealing (Phase 1 M5 and Phase 2: the embedding encodes F0 and so source
  pitch drives swaps), but F0 level is part of identity and Phase 4 already supplies speaker-relative pitch and F0
  matching (+0.06). Content removal is largely done by the FFN (M3 0.505) and by cross-utterance training.
- Cost: auxiliary classifiers on cached phone / codebook labels; instability risk. Prior 2; consider only for F0 after
  the others, as a small adversary on the 64-d embedding (median F0 regression head with reversal), evaluated on swaps.

## 4. Cross-cutting design remarks

1. The l6 FFN input is a mean only; every H1/H3/H6/H11 variant adds a class-wise partition of the same mean. A single
   stats cache (per utterance: per class, per layer in {1, 6}, mean, optional std, class frame counts, plus smoothed
   log-mel mean per class) supports H1, H3 (fixed codebook), H6 (fixed), H7 (voiced std), H8, H9 and H11 without a frame
   cache. Frame caches are only needed for H2, H4, H5, H10, H12 and learned H6. Recommend building the stats cache first.
2. Missing-class handling is the central design risk for short references: always provide class counts (or a presence
   mask), train with random class dropout and short crops (1-10 s), and fall back to the utterance mean.
3. Evaluate with the Phase 4 protocol (swap similarity with ECAPA, WavLM-SV and CAM++; margin 0.0266; three pools) and
   the Phase 3 T2 table; compare with at least two seeds for finalists. Include the 1-2 s reference slices where the
   baseline fails (Phase 1 M4).
4. Cheap pre-filter, keeping Phase 1 lessons in mind: fit the M1 probe and M3 leakage on voiced-only, unvoiced-only and
   combined stats for held-out speakers; a difference in M1 between the unvoiced branch and a content-matched voiced branch
   shows whether new timbre information exists. A probe pass does not decide, but a null result is informative: if
   unvoiced stats add nothing beyond voiced stats in M1 and in a speaker-identification probe on the same cached
   utterances, H1/H11's value is likely small.
5. Spurious gains to guard against: higher similarity from reference-content leakage (T1 own-utterance versus T2
   other-utterance; Phase 3 S1), channel matching on VCTK or GLOBE (check on LibriTTS-R and per-pool), and UTMOS drops
   from noisy-floor reproduction.

## 5. Ranked priors

| rank | id | variant in plain words | score |
|---|---|---|---|
| 1 | H1 | separate voiced and unvoiced (and optional silence) means, concatenated and projected | 4 |
| 1 | H11 | pooled smoothed log-mel (spectral envelope and tilt) per voicing class as a side branch | 4 |
| 3 | H3 | per broad sound class means with back-off (fixed k-means or phone classes) | 3 |
| 3 | H6 | fixed k-means or GhostVLAD residual cluster means, projected down | 3 |
| 3 | H13 | better fusion across several references (pooled statistics, duration weights, learned weights) | 3 |
| 3 | H4 | attentive mean/std pooling (best on top of H1 voicing gates) | 3 |
| 7 | H2, H5, H7, H8, H9, H10, H14 | see text | 2 |
| 14 | H12 | frame-level network before pooling | 1 |

Combined first batch recommended for the screen: baseline, H1 (two and three-way), H11 added to baseline and to H1,
H9 (layer 1 plus layer 6) as the control that tests whether more raw information alone helps, H3/H6 fixed codebook, and
H13 fusion variants at inference only.
