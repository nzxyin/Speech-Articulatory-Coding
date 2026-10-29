# Changelog

All notable changes to this fork. Earlier history is in `git log`.

## Unreleased

### Added
- Periodicity analysis revised after review (`docs/vocoder_periodicity_analysis.md`, `vocoders/eval/periodicity_analysis.py`): voicing judged by pYIN (independent of the vocoder's input tracker) and by forced-alignment phone voicing, the original measure kept as agreement with the input tracker, three voicing-transition class definitions (input tracker, pYIN, phone boundary), h as its own class, a second training seed of the fork recipe (`p4_b0_s1`, system `ref_s1`) as a noise reference with a seed-spread verdict in every table, `pyin` stage and `cfgdiff --pair`, run-comparability section (config differences, code states, RNG caveat), tests.
- Phase 9 shared pieces (`docs/spkemb_phase9_k12_inverter.md`): `vocoders/features/k12_inverter.py` (the WavLM Large k12 + BiGRU -> conv 9 inverters of the speech-to-EMA work applied as trained: truncated WavLM, 10 Hz zero-phase low-pass in float16, stored shift, stored target mean / std; `mngu0` de-normalizes to MNGU0 units, `all7` stays in per-speaker z units; output on the vocoder's 50 Hz grid with `feature_length(n24)` frames), `vocoders/features/inverter_pack.py` (resumable, sharded extraction of a packed split's 12 articulatory channels into chunk files, merge into a new packed split with every other array byte-identical, normalization statistics from the new channels over the given LibriTTS-R training splits written to a new file), `scripts/spkemb/p9_check_inverter.py` (reproduces the stored test scores, packed-grid alignment, live-vs-packed, timing), `tests/test_phase9_inverter.py`. Switches with unchanged defaults: `FeatureEncoder(inverter=...)` / `extract_pool(inverter=...)` / `RefConditions(inverter=...)` and the `inverter.name` config key (cloning evaluation encodes references and contents with the run's inverter; the reference-encoding cache key only changes when an inverter is set), and `vocoder.generator.ema_denorm` (HiFi-GAN: per-channel scale and offset of the 12 normalized articulatory inputs, linear in the speaker code, zero-initialized; default off leaves the module and state dict unchanged).
- Conditioning screen pre-registration frozen (`docs/spkemb_conditioning_screen.md`): arm 3 caveat (near-null under `k5_f0_loud`) with a reported, non-decisive `k5` variant for that arm and `p5_fv`, cost basis, validation own-speaker guard scope, test-pool variant, margin provenance; synthesis check on smoke checkpoints of four arms.
- Phase 8.3 listening-test preparation (`scripts/spkemb/listening/`, `artifacts/listening/`, `tests/test_listening_design.py`, `docs/spkemb_phase8.md`): fixed item list (34 VCTK, 8 GLOBE, 6 LibriTTS-R control), loudness-normalized blind-named stimuli, counterbalanced trial lists with attention checks, static listening page draft, mixed-model analysis and simulation-based power calculator, tested on simulated responses.
- Phase 8.2 constrained speaker-code fitting (`vocoders/eval/ood_svfit_p8.py`, `docs/spkemb_phase8.md`, `tests/test_ood_svfit_p8.py`): ECAPA-loss code fit with an L2 penalty toward the standard code and a log-mel-statistics naturalness term, CAM++-only selection on validation pools, WeSpeaker ResNet34 held out, own-speaker and swap test items, pre-registered advance rule; synthesized audio for the listening test saved. Also: acceptance is NaN (not available) instead of 0 for a cloning pool whose speakers all have one reference (`clone.calibration_stats` returns a NaN threshold, `item_table` masks acceptance), with a unit test.
- Periodicity analysis of the HiFi-GAN articulatory vocoder (`docs/vocoder_periodicity_analysis.md`): `FeatureFrontend(drop_periodicity=True)`
  and `models/hifigan_inputs.py` (HiFi-GAN with the original SPARC input set, 12 EMA + pitch + loudness = 14 channels, no
  periodicity input; speaker pooling, losses and data unchanged), experiment `periodicity_none_100k` (the `ablation_100k` recipe),
  `vocoders/eval/periodicity_analysis.py` (own-utterance resynthesis of fixed LibriTTS-R dev/test and VCTK validation/test subsets,
  inference-time reliance probes on the reference model, global, voicing and per-sound-class metrics from MFA forced alignments,
  speaker-level paired bootstrap), `scripts/spkemb/periodicity_mfa.sh`, tests.
- Conditioning-mechanism switches for the 24 kHz HiFi-GAN (Phase 8 conditioning study, `docs/spkemb_conditioning_screen.md`, `tests/test_conditioning.py`): `vocoder.generator.cond_type` (`film` | `adaln_zero`, new `AdaLNZero` in `models/film.py`), `cond_sites` (`residual` | `all`: zero-initialized modulation after the input convolution and every upsampling stage), `speaker.cond_stats` (pitch range of the references and source loudness level added to the code through a zero-initialized `StatsCode`; loudness `added` or `moved` out of the frame input). Defaults reproduce the previous generator bit for bit (same state-dict keys and outputs). Dataset keys `ref_f0_std`/`utt_loud`, cloning renderer support, five `p8c_*` experiment configs; no 100k run submitted.
- Conditioning screen revised after review (`docs/spkemb_conditioning_screen.md`): `cond_type: film_gate` (FiLM plus a zero-initialized gate, to separate the gate from normalization and modulation position in AdaLN-zero), `cond_contour: none | added | only` with `contour_smooth` (frame pitch and loudness contours drive a time-varying FiLM in every residual unit, in addition to or instead of the input), new configs `p8c_film_gate`, `p8c_film_f0range`, `p8c_film_contour_added`, `p8c_film_contour_only`; the screen is now five arms with the Phase 4/5 decision rule on the validation pools, a corrected smoke-run account and measured L40S step times. Defaults still bit-identical; tests extended.
- Phase 8.1 channel ceiling and input-path controls (`vocoders/eval/ood_channel.py`, `docs/spkemb_phase8_channel.md`, tests): own-speaker items on validation and test halves with the recording through Vocos mel copy synthesis, other-utterance F0/loudness controls and a different-speaker content control; variants chosen on validation, test run once; ECAPA, WavLM-SV, CAM++, UTMOS, speaker-level bootstrap.
- Out-of-domain speaker similarity recommendations (`docs/spkemb_ood_recommendations.md`) revised after review: evidence labelled validation-selected or test-judged, verifier-measured wording for the code-fitting gain, Phase 6 gains and data-arm figures cited by arm, existing constrained sweep rows reported, three redesigned next experiments (channel ceiling and input-path controls, constrained fitting with a non-selection verifier, listening test of the validation-chosen fit); correction note in `docs/spkemb_ood_gap.md`.
- SV-driven code fitting through the frozen generator (`vocoders/eval/ood_svfit.py`, new section in `docs/spkemb_ood_gap.md`): the 64-d speaker code or the speaker network input is fitted to maximize differentiable ECAPA similarity, scored on CAM++ and WeSpeaker ResNet34 (not in the loss) with UTMOS and WER guards; settings chosen on validation speakers, test items run once; plus a code-swap sensitivity check.
- Out-of-domain gap diagnosis (`vocoders/eval/ood_gap.py`, `docs/spkemb_ood_gap.md`): own-speaker resynthesis with the standard code, an oracle code, and test-time code fitting on the references or on the target utterance (steps and learning rate tuned on a LibriTTS-R dev set), against ground truth, on LibriTTS-R, VCTK and GLOBE subsets; speaker-level bootstrap summaries.
- Cloning split scoring from the split's own speakers (`clone.rescore_split`, `split_table`): `summarize-split` and
  `compare --split` rebuild the per-output table from stored embeddings, with the acceptance threshold and ground-truth
  calibration from the split's speakers, top-1 ranked among the split's speakers, only swaps whose target is in the split
  (cross-split source counts reported) and per-split stratum counts; `item_table(rank_speakers=...)`; tests against a direct
  recomputation for both halves.
- Real-speaker GLOBE test pool `globe_spk_test` (relaxed rule: test contributors with 2 utterances of >= 3 s, 1 reference +
  1 content utterance, 400 speakers, swap plan only), `PoolSpec.splits` / `min_long`, `pool-checks --pool`; split file
  `cloning.test.globe_spk_test`. Protocol doc: VCTK screen speakers, valid selection metrics, Amendment history, Phase 4
  inference-variant selection.
- Out-of-domain screen, clean validation scoring (`docs/spkemb_ood_screen.md`, Amendment 2): probes fitted only on held-out
  speakers in neither split (`splits.screen_scored`, `splits.probe_fit_mask`), VCTK scored speakers are the 40 cloning
  speakers, `ood_center` means taken per (corpus, fit / validation / test) group (`ood_embed.center_groups`); all variants
  re-scored, tests in `tests/test_spkscreen_ood_clean_scoring.py`.
- Out-of-domain encoder screen re-scored with selection on validation speakers (`docs/spkemb_ood_screen.md`, Amendment 1
  and Results): all six variants scored with `ood_report --split val|test`; only channel augmentation passes the rule on
  validation, and it is not confirmed on test (VCTK). `ood_embed` accepts `<run>:<architecture>`; tests in
  `tests/test_spkscreen_ood.py`.
- Validation / test protocol for speaker-embedding selection (`docs/spkemb_validation_protocol.md`, splits in
  `docs/spkemb_validation_splits.json`): `eval/splits.py` (split builder, `restrict_items`, `screen_masks`),
  `clone summarize-split` and `clone compare --split val|test` re-score stored cloning outputs on a speaker subset without
  re-synthesis (whole-pool re-score reproduces `summary.json` exactly), `ood_report --split` and
  `precond.heldout_stats(pooled=...)` for the encoder screen. No GLOBE test contributor has 3 utterances, so `globe_spk`
  is validation-only. Tests in `tests/test_validation_splits.py`.
- Phase 7 real GLOBE speaker labels (pre-registration `docs/spkemb_phase7_globe_speakers.md`):
  `features/globe_speakers.py` joins the GLOBE manifest with the Common Voice 21 transcript tables
  (`/data/user_data/xoy/common_voice_21_en_meta/`, CC0; `client_id` kept as an opaque hash, integer index
  `client_idx`, label `gcv<index>`) and writes `phase7/globe_train/{speakers,list_spk}.parquet` and the packed split
  `phase7/packed/globe_f_spk/globe.train` (same rows and linked features as the Phase 5 filtered split, only `speaker`
  changed); `report` / `pool-checks` give speaker statistics and the exact-overlap and ECAPA leak checks.
  `clone_pools` has a new pool `globe_spk` (GLOBE dev + test grouped by real contributor, `globe`'s size rules).
  Experiment `p7_gspk` (= `p5_fv` with the real-speaker GLOBE split) and `scripts/spkemb/p7_validate_configs.py`.
- Phase 6 speaker-encoder arms (code only; no vocoder run is started until the Phase 5 winner is known):
  `speaker.layer` accepts `l1l6` (periodicity-weighted means of WavLM Large layers 1 and 6, 2048-d), `vus3_l6` (the
  screen's voiced / unvoiced speech / silence layer 6 means, 3072-d) and `l6_rep3` (layer 6 mean repeated three times,
  3072-d, width control), all into the same speaker FFN (hidden 1024, output 64) with the z-score taken from
  `phase6/stats.npz`. `features/spk_phase6.py`: per-utterance side vectors (layer 1 mean, three-way layer 6 means)
  pooled with the screen's own code (`segment_masks`, `class_means`, `pool_weights`), extraction from a truncated WavLM
  forward (layers 1 and 6; checked against `spk_l6` at cosine >= 1 - 1e-6 on every utterance), sharded and resumable,
  stored by tag under `$SPKEMB_P6_CACHE` and found by utterance id (so the Phase 5 subset splits need no files of
  their own); `PackedStore` / `build_speaker` / `clone_pools.speaker_vector` / `FeatureEncoder.encode` use it, so
  training, validation and cloning-evaluation references (clean and re-encoded from audio) share the pooling. The `l6`
  path is untouched. Experiments `p6_<e1|e2|e3>_<p5_r|p5_f|p5_fv>` (include the Phase 5 experiment, change only
  `experiment_name` and `speaker.layer`); check `scripts/spkemb/p6_validate_configs.py`; tests
  `tests/test_spkemb_phase6.py`.
- Speaker-encoder screen precondition checks: `spkscreen/precond.py` (seed-paired speaker-level jackknife inference, Holm,
  per-corpus and GLOBE statistics), `precond_report.py`, `globe_embed.py`, `cache.py globe-manifest` and the `globe` cache
  set, capacity-matched control variants `cap_rep3` / `cap_rep2` (`variants/capacity.py`); results in
  `docs/spkemb_encoder_screen_results.md` ("Precondition checks").
- Phase 5 training data: `quality.py select` (thresholds UTMOS >= 3.0, Squim PESQ >= 2.5, Squim SI-SDR >= 15 dB, Squim
  STOI >= 0.95, bandwidth floor >= 10.5 kHz, clipping <= 1e-4, CER <= 0.10 where transcribed; GLOBE per-accent floor
  keeping at least the best 40 % of each accent by mean rank of UTMOS and PESQ; output `phase5/select/`),
  `features/subset_pack.py` (copies kept rows of a packed split into a new packed split, offsets recomputed), the
  `vctk_train.py` `extract` / `pack` hooks (GLOBE extractor and packer, real VCTK speaker ids), experiments `p5_r`
  (unfiltered LibriTTS-R + GLOBE), `p5_f` (filtered LibriTTS-R + GLOBE) and `p5_fv` (+ filtered VCTK): the Phase 4
  combination recipe at 100k steps, differing only in `data.train_splits`; config check
  `scripts/spkemb/p5_validate_configs.py`; tests `tests/test_subset_pack.py`, updated `tests/test_quality.py`,
  `tests/test_vctk_train.py`.
- Speaker-encoder screen write-up (`docs/spkemb_encoder_screen_results.md`): plain-word results tables for all 14 hypotheses, review-adjusted ranking, ideas to drop, and the specification of the 100k-step vocoder-in-the-loop test (three-way voiced / unvoiced / silence pooling, two-way split, layer 1 plus layer 6, with a capacity control).
- Speaker-encoder screen, multi-scale family (`spkscreen/variants/multiscale.py`, `spkscreen/multiref.py`, `scripts/spkscreen/submit_multiscale.sh`, tests `tests/test_spkscreen_multiscale.py`): layer 1 plus layer 6 means, learned layer weights, 0.5 s segment pooling, multi-layer aggregation before pooling, dilated-convolution and transformer frame networks, gradient reversal against content, F0 and both; plus an inference-only study of multi-reference fusion on the trained baseline encoders. 3 seeds each, table in `/data/user_data/xoy/spkemb/encoder_screen/results_multiscale.md`.
- Speaker-encoder screen, pooling-operator family (`spkscreen/variants/pooling_ops.py`, tests `tests/test_spkscreen_pooling.py`): attentive statistics (two), multi-head and multi-query attentive pooling, NetVLAD, GhostVLAD, fixed k-means codebook, mean+std, skewness, low-rank covariance, trimmed mean and quantiles; 3 seeds each, table in `/data/user_data/xoy/spkemb/encoder_screen/results_pooling.md`.
- Speaker-encoder screen, segment-aware family (`spkscreen/variants/segment.py`, tests `tests/test_spkscreen_segment.py`): voiced / unvoiced / silence split pooling (three forms), learned soft gates (two), fixed k-means codebooks (K 6 and 16) and a cepstral log-mel side branch (two); 3 seeds each, table in `/data/user_data/xoy/spkemb/encoder_screen/results_segment.md`.
- `sparc.vocoders.spkscreen` (speaker-encoder architecture screen, `docs/spkemb_encoder_screen.md`): resumable frame
  cache (WavLM Large layers 0, 1, 3, 6, 12, 24 in fp16, SPARC periodicity / loudness / F0, 80-bin log-mel; 6344 training
  utterances of 212 LibriTTS-R train.clean.100 speakers and the 5470 Phase 1 utterances, 41 GB), pluggable pooling-module
  interface with a variant registry (`spkscreen/variants/`), AAM-softmax plus auxiliary M1-head trainer, the Phase 1
  metric suite with speaker-level SEs, the pre-registered selection report, and the layer 6 periodicity-mean FFN
  baseline. Evaluation utterances under 0.4 s are padded with silence for extraction and cut back to their own frames.
- `sparc.vocoders.features.quality` (Phase 5): one shared, sharded, atomic and resumable quality scorer for every corpus
  (WADA-SNR and Squim SI-SDR, Squim STOI as the reverberation proxy because Brouhaha would downgrade the shared venv's
  speechbrain, UTMOS, Squim PESQ, 99 % energy bandwidth on the native-rate audio with an upsampled flag, clipping
  fraction, Whisper large-v3 WER/CER with the Phase 4 normalization), list builders for LibriTTS-R train / test.clean,
  the VCTK evaluation pool and the GLOBE training list, and a `report` subcommand (quantiles per corpus and GLOBE accent,
  retention at thresholds anchored on LibriTTS-R test.clean and the VCTK pool, accent-loss flag). Runner scripts
  `scripts/spkemb/p5_job.sh`, `p5_quality_anchors.sh`. Test: `tests/test_quality.py`.
- Phase 5 quality scorer: `vctk_train` corpus (leak-filtered list, trimmed wav24) and `vctk_eval_trim` anchor (pool scored on the
  training list's trim span); the report's retention and joint include WER/CER where every row has a transcript score.
- `sparc.vocoders.features.vctk_train` (Phase 5): VCTK training list (all speakers except the 40 evaluation-pool speakers,
  trimmed 1-15 s, native trim bounds recorded), resumable sharded trimmed 24 kHz wav preparation, and the ECAPA leak
  check (cosine >= 0.70) against the LibriTTS-R, VCTK and GLOBE evaluation pools; `extract` / `pack` are disabled hooks
  until the Phase 5 quality thresholds exist. Test: `tests/test_vctk_train.py`.
- Speaker-embedding Phase 3 (branch `spkemb-phase3`; design in `docs/speaker_embedding_plan.md` on `multispeaker-eval`):
  - `sparc.vocoders.features.spk_layers`: periodicity-weighted mean and std of all 25 WavLM Large hidden states per
    packed utterance (fp16, `<split>/{mean,std}.npy` + `stats.npz`); layers 0 and 6 are checked against the cached
    `spk_l0` / `spk_l6` on every utterance.
  - `LayerPoolSpeakerEncoder` (learned softmax layer weights over the z-scored statistics, MLP to 64-d; optional VAE
    with a product-of-Gaussians posterior over references including the prior); `SpeakerFFN.embed` for K references;
    `gaussian_kl`, `total_correlation` (minibatch-weighted sampling).
  - `PackedStore(speaker_layer="layers")`, `CropDataset(n_refs=...)` (1-3 references per crop) and
    `FullUtteranceDataset(multi_ref=True)` (T3 = every pool reference through the encoder).
  - `training/speaker_terms.py` and module hooks: capacity-controlled KL, beta-TC side batch, mixed / prior codes with
    adversarial-only loss, latent cycle through frozen WavLM; `speaker.*` config keys (defaults keep the comparison's
    behaviour); experiments `p3_b1` .. `p3_b6`; `scripts/spkemb/p3_job.sh`, `p3_train.sh`.
- Experiment `main_seed1`: the primary run (`main`) with training seed 1, for a seed-to-seed comparison of the three vocoders.
- `CLAUDE.md` with the project's current direction and findings, and this changelog.
- `docs/vocoders/INTERFACES.md`: module contract for the 24 kHz articulatory vocoder comparison (feature cache, data,
  models, losses, training, CLI and SLURM scripts).
- `sparc.vocoders` package skeleton with shared pieces: constants and the SPARC feature-length formula, the `Vocoder`
  base class, `FeatureFrontend` (feature normalization), `FiLM` and `FiLMLayerNorm`, and `SpeakerFFN`.
- Hydra config groups `paths`, `data` and the root `vocoder_config.yaml`; console scripts `sparc-cache` and
  `sparc-predict` (implemented in following commits).
- `third_party/` with the MIT license texts of Vocos and DDSP-Articulatory-Vocoder, from which code is ported.
- Direct dependencies `pandas` and `pyarrow`; dev dependency `pytest`.

- Feature cache (`sparc.vocoders.features`, `sparc-cache`): Parquet manifest of all 358,503 filtered LibriTTS-R
  utterances; a SPARC en+ extractor with the refit linear head (hash- and weight-checked), per-utterance pitch seeding,
  un-normalized loudness, layer-0/layer-6 speaker pooling and the frozen en+ embedding; resumable `trainer.predict`
  extraction with atomic per-utterance files; packing into memory-mapped split arrays; training-set statistics; a
  preempt array job with a GPU preflight.
- Data pipeline (`sparc.vocoders.data`): sox-equivalent peak gain, a resumable counter-based sampler, aligned random
  crops with loudness rescaled by the gain, cross-utterance speaker references (`p_cross`), and full-utterance
  evaluation datasets for the same-utterance, cross-utterance and speaker-mean conditions.
- Vocoders (`sparc.vocoders.models`): HiFi-GAN with upsampling [8, 5, 4, 3] and the shared FiLM; the DDSP24
  harmonic-plus-noise vocoder (200 Hz controls, F0 anchored at the CREPE frame centre, float32 synthesis); Vocos with
  articulatory input (options A/B/C) and FiLM layer norms. Size variants as configs.
- Losses and discriminators (`sparc.vocoders.losses`): MPD and MRD (from Vocos), MSD with the pooling fixed, hinge and
  LSGAN losses, feature matching, the 24 kHz mel loss and the DDSP-AV multi-scale spectral loss.
- Training (`sparc.vocoders.training`, `sparc-train vocoder=... experiment=...`, `sparc-predict`): a manual-optimization
  GAN module with generator-step counters, mel-only warm-up, warm-up plus cosine schedules, validation with audio
  logging, and preemption-safe checkpoints at batch boundaries with exact resume; `sparc-train` dispatches between the
  new and the legacy 16 kHz training; Slurm launch scripts for `preempt`.
- Tests for every module (`tests/`, 297 tests).
- Experiment config `ablation_100k` (100k generator steps, main recipe) for ablations and the Vocos option that lost
  the A/B pilot.
- Ablation experiment configs (decision D10), 100k generator steps each: `abl_pcross0` (speaker reference always the
  target utterance), `abl_msd` (MPD + MSD), `abl_native` (the fork's loss, Adam 0.5/0.9 at lr 1e-4, no mel-only
  warm-up). Each is compared with `ablation_100k` for the same vocoder.
- Phase 3 evaluation (`sparc.vocoders.eval`, `sparc-eval`, contract `docs/vocoders/EVALUATION.md`): resumable chunked
  stages for gt audio, reference systems (Vocos mel copy synthesis, the authors' en+ decoder at 16 kHz), vocoder
  synthesis, signal metrics (PESQ wb, MCD, MR-STFT, mel L1 and three bands), UTMOS, Whisper large-v3 WER/CER, ECAPA and
  WavLM-SV speaker embeddings, no-dither SPARC re-extraction with prosody and EMA consistency, controllability probes,
  efficiency (parameters, RTF, measured lookahead), aggregation with speaker-cluster bootstrap CI95 and paired
  differences, and matched samples; Slurm scripts `eval.sh` and `submit_eval.sh`; tests under `tests/eval/`.
- `SparcFeatureExtractor`: opt-in `extractor.dither=false` disables CREPE's pitch dither (default unchanged).
- `sparc-predict`: `predict.skip_existing` (default false) leaves out utterances whose WAV exists; `run()` accepts
  extra callbacks.
- Dependencies jiwer, pesq, pysptk, vocos, speechbrain; pytest marker `slow`.
- Ablation evaluation: `eval_systems=ablations` selects the ablation systems and `eval.table_name` writes their
  tables under `tables/<split>/<name>/`, leaving the main tables untouched.
- Results: `docs/vocoders/RESULTS.md` (Phase 3 write-up) and `docs/vocoders/results/` (main and ablation tables,
  paired comparisons, controllability, efficiency and failure-case analyses).
- Results: section 5.3 of `docs/vocoders/RESULTS.md` (paired X-Y EMA probes) and `docs/vocoders/results/xy_probes.md`.

- Evaluation stage `probes_xy` (`eval/probes_xy.py`): EMA edits per sensor along X only, Y only, the natural principal
  and the orthogonal direction, as constant offsets, speech-only offsets, dynamic scaling and local bumps; projection
  gain, leakage and angle from stored moments. System set `eval_systems=seeds` (seed-0 and seed-1 runs, seed-to-seed
  pairs).

- Experiment `main_800k` (the primary run with an 800k-step budget, seed 0) and evaluation system set
  `eval_systems=long` (800k against the two 400k seeds of HiFi-GAN and Vocos), to test whether 400k steps suffice.
- Results: sections 3.1 (test.other), 4.1 (run-to-run variation, second seeds) and 6.1 (compute and training length)
  of `docs/vocoders/RESULTS.md`; tables in `docs/vocoders/results/{test.other,seeds}/` and `results/convergence.md`.
- Results: section 6.1.3 of `docs/vocoders/RESULTS.md` (the 800k runs of HiFi-GAN and Vocos against the two 400k
  seeds), with the summary, open questions and reproducibility notes updated; tables in `docs/vocoders/results/long/`
  and section 3 of `results/convergence.md`.

### Changed
- `uv.lock` is now committed (removed from `.gitignore`).
- Training: validation and prediction synthesis run under a fixed torch RNG (the DDSP noise branch draws from it);
  `CUBLAS_WORKSPACE_CONFIG` is set for deterministic runs; the Slurm training step passes `--cpus-per-task` to srun;
  committed Slurm scripts default to relative log paths; `cudnn.benchmark` is off by default.

### Fixed
- Evaluation WER: Whisper's text normalizer deleted words in parentheses and brackets, which LibriTTS-R references
  use for spoken parentheticals (40 test-clean texts), so they counted as insertions for every system (absolute WER
  was about 11 % too high). The bracket characters are now removed first; stage `asr_rescore` recomputes stored counts.
- Efficiency stage: GPU and CPU timing runs shared one DONE marker, so the first to finish made the other skip.
- Evaluation samples: the samples directory now depends on the split (`samples_<split>` for splits other than
  test.clean), so a test.other run cannot overwrite the test.clean samples.
