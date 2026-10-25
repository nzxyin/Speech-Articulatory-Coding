# Changelog

All notable changes to this fork. Earlier history is in `git log`.

## Unreleased

### Added
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
