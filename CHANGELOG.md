# Changelog

All notable changes to this fork. Earlier history is in `git log`.

## Unreleased

### Added
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
- Results: seed-1 controllability probes and natural-variation baseline (section 5.2 of `docs/vocoders/RESULTS.md`) and
  seed-1 X-Y probes (section 5.3); `docs/vocoders/results/{probes_seeds,xy_probes_seeds,xy_probes_s1}.md`.

### Changed
- `uv.lock` is now committed (removed from `.gitignore`).
- Training: validation and prediction synthesis run under a fixed torch RNG (the DDSP noise branch draws from it);
  `CUBLAS_WORKSPACE_CONFIG` is set for deterministic runs; the Slurm training step passes `--cpus-per-task` to srun;
  committed Slurm scripts default to relative log paths; `cudnn.benchmark` is off by default.
- Merged `main` (legacy 16 kHz fixes for issues #8-#16, listed under Fixed) on 2026-10-20. `sparc.cli.train` keeps
  both the vocoder dispatcher and the legacy path's SLURM handling: the dispatcher still sets
  `SLURM_JOB_NAME=interactive` before the legacy trainer, and the job name recorded at import keeps run naming,
  auto-resume and requeue working. README gains a section on the 24 kHz vocoder comparison; `CLAUDE.md` and
  `docs/vocoders/INTERFACES.md` note which constraints the merged fixes resolved.

### Fixed
#### Legacy 16 kHz pipeline (`sparc.training`, `sparc-encode` / `sparc-decode`, `SPARC`; merged from `main`)
- `MultiScaleDiscriminator` mean-pools are both `AvgPool1d(4, 2, padding=2)`, so the three scales see 1x/2x/4x input
  instead of 1x/2x/8x (#9). Existing state dicts still load.
- Vocoder lr schedule halves every 200k steps by default (was 8k, which drove the lr to ~1e-12 by 320k);
  `lr_static_after` defaults to null; a warning is emitted if the multiplier at `max_steps` falls below 1e-3 (#10).
- `HiFiGANGenerator`: forward no longer mutates its input; the N(0, 0.01) init now runs before weight norm and takes
  effect for fresh models (pretrained decode is bit-identical); `SoftClamp` honors `temp`; unsupported paddings raise
  instead of calling `exit()` (#14).
- Vocoder training (#15): `max_steps`/`checkpoint_every_n_steps` count batches; SLURM SIGUSR1 requeue
  (`--requeue --signal=B:USR1@120`), `save_on_exception`, RNG state in checkpoints, a per-run checkpoint dir
  (`run_name`, default `slurm_<jobid>` under sbatch) with auto-resume; configurable `devices`/`strategy` (multi-GPU uses
  `ddp_find_unused_parameters_true` and falls back to `LightningEnvironment` under single-task sbatch);
  `toggle_optimizer` keeps discriminator gradients out of the generator step.
- Encode (#11, #12, #13): an explicit `linear_model_path` overrides the checkpoint head and is exposed as
  `model.linear_model_path`; `filtfilt` runs per utterance over its valid length (batched encode no longer corrupts
  short utterances, short utterances no longer crash); `sparc-encode` logs failures, exits non-zero, writes atomically
  and skips only when all outputs exist; zero periodicity weights fall back to uniform speaker pooling instead of NaN;
  `encode(seed=...)` and `deterministic_pitch` (default on in `sparc-encode`, seed `crc32(id)`) make CREPE pitch
  reproducible; `use_penn` is reported as unimplemented; `SourceExtractor._extract_pitch` works with file paths;
  `prepare_spk_raw.py --en-plus-compatible`; `configs/feature_extraction.yaml` matches the hub copy.
- `SPARC.encode(..., concat=True)` works for multi-utterance batches and returns one `{features, spk_emb}` dict per
  utterance; `SourceExtractor` pitch statistics fall back to uniform weights over the valid frames when every
  periodicity weight is 0 (were NaN); `scripts/prepare_spk_raw_slurm.sh` usage text lists the `[device]` argument
  that precedes `[limit]` (#16).
- Download scripts (gitignored, local only) moved off the removed `cpu` partition to `msp-cpu`/`msp_cpu_qos` and from
  `huggingface-cli` to `hf download` (#8).
- Regression tests under `tests/` for all of the above.
- `sparc-encode` writes temp files as `.<id>.npy.tmp`, so a killed job no longer leaves files that match `*.npy`;
  `sparc-decode` pairs features and speaker embeddings by file stem (was by sorted position) and warns about unpaired
  files (#12).
- `sparc-encode` records the model, config, head (sha256) and `deterministic_pitch` in `<save_dir>/encode_meta.json`
  and exits if a later run's settings differ (`allow_meta_mismatch=true` overrides); `save_dir`s with outputs but no
  manifest only get a warning; a missing `model.linear_model_path` is reported before the model loads (#12).
- `SourceExtractor` restores the global numpy RNG when pitch decoding raises; mutable `outputs={}` defaults in the
  extractors are gone; `use_penn` is logged at info level instead of warning on every `load_model("en+")`, and the
  local en+ config sets it to false; `spk_raw_subdir` (`VocoderDataset`, `train_config.yaml`, `training/sample.py`,
  default `spk_raw`) lets training and sampling use `prepare_spk_raw --en-plus-compatible` output (#13).
- `HiFiGANGenerator` accepts a negative `pitch_axis` again (-1 produced twice the channels after the #14 rewrite);
  `load_model` accepts configs without a `linear_model_path` key (#11, #14).
- `max_steps=-1` passes through to Lightning unchanged (was turned into -2); `sparc.cli.train` records
  `SLURM_JOB_NAME` at import (keyed by job id), so a wrapper that overwrites it before calling the trainer no longer
  disables the per-run checkpoint dir, auto-resume and SIGUSR1 requeue; `train_slurm.sh` exits with the trainer's
  real status after a trapped signal (#15).
- Tests for the encode/decode CLIs (`tests/test_encode_cli.py`), extractor RNG and file paths
  (`tests/test_extractor_fixes.py`), head selection (`tests/test_load_model_head.py`), generator pitch handling and
  save-on-exception resume.

#### 24 kHz vocoder comparison (`sparc.vocoders`)
- Evaluation WER: Whisper's text normalizer deleted words in parentheses and brackets, which LibriTTS-R references
  use for spoken parentheticals (40 test-clean texts), so they counted as insertions for every system (absolute WER
  was about 11 % too high). The bracket characters are now removed first; stage `asr_rescore` recomputes stored counts.
- Efficiency stage: GPU and CPU timing runs shared one DONE marker, so the first to finish made the other skip.
- Evaluation samples: the samples directory now depends on the split (`samples_<split>` for splits other than
  test.clean), so a test.other run cannot overwrite the test.clean samples.
