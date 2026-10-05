# Changelog

All notable changes to this fork. Earlier history is in `git log`.

## Unreleased

### Added
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

### Changed
- `uv.lock` is now committed (removed from `.gitignore`).
- Training: validation and prediction synthesis run under a fixed torch RNG (the DDSP noise branch draws from it);
  `CUBLAS_WORKSPACE_CONFIG` is set for deterministic runs; the Slurm training step passes `--cpus-per-task` to srun;
  committed Slurm scripts default to relative log paths; `cudnn.benchmark` is off by default.
