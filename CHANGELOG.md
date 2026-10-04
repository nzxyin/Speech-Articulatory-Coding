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

### Changed
- `uv.lock` is now committed (removed from `.gitignore`).
