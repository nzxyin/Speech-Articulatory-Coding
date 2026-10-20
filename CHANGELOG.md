# Changelog

All notable changes to this fork. Earlier history is in `git log`.

## Unreleased

### Fixed
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
  reproducible; `use_penn` warns; `SourceExtractor._extract_pitch` works with file paths;
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
