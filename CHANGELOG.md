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
- Download scripts (gitignored, local only) moved off the removed `cpu` partition to `msp-cpu`/`msp_cpu_qos` and from
  `huggingface-cli` to `hf download` (#8).
- Regression tests under `tests/` for all of the above.
