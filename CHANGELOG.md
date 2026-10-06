# Changelog

All notable changes to this fork. Earlier history is in `git log`.

## Unreleased

### Added
- `sparc.compression`: toolkit for studying structural compression of SSL speech encoders (XLS-R 300M/1B/2B,
  WavLM Large) for speech-to-EMA prediction on MNGU0.
  - `mngu0`: the corpus's standard train/validation/test file sets, EMA de-normalized to mm, and EMA/audio
    alignment derived from the forced-alignment labels (end of leading silence) instead of from a pretrained
    model's predictions; utterances shorter than 0.2 s of EMA are excluded.
  - `metrics`: RMSE/MAE/PCC (per-utterance and pooled), velocity error, per-channel, per-articulator and
    per-phone-class errors, bootstrap confidence intervals.
  - `encoders`: layer-subset encoders (prefix truncation or any increasing subset) that reuse the Hugging Face
    forward pass.
  - `compute`: parameters, FLOPs, activation memory and latency/RTF of every truncation depth.
  - `extract` / `probe`: per-layer hidden-state caching and ridge probes with validation-selected alignment shift
    and regularization.
  - `lora` / `heads` / `train`: independent, shared-A and shared-with-per-layer-gate LoRA; linear, local
    convolution and windowed attention heads (causal or centered) with optional layer pooling;
    preemption-safe training with validation early stopping and a merged-weights test check.
  - `prune`: non-contiguous layer selection (greedy backward elimination and Block Influence) evaluated
    against prefixes of the same size.
  - `evaluate`: robustness of finished runs under white noise (20/10/5/0 dB SNR) and ±10% speed perturbation.
  - `analyze`: performance-vs-FLOPs and performance-vs-compression figures, a compute-matched table, seed
    aggregation, non-contiguous selection and robustness tables.
  - The linear head standardizes its input with train statistics, so gradient training starts at the ridge
    probe solution instead of moving away from it on the first steps.
  - Job scripts set `HF_HUB_OFFLINE=1` (all models are pre-downloaded to the shared cache).
  - Layer pooling over retained layers (`--pool static|attn`, `--per-articulator`, `--pool-norm`, `--pool-lr`):
    static softmax weights or frame-wise attention weights, global or per articulator with articulator-specific
    head projections. Pooled linear heads are ridge-initialized on the uniform pool and re-standardize their
    input every epoch in a function-preserving way. Mean pool weights are reported in `results.json`.
  - `components`: iterative structured pruning of attention heads and FFN neurons (Taylor importance on mask
    variables, physical removal with exact constant-bias replacement of emptied blocks, LoRA recovery per round,
    resumable), with a random-importance control.
  - `train.fit()` factored out of `train.main()`; random-init linear controls are named `_randinit`.
  - Tests: `tests/test_components.py`, `tests/test_pooling_stats.py`.
  - `scripts/xlsr_ema_{probe,compute,job}_slurm.sh` job scripts; tests in `tests/test_compression.py`.

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
