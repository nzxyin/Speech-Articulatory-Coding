# Articulatory vocoder comparison: module contract

This file fixes module boundaries, data formats and signatures for the 24 kHz vocoder comparison (HiFi-GAN, DDSP,
Vocos driven by SPARC features). Read it before writing code; change a signature here first if it must change.
Design rationale: the Phase 1 report (decisions D1-D10, all approved). New code lives in `src/sparc/vocoders/`; the
existing `sparc` modules are reused, not edited, except where noted.

## 0. Conventions

- Audio: 24 kHz mono float32. Feature rate 50 Hz. One frame = `HOP = 480` samples. Every vocoder maps T frames to
  exactly `480·T` samples.
- Feature tensor in batches: `features[B, 15, T]`, float32, **raw (un-normalized) values**:

  | channel | content | unit |
  |---|---|---|
  | 0-11 | EMA from the refit linear head: TDX, TDY, TBX, TBY, TTX, TTY, LIX, LIY, ULX, ULY, LLX, LLY | head units |
  | 12 | F0 from CREPE (continuous through unvoiced frames) | Hz |
  | 13 | loudness = `loud_raw · g`: mean absolute amplitude per frame of the **un-normalized** 16 kHz waveform, times the training/eval gain g | linear |
  | 14 | periodicity after SPARC thresholding (0 or in [0.4, 1)) | 0-1 |

  The cache also keeps SPARC's own z-scored loudness (`zloud`); models never read it.
- Speaker conditioning: the LightningModule owns one `SpeakerFFN` that maps a cached 1024-d pooled WavLM vector to
  `spk[B, 64]`. Vocoders receive `spk[B, 64]`.
- Frame t of the features corresponds to 24 kHz samples `[480t, 480(t+1))` (decision D2). Feature centres inside
  SPARC (24 kHz samples): loudness at `480t` (measured), F0/periodicity at `480t + 120` (measured), EMA at
  `480t + 270 ± 40` (measured; the WavLM CNN support centre is `480t + 300`). Hand-built paths (the DDSP oscillator and
  any explicit loudness gain) must anchor each stream at its own centre.
- Constants live in `sparc/vocoders/constants.py`. No paths or hyperparameters are hard-coded in Python: everything
  comes from Hydra configs (`src/sparc/conf/`), and paths come from environment variables through
  `${oc.env:...}` interpolation.
- Randomness that affects data (crop start, gain, speaker reference) is drawn from a counter-based generator
  `np.random.default_rng([seed, epoch, position])`, never from worker or global RNG state, so a resumed run replays
  exactly the same stream.

## 1. Feature cache (`sparc.vocoders.features`)

### 1.1 Manifest (`manifest.py`)
`build_manifest(cfg) -> pandas.DataFrame`, written to `${paths.cache_root}/manifest.parquet`. One row per filtered
LibriTTS-R utterance (all 7 splits; names as in the HF dataset: `train.clean.100`, `train.clean.360`,
`train.other.500`, `dev.clean`, `dev.other`, `test.clean`, `test.other`). Columns:
`id, split, speaker, chapter, wav_path, n24 (int64), duration (float32), T (int32, predicted feature length),
encodable (bool, T >= 19), text`. `T = floor((ceil(2·n24/3) - 80) / 320)`. Source: the filtered id lists (parquet
`id` column only) mapped to `${paths.libritts_r_raw}/<split-dir>/<spk>/<chapter>/<id>.wav`. Never read the parquet
audio column. Validated Phase 1 TSVs exist at `/data/user_data/xoy/sparc-vocoders/phase1/data/manifest_{clean,other}.tsv`
(read them with `quoting=csv.QUOTE_NONE`).

### 1.2 Extractor (`extractor.py`)
`SparcFeatureExtractor(cfg)`:
- Builds `sparc.load_model("en+", device=..., linear_model_state_dict=<from the refit .npz>)` with `HF_HUB_OFFLINE=1`
  and the project's private pinned hub cache (`HF_HUB_CACHE`, SPARC snapshot `2e6a07d4…`, WavLM-large `c1423ed9…`),
  asserts the `.npz` sha256 equals `cfg.inversion.linear_sha256`, and asserts the loaded head weights equal the npz
  weights (never use `linear_model_path=` alone: it is silently ignored, fork issue #11). Also assert
  `coder.source_extractor.q == 2` and `pitch_hop_length == 160`: loading the checkpoint without the en+ yaml silently
  gives `pitch_q = 4` and moves the pitch centre.
- `extract(wav24: np.ndarray, seed: int) -> dict`, one utterance (never batched: issue #12):
  1. `raw16 = librosa.resample(wav24, orig_sr=24000, target_sr=16000)` (default `soxr_hq`, as SPARC does).
  2. Run SPARC's own path on `raw16` (it z-scores internally) with `np.random.seed(seed)` set immediately before CREPE
     (torchcrepe's pitch dither uses the global numpy RNG).
  3. Capture WavLM `hidden_states[0]`, `[6]` and `[9]` from one forward pass; EMA must equal what `coder.encode`
     returns (test: max abs difference < 1e-5 on the same device).
  4. `loud_raw`: `AmplitudeHistogram(320)` (from `sparc.src_extractor`) on `raw16` **before** z-scoring.
  5. Speaker vectors: periodicity-weighted mean over frames of `hidden_states[0]` and `[6]`, using the final
     (thresholded, gated) periodicity, truncated to the shorter length, exactly like
     `SpeakerEncoder._get_spk_emb`. If the weights sum to 0, use the uniform mean and set `spk_fallback = True`.
     Also store the frozen en+ 64-d embedding (en+ FFN applied to the layer-6 vector) for the ablation.
  6. Truncate every per-frame array to `T` = EMA length; assert `T == manifest T`.
- Returns: `feats (T, 15) float32` = [EMA(12), F0 Hz, zloud, periodicity] (SPARC's column order),
  `loud_raw (T,) float32`, `spk_l0 (1024,)`, `spk_l6 (1024,)`, `spk_enplus64 (64,)`, `spk_wsum float`,
  `spk_fallback bool`, `n24 int`, `T int`, `peak24 float` (max |x| of the 24 kHz file), `seed int`.
- `seed = zlib.crc32(id.encode()) & 0x7FFFFFFF`.

### 1.3 Per-utterance cache files (`cache.py`)
`${paths.cache_root}/utt/<split>/<id>.npz`, written with `np.savez` to `<id>.npz.tmp` then `os.replace`.
`is_valid(path, T_expected)` loads and checks keys, shapes and finiteness. The extraction job skips valid files and
rewrites invalid ones. A single `${paths.cache_root}/meta.json` records: fork git commit, torch/torchaudio/
transformers/torchcrepe/librosa versions, en+ ckpt sha256, HF snapshot hash, refit npz sha256, extractor config,
seed rule. Each npz stores `meta_hash` (sha256 of meta.json) and `gpu` (device name).

### 1.4 Extraction driver (`cache_module.py`)
Lightning `trainer.predict` over a manifest shard: `ManifestShardDataModule(shard_index, num_shards, skip_valid=True)`
(DataLoader workers read audio with `soundfile`, batch size 1), `FeatureCacheModule(LightningModule)` whose
`predict_step` calls the extractor, and a `CacheWriter(BasePredictionWriter)` that writes the npz files. Errors are
caught per utterance, logged to `${paths.cache_root}/errors/<shard>.jsonl`, and never silently dropped; the job exits
non-zero if any utterance other than the known `encodable == False` ones failed.

### 1.5 Packing (`pack.py`)
`pack_split(split)` writes `${paths.cache_root}/packed/<split>/`: `feats.npy (N_frames, 15) float32`,
`loud_raw.npy (N_frames,) float32`, `spk_l0.npy (N_utt, 1024) float32`, `spk_l6.npy (N_utt, 1024) float32`,
`spk_enplus64.npy (N_utt, 64) float32`, `index.parquet` with `id, speaker, chapter, wav_path, n24, T, offset
(frame offset into feats.npy), peak24, seed, duration, spk_wsum, spk_fallback`. All `.npy` files are loaded with
`np.load(..., mmap_mode="r")`.

### 1.6 Statistics (`stats.py`)
`compute_stats(cfg) -> dict`, saved to `${paths.cache_root}/stats/train_stats.json`, over the training splits listed
in `cfg.data.train_splits`:
`ema_mean[12], ema_std[12]`; `logf0_mean, logf0_std` (ln F0, all frames); `loud_log_mean, loud_log_std` of
`ln(loud_raw · g + 1e-4)` with g drawn per utterance from the training gain distribution (fixed seed);
`per_mean, per_std`; `spk_l0_mean[1024], spk_l0_std[1024], spk_l6_mean[1024], spk_l6_std[1024]` (std floored at
1e-6); frame and utterance counts; `meta_hash`.

### 1.7 CLI
`sparc-cache` (Hydra, `config_name=cache_config`): `stage=manifest | extract | pack | stats`, with
`shard_index`/`num_shards` for `extract` (defaults from `SLURM_ARRAY_TASK_ID`/`SLURM_ARRAY_TASK_COUNT`).

## 2. Data (`sparc.vocoders.data`)

- `gain.py`: `sample_gain_db(rng) -> float` = `round(rng.uniform(-6, -1), 2)`; `gain_factor(peak24, db) -> float` =
  `10**(db/20) / peak24` (sox `norm` equivalent; applied to the whole utterance, so computed from the full-file peak);
  eval uses `db = -3.0`.
- `sampler.py`: `ResumableSampler(num_items, seed, rank=0, world_size=1)`. Each epoch is a permutation from
  `default_rng([seed, epoch])`, strided by rank. It yields keys `(epoch, position, item_index)`.
  `state_dict() -> {"epoch", "position"}` and `load_state_dict()`; the training module records the number of
  samples consumed and restores it, so the stream continues exactly where it stopped.
- `dataset.py`:
  - `CropDataset(packed_dirs, stats, crop_frames=64, p_cross=0.5, ref_min_dur=3.0, speaker_layer="l6", seed)`.
    Items: utterances with `T >= crop_frames`. `__getitem__(key)`: rng = `default_rng([seed, epoch, position])`;
    `t = rng.integers(0, T - N + 1)`; `db = sample_gain_db(rng)`; `g = gain_factor(peak24, db)`;
    `audio = soundfile.read(wav_path, start=480t, frames=480N)[0] · g`; `features` = packed `feats[offset+t :
    offset+t+N]` with channel 13 replaced by `loud_raw[...] · g`; speaker: with probability `p_cross` the raw vector of
    another utterance of the same speaker from the training pool (different chapter preferred, duration >= 3 s;
    fallbacks: any other utterance of the speaker, then itself), otherwise its own. Returns a dict with
    `features (15, N)`, `audio (1, 480N)`, `spk_raw (1024,)`, `gain_db`, `utt_index`, `ref_index`, `is_cross`,
    `position`.
  - `FullUtteranceDataset(packed_dir, ids=None, gain_db=-3.0, condition="T1"|"T2"|"T3", speaker_layer="l6")`:
    full T frames, audio = first `480·T` samples of the file times g, speaker vector by condition: T1 own; T2 one
    reference chosen deterministically (pool = same split and speaker, other utterances >= 3 s, different chapter if
    any; index drawn with `default_rng([0, int(sha1(id)[:16], 16)])`); T3 the `spk_wsum`-weighted mean of the pool.
    Returns `features (15, T)`, `audio (1, 480T)`, `spk_raw`, `id`, `condition`, `ref_id`. Batch size 1.
- `datamodule.py`: `VocoderDataModule(cfg)` with `train_dataloader` (CropDataset + ResumableSampler,
  `drop_last=True`), `val_dataloader` (fixed dev.clean subset, T1, plus the fixed 20-utterance logging subset), and
  `predict_dataloader` (split and conditions from config).

## 3. Models (`sparc.vocoders.models`)

Already written (shared, do not change signatures without updating this file):
- `base.py`: `Vocoder(nn.Module)`, abstract `forward(features[B,15,T], spk[B,64]) -> wav[B,1,480T]`.
- `frontend.py`: `FeatureFrontend(stats, pitch_mode="log", voiced_flag=False)`; `forward(features) ->
  FrontendOut(x[B, C_in, T], f0_hz[B,1,T], periodicity[B,1,T], loudness[B,1,T])`, out of place; `C_in = 15 + voiced_flag`.
- `film.py`: `FiLM(cond_dim, channels)`; `forward(x, cond, channel_dim=1)` returns `x·(1 + γ) + β`; zero-initialized.
- `speaker.py`: `SpeakerFFN(in_dim=1024, hidden_dim=1024, out_dim=64, dropout=0.2, mean=None, std=None)`.

To implement:
- `hifigan.py`: `HiFiGANVocoder(stats, in_channels=15, channels=512, upsample_scales=(8,5,4,3),
  upsample_kernel_sizes=(16,10,8,6), resblock_kernel_sizes=(3,7,11), resblock_dilations=((1,3,5),)*3,
  spk_dim=64, pitch_mode="log", voiced_flag=False)`. Port of `sparc/generator.py` and `sparc/block.py` with: the
  shared `FiLM` on every residual unit (after the second conv of each dilation pair, before the residual add,
  replacing the MLP + SoftClamp FiLM); no in-place input mutation; keep PyTorch's default conv init (the fork's
  N(0, 0.01) init never took effect because it ran after weight norm, issue #14, so the validated recipe uses the
  default; delete the dead code instead of activating it); `torch.nn.utils.parametrizations.weight_norm`;
  LeakyReLU(0.1); tanh output.
- `ddsp.py`: `DDSPVocoder(stats, channels=256, n_harmonics=100, n_noise_bands=129, control_rate=200,
  noise_gain_cap=0.25, post_filter_taps=1537, f0_anchor_offset=120, spk_dim=64, ...)`, the DDSP24 design (report
  section 6; prototype `~/sparc-vocoders-work/probes/ddsp/ddsp24.py`, which lacks the F0 anchor and the autocast
  guard). Synthesis in float32 with autocast disabled; float64 phase accumulation of the fundamental; F0 clamped to
  [50, 550] and anchored at `480j + f0_anchor_offset` (default 120) when interpolated to 24 kHz; any explicit loudness
  path anchored at `480j`. Config options for ablations: `control_rate` (100, 200, 400), `trunk_stacks` (1 or 2;
  the 2-stack trunk's receptive field of about ±66 frames exceeds the 64-frame crop), `periodicity_gate` (bool),
  `f0_anchor_offset`.
- `vocos.py`: `VocosVocoder(stats, option="B", dim=512, intermediate_dim=1536, num_layers=8, spk_dim=64, ...)`;
  option A (no upsampling, n_fft 1920, hop 480), B (2× linear interpolation, n_fft 960, hop 240), C (4×, 480, 120);
  `same` padding; magnitude clip at `n_fft/2`; `FiLMLayerNorm` (LayerNorm without affine followed by `FiLM`) in the
  embed norm and all blocks; FiLM re-zeroed after the backbone's `_init_weights`; head in float32 with autocast
  disabled. Ported from Vocos (MIT; keep the notice).
- Every vocoder takes the stats dict (or a path) and builds its own `FeatureFrontend`; the stats are buffers, so they
  are saved in checkpoints.

## 4. Losses and discriminators (`sparc.vocoders.losses`)

- `discriminators.py`: `MultiPeriodDiscriminator(periods=(2,3,5,7,11))` and `MultiResolutionDiscriminator(
  fft_sizes=(2048,1024,512), num_bands=5, channels=32)` ported from Vocos (replace `einops.rearrange` with
  `permute`); `MultiScaleDiscriminator` from the fork with the pooling bug fixed (scales 1, 1/2, 1/4; issue #9).
  Interface: `forward(y[B,1,L], y_hat[B,1,L]) -> (real_logits: list, fake_logits: list, real_fmaps: list[list],
  fake_fmaps: list[list])`.
- `losses.py`: `hinge_d_loss`, `hinge_g_loss`, `lsgan_d_loss`, `lsgan_g_loss` (each averaged over sub-discriminators,
  returning `(total, per_disc)`); `feature_matching_loss(real_fmaps, fake_fmaps)` (Vocos form: sum over layers of
  mean |r - g|, divided by the number of sub-discriminators); `MelSpectrogramLoss(sample_rate=24000, n_fft=1024,
  win_length=1024, hop_length=256, n_mels=100, f_min=0, f_max=12000, power=1, center=True, mel_scale="htk",
  clamp=1e-5)` (L1 of natural-log mel); `MultiScaleSpectralLoss(fft_sizes=(2048,1024,512,256,128,64),
  hop_ratio=0.25, alpha=1.0)` (DDSP-AV form, summed over scales); `mr_stft_distance` for validation.

## 5. Training (`sparc.vocoders.training`)

`VocoderGANModule(cfg)` (LightningModule; the resolved config is saved with `save_hyperparameters`):
- Builds the generator with `hydra.utils.instantiate(cfg.vocoder.generator, stats=stats)`, the `SpeakerFFN`
  (z-score stats for the chosen layer), the discriminators and losses from `cfg.loss`.
- Manual optimization. `opt_g` = AdamW over generator + speaker FFN; `opt_d` = AdamW over discriminators; lr
  `cfg.optim.lr` (2e-4), betas (0.8, 0.9), weight decay 0.01. Schedules: linear warm-up (`cfg.optim.warmup_steps`,
  1000) then cosine to 0 at `cfg.train.max_g_steps`; the generator schedule is indexed by `g_step`, the discriminator
  schedule by `d_step`.
- Counters `g_step`, `d_step`, `samples_consumed` are stored in the checkpoint (`on_save_checkpoint` /
  `on_load_checkpoint`). Training stops when `g_step >= cfg.train.max_g_steps` (`trainer.max_steps = -1`).
  Do not use `trainer.global_step` for any schedule (it counts both optimizers).
- Step: `spk = ffn(spk_raw)`; `wav_hat = G(features, spk)`; assert shape `[B, 1, 480N]`. While
  `g_step < cfg.train.mel_warmup_steps` (10k): generator loss = `45 · mel` only; discriminators untouched. After:
  discriminator update (hinge, MPD weight 1.0, MRD weight 0.1, generator output detached), then generator update
  with discriminator parameters frozen (`toggle_optimizer`): `45 · mel + adv + fm` (+ `cfg.loss.mss_weight · MSS`,
  0 in the primary runs). Log losses, learning rates and gradient norms (no clipping).
- Validation every `cfg.train.val_every_g_steps`: mel L1 and MR-STFT distance on the fixed dev subset (full
  utterances, T1); audio and mel images for the 20-utterance logging subset.
- `predict_step`: full-utterance synthesis; a `WavWriter(BasePredictionWriter)` saves
  `${run_dir}/predictions/<split>/<condition>/<id>.wav` (24 kHz, float32 WAV).
- Checkpointing and preemption (design verified bit-exact in `~/sparc-vocoders-work/probes/compute/resume_demo.py`
  and `sbatch_train_template.sh`; evidence in `~/sparc-vocoders-work/phase1/compute.md`). Do NOT rely on Lightning's
  stock SLURM path: in a plain `sbatch` script `SLURM_NTASKS` is unset, so SIGUSR1 kills Python; the stock handler
  checkpoints mid-step; Lightning's SIGTERM path exits without a checkpoint; `last.ckpt` gets versioned
  (`last-v1.ckpt`); `hpc_ckpt_N` files go stale. Instead:
  - `PreemptionCheckpoint(Callback)`: in `on_train_start` install handlers for SIGUSR1 and SIGTERM that only set a
    flag (DataLoader workers ignore both signals through `worker_init_fn`). In `on_train_batch_end`, after the full
    D and G update: if the flag is set, save `${run_dir}/ckpt/step{g_step:09d}.ckpt` with
    `trainer.save_checkpoint` (atomic), run `$REQUEUE_CMD` if that environment variable is set (only the sbatch script
    sets it; never requeue implicitly, because ssh sessions adopted into a job also carry `SLURM_JOB_ID`), then set
    `trainer.should_stop = True`. Also save every `cfg.train.checkpoint_every_minutes` (time-based) and keep the last
    3 such files plus milestones every `cfg.train.milestone_every_g_steps`.
  - Resume: the CLI picks the newest *loadable* `step*.ckpt` by step number (never by mtime); no `ModelCheckpoint`
    with `save_last`.
  - `RNGStateCallback` saves and restores python, numpy, torch and CUDA RNG states; the sampler resumes from
    `samples_consumed`.
  - `trainer.plugins = [LightningEnvironment()]` explicitly (no SLURM plugin); never spoof `SLURM_JOB_NAME`.
- Precision from `cfg.trainer.precision` (`32-true` or `bf16-mixed`; note PyTorch's default TF32 convolutions in the
  fp32 arm). Strategy `auto` on one GPU, `ddp_find_unused_parameters_true` on several (one srun task per GPU).

## 6. CLI, configs, scripts

- `sparc-train` stays the entry point. `sparc.cli.train:main` dispatches on the root key of each override
  (`arg.lstrip("+~").split("=")[0].split(".")[0]`): if any root key is one of `vocoder, experiment, data, trainer,
  loss, train, optim, speaker, paths, run_dir` or `--config-name vocoder_config` is given, it runs
  `sparc.cli.train_vocoder:main` (Hydra `config_name=vocoder_config`); otherwise the legacy 16 kHz path runs
  unchanged. Example: `uv run sparc-train vocoder=vocos experiment=main`. Lightning's DDP subprocess launcher re-enters
  the same dispatcher, which must still route correctly.
- `sparc-predict` (new console script): synthesis over a split for given conditions from a checkpoint.
- Config groups under `src/sparc/conf/`: `vocoder/{hifigan,ddsp,vocos,...}.yaml`, `data/librittsr_filtered.yaml`,
  `trainer/{default,overfit,smoke}.yaml`, `experiment/{main,overfit,precision_pilot,vocos_ab_pilot,...}.yaml`;
  roots `vocoder_config.yaml` and `cache_config.yaml`. Paths: `paths.cache_root = ${oc.env:SPARC_VOC_CACHE}`,
  `paths.runs_root = ${oc.env:SPARC_VOC_RUNS}`, `paths.libritts_r_raw = ${oc.env:LIBRITTSR_RAW}`,
  `paths.filtered_list = ${oc.env:LIBRITTSR_FILTERED}`, `paths.refit_npz = ${oc.env:SPARC_REFIT_NPZ}`.
- Run directory: `${paths.runs_root}/${experiment.name}/${vocoder.name}` (fixed, so a requeued job finds it).
  Resume order: newest `hpc_ckpt_*.ckpt`, else `ckpt/last.ckpt`, else a fresh start.
- `scripts/slurm/cache_features.sh` (array job) and `scripts/slurm/train_vocoder.sh`, modelled on
  `~/sparc-vocoders-work/probes/compute/sbatch_train_template.sh`: `#!/bin/bash`; source `$SPARC_ENV_FILE`;
  `--partition=preempt --requeue --open-mode=append --signal=USR1@120` (no `B:`: the signal must reach the srun
  tasks); `export REQUEUE_CMD="scontrol requeue $SLURM_JOB_ID"`; a `DONE` guard so a finished run is never redone;
  a GPU preflight (`torch.zeros(1).cuda()`) that on failure adds the node to the job's `ExcNodeList` and requeues
  (4.5 % of preempt GPU starts fail within minutes and Slurm does not requeue FAILED jobs); then
  `srun --kill-on-bad-exit=1 uv run --no-sync sparc-train ...` with one task per GPU. Idempotent from line 1.

## 7. Tests (`tests/`, pytest, CPU, seconds each)

Output length `480·N` for N in {1, 7, 64, 123} for every vocoder; FiLM identity at init (two different `spk` give the
same output at init); no in-place mutation of `features`; `gain_factor` and `loud_raw · g` exactness against
recomputation on a synthetic signal; sampler resume determinism (consume k items, save, reload, continue equals an
uninterrupted stream); discriminator and loss shapes; cache write/read/validate round trip and atomic write; T
formula and crop alignment (a synthetic burst at known sample positions lands in the same relative frame of cropped
features and cropped audio).

## 8. Implementation notes (added after the first implementation pass)

- Feature cache: constructors take the Hydra config first (`SparcFeatureExtractor(cfg, device)`,
  `ManifestShardDataModule(cfg, shard_index, num_shards, skip_valid)`, `pack_split(cfg, split)`,
  `compute_stats(cfg)`). `sparc-cache` also has `stage=validate`. `cache_config.yaml` has a `cache:` block with the
  directory layout. The extractor disables TF32 so cached values do not depend on the GPU model, and `meta.json` may
  record `fork_commit` with a `+dirty` suffix. A stop signal ends extraction with exit code 75 after the current
  utterance. Smoke runs must use a separate cache root (the DONE markers are keyed by shard only).
- DataLoader workers use `sparc.vocoders.data.datamodule.ignore_stop_signals` (ignore SIGUSR1; exit on SIGTERM only
  when the parent sends it), so a crashing main process never waits on its workers.
- `FiLM` computes its projection in at least float32 even under autocast.
- Losses use `losses/ops.py` (deterministic reflect padding and centred STFT) so `trainer.deterministic=true` works.
- Trainer settings live under `cfg.trainer.trainer.*`; `use_distributed_sampler=False` (the sampler strides by rank).
  Bit-exact resume on GPU needs `trainer.trainer.deterministic=true` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; otherwise a
  resumed run is statistically equivalent but not bit-identical.
