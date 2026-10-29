# Phase 9 tools: k12 inverter features for the vocoder

Implements the shared pieces of `docs/spkemb_phase9_k12_inverter.md` (pre-registration unchanged). No full extraction yet.

## Pieces

- `sparc/vocoders/features/k12_inverter.py`: `K12Inverter(mode)` with `mode` `mngu0` or `all7`. WavLM Large truncated to layer 12
  (fp32, TF32 off) on the zero-mean / unit-variance 16 kHz waveform, 10 Hz zero-phase low-pass over the whole utterance in float16
  (as the training cache), the BiGRU -> conv 9 head, then `y * ys + ym` with the stored target statistics. `mngu0`: MNGU0 mm;
  `all7`: per-speaker (per recording-session group) z units. Channel `t` is the head output at frame `t + shift` (stored shift 1;
  the last frame repeats), so the frame count is `feature_length(n24)`. `channels24(wav24)` is the entry point.
- `sparc/vocoders/features/inverter_pack.py`: `extract` (chunks of 256 rows, `<out>.parts/part_*.npy`, sharded, resumable),
  `merge` (new packed split: columns 0-11 of `feats.npy` replaced, columns 12-14 and every other file byte-identical,
  `inverter.json` with checkpoint sha256 added), `stats` (copy of `train_stats.json` with `ema_mean` / `ema_std` from the new channels over
  the splits given; refuses to overwrite the existing file).
- `FeatureEncoder(inverter=...)`, `extract_pool(..., inverter=...)`, `clone_pools extract --inverter`, `RefConditions(inverter=...)`
  and the config key `inverter.name` (read by `clone.py synth`): references and contents are encoded with the run's inverter. Use a
  separate pool root and `--refcond-dir` for each inverter (the reference cache key includes the inverter only when one is set).
- `vocoder.generator.ema_denorm: true` (HiFi-GAN): `x_ema * (1 + s(spk)) + o(spk)` on the 12 normalized articulatory inputs,
  `[s, o] = Linear(64, 24)(code)`, zero-initialized. Off by default: module and state dict unchanged. Set it from an experiment file
  (the base `hifigan.yaml` and its variants do not carry the key).
- `scripts/spkemb/p9_check_inverter.py` (`sanity`, `align`, `timing`, `live`); results in `/data/user_data/xoy/spkemb/phase9/checks/`.

## Running extraction

Per inverter `X` in `mngu0`, `all7`, per source split `S` (a packed split directory):

```
slurm-alloc submit --agent sparc-xlsr-ema --restartable --no-wait --gpus 1 --cpus 8 --mem 48G --time 04:00:00 \
  --sbatch "--constraint=L40S|L40|6000Ada|RTX_PRO_6000|A100_80GB|A100_80G|A100_40GB|H100|H200" \
  -- scripts/spkemb/p5_job.sh -m sparc.vocoders.features.inverter_pack extract --inverter X --src S --out OUT \
     --shard K --nshards N          # one job per shard
python -m sparc.vocoders.features.inverter_pack merge --src S --out OUT        # CPU, after all shards
python -m sparc.vocoders.features.inverter_pack stats --splits OUT_train.clean.100 ... --out stats_X.json   # LibriTTS-R training splits only
```

## Checks (2026-10-29)

- Stored test scores reproduced from audio with the training-time alignment: MNGU0 61 utterances RMSE 0.76546 mm / rmse_z 0.37021 / PCC
  0.92808 (stored 0.76549 / 0.37022 / 0.92808); all7 397 seen-speaker test sentences rmse_z 0.56870 / PCC 0.80451 (stored 0.56870 / 0.80451), every
  speaker agrees to 1e-5. Live features vs the cached ones of the speech-to-EMA work: max relative difference 1.6e-3 (MNGU0), 4.7e-3 (all7), the
  float16 and TF32 level of the cache.
- Frame count equals the packed `T` on every checked utterance.
- Alignment versus the packed layer-9 channels (20 LibriTTS-R utterances, 12 channels): with the stored shift 1 the peak of the cross-correlation sits
  at lag -1 or -2 frames (the new channels lead the layer-9 ones by 1.5 frames on average, 30 ms; mean peak correlation 0.89), never at 0. Grid shift 0
  gives lags -1/0, grid shift -1 gives 0/+1, so the layer-9 head's channels sit about half a frame later than the inverter's
  EMA-time convention at shift 0. The default stays the stored shift (pre-registration: "shifted as trained"); `channels24(wav, shift=...)` exists for the
  checks. The vocoder's input convolution has kernel 7 (3 frames each side) and learns any constant lag.
- Packing tool vs live encoder on 8 (mngu0) / 4 (all7) utterances: channels 0-11 differ by <= 4e-6, channels 12-14 and loud_raw match the packed source
  (<= 5e-7 / 0).
