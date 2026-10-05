# Phase 3 evaluation: module contract

Contract for `sparc.vocoders.eval` and the `sparc-eval` CLI. Read `docs/vocoders/INTERFACES.md` first; its
conventions (channels, `HOP = 480`, `feature_length`, eval gain -3 dBFS, T1/T2/T3) apply here unchanged.

## 0. Ground rules

- **Isolation.** Training jobs run from `~/Speech-Articulatory-Coding` (branch `vocoder-comparison`) and its venv
  `/data/user_data/xoy/venvs/sparc-fork`. Phase 3 is developed in the worktree `~/sparc-eval-wt` (branch
  `phase3-eval`) with its own venv `/data/user_data/xoy/venvs/sparc-eval` (`~/sparc-eval-wt/.venv` is a symlink to it).
  Never run `uv sync`, `uv add` or `uv pip` in `~/Speech-Articulatory-Coding`, and never edit files there.
- New code goes into new files (`src/sparc/vocoders/eval/`, `src/sparc/cli/eval_vocoder.py`,
  `src/sparc/conf/eval_config.yaml`, `src/sparc/conf/eval/`, `scripts/slurm/eval.sh`, `scripts/slurm/submit_eval.sh`,
  `tests/eval/`). Changes to existing files are limited to: `pyproject.toml`/`uv.lock` (new dependencies, console script
  `sparc-eval`), `src/sparc/cli/predict_vocoder.py` and `datamodule.predict_dataloader` (skip-existing, section 3), and
  `src/sparc/vocoders/features/extractor.py` (an opt-in `dither` switch, section 5). Training behaviour must not change.
- All Hydra; no hard-coded paths or hyperparameters in Python. Paths: `paths.eval_root = ${oc.env:SV_ROOT}/eval` plus the
  existing `paths.*`. Model downloads go to the private hub cache (`HF_HUB_CACHE=$SV_ROOT/hf_hub`) and
  `TORCH_HOME=/data/user_data/xoy/.cache/torch`; pin model revisions in config and record them.
- Every stage is resumable: work is split into fixed chunks (`eval.chunk_size = 256` utterances of the sorted id list),
  each chunk's output is written atomically (temporary file + `os.replace`) and existing chunk outputs are skipped. A
  stop signal (SIGUSR1/SIGTERM) sets a flag; the stage finishes the current chunk and exits with code 75.
- Each stage records provenance in a sidecar `meta.json` next to its outputs: fork commit, package versions, model ids
  and revisions, GPU name, config.
- Evaluation models run in fp32 except Whisper (fp16 on GPU). Fixed seeds everywhere.

## 1. Test set, systems and conditions

- Primary split `test.clean` (filtered: 4,687 utterances after the 171 not-encodable rows; the two utterances under
  0.385 s are among them). `test.other` is secondary and uses the same code (`eval.split=test.other`).
- Reference audio ("gt"): the first `480·T` samples of the LibriTTS-R file times the eval gain (`gain_factor(peak24,
  -3.0)`), exactly as `FullUtteranceDataset` builds `audio`. Materialized once as float32 WAV under
  `eval_root/audio/gt/<split>/T1/<id>.wav` (stage `gt`).
- Systems (config `eval.systems`, a dict name -> spec):

  | name | kind | sample rate | conditions | EMA head | audio |
  |---|---|---|---|---|---|
  | `gt` | gt | 24000 | T1 | refit | `eval_root/audio/gt/...` |
  | `vocos_mel` | reference | 24000 | T1 (copy synthesis) | refit | `eval_root/audio/vocos_mel/...` |
  | `enplus16` | reference | 16000 | T1, T2 | shipped | `eval_root/audio/enplus16/...` |
  | `hifigan`, `ddsp`, `vocos` | vocoder | 24000 | T1, T2, T3 | refit | `${paths.runs_root}/<experiment>/<vocoder>/predictions/<split>/<cond>/` |

  A vocoder spec has `experiment` and `vocoder` keys (for example `vocos` may point to `vocoder: vocos_a` if the A/B
  pilot picks option A). Ablation runs are added the same way.
- T2 references come from `FullUtteranceDataset` (`ref_id`); record `ref_id` per utterance so T2 can be stratified by
  same versus different chapter (chapter = second `_` field of the id). Speaker 61 has no T2 reference and is absent
  from T2.
- Every system is evaluated at its final checkpoint (no selection on test metrics).

## 2. Output layout (`eval_root`)

```
audio/<system>/<split>/<cond>/<id>.wav                      gt and reference systems
results/<split>/<system>/<cond>/<group>/part-<k:04d>.parquet one row per utterance (section 4)
arrays/<split>/<system>/<cond>/reextract/part-<k:04d>.npz   re-extracted features (section 5)
embeddings/<split>/<system>/<cond>/<model>/part-<k:04d>.npz speaker embeddings (section 4.4)
probes/<system>/...                                         section 6
efficiency/<system>.json                                    section 7
tables/<split>/{table.md,table.csv,paired.csv,per_speaker.csv,meta.json}   section 8
samples/<id>/<system>_<cond>.wav, samples/index.csv          section 9
```

## 3. Synthesis

- Vocoders: `sparc-predict vocoder=<v> experiment=<e> data.predict_split=<split>` (existing). Add
  `predict.skip_existing: true` (default true in the eval path, false keeps today's behaviour): per condition, ids whose
  WAV already exists are dropped from the dataset before `trainer.predict`. WAV writes become atomic.
- `vocos_mel` (stage `refs`): `charactr/vocos-mel-24khz` (pip `vocos`) on the gt audio; output trimmed or zero-padded
  to `480·T`; level unchanged. One condition, stored as `T1`.
- `enplus16` (stage `refs`): `sparc.load_model("en+")` with its shipped head; encode the gt audio resampled to 16 kHz
  (the shipped pipeline, `normalize: true`), decode with the utterance's own en+ speaker embedding (T1) and with the
  T2 reference's embedding (T2); trim or pad to `320·T` samples; peak-normalize to -3 dBFS (SPARC normalizes its input,
  so its output level is arbitrary; record this). Stored at 16 kHz.
- Resampling everywhere: `librosa.resample(..., res_type="soxr_hq")`. A 16 kHz system is upsampled to 24 kHz for 24 kHz
  metrics, and its 8-12 kHz band metrics are reported as not applicable.

## 4. Metric groups (per utterance)

All read the system WAV and, where needed, the gt WAV of the same id. Rows carry `id, speaker, chapter, dur_s, ref_id`
plus the group's columns; a failure on one utterance writes NaN and an `err` string, never aborts the chunk.

### 4.1 `signal` (CPU)
- `pesq_wb`: `pesq.pesq(16000, ref16, deg16, "wb")` after resampling both to 16 kHz.
- `mcd`: mel-cepstral distortion, no DTW (signals are time-aligned): Hann window 1024, hop 240, power spectrum ->
  mel-cepstrum of order 24 with all-pass constant `alpha = 0.466` (pysptk `sp2mc`, or an equivalent `freqt`
  implementation checked against pysptk), `c0` excluded, `MCD = (10 / ln 10) · sqrt(2 · Σ_d (Δc_d)²)` averaged over
  frames whose gt frame energy is within 40 dB of the utterance's loudest gt frame.
- `mrstft`: mean over resolutions (n_fft, hop, win) = (1024, 120, 600), (2048, 240, 1200), (512, 50, 240), Hann, of
  spectral convergence plus mean |log(max(|Y|, 1e-7)) - log(max(|X|, 1e-7))| (auraloss defaults, own torch code).
- `mel_l1` (the training mel: 24 kHz, n_fft 1024, hop 256, 100 mels, 0-12 kHz, clamp 1e-5, natural log) and the same L1
  restricted to mel bins centred in [0, 4), [4, 8) and [8, 12] kHz: `mel_l1_0_4k`, `mel_l1_4_8k`, `mel_l1_8_12k`.

### 4.2 `utmos` (GPU)
- UTMOS22 strong (`torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)`), 16 kHz input.

### 4.3 `asr` (GPU)
- `openai/whisper-large-v3` (transformers, fp16, greedy, `language="en"`, `task="transcribe"`, no prompt), batch by
  length; utterances over 30 s use long-form decoding. Reference: the LibriTTS-R `<id>.normalized.txt`. Both sides go
  through Whisper's `EnglishTextNormalizer` (with the tokenizer's English spelling map). Columns: `hyp`, `ref_norm`,
  `hyp_norm`, `word_errors`, `word_ref_len`, `char_errors`, `char_ref_len` (Levenshtein; `jiwer` if available). WER and
  CER are corpus-level in the tables (sum of errors / sum of reference lengths).

### 4.4 `spk` (GPU)
- Embeddings, stored per utterance: ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, 16 kHz) as the primary model;
  `microsoft/wavlm-base-plus-sv` (transformers `WavLMForXVector`) as the secondary model. If speechbrain cannot run with
  the installed torch/torchaudio, record why and report WavLM-SV only.
- Similarities (computed at aggregation): `sim_same` = cosine with the gt of the same utterance; `sim_spk` = cosine with
  the mean of the L2-normalized gt embeddings of the speaker's other test utterances, excluding the target and its T2
  reference. The gt row's `sim_spk` is the ceiling.

### 4.5 `reextract` (GPU) and `prosody` (CPU)
- `reextract`: run `SparcFeatureExtractor.extract` (refit head; the shipped head for `enplus16`) on the system audio at
  24 kHz with **CREPE dither disabled**, batch size 1. Save `feats (T, 15)` and `loud_raw (T,)` per utterance. Also run it
  on gt (the reference for pitch, voicing and periodicity).
- `prosody` compares a system's re-extraction with:
  - the gt re-extraction (no dither) for pitch, voicing and periodicity;
  - the input features for EMA and loudness (cached `feats[:, :12]` and `loud_raw · g_eval`). For `enplus16`, the
    shipped-head gt re-extraction replaces the cached EMA.
- Voiced = periodicity > 0 (SPARC's own thresholding and loudness gate are already applied).
- Columns: `f0_rmse_cents` (frames voiced in both; `1200·log2(f_sys/f_ref)`), `f0_med_abs_cents`, `f0_within50`,
  `n_voiced_both`, `vde` (fraction of frames whose voicing decision differs), `per_mae`, `per_rmse`, `loud_db_rmse` and
  `loud_db_bias` (`20·log10((l_sys + 1e-4)/(l_ref + 1e-4))` on frames where `l_ref` is above the 5th percentile of the
  utterance), `ema_r_<c>` and `ema_rmse_<c>` for the 12 channels (RMSE in units of the training std of the channel),
  `ema_r_mean`, `ema_rmse_mean`.
- The gt row compares gt re-extraction (no dither) with the cached input (with dither): the extraction floor.

## 5. Extractor change

`SparcFeatureExtractor` gains `dither: bool = True` (config `extractor.dither`, default true, so caching is unchanged).
With `dither=False`, torchcrepe's dither is replaced by the identity for the duration of the call (context manager that
restores the original function). Test: two no-dither extractions of the same audio are bit-identical, and the default
path is unchanged.

## 6. Controllability probes (stage `probes`, GPU)

- Subset: 50 test.clean utterances of 3-10 s, at most 2 per speaker, chosen with `default_rng(0)` from the sorted ids;
  T1 speaker vectors.
- Edits on the raw 15-channel input (before the frontend): F0 × 2^(k/12) for k in {-4, -2, +2, +4} on voiced frames
  only; loudness × {0.5, 2}; periodicity set to 0; each EMA channel c plus s·std_c for s in {-1, +1} (training std).
- For each vocoder and edit: synthesize, re-extract (no dither), and compare with the unedited synthesis:
  - F0: median measured shift in cents on frames voiced in both, against the target 100·k;
  - loudness: median measured change in dB of `loud_raw` on frames above the 5th percentile, against ±6.02 dB;
  - periodicity 0: voiced-frame fraction before and after;
  - EMA: control gain `mean(Δ re-EMA_c) / (s · std_c)` and leakage `mean_{c' != c} mean |Δ re-EMA_c'| / std_c'`.
- Outputs: `probes/<system>/results.parquet` (one row per utterance and edit) and example WAVs for the first 3
  utterances under `probes/<system>/wav/<id>/<edit>.wav`.

## 7. Efficiency (stage `efficiency`)

- Parameters: generator, speaker FFN, total (the frontend has buffers only).
- Real-time factor `RTF = compute time / audio duration` on 50 test utterances, batch 1, fp32, 5 warm-up calls,
  `torch.cuda.synchronize` around timing: on GPU, and on CPU with 1 and with 8 threads (CPU timing runs as a separate
  CPU job). Includes the frontend and speaker FFN; feature extraction is reported once, separately.
- Receptive field and lookahead, measured: on a 400-frame input taken from a real utterance, add 0.5 training std to
  every channel of frame 200; with the same RNG state for both calls, find the first and last output samples whose
  absolute change exceeds `1e-4 · max|y|`. Lookahead = `(480·200 - first)/24000` s, past context =
  `(last - 480·201)/24000` s. Also state the analytic receptive field from the architecture.
- `vocos_mel` and `enplus16` are measured too (their own inputs).

## 8. Aggregation (stage `aggregate`, CPU)

- Per system and condition: mean and CI95 of each metric over utterances (speaker-cluster bootstrap: resample speakers
  with replacement, keep all their utterances; 10,000 resamples, seed 0); corpus WER/CER with the same bootstrap.
- Paired differences for every pair of vocoders under the same condition, and each vocoder against `vocos_mel` and
  `enplus16` on shared ids: mean difference with the same speaker-cluster bootstrap CI95, and the fraction of speakers
  whose mean moves in the same direction. Only ids present in both systems.
- T2 stratified by same and different chapter.
- `table.md` (rows: system × condition; columns: UTMOS, PESQ, MCD, MR-STFT, mel L1 and its three bands, WER, CER, F0
  RMSE, F0 median, F0 within 50 cents, VDE, periodicity MAE, loudness dB RMSE, EMA r mean, EMA RMSE mean, sim_same,
  sim_spk), `table.csv`, `paired.csv`, `per_speaker.csv`, and `meta.json` (commits, checkpoints with their step
  counts, models, n per cell). State in the table header that each arm is a single training run.

## 9. Samples (stage `samples`)

- 20 test.clean utterances: 10 female and 10 male speakers (LibriTTS `SPEAKERS.txt`), 3-12 s, distinct speakers,
  `default_rng(0)` on the sorted ids. For each: gt and every system and condition, as 16-bit PCM WAV at the system's
  sample rate, under `samples/<id>/<system>_<cond>.wav`; `index.csv` with id, speaker, sex, duration, transcript and
  T2 `ref_id`.

## 10. CLI and Slurm

- `uv run sparc-eval stage=<gt|refs|signal|utmos|asr|spk|reextract|prosody|probes|efficiency|aggregate|samples>
  system=<name|all> condition=<T1|T2|T3|all> eval.split=test.clean`.
- `scripts/slurm/eval.sh` (new, bash, `preempt`, `--requeue --open-mode=append --signal=USR1@120`, GPU preflight as in
  `train_vocoder.sh`, runs a list of `stage:system:condition` items and skips finished ones, requeues on exit code 75)
  and `scripts/slurm/submit_eval.sh` (`--cpu` for CPU stages on `preempt_cpu_qos`). Logs under `$SV_ROOT/slurm_logs`.
  At most one GPU for evaluation while the training runs hold three.

## 11. Tests (`tests/eval/`, CPU, seconds each)

Metric sanity on synthetic signals (identical inputs give PESQ ≈ 4.6, MCD 0, MR-STFT 0, mel L1 0; a known gain gives
the expected loudness bias; a known pitch shift gives the expected cents); chunking and skip-existing; atomic writes;
the bootstrap on a toy table with a known answer; corpus WER on toy strings; probe edit functions (only the intended
channel changes; F0 edit leaves unvoiced frames alone); no-dither extraction determinism (marked slow / GPU-optional).
