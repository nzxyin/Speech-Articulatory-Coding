# Speech Articulatory Coding (SPARC) — Unofficial Fork
[Paper](https://arxiv.org/abs/2406.12998) | [Audio Samples](https://berkeley-speech-group.github.io/sparc-demo) | [Colab Demo](https://colab.research.google.com/drive/1TVGJJpOzPiesLPo46gZNCQLMl-y_QIKe#scrollTo=uBemLVlk-s7W)

<div align="center">
    <img src="images/articulatory_coding.png" alt="drawing" width="600"/>
</div>

This is an **unofficial fork** of the official code base for [Coding Speech through Vocal Tract Kinematics](https://arxiv.org/abs/2406.12998), maintained independently at [nzxyin/Speech-Articulatory-Coding](https://github.com/nzxyin/Speech-Articulatory-Coding). It is not affiliated with or endorsed by the original authors. For the official release, see [Berkeley-Speech-Group/Speech-Articulatory-Coding](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding).

This fork modernizes the packaging (Python 3.13+, `uv`-managed environment, `src/` layout) and trims unused code; the model code and checkpoints are unchanged from upstream.

## Installation

Install `uv`, then either install directly from this fork:
```
uv add git+https://github.com/nzxyin/Speech-Articulatory-Coding.git
```
or clone and set up a local environment:
```
git clone https://github.com/nzxyin/Speech-Articulatory-Coding.git
cd Speech-Articulatory-Coding
uv venv --python 3.13
source .venv/bin/activate
uv sync
```

## Usage

#### Load Model

```python
from sparc import load_model
coder = load_model("en", device= "cpu")     # For using CPU
coder = load_model("en", device= "cuda:0")  # For using GPU
```

Pitch tracking uses [torchcrepe](https://github.com/maxrmorrison/torchcrepe).

For inversion only, you can use the following,

```python
coder = load_model("feature_extraction") 
coder_from_config = load_model(config="configs/feature_extraction.yaml")
```

The following model checkpoints are offered. You can replace `en` with other models (`multi` or `en+`) in `load_model`.

| Model  | Language |     Training Dataset    | 
|--------|:--------:|:--------------:|
| en     |  English |LibriTTS-R|
| multi  |   Multi  |LibriTTS-R, Multilignual LibriSpeech, AISHELL, JVS, KSS    |
| en+     |  English |LibriTTS-R, LibriTTS, EXPRESSO|


#### Articulatory Analysis

```python
code = coder.encode(WAV_FILE)          # Single inference
codes = coder.encode([WAV_FILE1, WAV_FILE2, ...]) # Batched processing
```


The articulatory code outputs have the following format.

```python
# All features are in 50 Hz except speaker encoding
{"ema": (L, 12) array, #'TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY'
 "loudness": (L, 1) array, 
 "pitch": (L, 1) array, 
 "periodicity": (L, 1) array, # auxiliary output of pitch tracker
 "pitch_stats": (pitch mean, pitch std),
 "spk_emb": (spk_emb_dim,) array, # all shared models use spk_emb_dim=64
 "ft_len": Length of features, # usefull when batched processing with padding
}
```

#### Articulatory Synthesis

```python
wav = coder.decode(**code)
sr = coder.sr
```

#### Voice Conversion

```python
wav = coder.convert(SOURCE_WAV_FILE, TARGET_WAV_FILE)
sr = coder.sr
```
#### Demo

Please check `notebooks/demo.ipynb` for a demonstration of the functions.


## CLI Inference

Feature extraction and resynthesis are exposed as Hydra-configured CLIs, installed as console scripts:

```
uv run sparc-encode dataset=vctk                 # uses the en+ model by default
uv run sparc-encode dataset=vctk model=en         # override the model
uv run sparc-decode dataset=vctk                  # resynthesize wavs from extracted features
```

`dataset` and `model` are Hydra config groups defined in `src/sparc/conf/dataset` and
`src/sparc/conf/model`; add a new YAML file there (with `wav_dir`, `save_dir`, `decode_dir`) to
support another dataset. On SLURM, `scripts/encode_slurm.sh` and `scripts/decode_slurm.sh` wrap
the same CLIs — see the usage comments at the top of each script for single-dataset and
array-job invocations.


## Vocoder Training (Reproduction)

This reproduces the trainable part of the SPARC paper's methodology (Section III-B / Appendix
B): a HiFi-GAN generator and small speaker-encoder FFN, trained adversarially against
multi-period/multi-scale discriminators. Everything else in the pipeline (WavLM feature
extraction, the linear EMA-inversion head, CREPE pitch tracking, loudness) stays frozen, exactly
as in the paper.

#### 1. Prepare training data

Encode a dataset first (see [CLI Inference](#cli-inference) above), then precompute the raw
(pre-FFN) speaker feature training needs — this is separate from the `spk_emb` that
`sparc-encode` already caches, since that one already has the *pretrained* FFN applied:

```
uv run sparc-encode dataset=librittsr_train_clean_100
uv run python -m sparc.training.prepare_spk_raw <wav_dir> <sparc_dir>
```

#### 2. Train

```
uv run sparc-train dataset=librittsr_train_clean_100
```

Optimizer, LR schedule, loss weights, and 320ms random-crop batching default to the paper's
Appendix B values. See `src/sparc/conf/train_config.yaml` for the full list, and override any of
them on the command line, e.g.:

```
uv run sparc-train dataset=librittsr_train_clean_100 batch_size=64 max_steps=1500000
```

On SLURM, `scripts/prepare_spk_raw_slurm.sh` and `scripts/train_slurm.sh` wrap the same
commands — see the usage comments at the top of `scripts/train_slurm.sh` for single-run and
long-run (`preempt` partition) invocations.

#### 3. Monitor with TensorBoard and/or Weights & Biases

Both logging backends are supported, individually or together:

```
uv run sparc-train dataset=librittsr_train_clean_100 logger_backends='[tensorboard]'       # default
uv run sparc-train dataset=librittsr_train_clean_100 logger_backends='[wandb]'
uv run sparc-train dataset=librittsr_train_clean_100 logger_backends='[tensorboard,wandb]'  # both
```

TensorBoard logs go to `<dataset.save_dir>/tb_logs` by default:

```
tensorboard --logdir <dataset.save_dir>/tb_logs
```

wandb defaults to `wandb_mode=offline` (no API key required) with logs under
`<dataset.save_dir>/wandb_logs`; sync them later with `wandb sync <run_dir>`, or set
`wandb_mode=online` after `wandb login` to stream live. Override the destination project/entity/
run name via `wandb_project`, `wandb_entity`, `wandb_run_name`.

Both scalars (losses, learning rate) and periodic audio/mel-spectrogram samples
(`log_audio_every_n_steps`) are logged to every enabled backend.

#### 4. Checkpoints and resuming

Checkpoints save to `<dataset.save_dir>/vocoder_ckpt` every `checkpoint_every_n_steps` (default
1000), keeping the `keep_last_n_checkpoints` most recent plus `last.ckpt`. Resume a run with:

```
uv run sparc-train dataset=librittsr_train_clean_100 resume_from_checkpoint=<path-to>/last.ckpt
```

#### 5. Listen to samples

```
uv run python -m sparc.training.sample <checkpoint>.ckpt <sparc_dir> <out_dir> [n_samples]
```

Synthesizes standalone wav files from a trained checkpoint for qualitative listening outside
the logging UI.

#### Scope

Full reproduction (paper: 1.5M steps, batch 64, ~555h of LibriTTS-R) is well beyond a single
run's practical scope here; the pipeline has been verified end-to-end — correct architecture,
losses, and data pipeline, with loss curves behaving as expected — but not run to paper-matching
scale or quality. The linear EMA-inversion head is a separately frozen component, refit from
scratch below rather than trained jointly with the vocoder — matching how the paper itself
treats it.

## Linear AAI Head Refit (MNGU0)

This reproduces the paper's Section III-A1 / Appendix A.1 methodology: WavLM Large layer-9
features → 12-dim EMA (tongue dorsum/blade/tip, jaw, upper lip, lower lip × X/Y) via ordinary
least squares, fit on MNGU0. Requires the MNGU0 "day1"/"s1" release (manual dataset-owner
approval) laid out as:

```
<mngu0_root>/ema_norm/<release>/*.ema   # head-corrected EMA, EST_Track binary format, 200Hz
<mngu0_root>/wav_16kHz/<release>/*.wav  # matching audio, 16kHz
```

Set `MNGU0_ROOT` at the top of `src/sparc/training/refit_mngu0_linear_aai.py` (and the other
scripts below, which import their paths from it) to point at your copy.

#### 1. Verify the layer choice

`mngu0_layer_sweep.py` probes every WavLM Large transformer layer (not just 9) with a single
held-out split per layer, to check the paper's Fig. 7 finding that layer 9 is the peak before
committing to it:

```
uv run python -m sparc.training.mngu0_layer_sweep [cuda:0]
```

Layers 9–10 come out statistically tied for the peak, matching the paper's rise → plateau →
decline curve shape — confirming layer 9 is the right (or an indistinguishable-from-optimal)
choice.

#### 2. Correct the EMA/audio sync lag, then fit

MNGU0's EMA capture and audio recording aren't triggered at the same instant despite both
files' internal clocks nominally starting at t=0 — there's a hardware/software sync lag between
the two subsystems that must be corrected before pairing frames, and it isn't perfectly
constant across utterances. Two variants are provided:

```
uv run python -m sparc.training.refit_mngu0_linear_aai [cuda:0]   # single global offset
uv run python -m sparc.training.mngu0_peralign_refit [cuda:0]     # per-utterance offset (recommended)
```

`refit_mngu0_linear_aai.py` uses one fixed offset (`EMA_AUDIO_OFFSET_FRAMES`), found once via
cross-correlation against the shipped model's predictions on a handful of utterances.
`mngu0_peralign_refit.py` instead computes that offset per utterance: cross-correlating each
utterance's ground-truth EMA against the *shipped model's own prediction for that utterance*
(audio-locked by construction, since it's computed directly from that utterance's audio) over a
search window, and taking the shift that maximizes mean per-channel correlation. This uses the
shipped model only as a timing reference, never as a value target — the EMA values fit against
are always the real measured ground truth.

`mngu0_peralign_control_test.py` validates that the per-utterance result isn't a search/leakage
artifact: it repeats the identical procedure but aligns each utterance against a *different,
randomly assigned* utterance's shipped prediction, where no genuine temporal correspondence
should exist:

```
uv run python -m sparc.training.mngu0_peralign_control_test [cuda:0]
```

#### Results

5-fold cross-validated mean Pearson correlation, matching the paper's evaluation protocol:

| Method | Mean PCC |
|---|---|
| Global offset (`refit_mngu0_linear_aai.py`) | 0.7021 ± 0.0090 |
| Shipped checkpoint's own head, evaluated against this ground truth | 0.6855 ± 0.278 |
| **Per-utterance offset (`mngu0_peralign_refit.py`)** | **0.9186 ± 0.0010** |
| Mismatched-pairing control | 0.0590 ± 0.0053 |
| Paper (Appendix A.1) | 0.878 ± 0.012 |

The mismatched-pairing control collapsing to near-zero shows the per-utterance result requires
genuine matched alignment, not an artifact of searching over many candidate shifts. The found
offsets are also tightly and physically plausibly distributed around the global-offset method's
constant (mean 23.2 frames, std 3.2, both at a 50Hz frame rate), consistent with correcting real
per-utterance timing drift rather than fitting noise. 143 of 1189 utterances are excluded from
the per-utterance variant — MNGU0's shortest utterances, under the 50-frame (~1s) minimum needed
to estimate a reliable per-utterance offset.

## TODO

- Scale vocoder training to the paper's full schedule and evaluate against the paper's reported
  metrics (WER/CER/MOS/UTMOS).
- Multilingual fine-tuning.

## License

This fork's modifications are released under the [MIT License](LICENSE). The underlying SPARC model code and checkpoints are used with permission from the original authors; see the LICENSE file for details.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{cho2024coding,
  title={Coding Speech through Vocal Tract Kinematics},
  author={Cho, Cheol Jun and Wu, Peter and Prabhune, Tejas S. and Agarwal, Dhruv and Anumanchipalli, Gopala K.},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={18},
  number={8},
  pages={1427--1440},
  year={2024},
  publisher={IEEE}
}
```

