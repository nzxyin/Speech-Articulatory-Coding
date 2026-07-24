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
scale or quality. The linear EMA-inversion head is reused from the shipped checkpoint rather
than refit from scratch, since that requires the MNGU0 corpus (manual dataset-owner approval, no
automated access) — this matches what the paper itself treats as a fixed component when
describing vocoder training.

## TODO

- Refit the linear EMA-inversion head from scratch on MNGU0 (currently reused from the shipped
  checkpoint; blocked on manual dataset access approval).
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

