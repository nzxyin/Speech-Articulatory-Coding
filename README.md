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


### Training

#### Feature extraction

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


## TODO

- Add training codes.

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

