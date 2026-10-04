# Third-party code

Parts of `src/sparc/vocoders/` are ported from the projects below. Each ported file names its source in a header
comment. The original license texts are in `licenses/`.

| project | commit | license | ported into |
|---|---|---|---|
| [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) | `eb39abf` | MIT, Copyright (c) 2023 Charactr Inc. | `models/vocos.py`, `losses/discriminators.py`, `losses/losses.py` |
| [Louis0324/DDSP-Articulatory-Vocoder](https://github.com/Louis0324/DDSP-Articulatory-Vocoder) | `dc6df4c` | MIT, Copyright (c) 2024 Louis Liu, Drake Lin | `models/ddsp.py`, `losses/losses.py` |

RT-VC (Berkeley-Speech-Group/RT-VC) has no license, so none of its code is copied; only ideas described in its paper
(periodicity as an input, FiLM in the heads) are reimplemented.
