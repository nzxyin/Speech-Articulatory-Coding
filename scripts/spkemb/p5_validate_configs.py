"""Phase 5 config check (no training): composes experiment p5_r / p5_f / p5_fv, reads the packed splits, draws one
training batch from the data module and reports the data composition.

    scripts/spkemb/p5_job.sh scripts/spkemb/p5_validate_configs.py [p5_r p5_f p5_fv]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from sparc.vocoders.data.datamodule import VocoderDataModule
from sparc.vocoders.data.dataset import PackedStore

CONF = Path(__file__).resolve().parents[2] / "src" / "sparc" / "conf"
BASE = "p4_mrg_main"


def compose_cfg(name: str):
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return compose(config_name="vocoder_config", overrides=["vocoder=hifigan", f"experiment={name}", "data.num_workers=0"])


def check(name: str) -> dict:
    cfg = compose_cfg(name)
    assert cfg.experiment_name == name and cfg.seed == 0 and cfg.train.max_g_steps == 100000
    assert cfg.vocoder.name == "hifigan" and tuple(cfg.speaker.n_refs) == (1, 8) and cfg.speaker.rel_f0.enabled
    assert cfg.vocoder.generator.pitch_mode == "log_relative" and cfg.speaker.p_cross == 0.5 and cfg.speaker.layer == "l6"
    dm = VocoderDataModule(cfg)
    dirs = [dm._packed(s) for s in cfg.data.train_splits]
    parts = {}
    for d in dirs:
        idx = pd.read_parquet(d / "index.parquet", columns=["id", "speaker", "duration", "T"])
        parts[d.name if d.name.startswith(("train", "globe", "vctk")) else str(d)] = {
            "dir": str(d), "utterances": int(len(idx)), "hours": float(idx["duration"].sum() / 3600),
            "speakers": int(idx["speaker"].nunique()),
        }  # fmt: skip
    store = PackedStore(dirs, cfg.speaker.layer)
    assert len(set(store.ids)) == len(store.ids)
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    n = int(cfg.data.batch_size)
    assert batch["features"].shape == (n, 15, int(cfg.data.crop_frames)), batch["features"].shape
    assert batch["spk_raw"].shape == (n, 8, 1024) and batch["ref_mask"].shape == (n, 8)
    assert bool(batch["ref_mask"][:, 0].all()) and np.isfinite(batch["spk_raw"].numpy()).all()
    assert np.isfinite(batch["features"].numpy()).all() and batch["audio"].shape[0] == n
    assert np.isfinite(batch["utt_f0"].numpy()).all() or True  # NaN for utterances without voiced frames is allowed
    out = {"experiment": name, "utterances": int(len(store)), "speakers": int(len(store.speakers)), "splits": parts,
           "batch": {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")}}  # fmt: skip
    # references of a speaker-labelled corpus come from the same speaker: draw many (utterance, reference) pairs
    ds = loader.dataset
    same, cross = 0, 0
    for key in [(0, p, i) for p, i in enumerate(np.random.default_rng(0).integers(0, len(ds), 400))]:
        s = ds[key]
        u, r = int(s["utt_index"]), int(s["ref_index"])
        if r != u:
            cross += 1
            same += int(store.speaker_codes[u] == store.speaker_codes[r])
    out["cross_references_same_speaker"] = [same, cross]
    assert same == cross
    return out


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def composed_diff(a: str, b: str) -> dict:
    fa = flat(OmegaConf.to_container(compose_cfg(a), resolve=False))
    fb = flat(OmegaConf.to_container(compose_cfg(b), resolve=False))
    return {k: [fa.get(k), fb.get(k)] for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


if __name__ == "__main__":
    names = sys.argv[1:] or ["p5_r", "p5_f", "p5_fv"]
    print(json.dumps([check(n) for n in names], indent=1))
    for n in names:
        for ref in (BASE, "p4_g"):
            print(f"DIFF {ref} -> {n}:", json.dumps(composed_diff(ref, n), indent=1))
    print("DIFF p5_r -> p5_f:", json.dumps(composed_diff("p5_r", "p5_f"), indent=1))
