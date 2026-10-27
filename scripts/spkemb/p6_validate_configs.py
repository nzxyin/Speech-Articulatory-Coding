"""Phase 6 config check (no training): composes p6_<e1|e2|e3>_<p5_r|p5_f|p5_fv>, draws one training batch from the data
module on the real packed splits and Phase 6 side files, runs the speaker network on it, checks the vector layout
against the packed ``spk_l6`` and prints the composed-config difference to the Phase 5 arm.

    scripts/spkemb/p5_job.sh scripts/spkemb/p6_validate_configs.py [p6_e1_p5_r ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from sparc.vocoders.data.datamodule import VocoderDataModule
from sparc.vocoders.data.dataset import PackedStore
from sparc.vocoders.models.frontend import load_stats
from sparc.vocoders.training.speaker_terms import build_speaker

CONF = Path(__file__).resolve().parents[2] / "src" / "sparc" / "conf"
LAYERS = {"e1": "l1l6", "e2": "vus3_l6", "e3": "l6_rep3"}
WIDTH = {"l1l6": 2048, "vus3_l6": 3072, "l6_rep3": 3072}


def compose_cfg(name: str):
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return compose(config_name="vocoder_config", overrides=["vocoder=hifigan", f"experiment={name}", "data.num_workers=0"])


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def check(name: str) -> dict:
    _, arm, p5 = name.split("_", 2)
    layer = LAYERS[arm]
    cfg = compose_cfg(name)
    fa = flat(OmegaConf.to_container(compose_cfg(p5), resolve=False))
    fb = flat(OmegaConf.to_container(cfg, resolve=False))
    diff = {k: [fa.get(k), fb.get(k)] for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}
    assert set(diff) == {"experiment_name", "speaker.layer"}, diff
    assert cfg.speaker.layer == layer and cfg.seed == 0 and cfg.train.max_g_steps == 100000 and cfg.vocoder.name == "hifigan"
    dm = VocoderDataModule(cfg)
    dirs = [dm._packed(s) for s in cfg.data.train_splits]
    store = PackedStore(dirs, layer)  # raises when an utterance has no side vector
    l6 = PackedStore(dirs, "l6")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    n = int(cfg.data.batch_size)
    spk = batch["spk_raw"]
    assert spk.shape == (n, 8, WIDTH[layer]) and np.isfinite(spk.numpy()).all()
    mask = batch["ref_mask"]
    # first reference of every sample: the layout is built from the packed spk_l6 (and the side vectors)
    first = spk[:, 0].numpy()
    ref = np.stack([l6.speaker_vector(int(r)) for r in batch["ref_index"]])
    if layer == "l1l6":
        assert np.array_equal(first[:, 1024:], ref)
    elif layer == "l6_rep3":
        assert all(np.array_equal(first[:, i * 1024 : (i + 1) * 1024], ref) for i in range(3))
    speaker = build_speaker(cfg, load_stats(cfg.stats_path))
    z, _ = speaker.embed(spk, mask)
    assert z.shape == (n, 64) and torch.isfinite(z).all()
    assert speaker.net[0].in_features == WIDTH[layer]
    return {"experiment": name, "layer": layer, "utterances": int(len(store)), "spk_raw": list(spk.shape),
            "z_std": float(z.std()), "config_diff_to_" + p5: diff}  # fmt: skip


if __name__ == "__main__":
    names = sys.argv[1:] or [f"p6_{a}_p5_{p}" for p in ("r", "f", "fv") for a in ("e1", "e2", "e3")]
    print(json.dumps([check(n) for n in names], indent=1))
