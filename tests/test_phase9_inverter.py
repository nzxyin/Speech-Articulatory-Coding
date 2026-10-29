"""Phase 9: k12 inverter plumbing (no checkpoints), packing of inverter channels, de-normalization switch."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from sparc.vocoders.constants import N_EMA, feature_length
from sparc.vocoders.features import inverter_pack as ip
from sparc.vocoders.features import k12_inverter as k12
from sparc.vocoders.models.hifigan import HiFiGANVocoder
from test_data import build_split
from test_model_hifigan import SMALL, make_features, stats  # noqa: F401  (fixture)

ROWS = [(f"u{i}", f"s{i % 2}", f"c{i}", 0.6 + 0.2 * i) for i in range(7)]


def test_to_grid_shift():
    out = np.arange(10, dtype=np.float64)[:, None] * np.ones((1, 12))
    g = k12.to_grid(out, 1)
    assert g[0, 0] == 1 and g[7, 0] == 8 and g[9, 0] == 9  # last frame repeated
    assert np.array_equal(k12.to_grid(out, 0), out)
    assert k12.to_grid(out, -1)[0, 0] == 0 and k12.to_grid(out, -1)[3, 0] == 2


def test_head_shapes_and_state_keys():
    head = k12.GruConvHead().eval()
    keys = {"front.norm.weight", "front.norm.bias", "front.inp.weight", "front.inp.bias", "conv.weight", "conv.bias",
            "rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0", "rnn.weight_ih_l0_reverse",
            "rnn.weight_hh_l0_reverse", "rnn.bias_ih_l0_reverse", "rnn.bias_hh_l0_reverse", "out.weight", "out.bias"}
    assert set(head.state_dict()) == keys
    assert head(torch.randn(2, 31, 1024)).shape == (2, 31, 12)


class FakeInverter:
    mode, sha256 = "mngu0", "0" * 64

    def describe(self):
        return {"mode": self.mode, "sha256": self.sha256}

    def channels24(self, wav):
        T = feature_length(len(wav))
        return np.tile(np.linspace(-1, 1, T, dtype=np.float32)[:, None], (1, 12)) * (1 + float(np.abs(wav).max()))


def _existing(tmp_path):
    p = tmp_path / "existing.json"
    p.write_text(json.dumps({"ema_mean": [0.0] * 12, "ema_std": [1.0] * 12, "per_mean": 0.5, "logf0_std": 0.3}))
    return p


def test_extract_merge_keeps_other_columns(tmp_path, monkeypatch):
    src = build_split(tmp_path, "src", ROWS)
    monkeypatch.setattr(ip, "CHUNK", 3)
    monkeypatch.setattr(k12, "K12Inverter", lambda mode, device: FakeInverter())
    out = tmp_path / "new" / "src"
    assert ip.run_extract(src, out, "mngu0", 0, 2, "cpu", None) == 2  # chunks 0 and 2 of 3
    assert ip.run_extract(src, out, "mngu0", 0, 2, "cpu", None) == 0  # resumable
    with pytest.raises(FileNotFoundError):
        ip.run_merge(src, out)
    assert ip.run_extract(src, out, "mngu0", 1, 2, "cpu", None) == 1
    ip.run_merge(src, out)
    old, new = np.load(src / "feats.npy"), np.load(out / "feats.npy")
    assert new.shape == old.shape and new.dtype == old.dtype
    assert np.array_equal(old[:, N_EMA:], new[:, N_EMA:])
    assert not np.allclose(old[:, :N_EMA], new[:, :N_EMA])
    for name in ip.COPIED:
        assert (src / name).read_bytes() == (out / name).read_bytes()
    stats_out = tmp_path / "stats.json"
    s = ip.run_stats([out], stats_out, existing=_existing(tmp_path))
    assert len(s["ema_mean"]) == 12 and s["inverter_stats"]["frames"] == len(new)
    assert np.allclose(s["ema_mean"], new[:, :N_EMA].mean(0), atol=1e-6)
    assert s["per_mean"] == 0.5 and s["logf0_std"] == 0.3  # other fields copied
    index = pd.read_parquet(out / "index.parquet")
    assert len(index) == len(ROWS)


def test_ema_denorm_default_unchanged_and_identity_at_start(stats):  # noqa: F811
    torch.manual_seed(0)
    base = HiFiGANVocoder(stats, **SMALL).eval()
    torch.manual_seed(0)
    default = HiFiGANVocoder(stats, ema_denorm=False, **SMALL).eval()
    assert list(base.state_dict()) == list(default.state_dict()) and base.ema_denorm is None
    assert not any("ema_denorm" in k for k in base.state_dict())
    torch.manual_seed(0)
    switched = HiFiGANVocoder(stats, ema_denorm=True, **SMALL).eval()
    extra = set(switched.state_dict()) - set(base.state_dict())
    assert extra == {"ema_denorm.weight", "ema_denorm.bias"}
    missing = switched.load_state_dict(base.state_dict(), strict=False)
    assert not missing.unexpected_keys and set(missing.missing_keys) == extra
    feats, spk = make_features(2, 40), torch.randn(2, 64)
    with torch.no_grad():
        assert torch.equal(base(feats, spk), switched(feats, spk))
        assert torch.equal(base(feats, spk), default(feats, spk))
    switched.train()
    switched(feats, spk).square().mean().backward()
    assert switched.ema_denorm.weight.grad.abs().sum() > 0  # the switch can learn from the start
    with torch.no_grad():
        switched.ema_denorm.bias[:N_EMA] += 0.5  # an offset changes the output
        switched.eval()
        assert not torch.equal(base(feats, spk), switched(feats, spk))


def test_experiment_yaml_can_switch_on_denorm_and_inverter(tmp_path):
    """An experiment file (not the base configs) enables the switches; the base configs keep them off."""
    import shutil
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import sparc.conf

    conf = tmp_path / "conf"
    shutil.copytree(Path(sparc.conf.__file__).parent, conf, ignore=shutil.ignore_patterns("__pycache__"))
    (conf / "experiment" / "p9_probe.yaml").write_text(
        "# @package _global_\nexperiment_name: p9_probe\ninverter:\n  name: mngu0\nvocoder:\n  generator:\n    ema_denorm: true\n"
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        base = compose(config_name="vocoder_config")
        cfg = compose(config_name="vocoder_config", overrides=["experiment=p9_probe"])
    GlobalHydra.instance().clear()
    assert base.inverter.name is None and "ema_denorm" not in base.vocoder.generator
    assert cfg.inverter.name == "mngu0" and cfg.vocoder.generator.ema_denorm is True
