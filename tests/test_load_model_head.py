"""Tests for the linear AAI head selection in load_model (issue #11).

Checkpoint-gated tests are skipped when the en+ files are not in the Hugging Face cache.
"""
import sys

import numpy as np
import pytest
import torch


def _en_plus_files():
    from sparc.sparc import download_huggingface, model_name_map
    name = model_name_map["en+"]
    try:
        return download_huggingface(f"{name}.yaml"), download_huggingface(f"{name}.ckpt")
    except Exception as e:  # checkpoint/network unavailable
        pytest.skip(f"en+ checkpoint unavailable: {e}")


def _ckpt_head(ckpt_path):
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]["linear_model"]


def _assert_head(coder, state_dict):
    lin = coder.inverter.linear_model
    for k, v in state_dict.items():
        np.testing.assert_array_equal(lin.state_dict()[k].numpy(), v.numpy())


def test_default_keeps_checkpoint_head():
    from sparc import load_model
    _, ckpt = _en_plus_files()
    coder = load_model("en+", device="cpu")
    _assert_head(coder, _ckpt_head(ckpt))


def test_state_dict_alone_gives_that_head():
    from sparc import load_model
    _, ckpt = _en_plus_files()
    sd = {k: v + 1.0 for k, v in _ckpt_head(ckpt).items()}
    coder = load_model("en+", device="cpu", linear_model_state_dict=sd)
    _assert_head(coder, sd)


def test_config_without_linear_model_path_key_loads():
    import yaml
    from sparc import load_model
    cfg_path, ckpt = _en_plus_files()
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.pop("linear_model_path", None)
    coder = load_model(config=config, ckpt=ckpt, device="cpu")
    _assert_head(coder, _ckpt_head(ckpt))


def test_hydra_model_linear_model_path_reaches_load_model(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_module
    from sparc.cli import encode

    class _Stop(Exception):
        pass

    calls = []

    def fake_load_model(*args, **kwargs):
        calls.append((args, kwargs))
        raise _Stop

    monkeypatch.setattr(encode, "load_model", fake_load_model)
    run = encode.main.__wrapped__  # the undecorated function, called with a composed config
    head = tmp_path / "head.pkl"
    head.write_bytes(b"not loaded")  # sparc-encode hashes it for encode_meta.json
    with initialize_config_module(config_module="sparc.conf", version_base=None):
        for i, override in enumerate((None, str(head))):
            calls.clear()
            # separate save_dirs: encode_meta.json would reject the second setting in the first dir
            overrides = ["dataset=esd", f"dataset.save_dir={tmp_path / f'out{i}'}", "device=cpu"]
            overrides += [f"model.linear_model_path={override}"] if override else []
            cfg = compose(config_name="config", overrides=overrides)
            assert cfg.model.model_name == "en+"
            with pytest.raises(_Stop):
                run(cfg)
            assert calls[0][0] == ("en+",)
            assert calls[0][1].get("linear_model_path") == override


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
