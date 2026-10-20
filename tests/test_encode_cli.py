"""CPU-only tests for sparc-encode / sparc-decode plumbing (issue #12); no checkpoint needed."""
import json

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from sparc.cli import decode as decode_cli
from sparc.cli import encode as encode_cli


class FakeCoder:
    sr = 16000

    def __init__(self, fail_on=()):
        self.fail_on = fail_on
        self.calls = []

    def encode(self, audio, concat=True, seed=None):
        self.calls.append(seed)
        if len(sf.read(audio)[0]) in self.fail_on:
            raise RuntimeError("boom")
        return {"features": np.zeros((5, 15), dtype=np.float32), "spk_emb": np.zeros(64, dtype=np.float32)}


def _cfg(tmp_path, linear_model_path=None, **extra):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir(exist_ok=True)
    cfg = {
        "device": "cpu",
        "deterministic_pitch": True,
        "dataset": {"wav_dir": str(wav_dir), "save_dir": str(tmp_path / "out")},
        "model": {"model_name": "en+", "config_path": None, "linear_model_path": linear_model_path},
    }
    cfg.update(extra)
    return OmegaConf.create(cfg)


def _wavs(tmp_path, lengths=(1000, 2000)):
    d = tmp_path / "wavs"
    d.mkdir(exist_ok=True)
    for n in lengths:
        sf.write(d / f"u{n}.wav", np.zeros(n), 16000)


def _patch(monkeypatch, coder):
    monkeypatch.setattr(encode_cli, "load_model", lambda *a, **k: coder)
    return coder


def test_atomic_save_tmp_name_and_cleanup(tmp_path, monkeypatch):
    target = tmp_path / "a.npy"
    seen = []
    real_replace = encode_cli.os.replace

    def spy(src, dst):
        seen.append(src.name)
        return real_replace(src, dst)

    monkeypatch.setattr(encode_cli.os, "replace", spy)
    encode_cli._atomic_save(target, np.arange(3))
    assert not seen[0].endswith(".npy")
    assert [p.name for p in tmp_path.iterdir()] == ["a.npy"]
    np.testing.assert_array_equal(np.load(target), np.arange(3))

    def fail(*a):
        raise OSError("x")

    bad = tmp_path / "b.npy"
    monkeypatch.setattr(encode_cli.os, "replace", fail)
    with pytest.raises(OSError):
        encode_cli._atomic_save(bad, np.arange(3))
    assert not bad.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.npy"]


def test_skip_only_when_both_outputs_exist(tmp_path, monkeypatch):
    _wavs(tmp_path)
    cfg = _cfg(tmp_path)
    save = tmp_path / "out"
    (save / "emasrc").mkdir(parents=True)
    (save / "spk_emb").mkdir(parents=True)
    np.save(save / "emasrc" / "u1000.npy", np.zeros(1))
    np.save(save / "spk_emb" / "u1000.npy", np.zeros(1))
    np.save(save / "emasrc" / "u2000.npy", np.zeros(1))  # only one of the two
    coder = _patch(monkeypatch, FakeCoder())
    encode_cli.run(cfg)
    assert len(coder.calls) == 1
    assert (save / "spk_emb" / "u2000.npy").exists()
    assert np.load(save / "emasrc" / "u2000.npy").shape == (5, 15)


def test_failure_exits_nonzero(tmp_path, monkeypatch):
    _wavs(tmp_path)
    _patch(monkeypatch, FakeCoder(fail_on=(2000,)))
    with pytest.raises(SystemExit) as e:
        encode_cli.run(_cfg(tmp_path))
    assert e.value.code == 1
    save = tmp_path / "out"
    assert (save / "emasrc" / "u1000.npy").exists()
    assert not (save / "emasrc" / "u2000.npy").exists()
    assert not list(save.rglob("*.tmp"))


def test_manifest_written_and_matches(tmp_path, monkeypatch):
    _wavs(tmp_path)
    head = tmp_path / "head.pkl"
    head.write_bytes(b"abc")
    _patch(monkeypatch, FakeCoder())
    encode_cli.run(_cfg(tmp_path, linear_model_path=str(head)))
    meta = json.loads((tmp_path / "out" / "encode_meta.json").read_text())
    assert meta["model_name"] == "en+"
    assert meta["linear_model_path"] == str(head)
    assert len(meta["linear_model_sha256"]) == 64
    encode_cli.run(_cfg(tmp_path, linear_model_path=str(head)))  # same settings: fine


def test_manifest_mismatch_and_override(tmp_path, monkeypatch):
    _wavs(tmp_path)
    coder = _patch(monkeypatch, FakeCoder())
    encode_cli.run(_cfg(tmp_path))
    n = len(coder.calls)
    with pytest.raises(SystemExit) as e:
        encode_cli.run(_cfg(tmp_path, deterministic_pitch=False))
    assert "deterministic_pitch" in str(e.value.code)
    assert len(coder.calls) == n  # failed before loading/encoding
    encode_cli.run(_cfg(tmp_path, deterministic_pitch=False, allow_meta_mismatch=True))
    meta = json.loads((tmp_path / "out" / "encode_meta.json").read_text())
    assert meta["deterministic_pitch"] is True  # stored manifest is not rewritten


def test_manifest_head_content_change_detected(tmp_path, monkeypatch):
    _wavs(tmp_path)
    head = tmp_path / "head.pkl"
    head.write_bytes(b"abc")
    _patch(monkeypatch, FakeCoder())
    encode_cli.run(_cfg(tmp_path, linear_model_path=str(head)))
    head.write_bytes(b"abd")
    with pytest.raises(SystemExit):
        encode_cli.run(_cfg(tmp_path, linear_model_path=str(head)))


def test_manifest_same_head_under_another_path(tmp_path, monkeypatch):
    _wavs(tmp_path)
    head = tmp_path / "head.pkl"
    head.write_bytes(b"abc")
    copy = tmp_path / "copy_of_head.pkl"
    copy.write_bytes(b"abc")
    _patch(monkeypatch, FakeCoder())
    encode_cli.run(_cfg(tmp_path, linear_model_path=str(head)))
    encode_cli.run(_cfg(tmp_path, linear_model_path=str(copy)))  # same sha256: not a mismatch
    assert not list((tmp_path / "out").glob(".*.tmp"))


def test_legacy_dir_warns_without_manifest(tmp_path, monkeypatch, capsys):
    _wavs(tmp_path)
    save = tmp_path / "out"
    (save / "emasrc").mkdir(parents=True)
    np.save(save / "emasrc" / "old.npy", np.zeros(1))
    _patch(monkeypatch, FakeCoder())
    encode_cli.run(_cfg(tmp_path))
    assert "no encode_meta.json" in capsys.readouterr().err
    assert not (save / "encode_meta.json").exists()


def test_decode_pairs_by_stem(tmp_path, capsys):
    ft, spk = tmp_path / "emasrc", tmp_path / "spk_emb"
    ft.mkdir()
    spk.mkdir()
    for s in ["a", "b", "c"]:
        np.save(ft / f"{s}.npy", np.zeros(1))
    for s in ["b", "c", "d"]:
        np.save(spk / f"{s}.npy", np.zeros(1))
    (ft / ".stray.npy.tmp").write_bytes(b"")
    np.save(ft / "c.tmp.npy", np.zeros(1))  # partial write left by the old temp naming
    np.save(spk / "c.tmp.npy", np.zeros(1))
    pairs = decode_cli._pair_by_stem(ft, spk)
    assert [(f.stem, s.stem) for f, s in pairs] == [("b", "b"), ("c", "c")]
    err = capsys.readouterr().err
    assert "skipping a" in err and "skipping d" in err

