"""Tests for the extractor fixes (issue #13 follow-up): file-path pitch extraction, RNG restoration,
use_penn handling and the configurable spk_raw directory. CPU-only; torchcrepe ships its weights."""
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torchcrepe

from sparc.src_extractor import SourceExtractor
from sparc.training.dataset import VocoderDataset

HERE = Path(__file__).parent
SAMPLE = HERE.parent / "sample_audio" / "sample1.wav"


@pytest.fixture(scope="module")
def short_wav(tmp_path_factory):
    import librosa
    wav, sr = sf.read(SAMPLE)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)[: 16000]
    path = tmp_path_factory.mktemp("wav") / "short.wav"
    sf.write(path, wav, 16000)
    return str(path)


@pytest.fixture(scope="module")
def extractor():
    return SourceExtractor(device="cpu", crepe_model="tiny")


def _state_equal(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_extract_pitch_with_file_path(extractor, short_wav):
    out = extractor._extract_pitch(short_wav, seed=0)
    assert out["pitch"].shape == out["periodicity"].shape
    assert out["pitch"].shape[0] == 1 and out["pitch"].shape[-1] == 1
    assert out["pitch_stats"].shape == (1, 2)


def test_outputs_default_not_shared(extractor, short_wav):
    first = extractor._extract_pitch(short_wav, seed=0)
    second = extractor._extract_pitch(short_wav, seed=0)
    assert first is not second


def test_rng_restored_on_success(extractor, short_wav):
    np.random.seed(123)
    before = np.random.get_state()
    extractor._extract_pitch(short_wav, seed=5)
    assert _state_equal(before, np.random.get_state())


def test_rng_restored_on_exception(extractor, short_wav, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("forced")
    monkeypatch.setattr(torchcrepe, "predict", boom)
    np.random.seed(123)
    before = np.random.get_state()
    with pytest.raises(RuntimeError):
        extractor._extract_pitch(short_wav, seed=5)
    assert _state_equal(before, np.random.get_state())


def test_seed_behaviour(extractor, short_wav):
    a = extractor._extract_pitch(short_wav, seed=1)["pitch"]
    a2 = extractor._extract_pitch(short_wav, seed=1)["pitch"]
    b = extractor._extract_pitch(short_wav, seed=2)["pitch"]
    np.testing.assert_array_equal(a, a2)
    assert not np.array_equal(a, b)


def test_seeded_call_leaves_no_seed_behind(extractor, short_wav):
    np.random.seed(7)
    expected_state = np.random.get_state()
    extractor._extract_pitch(short_wav, seed=99)
    assert _state_equal(expected_state, np.random.get_state())


def test_shipped_config_use_penn_false():
    import yaml
    cfg = yaml.safe_load((HERE.parent / "configs" / "model_englishplus_2M.yaml").read_text())
    assert cfg["use_penn"] is False


def test_use_penn_does_not_warn(caplog):
    from sparc import load_model
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with caplog.at_level(logging.INFO, logger="sparc.sparc"):
                load_model("en+", device="cpu")
    except Exception as e:
        pytest.skip(f"en+ checkpoint unavailable: {e}")
    assert not [x for x in w if "use_penn" in str(x.message)]


def _fake_sparc_dir(root, subdirs):
    wav_dir = root / "wav"
    wav_dir.mkdir()
    sparc_dir = root / "sparc"
    (sparc_dir / "emasrc").mkdir(parents=True)
    for sub in subdirs:
        (sparc_dir / sub).mkdir()
    sf.write(wav_dir / "utt.wav", np.zeros(16000 * 2, dtype=np.float32), 16000)
    np.save(sparc_dir / "emasrc" / "utt.npy", np.zeros((100, 15), dtype=np.float32))
    for sub in subdirs:
        np.save(sparc_dir / sub / "utt.npy", np.zeros(1024, dtype=np.float32))
    return wav_dir, sparc_dir


def test_dataset_default_spk_raw_dir(tmp_path):
    wav_dir, sparc_dir = _fake_sparc_dir(tmp_path, ["spk_raw_enplus"])
    assert len(VocoderDataset(wav_dir, sparc_dir)) == 0  # default looks in spk_raw/ only


def test_dataset_configured_spk_raw_dir(tmp_path):
    wav_dir, sparc_dir = _fake_sparc_dir(tmp_path, ["spk_raw_enplus"])
    ds = VocoderDataset(wav_dir, sparc_dir, spk_raw_subdir="spk_raw_enplus")
    assert len(ds) == 1
    assert ds.items[0][3].parent.name == "spk_raw_enplus"
