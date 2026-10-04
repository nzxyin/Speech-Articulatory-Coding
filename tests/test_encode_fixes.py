"""Regression tests for the encode() fixes (issues #11, #12, #13).

CPU-only; model-dependent tests are skipped when the en+ checkpoint is not in the Hugging Face
cache. Run with pytest or plain `python tests/test_encode_fixes.py`.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

HERE = Path(__file__).parent
SAMPLE = HERE.parent / "sample_audio" / "sample1.wav"
BASELINE = HERE / "data" / "enplus_sample1_3s_baseline.npz"

try:
    import pytest
    skip = pytest.skip
except ImportError:  # plain-python mode
    class _Skip(Exception):
        pass

    def skip(msg):
        raise _Skip(msg)

_coder = None


def _load_wav(seconds=None):
    wav, sr = sf.read(SAMPLE)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav if seconds is None else wav[: int(16000 * seconds)]


def _get_coder():
    global _coder
    if _coder is None:
        from sparc import load_model
        try:
            _coder = load_model("en+", device="cpu")
        except Exception as e:  # checkpoint/network unavailable
            skip(f"en+ checkpoint unavailable: {e}")
    return _coder


def test_batch1_matches_baseline():
    # Golden output produced by the pre-fix code (np.random.seed(0) before encode).
    coder = _get_coder()
    base = np.load(BASELINE)
    np.random.seed(0)
    out = coder.encode(_load_wav(3), concat=True)
    assert out["features"].shape == base["features"].shape
    np.testing.assert_allclose(out["features"], base["features"], atol=1e-3)
    np.testing.assert_allclose(out["spk_emb"], base["spk_emb"], atol=1e-3)


def test_batched_short_matches_single():
    coder = _get_coder()
    short, long = _load_wav(1.0), _load_wav(3.0)
    single = coder.encode(short, seed=1)
    batched = coder.encode([short, long], seed=1)
    for key in ["ema", "pitch", "loudness", "periodicity"]:
        n = single[key].shape[0]  # frames beyond the short utterance's own length are padding
        np.testing.assert_allclose(batched[0][key][:n], single[key], atol=1e-3, err_msg=key)
    np.testing.assert_allclose(batched[0]["spk_emb"], single["spk_emb"], atol=1e-3)


def test_short_utterance_does_not_crash():
    coder = _get_coder()
    out = coder.encode(_load_wav(0.25), concat=True, seed=1)  # fewer frames than filtfilt padlen
    assert np.isfinite(out["features"]).all()
    assert np.isfinite(out["spk_emb"]).all()


def test_seed_makes_pitch_reproducible():
    coder = _get_coder()
    wav = _load_wav(1.0)
    a = coder.encode(wav, seed=7)["pitch"]
    b = coder.encode(wav, seed=7)["pitch"]
    np.testing.assert_array_equal(a, b)


def test_zero_weight_spk_emb_is_finite():
    from sparc.spk_encoder import SpeakerEncoder
    enc = SpeakerEncoder(spk_ft_ckpt=None, speech_model=None, device="cpu")
    acoustics = np.random.RandomState(0).randn(2, 10, 8).astype(np.float32)
    weights = np.zeros((2, 10, 1), dtype=np.float32)
    weights[1, :5] = 1.0
    out = enc._get_spk_emb(acoustics, weights, axis=1, lengths=[10, 5])
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0], acoustics[0].mean(0), atol=1e-5)


def test_filter_uses_valid_length_only():
    from sparc.inversion import butter_bandpass_filter, butter_bandpass_filter_padded
    rng = np.random.RandomState(0)
    x = rng.randn(2, 40, 3)
    x[0, 25:] = 0.0  # padding
    y = butter_bandpass_filter_padded(x, [25, 40], 10, 50)
    np.testing.assert_allclose(y[0, :25], butter_bandpass_filter(x[:1, :25], 10, 50)[0], atol=1e-10)
    np.testing.assert_allclose(y[1], butter_bandpass_filter(x[1:], 10, 50)[0], atol=1e-10)
    # shorter than the default padlen (18): must not raise
    z = butter_bandpass_filter_padded(x, [10, 5], 10, 50)
    assert np.isfinite(z).all()


def test_explicit_linear_model_path_overrides_checkpoint_head(tmp_path=None):
    import pickle
    import tempfile
    import types
    from sparc import load_model
    _get_coder()  # skips when the checkpoint is unavailable
    rng = np.random.RandomState(0)
    head = types.SimpleNamespace(coef_=rng.randn(12, 1024).astype(np.float32) * 0.01,
                                 intercept_=rng.randn(12).astype(np.float32))
    with tempfile.TemporaryDirectory() as d:
        pkl = Path(d) / "head.pkl"
        pkl.write_bytes(pickle.dumps(head))
        coder = load_model("en+", device="cpu", linear_model_path=str(pkl))
    np.testing.assert_allclose(coder.inverter.linear_model.weight.numpy(), head.coef_, atol=1e-7)
    np.testing.assert_allclose(coder.inverter.linear_model.bias.numpy(), head.intercept_, atol=1e-7)


def test_load_model_rejects_both_heads():
    from sparc import load_model
    try:
        load_model("en+", device="cpu", linear_model_path="a", linear_model_state_dict={})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_batched_concat_returns_per_utterance():
    # Issue #16: concat=True on a multi-utterance batch indexed a list with a string key.
    coder = _get_coder()
    short, long = _load_wav(1.0), _load_wav(3.0)
    batched = coder.encode([short, long], concat=True, seed=1)
    assert isinstance(batched, list) and len(batched) == 2
    for wav, out in zip([short, long], batched):
        single = coder.encode(wav, concat=True, seed=1)
        n = single["features"].shape[0]
        assert out["features"].shape[1] == single["features"].shape[1]
        np.testing.assert_allclose(out["features"][:n], single["features"], atol=1e-3)
        np.testing.assert_allclose(out["spk_emb"], single["spk_emb"], atol=1e-3)


def test_zero_weight_pitch_stats_is_finite():
    # Issue #16: all-zero periodicity weights made the pitch statistics NaN.
    from sparc.src_extractor import SourceExtractor
    ext = SourceExtractor(device="cpu")
    pitch = np.array([100.0, 200.0, 300.0, 0.0, 0.0])
    stats = ext._pitch_stats(pitch, np.zeros(5), length=3)  # last two frames are batch padding
    np.testing.assert_allclose(stats, [200.0, np.std([100.0, 200.0, 300.0])])
    weights = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(ext._pitch_stats(pitch, weights, length=3), [150.0, 50.0])


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                if type(e).__name__ == "_Skip":
                    print("SKIP", name, e)
                else:
                    failed += 1
                    print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)
