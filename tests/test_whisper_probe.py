"""Whisper encoder probing: frame geometry, padding and slicing, prefix equivalence, cache writer, FLOP accounting.

Uses small randomly initialised encoders and a local feature extractor, so no checkpoint is needed.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from sparc.compression import whisper_extract as we
from sparc.compression.extract import lowpass
from sparc.compression.whisper_encoders import (
    MEL_FRAMES, OUT_FRAMES, PrefixEncoder, hidden_states, log_mel, n_real_frames,
)


def tiny_encoder(layers=3, d=64, heads=4, mel=80, seed=0):
    torch.manual_seed(seed)
    cfg = WhisperConfig(num_mel_bins=mel, d_model=d, encoder_layers=layers, encoder_attention_heads=heads,
                        encoder_ffn_dim=2 * d, max_source_positions=OUT_FRAMES)
    return WhisperEncoder(cfg).eval()


@pytest.fixture(scope="module")
def fe():
    return WhisperFeatureExtractor(feature_size=80)


def test_n_real_frames_is_centre_inside_audio():
    for n in [1, 159, 160, 161, 319, 320, 321, 640, 1000, 16000, 63999, 64000, 480000]:
        assert n_real_frames(n) == sum(1 for j in range(OUT_FRAMES) if 320 * j < n)


def test_n_real_frames_vs_wavlm_convention():
    # WavLM / XLS-R: floor((n - 400) / 320) + 1 frames; Whisper keeps one or two more, so the probe's aligned() truncation
    # (min of feature and EMA lengths) is unaffected.
    for n in range(8000, 8000 + 1600, 7):
        d = n_real_frames(n) - ((n - 400) // 320 + 1)
        assert d in (1, 2)


def test_impulse_geometry(fe):
    j0 = 40
    n = 320 * j0
    wav = np.zeros(32000, dtype=np.float32)
    wav[n] = 1.0
    mel = log_mel(fe, [wav])[0]  # (80, 3000)
    peak = int(mel.max(0).values.argmax())
    assert peak == round(n / 160) == 2 * j0  # mel frame 2 j0 feeds encoder frame j0 (stride 2), centred on sample 320 j0
    assert mel.shape == (80, MEL_FRAMES)


def test_rejects_over_30_seconds(fe):
    with pytest.raises(ValueError):
        log_mel(fe, [np.zeros(480001, dtype=np.float32)])
    log_mel(fe, [np.zeros(480000, dtype=np.float32)])  # exactly 30 s is fine


def test_layer_count_and_prefix_equals_hidden_states(fe):
    enc = tiny_encoder(layers=3)
    wav = np.random.default_rng(0).standard_normal(24000).astype(np.float32) * 0.1
    mel = log_mel(fe, [wav])
    hs = hidden_states(enc, mel)
    assert len(hs) == 3 + 1
    with torch.no_grad():
        for k in range(4):
            out = PrefixEncoder(enc, k)(mel)
            torch.testing.assert_close(out, hs[k], atol=1e-5, rtol=1e-5)
    # layer 0 = convolutions + positional embedding (no layer norm); the last layer carries the final LayerNorm
    # (the default LayerNorm gives zero mean per frame; the pre-norm residual stream does not)
    assert hs[3].mean(-1).abs().max() < 1e-4
    assert hs[2].mean(-1).abs().max() > 1e-3


def test_prefix_rejects_short_window(fe):
    enc = tiny_encoder(layers=2)
    with pytest.raises(ValueError):
        PrefixEncoder(enc, 1)(torch.zeros(1, 80, 1000))


def test_batch_of_8_equals_batch_of_1(fe):
    enc = tiny_encoder(layers=2)
    rng = np.random.default_rng(1)
    wavs = [rng.standard_normal(int(16000 * (0.6 + 0.3 * i))).astype(np.float32) * 0.1 for i in range(8)]
    mel = log_mel(fe, wavs)
    hs8 = hidden_states(enc, mel)
    for i in (0, 3, 7):
        hs1 = hidden_states(enc, mel[i : i + 1])
        for k in range(3):
            torch.testing.assert_close(hs8[k][i], hs1[k][0], atol=1e-4, rtol=1e-4)
    # the front end is also per row: a row does not depend on its batch mates
    single = log_mel(fe, [wavs[5]])
    torch.testing.assert_close(mel[5 : 6], single, atol=1e-5, rtol=1e-5)


def test_slicing_happens_before_the_lowpass():
    rng = np.random.default_rng(2)
    h = rng.standard_normal((OUT_FRAMES, 8)).astype(np.float32)
    h[200:] += 50.0  # a padding-like state change after the real frames
    n = 150
    got = we.slice_and_filter(h, n)
    assert got.shape == (n, 8)
    np.testing.assert_allclose(got, lowpass(h[:n]), atol=1e-5)
    assert np.abs(got - lowpass(h)[:n]).max() > 1e-3  # filtering the whole window first would differ
    raw = we.slice_and_filter(h, n, raw=True)
    np.testing.assert_array_equal(raw, h[:n])


class FakeDS:
    """Three short utterances; the interface `extract_whisper` needs."""

    def __init__(self, n_utts=3, seconds=(1.1, 1.5, 2.0)):
        rng = np.random.default_rng(3)
        self.wavs = {f"u{i}": (rng.standard_normal(int(16000 * seconds[i % len(seconds)])) * 0.1).astype(np.float32)
                     for i in range(n_utts)}

    def available_stems(self):
        return sorted(self.wavs)

    def n_samples(self, s):
        return len(self.wavs[s])

    def splits(self):
        st = self.available_stems()
        return {"train": st[:1], "valid": st[1:2], "test": st[2:]}

    def load_utterance(self, s):
        w = self.wavs[s]
        return SimpleNamespace(wav=w, ema_mm=np.zeros((len(w) // 320 - 6, 12), np.float32), base_offset=2)


def _read(out, k):
    return np.load(out / f"layer_{k:02d}.npy", mmap_mode="r")


def test_cache_layout_and_values(tmp_path, fe):
    enc, ds = tiny_encoder(layers=2), FakeDS()
    out = we.extract_whisper("whisper-medium", tmp_path, device="cpu", batch=2, bundle=(enc, fe), ds=ds, threads=2)
    index = json.loads((out / "index.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    assert meta["complete"] and meta["num_layers"] == 2 and meta["dim"] == 64 and meta["padded"] and meta["fp32_layers"] == []
    start = 0
    for s in ds.available_stems():
        n = n_real_frames(ds.n_samples(s))
        assert index[s] == [start, n]  # frame count follows ceil(n / 320), not the WavLM count
        start += n
    assert meta["n_frames"] == start
    for k in range(3):
        m = _read(out, k)
        assert m.dtype == np.float16 and m.shape == (start, 64)
    # values: layer 2 of utterance u1 equals real-frame slice then low-pass of the stock padded forward
    mel = log_mel(fe, [ds.wavs["u1"]])
    h = hidden_states(enc, mel)[2][0].numpy()
    a, n = index["u1"]
    np.testing.assert_allclose(_read(out, 2)[a : a + n].astype(np.float32), lowpass(h[:n]), atol=2e-3, rtol=2e-3)


def test_fp16_overflow_falls_back_to_float32(tmp_path, fe):
    enc, ds = tiny_encoder(layers=2), FakeDS()
    ref = we.extract_whisper("whisper-medium", tmp_path / "ref", device="cpu", batch=2, bundle=(enc, fe), ds=ds, threads=2)
    peaks = [float(np.abs(_read(ref, k).astype(np.float32)).max()) for k in range(3)]
    limit = float(np.median(peaks)) * 0.999
    big = [k for k in range(3) if peaks[k] > limit]
    assert 0 < len(big) < 3
    out = we.extract_whisper("whisper-medium", tmp_path / "fp", device="cpu", batch=2, bundle=(enc, fe), ds=ds,
                             threads=2, fp16_limit=limit)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["fp32_layers"] == big
    for k in range(3):
        assert _read(out, k).dtype == (np.float32 if k in big else np.float16)
        np.testing.assert_allclose(_read(out, k).astype(np.float32), _read(ref, k).astype(np.float32), atol=2e-3, rtol=2e-3)


def test_resume_after_interruption(tmp_path, fe, monkeypatch):
    enc, ds = tiny_encoder(layers=2), FakeDS(n_utts=6)
    ref = we.extract_whisper("whisper-medium", tmp_path / "ref", device="cpu", batch=1, bundle=(enc, fe), ds=ds, threads=2)
    monkeypatch.setattr(we, "CHECK_EVERY", 1)
    calls = {"n": 0}

    def flaky(fe_, wavs, *a, **k):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("preempted")
        return real(fe_, wavs, *a, **k)

    real = we.log_mel
    monkeypatch.setattr(we, "log_mel", flaky)
    with pytest.raises(RuntimeError):
        we.extract_whisper("whisper-medium", tmp_path / "run", device="cpu", batch=1, bundle=(enc, fe), ds=ds, threads=2)
    assert json.loads((tmp_path / "run" / "whisper-medium" / "progress.json").read_text())["done"] == 3
    assert not (tmp_path / "run" / "whisper-medium" / "meta.json").exists()
    monkeypatch.setattr(we, "log_mel", real)
    seen = []
    monkeypatch.setattr(we, "log_mel", lambda f, w, *a, **k: (seen.append(len(w)), real(f, w, *a, **k))[1])
    out = we.extract_whisper("whisper-medium", tmp_path / "run", device="cpu", batch=1, bundle=(enc, fe), ds=ds, threads=2)
    assert len(seen) == 3  # only the three unfinished utterances were redone
    for k in range(3):
        np.testing.assert_array_equal(_read(out, k), _read(ref, k))
    # a complete cache is not redone
    seen.clear()
    we.extract_whisper("whisper-medium", tmp_path / "run", device="cpu", batch=1, bundle=(enc, fe), ds=ds, threads=2)
    assert seen == []


def test_extraction_refuses_over_30_seconds(tmp_path, fe):
    enc = tiny_encoder(layers=1)
    ds = FakeDS(n_utts=1, seconds=(31.0,))
    with pytest.raises((AssertionError, ValueError)):
        we.extract_whisper("whisper-medium", tmp_path, device="cpu", bundle=(enc, fe), ds=ds, threads=1)


def _flops(cfg, ideal=False):
    from sparc.compression.whisper_compute import count_flops

    with torch.device("meta"):
        enc = WhisperEncoder(cfg).eval()
    return count_flops(PrefixEncoder(enc, cfg.encoder_layers), ideal=ideal)


@pytest.mark.parametrize("name,cfg,padded,ideal", [
    ("large-v3", dict(d_model=1280, encoder_layers=32, encoder_attention_heads=20, encoder_ffn_dim=5120, num_mel_bins=128),
     2.27e12, 65e9 * 4),
    ("medium", dict(d_model=1024, encoder_layers=24, encoder_attention_heads=16, encoder_ffn_dim=4096, num_mel_bins=80),
     1.14e12, 32e9 * 4),
])
def test_flops_match_hand_estimates(name, cfg, padded, ideal):
    cfg = WhisperConfig(max_source_positions=OUT_FRAMES, **cfg)
    assert _flops(cfg) == pytest.approx(padded, rel=0.03)
    assert _flops(cfg, ideal=True) == pytest.approx(ideal, rel=0.03)


def test_padded_cost_is_fixed_and_ideal_is_cheaper():
    cfg = WhisperConfig(max_source_positions=OUT_FRAMES, d_model=64, encoder_layers=2, encoder_attention_heads=4,
                        encoder_ffn_dim=128, num_mel_bins=80)
    from sparc.compression.whisper_compute import count_flops

    with torch.device("meta"):
        enc = WhisperEncoder(cfg).eval()
    pre = PrefixEncoder(enc, 2)
    # the padded forward costs the same whatever the reference duration; the idealized one grows with it
    assert count_flops(pre, seconds=2.0) == count_flops(pre, seconds=8.0)
    assert count_flops(pre, ideal=True, seconds=2.0) < count_flops(pre, ideal=True, seconds=8.0) < count_flops(pre)


def test_profile_prefix_keys_and_units():
    from sparc.compression.whisper_compute import profile_prefix

    cfg = WhisperConfig(max_source_positions=OUT_FRAMES, d_model=64, encoder_layers=3, encoder_attention_heads=4,
                        encoder_ffn_dim=128, num_mel_bins=80)
    enc = tiny_encoder(layers=3)
    for k in (1, 3):
        r = profile_prefix(PrefixEncoder(enc, k), "cpu")
        assert r["gflops_per_audio_s"] == pytest.approx(r["flops_per_window_padded"] / 4.0 / 1e9)
        assert r["gflops_per_audio_s_30s"] == pytest.approx(r["flops_per_window_padded"] / 30.0 / 1e9)
        assert r["gflops_per_audio_s_ideal"] < r["gflops_per_audio_s"]
        assert r["window_seconds"] == 30 and r["ref_seconds"] == 4.0
    assert profile_prefix(PrefixEncoder(enc, 3), "cpu")["params"]["final_norm"] > 0
    assert profile_prefix(PrefixEncoder(enc, 2), "cpu")["params"]["final_norm"] == 0
    assert cfg.encoder_layers == 3
