"""Tests of the pure helpers of sparc.vocoders.eval.periodicity_analysis (class mapping, frame masks, HNR, probes)."""

import numpy as np
import pytest

from sparc.vocoders.eval import periodicity_analysis as pa

SR = 24000


def test_phone_classes():
    expected = {
        "a": "vowels", "ɪ": "vowels", "iː": "vowels", "aj": "vowels", "əw": "vowels", "ɚ": "vowels",
        "n": "nasals", "m̩": "nasals", "ŋ": "nasals",
        "p": "stops", "kʰ": "stops", "dʲ": "stops", "ɡ": "stops", "ʔ": "stops",
        "tʃ": "affricates", "dʒ": "affricates",
        "z": "voiced_fricatives", "ð": "voiced_fricatives", "v": "voiced_fricatives",
        "s": "voiceless_fricatives", "ʃ": "voiceless_fricatives", "f": "voiceless_fricatives", "h": "h",
        "l": "approximants", "ɫ": "approximants", "ɹ": "approximants", "j": "approximants", "w": "approximants", "ɾ": "approximants",
        "": "silence", "sil": "silence", "sp": "silence",
        "spn": None,
    }
    for phone, cls in expected.items():
        assert pa.phone_class(phone) == cls, phone


def test_phone_voicing():
    expected = {"a": "voiced", "n": "voiced", "l": "voiced", "z": "voiced", "b": "voiced", "ɡ": "voiced", "dʒ": "voiced",
                "p": "voiceless", "kʰ": "voiceless", "s": "voiceless", "θ": "voiceless", "tʃ": "voiceless",
                "": "silence", "sp": "silence", "h": None, "ʔ": None, "spn": None}
    for phone, v in expected.items():
        assert pa.phone_voicing(phone) == v, phone


def test_phone_sets_and_boundaries():
    n = 50
    # silence, voiced vowel, voiceless s, h (breaks adjacency), voiced n, spn
    phones = [(0.0, 0.2, ""), (0.2, 0.4, "a"), (0.4, 0.6, "s"), (0.6, 0.7, "h"), (0.7, 0.8, "n"), (0.8, 1.0, "spn")]
    py = np.zeros(n)
    py[10:20] = 1.0
    m, _ = pa.class_masks(phones, n, np.zeros(n), py)
    assert np.array_equal(np.flatnonzero(m["phone_voiced"]), np.r_[10:20, 35:40])
    assert np.array_equal(np.flatnonzero(m["sonorants"]), np.r_[10:20, 35:40])
    assert np.array_equal(np.flatnonzero(m["phone_voiceless"]), np.arange(20, 30))
    assert np.array_equal(np.flatnonzero(m["h"]), np.arange(30, 35))
    # boundaries: silence | a at 200 ms (frames 8..11), a | s at 400 ms (frames 18..21); s | h and h | n are not boundaries
    assert np.array_equal(np.flatnonzero(m["change_phone"]), np.r_[8:12, 18:22])
    # pYIN changes at frames 10 (onset) and 20 (offset)
    assert np.array_equal(np.flatnonzero(m["change_pyin"]), np.r_[8:12, 18:22])


def test_class_masks_stops_and_transitions():
    n = 50  # 1 s of frames
    phones = [(0.0, 0.2, ""), (0.2, 0.3, "t"), (0.3, 0.6, "a"), (0.6, 0.62, "p"), (0.62, 0.8, "s"), (0.8, 1.0, "spn")]
    voiced = np.zeros(n)
    voiced[15:30] = 1.0  # voiced from frame 15 to 29
    m, info = pa.class_masks(phones, n, voiced)
    c = (np.arange(n) + 0.5) * 0.02
    assert m["silence"][:10].all() and not m["silence"][10:].any()
    # stop 200-300 ms: burst = last 40 ms (centres >= 260 ms), closure the rest
    assert np.array_equal(np.flatnonzero(m["stop_closure"]), np.arange(10, 13))
    assert np.array_equal(np.flatnonzero(m["stop_burst"]), np.array([13, 14] + [30]))  # 620 ms stop: shorter than 60 ms, all burst
    assert m["vowels"][15:30].all() and m["voiceless_fricatives"][31:40].all()
    assert info["unknown"] == 10 and info["labelled"] == 40
    # onset at frame 15 (between frames 14 and 15), offset at 30
    assert np.array_equal(np.flatnonzero(m["voicing_onset"]), np.arange(13, 17))
    assert np.array_equal(np.flatnonzero(m["voicing_offset"]), np.arange(28, 32))
    assert m["voicing_onset_offset"].sum() == 8
    assert c[13] > 0


def test_smooth_runs_removes_flicker():
    v = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    assert np.array_equal(pa.smooth_runs(v, 3), np.array([1.0] * 8 + [0.0] * 4))
    w = np.array([0, 1, 0, 0, 0, np.nan, np.nan], dtype=float)  # a first short run is kept, NaN runs are left alone
    out = pa.smooth_runs(w, 3)
    assert out[0] == 0 and np.all(out[1:5] == 0) and np.isnan(out[5:]).all()


def test_hnr_periodic_vs_noise():
    t = np.arange(SR) / SR
    harmonic = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 10))
    noise = np.random.default_rng(0).standard_normal(SR)
    centres = pa.frame_centres(SR // pa.HOP - 2)[5:30]
    h, nz = pa.hnr_db(harmonic, centres), pa.hnr_db(noise, centres)
    assert np.nanmedian(h) > 30.0
    assert np.nanmedian(nz) < 0.0
    mixed = harmonic / harmonic.std() + noise
    assert np.nanmedian(nz) < np.nanmedian(pa.hnr_db(mixed, centres)) < np.nanmedian(h)


def test_hf_band_db_selects_band():
    t = np.arange(SR) / SR
    low, high = np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 8000 * t)
    centres = pa.frame_centres(20)[2:15]
    assert np.all(pa.hf_band_db(high, centres) - pa.hf_band_db(low, centres) > 60)


def test_probe_periodicity():
    per = np.array([0.0, 0.0, 0.9, 0.7, 0.8, 0.0, 0.0], dtype=np.float32)
    assert np.all(pa.probe_periodicity("probe_const", per, 0.437, 0.8) == np.float32(0.437))
    b = pa.probe_periodicity("probe_binary", per, 0.437, 0.8)
    assert np.allclose(b, [0, 0, 0.8, 0.8, 0.8, 0, 0])
    s = pa.probe_periodicity("probe_smooth", np.pad(per, (10, 10)), 0.437, 0.8)
    assert s.shape == per.shape[:0] + (27,) and s.max() < per.max() and s.sum() == pytest.approx(np.pad(per, (10, 10)).sum(), rel=1e-5)


def test_boot_and_paired():
    import pandas as pd

    a = pd.DataFrame({"id": list("abcd"), "speaker": ["s1", "s1", "s2", "s2"], "x": [1.0, 1.0, 3.0, 3.0]})
    b = a.assign(x=[0.0, 0.0, 2.0, 2.0])
    r = pa.paired(a, b, "x")
    assert r["diff"] == pytest.approx(1.0) and r["lo"] == pytest.approx(1.0) and r["hi"] == pytest.approx(1.0) and r["n_speakers"] == 2
