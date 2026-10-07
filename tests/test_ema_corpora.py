"""Tests for sparc.ema_corpora (synthetic data; corpus-dependent tests are skipped when /data is not mounted)."""
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from sparc.ema_corpora import corpora, preprocess
from sparc.ema_corpora.mview import MviewRecord, read_mview


def _write_mview(path, audio_sr=22050, ema_sr=100.08, dur=3.0, order=("TT", "TB", "TD", "UL", "LL", "JAW"), sig=None):
    n = int(dur * ema_sr)
    t = np.arange(n) / ema_sr
    elems = [{"NAME": "AUDIO", "SRATE": float(audio_sr), "SIGNAL": np.random.default_rng(0).normal(size=int(dur * audio_sr)) * 0.01}]
    for k, s in enumerate(order):
        x = sig(t, k) if sig else np.stack([t + k, -t - k, 0 * t + k], 1)
        elems.append({"NAME": s, "SRATE": float(ema_sr), "SIGNAL": x})
    dt = np.dtype([("NAME", object), ("SRATE", object), ("SIGNAL", object)])
    arr = np.empty(len(elems), dtype=dt)
    for i, e in enumerate(elems):
        arr[i] = (e["NAME"], e["SRATE"], e["SIGNAL"])
    sio.savemat(path, {path.stem: arr})


def test_read_mview_maps_sensors_by_name(tmp_path):
    p = tmp_path / "rec_a.mat"
    _write_mview(p, order=("UL", "LL", "JAW", "TD", "TB", "TT"))
    r = read_mview(p)
    assert set(r.ema) == {"TT", "TB", "TD", "UL", "LL", "JAW"}
    # sensor k in the file order gets x = t + k
    assert np.allclose(r.ema["UL"][:, 2], 0) and np.allclose(r.ema["TT"][:, 2], 5)
    assert r.audio_sr == 22050 and abs(r.ema_sr - 100.08) < 1e-9


def test_rotation_turns_jn_lip_line_vertical():
    # lip line tilted by +48 deg (UL anterior of LL), as measured for 5emo_jn
    ul = np.array([np.sin(np.radians(48)), np.cos(np.radians(48)), 0.0]) * 20
    out = preprocess.rotate_xy(ul[None], 48.0)[0]
    assert abs(out[0]) < 1e-9 and out[1] > 0 and np.isclose(np.linalg.norm(out), 20)


def test_fill_short_gaps_only_short():
    x = np.arange(30, dtype=float)[:, None].repeat(3, 1)
    x[5:8] = np.nan  # 3 samples: filled
    x[15:27] = np.nan  # 12 samples: kept
    f, bad = preprocess.fill_short_gaps(x, 10)
    assert np.allclose(f[5:8, 0], [5, 6, 7]) and not bad[5:8].any()
    assert np.isnan(f[15:27]).all() and bad[15:27].all()


def test_process_segment_alignment_and_lag(tmp_path, monkeypatch):
    """A step in tongue-tip height at t = 1.5 s (EMA clock) must appear at 1.5 + lag s on the audio clock."""
    def sig(t, k):
        return np.stack([0 * t, np.where(t >= 1.5, 10.0, 0.0), 0 * t], 1)

    p = tmp_path / "rec_b.mat"
    _write_mview(p, sig=sig, dur=3.0)
    monkeypatch.setitem(preprocess.LAG_S, "usc_TEST", 0.04)
    seg = corpora.Segment(corpus="usc_timit", speaker="usc_TEST", utt_id="u1", mat=p, start=0.5, end=2.5,
                          text="", norm_group="g")
    row, med = preprocess.process_segment(seg, tmp_path)
    z = np.load(tmp_path / row["ema"])
    tt_y = z["ema"][:, 5]  # TT_y
    assert z["ema"].shape == (100, 12) and z["valid"].all()
    t = 0.5 + np.arange(100) / 50 + preprocess.FRAME_CENTER
    # the low-passed step crosses half height at the step time (first native sample >= 1.5 s, 1.504 s for
    # 100.08 Hz, minus half a sample for the symmetric filter's crossing): interpolate between frames
    i = int(np.flatnonzero(tt_y >= 5.0)[0])
    half = t[i - 1] + (5.0 - tt_y[i - 1]) / (tt_y[i] - tt_y[i - 1]) * (t[i] - t[i - 1])
    assert abs(half - (1.504 + 0.04)) < 0.006
    # without the lag the step lands 40 ms earlier
    monkeypatch.setitem(preprocess.LAG_S, "usc_TEST", 0.0)
    row0, _ = preprocess.process_segment(seg, tmp_path)
    y0 = np.load(tmp_path / row0["ema"])["ema"][:, 5]
    j = int(np.flatnonzero(y0 >= 5.0)[0])
    half0 = t[j - 1] + (5.0 - y0[j - 1]) / (y0[j] - y0[j - 1]) * (t[j] - t[j - 1])
    assert abs((half - half0) - 0.04) < 0.003


def test_spans_from_trans_pads_into_silence_only():
    rows = [(0.0, 0.5, "sil", "", None), (0.5, 1.0, "dh", "this", 1), (1.0, 1.4, "s", "this", 1),
            (1.4, 1.6, "sil", "", None), (1.6, 2.0, "b", "be", 2), (2.0, 3.0, "sil", "", None)]
    sp = corpora.spans_from_trans(rows, 3.0, pad=0.2)
    assert np.allclose(sp[1], (0.3, 1.5)) and np.allclose(sp[2], (1.5, 2.2))


def test_locate_exact_excerpt():
    rng = np.random.default_rng(1)
    full = rng.normal(size=50000)
    assert corpora.locate(full[12345:20000], full) == 12345
    assert corpora.locate(rng.normal(size=5000), full) is None
    # an excerpt starting in (near-)silence: the probe must not rely on its first samples
    full2 = np.concatenate([np.zeros(8000), rng.normal(size=20000), np.zeros(5000), rng.normal(size=20000)])
    assert corpora.locate(full2[2000:30000], full2) == 2000


def test_usc_split_text_disjoint():
    assert {preprocess.usc_split(i) for i in (10, 20, 460)} == {"test"}
    assert preprocess.usc_split(15) == "valid" and preprocess.usc_split(13) == "train"


needs_data = pytest.mark.skipif(not corpora.USC_ROOT.exists(), reason="corpora not mounted")


@needs_data
def test_real_files_start_aligned_and_named():
    r = read_mview(corpora.USC_ROOT / "M1" / "mat" / "usctimit_ema_m1_001_005.mat")
    assert abs(r.audio_dur - r.ema_dur) < 0.01
    assert corpora.EMO_NAME.match("ema_5emo_jn_neu_norm_sent3_rep1_utt014.mat")["rep"] == "1"
