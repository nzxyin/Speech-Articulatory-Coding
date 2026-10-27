"""CPU tests of the Phase 5 quality scorer: signal measures on synthetic signals, chunked resumable scoring, report."""

import numpy as np
import pandas as pd
import soundfile as sf

from sparc.vocoders.features import quality as q


def _noise(seconds: float, sr: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(seconds * sr)).astype(np.float32) * 0.1


def _lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(len(x), 1 / sr) > cutoff] = 0
    return np.fft.irfft(spec, len(x)).astype(np.float32)


def test_bandwidth_of_tone_and_band_limited_noise():
    sr = 24000
    t = np.arange(sr * 2) / sr
    bw99, _ = q.spectrum_measures(0.5 * np.sin(2 * np.pi * 1000 * t), sr)
    assert 900 <= bw99 <= 1100
    for cutoff in (4000, 8000):
        x = _lowpass(_noise(4, sr), sr, cutoff)
        bw99, floor = q.spectrum_measures(x, sr)
        assert 0.95 * cutoff <= bw99 <= cutoff + 100
        assert abs(floor - cutoff) < 300  # the 60 dB floor sees the cut
    full = q.spectrum_measures(_noise(4, sr), sr)
    assert full[0] > 11000 and full[1] > 11500  # white noise fills the band


def test_bandwidth_uses_native_rate():
    sr = 48000
    bw99, _ = q.spectrum_measures(_lowpass(_noise(4, sr), sr, 20000), sr)
    assert 19000 < bw99 <= 20100


def test_clipping_fraction_and_peak():
    x = np.sin(2 * np.pi * 100 * np.arange(24000) / 24000)
    assert q.clipping(0.5 * x) == (0.0, max(abs(0.5 * x)))
    frac, peak = q.clipping(np.clip(3 * x, -1, 1))
    assert 0.7 < frac < 0.85 and peak == 1.0
    frac, _ = q.clipping(np.array([0.9985, -0.9995, 0.0, 1.0]))
    assert frac == 0.5


def test_wada_table_matches_the_published_one():
    db, g = q.wada_table()
    assert abs(g[0] - 0.40975) < 2e-3  # -20 dB
    assert abs(g[20] - 0.46221) < 5e-3  # 0 dB
    assert abs(g[30] - 0.66751) < 5e-3  # 10 dB
    assert abs(g[40] - 0.95655) < 5e-3  # 20 dB
    assert abs(g[-1] - 1.645) < 2e-2  # noise-free speech
    assert (np.diff(g) >= 0).all()


def test_wada_snr_on_the_model_signals():
    rng = np.random.default_rng(1)
    n = 600_000
    s = rng.gamma(0.4, 1.0, n) * rng.choice([-1.0, 1.0], n)
    noise = rng.standard_normal(n)
    for snr in (0, 10, 20, 30):
        z = s + noise * np.sqrt(np.mean(s**2) / 10 ** (snr / 10))
        assert abs(q.wada_snr(z) - snr) < 2.5, snr
    assert q.wada_snr(s) > 60
    assert q.wada_snr(noise) < -10  # Gaussian noise alone is at the floor of the scale
    assert np.isnan(q.wada_snr(np.zeros(100)))


def _write_list(tmp_path, n=7):
    rows = []
    for i in range(n):
        sr = 24000 if i % 2 == 0 else 16000
        path = tmp_path / f"u{i}.wav"
        sf.write(path, _lowpass(_noise(1.0 + 0.1 * i, sr, i), sr, 5000), sr)
        rows.append({"id": f"u{i}", "audio": str(path), "native_path": "", "corpus": "toy", "accent": "A" if i < 4 else "B",
                     "speaker": "s", "text": "", "do_wer": False})  # fmt: skip
    rows.append({"id": "bad", "audio": str(tmp_path / "missing.wav"), "native_path": "", "corpus": "toy", "accent": "B",
                 "speaker": "s", "text": "", "do_wer": False})  # fmt: skip
    path = tmp_path / "toy" / "list.parquet"
    path.parent.mkdir()
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_run_is_chunked_atomic_resumable_and_the_report_builds(tmp_path):
    path = _write_list(tmp_path)
    args = ["run", "--corpus", "toy", "--list", str(path), "--out", str(tmp_path / "toy"), "--chunk", "3", "--no-models", "--workers", "2"]
    # shard 1 of 2 owns chunk 1 only
    assert q.main([*args, "--shard", "1", "--nshards", "2"]) == 0
    files = sorted((tmp_path / "toy" / "scores").glob("chunk-*.parquet"))
    assert [f.name for f in files] == ["chunk-00001.parquet"]
    stamp = files[0].stat().st_mtime_ns
    assert q.main([*args, "--shard", "1", "--nshards", "2"]) == 0
    assert files[0].stat().st_mtime_ns == stamp  # finished chunks are not recomputed
    assert q.main([*args, "--shard", "0", "--nshards", "2"]) == 0
    assert not list((tmp_path / "toy" / "scores").glob("*.tmp*"))
    df = q.load_scores(tmp_path / "toy")
    assert len(df) == 8 and df["id"].is_unique
    bad = df[df["id"] == "bad"].iloc[0]
    assert bad["error"].startswith("load:") and np.isnan(bad["bw99_hz"])
    good = df[df["id"] == "u1"].iloc[0]
    assert good["native_sr"] == 16000 and good["upsampled"] and good["native_unavailable"]
    assert abs(good["bw99_hz"] - 5000) < 400 and good["clip_frac"] == 0
    assert q.main(["report", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").is_file()


def test_report_retention_and_accent_flag():
    rng = np.random.default_rng(0)
    n = 400
    anchor = pd.DataFrame({m: rng.normal(size=n) for m in q.MEASURES})
    anchor["clip_frac"] = 0.0
    anchor["duration_s"] = 3600.0 / n
    th = q.anchor_thresholds(anchor, q.MEASURES)
    assert th["utmos"][5] < th["utmos"][10] < th["utmos"][25]
    assert th["clip_frac"][5] == 0.0
    r = q.retention(anchor, th, q.MEASURES, 10)
    assert abs(r["utmos"]["frac"] - 0.9) < 0.02 and abs(r["hours"] - 1.0) < 1e-9
    assert r["joint"]["frac"] < r["utmos"]["frac"]


def test_load_list_fails_loudly_without_audio_column(tmp_path):
    import pytest

    path = tmp_path / "list.parquet"
    pd.DataFrame({"id": ["a"], "wav48": ["x.wav"], "trim_start": [0], "trim_end": [10]}).to_parquet(path)
    with pytest.raises(SystemExit, match="audio"):
        q.load_list(path)


def test_load_row_applies_native_and_audio_spans(tmp_path):
    sr = 24000
    x = np.concatenate([np.zeros(sr, np.float32), _noise(2, sr), np.zeros(sr, np.float32)])
    sf.write(tmp_path / "a.wav", x, sr)
    sf.write(tmp_path / "n.wav", x, sr)
    base = {"id": "u", "corpus": "c", "accent": "", "speaker": "s", "text": "", "do_wer": False,
            "audio": str(tmp_path / "a.wav"), "native_path": str(tmp_path / "n.wav")}  # fmt: skip
    whole = q.load_row({**base, "start_s": np.nan, "end_s": np.nan, "audio_start_s": np.nan, "audio_end_s": np.nan})
    cut = q.load_row({**base, "start_s": 1.0, "end_s": 3.0, "audio_start_s": 1.0, "audio_end_s": 3.0})
    assert whole["error"] == "" and cut["error"] == ""
    assert abs(whole["duration_s"] - 4.0) < 1e-3 and abs(cut["duration_s"] - 2.0) < 1e-3


def test_retention_includes_wer_when_all_rows_have_it_and_samples_otherwise():
    n = 100
    df = pd.DataFrame({"duration_s": 3600.0 / 50, "wada_snr": np.arange(n, dtype=float), "has_wer": True,
                       "wer": np.where(np.arange(n) < 20, 0.5, 0.0), "cer": 0.0})  # fmt: skip
    th = {"wada_snr": {10: 10.0}, "wer": {10: 0.1}, "cer": {10: 0.1}}
    r = q.retention(df, th, {"wada_snr": True}, 10)
    assert r["wer"]["kept"] == 80 and r["cer"]["kept"] == 100
    assert r["joint"]["includes_wer"] and r["joint"]["kept"] == 80  # snr >= 10 and wer <= 0.1 (rows 20..99)
    part = df.copy()
    part["has_wer"] = np.arange(n) % 2 == 0
    r = q.retention(part, th, {"wada_snr": True}, 10)
    assert r["wer"]["n_scored"] == 50 and not r["joint"]["includes_wer"] and r["joint"]["kept"] == 90
    assert r["joint_with_wer_sample"]["n_scored"] == 50 and r["joint_with_wer_sample"]["kept"] == 40


def _scores(n: int, accent: str, utmos, pesq, **over) -> pd.DataFrame:
    base = {"id": [f"{accent}{i}" for i in range(n)], "corpus": "globe_train", "accent": accent, "speaker": "",
            "duration_s": 3600.0 / n, "utmos": utmos, "squim_pesq": pesq, "squim_si_sdr": 20.0, "squim_stoi": 0.99,
            "bw_floor_hz": 11000.0, "clip_frac": 0.0, "has_wer": False, "cer": np.nan, "error": ""}  # fmt: skip
    base.update(over)
    return pd.DataFrame(base)


def test_select_thresholds_cer_and_scoring_errors():
    d = _scores(6, "A", [3.5, 2.9, 3.5, 3.5, 3.5, 3.5], [3.0, 3.0, 2.4, 3.0, 3.0, 3.0],
                squim_stoi=[0.99, 0.99, 0.99, 0.94, 0.99, 0.99], clip_frac=[0, 0, 0, 0, 2e-4, 0],
                has_wer=[True, True, True, True, True, False], cer=[0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    d.loc[5, "error"] = "boom"
    d.loc[0, "cer"] = 0.11
    out = q.select_rows(d, floor=False)
    assert out["keep"].tolist() == [False] * 6  # row0 cer, row1 utmos, row2 pesq, row3 stoi, row4 clip, row5 error
    assert out["scoring_error"].tolist() == [False] * 5 + [True]
    d.loc[0, "cer"] = 0.10
    assert q.select_rows(d, floor=False)["keep"].tolist()[0]


def test_select_floor_is_per_accent_union_and_ignores_errors():
    # accent A: 10 rows all failing utmos, the floor keeps the best ceil(0.4 * 10) = 4 by mean percentile rank
    a = _scores(10, "A", np.linspace(1.0, 2.9, 10), np.linspace(1.0, 2.0, 10))
    # accent B: 5 rows, 2 pass the thresholds; the floor adds ceil(0.4 * 5) = 2 of the best, which are the passers
    b = _scores(5, "B", [3.5, 3.4, 2.0, 1.0, 1.5], [3.0, 3.0, 1.0, 1.0, 1.0])
    # a scoring-error row with the best scores is never kept and does not take a floor slot
    c = _scores(5, "C", [9.0, 1.0, 1.1, 1.2, 1.3], [9.0, 1.0, 1.1, 1.2, 1.3])
    c.loc[0, "error"] = "x"
    out = q.select_rows(pd.concat([a, b, c], ignore_index=True), floor=True)
    by = {k: g for k, g in out.groupby("accent")}
    assert by["A"]["floor"].sum() == 4 and by["A"]["keep"].sum() == 4 and by["A"]["pass_all"].sum() == 0
    assert by["A"].loc[by["A"]["keep"], "id"].tolist() == ["A6", "A7", "A8", "A9"]
    assert by["B"]["keep"].sum() == 2 and by["B"]["pass_all"].sum() == 2
    assert not by["C"]["keep"].iloc[0] and by["C"]["floor"].sum() == 2  # ceil(0.4 * 4) of the four error-free rows
    rep = q.selection_report(out)
    assert rep["per_accent"]["A"]["kept_only_via_floor"] == 4 and rep["scoring_errors_dropped"] == 1
    assert q.select_rows(pd.concat([a, b], ignore_index=True), floor=False)["keep"].sum() == 2  # no floor: B passers only
