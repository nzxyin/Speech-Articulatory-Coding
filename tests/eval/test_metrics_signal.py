"""Signal metrics (PESQ, MCD, MR-STFT, mel L1) on synthetic signals and against independent references."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.signal import lfilter

import sparc.conf
from sparc.vocoders.eval.metrics import signal as S
from sparc.vocoders.losses.losses import MelSpectrogramLoss

CONF = Path(sparc.conf.__file__).parent
SR = 24000


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    return OmegaConf.load(CONF / "eval_metrics" / "default.yaml").signal


def synth_speech(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Voiced-like test signal at 24 kHz: harmonic source, moving formants, 2.5 Hz syllable envelope, weak noise floor."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    source = sum(np.cos(k * phase) / k for k in range(1, 40)) + 0.02 * rng.standard_normal(n)
    block = SR // 10
    parts = []
    for i in range(0, n, block):
        res = source[i : i + block]
        for f, bw in ((500 + 200 * np.sin(i / SR * 3), 80), (1500 + 400 * np.sin(i / SR * 2), 120), (2500, 160)):
            r, th = np.exp(-np.pi * bw / SR), 2 * np.pi * f / SR
            res = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], res)
        parts.append(res)
    env = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t - np.pi / 2)) ** 0.7
    y = np.concatenate(parts) * env
    y = y / np.abs(y).max() * 0.7 + 2e-3 * rng.standard_normal(n)  # noise floor keeps every mel bin above the clamp
    return y.astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return (x + rng.standard_normal(len(x)) * rms * 10 ** (-snr_db / 20)).astype(np.float32)


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    return synth_speech()


def test_identical_signals_are_perfect(speech, cfg):
    m = S.signal_metrics(speech, speech.copy(), SR, cfg)
    assert list(m) == ["pesq_wb", "mcd", "mrstft", "mel_l1", "mel_l1_0_4k", "mel_l1_4_8k", "mel_l1_8_12k"]
    assert m["pesq_wb"] == pytest.approx(4.64, abs=0.02)
    for key in ("mcd", "mrstft", "mel_l1", "mel_l1_0_4k", "mel_l1_4_8k", "mel_l1_8_12k"):
        assert m[key] == pytest.approx(0.0, abs=1e-6), key


def test_gain_shifts_log_mel_by_ln2_and_keeps_pesq_and_mcd(speech, cfg):
    ref = S.signal_metrics(speech, speech, SR, cfg)
    m = S.signal_metrics(speech, speech * 2.0, SR, cfg)  # +6.02 dB
    for key in ("mel_l1", "mel_l1_0_4k", "mel_l1_4_8k", "mel_l1_8_12k"):
        assert m[key] == pytest.approx(math.log(2.0), abs=2e-3), key
    assert m["pesq_wb"] == pytest.approx(ref["pesq_wb"], abs=0.05)  # PESQ aligns the levels first
    assert m["mcd"] < 0.01  # a gain only moves c0, which is excluded
    assert m["mrstft"] > 0.5


def test_metrics_grow_monotonically_with_noise(speech, cfg):
    rows = [S.signal_metrics(speech, add_noise(speech, snr), SR, cfg) for snr in (40, 30, 20, 10)]
    for key in ("mcd", "mrstft", "mel_l1", "mel_l1_0_4k", "mel_l1_4_8k", "mel_l1_8_12k"):
        values = [r[key] for r in rows]
        assert values == sorted(values) and len(set(values)) == 4, (key, values)
    pesq_values = [r["pesq_wb"] for r in rows]
    assert pesq_values == sorted(pesq_values, reverse=True)


def test_16khz_system_has_nan_only_in_the_top_band(speech, cfg):
    deg16 = S.resample(add_noise(speech, 30), SR, 16000)
    assert len(deg16) == len(speech) * 2 // 3
    m = S.signal_metrics(speech, deg16, 16000, cfg)
    assert math.isnan(m["mel_l1_8_12k"])
    for key in ("pesq_wb", "mcd", "mrstft", "mel_l1", "mel_l1_0_4k", "mel_l1_4_8k"):
        assert math.isfinite(m[key]), key
    # the missing 8-12 kHz band costs the full-band metrics a lot compared with the same signal at 24 kHz
    m24 = S.signal_metrics(speech, add_noise(speech, 30), SR, cfg)
    assert m["mel_l1"] > m24["mel_l1"] and m["mcd"] > m24["mcd"]


def test_pesq_failure_gives_nan_without_raising(speech, cfg):
    m = S.signal_metrics(speech, np.zeros_like(speech), SR, cfg)
    assert math.isnan(m["pesq_wb"])
    assert all(math.isfinite(m[k]) for k in ("mcd", "mrstft", "mel_l1"))
    short = speech[: 3 * SR // 50]  # 60 ms: shorter than PESQ can handle
    m = S.signal_metrics(short, short.copy(), SR, cfg)
    assert math.isnan(m["pesq_wb"])
    assert m["mel_l1"] == pytest.approx(0.0, abs=1e-6)


def test_length_and_rate_are_checked(speech, cfg):
    with pytest.raises(ValueError, match="expected"):
        S.signal_metrics(speech, speech[:-1], SR, cfg)
    with pytest.raises(ValueError, match="expected"):
        S.signal_metrics(speech, speech, 16000, cfg)
    with pytest.raises(ValueError, match="deg_sr"):
        S.signal_metrics(speech, speech, 22050, cfg)


def test_mel_l1_equals_the_training_mel_loss(speech, cfg):
    shared = OmegaConf.load(CONF / "loss" / "shared.yaml").mel
    for key, value in cfg.mel.items():  # the metric uses the training mel definition
        assert shared[key] == value, key
    train = MelSpectrogramLoss(**{k: v for k, v in shared.items() if k != "_target_"})
    deg = add_noise(speech, 15)
    expected = float(train(torch.from_numpy(deg)[None, None], torch.from_numpy(speech)[None, None]))
    assert S.mel_l1(speech, deg, cfg)["mel_l1"] == pytest.approx(expected, rel=1e-5)


def test_mel_bands_partition_the_bins(cfg):
    masks = S.mel_band_masks(cfg)
    assert list(masks) == ["0_4k", "4_8k", "8_12k"]
    stacked = np.stack(list(masks.values()))
    assert (stacked.sum(axis=0) == 1).all() and stacked.shape[1] == cfg.mel.n_mels
    # HTK bins are spaced evenly in mel, so the low band holds most of them
    assert [int(m.sum()) for m in masks.values()] == [66, 21, 13]


def reference_mrstft(ref: np.ndarray, deg: np.ndarray, n_fft: int, hop: int, win: int, eps: float) -> float:
    """One MR-STFT resolution with explicit framing in numpy, independent of torch.stft."""
    window = np.zeros(n_fft)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)  # periodic Hann
    window[(n_fft - win) // 2 : (n_fft - win) // 2 + win] = w

    def mag(x):
        x = np.pad(x.astype(np.float64), (n_fft // 2, n_fft // 2), mode="reflect")
        frames = np.stack([x[i : i + n_fft] for i in range(0, len(x) - n_fft + 1, hop)])
        return np.abs(np.fft.rfft(frames * window, axis=-1))

    x_mag, y_mag = mag(deg), mag(ref)
    sc = np.linalg.norm(y_mag - x_mag) / np.linalg.norm(y_mag)
    return float(sc + np.mean(np.abs(np.log(np.maximum(x_mag, eps)) - np.log(np.maximum(y_mag, eps)))))


def test_mrstft_matches_a_numpy_reference(speech, cfg):
    deg = add_noise(speech, 20)
    resolutions = [(1024, 120, 600), (512, 50, 240)]
    sub = OmegaConf.create({"resolutions": resolutions, "eps": cfg.mrstft.eps})
    expected = np.mean([reference_mrstft(speech, deg, *r, float(cfg.mrstft.eps)) for r in resolutions])
    assert S.mrstft(speech, deg, sub) == pytest.approx(expected, rel=2e-4)
    # the configured resolutions are the three of the contract
    assert [list(r) for r in cfg.mrstft.resolutions] == [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]


def freqt_numpy(c: np.ndarray, order: int, alpha: float) -> np.ndarray:
    """SPTK's frequency transformation (all-pass recursion) in numpy, vectorized over rows of ``c``."""
    c = np.atleast_2d(c)
    g = np.zeros((c.shape[0], order + 1))
    for i in range(c.shape[1] - 1, -1, -1):
        d = g.copy()
        g[:, 0] = c[:, i] + alpha * d[:, 0]
        g[:, 1] = (1.0 - alpha**2) * d[:, 0] + alpha * d[:, 1]
        for j in range(2, order + 1):
            g[:, j] = d[:, j - 1] + alpha * (d[:, j] - g[:, j - 1])
    return g


def warped_cepstrum_by_quadrature(c: np.ndarray, order: int, alpha: float, n: int = 40000) -> np.ndarray:
    """Mel-cepstrum from the definition: ``ln|H(w)| = sum_m c_a(m) cos(m * w_tilde(w))`` with the all-pass phase warp.

    With the log-magnitude ``L(w) = sum_m c(m) cos(m w)``, ``c_a(m)`` is the cosine transform of ``L`` sampled on the
    warped axis, ``w = w_tilde + 2 atan(-alpha sin(w_tilde) / (1 + alpha cos(w_tilde)))``.
    """
    w_tilde = (np.arange(n) + 0.5) * np.pi / n
    w = w_tilde + 2.0 * np.arctan(-alpha * np.sin(w_tilde) / (1.0 + alpha * np.cos(w_tilde)))
    log_mag = np.cos(np.outer(w, np.arange(len(c)))) @ c
    basis = np.cos(np.outer(w_tilde, np.arange(order + 1)))
    out = 2.0 / n * (log_mag @ basis)
    out[0] /= 2.0
    return out


def test_mel_cepstrum_matches_independent_references(cfg):
    rng = np.random.default_rng(0)
    order, alpha = int(cfg.mcd.order), float(cfg.mcd.alpha)
    # smooth log spectra from a short random cepstrum, so the truncated and the sampled definitions coincide
    c = rng.standard_normal((3, 41)) * np.exp(-np.arange(41) / 8.0)
    omega = np.pi * np.arange(513) / 512
    log_mag = np.stack([np.cos(np.outer(omega, np.arange(41))) @ row for row in c])
    power = np.exp(2.0 * log_mag)
    mc = S.mel_cepstrum(power, order, alpha, floor=1e-30)
    assert mc.shape == (3, order + 1)
    for row in range(3):
        quad = warped_cepstrum_by_quadrature(c[row], order, alpha)
        np.testing.assert_allclose(mc[row], quad, atol=2e-5)
    np.testing.assert_allclose(mc, freqt_numpy(np.fft.irfft(np.log(power), axis=-1) * np.r_[0.5, np.ones(1023)], order, alpha), atol=1e-8)


def test_mcd_matches_the_formula_on_known_cepstral_differences(cfg):
    ref_mc = np.zeros((4, 25))
    deg_mc = np.zeros((4, 25))
    deg_mc[:, 0] = 5.0  # c0 is excluded
    deg_mc[0, 3] = 0.1
    deg_mc[1, 1], deg_mc[1, 24] = 0.1, -0.2
    expected = (10.0 / math.log(10.0)) * np.sqrt(2.0 * np.array([0.01, 0.05, 0.0, 0.0]))
    np.testing.assert_allclose(S.mcd_frames(ref_mc, deg_mc), expected)


def test_mcd_leaves_out_frames_far_below_the_loudest(cfg):
    rng = np.random.default_rng(3)
    loud = (0.3 * rng.standard_normal(SR)).astype(np.float32)
    quiet = (3e-5 * rng.standard_normal(SR)).astype(np.float32)  # 80 dB below
    ref = np.concatenate([loud, quiet])
    deg = np.concatenate([loud, (3e-5 * rng.standard_normal(SR)).astype(np.float32)])  # other noise in the quiet half
    full = S.mcd(ref, deg, OmegaConf.merge(cfg.mcd, {"energy_range_db": 200.0}))
    selected = S.mcd(ref, deg, cfg.mcd)
    assert selected < 0.1 < full
    assert math.isnan(S.mcd(np.zeros(SR, dtype=np.float32), loud, cfg.mcd))
