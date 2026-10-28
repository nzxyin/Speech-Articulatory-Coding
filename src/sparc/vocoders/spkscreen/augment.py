"""Channel augmentation of 24 kHz waveforms for the out-of-domain encoder screen (docs/spkemb_ood_screen.md, variant 6).

One of four channel effects is drawn per utterance from its seed (equal probability), then a random gain in
[-12, +6] dB. The output has exactly the input's length, so frame counts and the clean utterance's targets stay valid.

- ``noise``: coloured noise (white noise shaped by a spectral tilt drawn from [-6, 0] dB/octave) at an SNR drawn from
  [5, 25] dB;
- ``reverb``: a synthetic room impulse response (direct path plus exponentially decaying Gaussian noise tail, RT60 drawn
  from [0.2, 0.9] s, direct-to-reverberant ratio drawn from [-2, 10] dB);
- ``bandlimit``: band-limiting through a lower sampling rate (6 kHz, 8 kHz or 12 kHz out of the 24 kHz input, soxr);
- ``telephone``: resampling to 8 kHz plus mu-law companding at 8 bits and a 300 Hz high-pass.
"""

import numpy as np

KINDS = ("noise", "reverb", "bandlimit", "telephone")


def _resample(x: np.ndarray, orig: int, target: int) -> np.ndarray:
    import librosa

    return librosa.resample(x, orig_sr=orig, target_sr=target, res_type="soxr_hq")


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.concatenate([x, np.zeros(n - len(x), x.dtype)])


def _noise(x, rng):
    n = len(x)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.fft.rfftfreq(n, 1 / 24000), 50.0)
    tilt = rng.uniform(-6.0, 0.0)  # dB per octave
    spec *= (f / 1000.0) ** (tilt / 6.02)
    noise = np.fft.irfft(spec, n)
    snr = rng.uniform(5.0, 25.0)
    p_sig, p_noise = float((x**2).mean()) + 1e-12, float((noise**2).mean()) + 1e-12
    return x + noise * np.sqrt(p_sig / p_noise / 10 ** (snr / 10))


def _reverb(x, rng):
    from scipy.signal import fftconvolve

    rt60 = rng.uniform(0.2, 0.9)
    drr = rng.uniform(-2.0, 10.0)
    n = int(rt60 * 24000)
    t = np.arange(n) / 24000
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail *= 10 ** (-drr / 20) / (np.sqrt((tail**2).sum()) + 1e-12)
    rir = np.concatenate([[1.0], tail])
    y = fftconvolve(x, rir)[: len(x)]
    return y * (np.sqrt((x**2).mean()) / (np.sqrt((y**2).mean()) + 1e-12))


def _bandlimit(x, rng):
    sr = int(rng.choice([6000, 8000, 12000]))
    return _fit(_resample(_resample(x, 24000, sr), sr, 24000), len(x))


def _telephone(x, rng):
    from scipy.signal import butter, sosfilt

    y = _resample(x, 24000, 8000)
    peak = np.abs(y).max() + 1e-9
    y = y / peak
    mu = 255.0
    comp = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    comp = np.round((comp + 1) * 127.5) / 127.5 - 1  # 8-bit
    y = np.sign(comp) * ((1 + mu) ** np.abs(comp) - 1) / mu * peak
    y = sosfilt(butter(2, 300, "highpass", fs=8000, output="sos"), y)
    return _fit(_resample(y, 8000, 24000), len(x))


_FN = {"noise": _noise, "reverb": _reverb, "bandlimit": _bandlimit, "telephone": _telephone}


def augment(wav24: np.ndarray, seed: int) -> np.ndarray:
    """One augmented copy of ``wav24`` (float32, same length); deterministic given ``seed``."""
    rng = np.random.default_rng(seed)
    kind = KINDS[int(rng.integers(len(KINDS)))]
    y = _FN[kind](np.asarray(wav24, np.float64), rng) * 10 ** (rng.uniform(-12.0, 6.0) / 20)
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * (0.99 / peak)
    return np.ascontiguousarray(y, dtype=np.float32)
