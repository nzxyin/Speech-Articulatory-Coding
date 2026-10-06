"""Signal-level metrics: PESQ-WB, mel-cepstral distortion, multi-resolution STFT distance and mel L1.

Implements EVALUATION.md section 4.1. All metrics compare a system output with the reference ("gt") of the same
utterance; both are time-aligned, so no alignment is applied. The reference is 24 kHz; a 16 kHz system is upsampled
to 24 kHz for the 24 kHz metrics and used as it is for PESQ, and the mel band above its Nyquist frequency is reported
as NaN. Every setting comes from ``cfg.eval_metrics.signal``.
"""

import functools
import logging
import math

import librosa
import numpy as np
import pesq
import pysptk
import torch
from omegaconf import DictConfig, OmegaConf

from sparc.vocoders.losses.losses import MelSpectrogramLoss

logger = logging.getLogger(__name__)

SYSTEM_RATES = (24000, 16000)


def resample(x: np.ndarray, orig_sr: int, target_sr: int, res_type: str = "soxr_hq") -> np.ndarray:
    """Resamples a mono float32 signal; returns ``x`` unchanged when the rates are equal."""
    if orig_sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    return librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)


def pesq_wb(ref16: np.ndarray, deg16: np.ndarray, cfg: DictConfig) -> float:
    """Wide-band PESQ of two 16 kHz signals; NaN when PESQ cannot score the pair (for example no utterance found)."""
    if not (np.isfinite(ref16).all() and np.isfinite(deg16).all()):
        logger.warning("PESQ skipped: non-finite samples")
        return math.nan
    if not (np.abs(ref16).max(initial=0.0) > 0.0 and np.abs(deg16).max(initial=0.0) > 0.0):
        logger.debug("PESQ skipped: a signal is digital silence")
        return math.nan
    try:
        return float(pesq.pesq(int(cfg.pesq.sample_rate), ref16, deg16, str(cfg.pesq.mode)))
    except (pesq.PesqError, ValueError) as err:
        logger.debug("PESQ failed: %s", err)
        return math.nan


def power_spectrogram(x: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Hann-windowed power spectrogram ``[frames, n_fft // 2 + 1]`` in float64 (reflect-padded centred frames)."""
    window = torch.hann_window(int(cfg.win_length), dtype=torch.float64)
    spec = torch.stft(
        torch.from_numpy(np.asarray(x, dtype=np.float64)),
        int(cfg.n_fft),
        hop_length=int(cfg.hop_length),
        win_length=int(cfg.win_length),
        window=window,
        center=bool(cfg.center),
        pad_mode="reflect",
        return_complex=True,
    )
    return (spec.real**2 + spec.imag**2).T.contiguous().numpy()


def mel_cepstrum(power: np.ndarray, order: int, alpha: float, floor: float) -> np.ndarray:
    """Mel-cepstrum ``[frames, order + 1]`` of a power spectrogram (SPTK convention, ``c0`` first).

    ``pysptk.sp2mc`` takes the log of the power spectrum, inverse-transforms it to the real cepstrum and warps it
    with the all-pass constant ``alpha`` (``freqt``), which is SPTK's ``mcep`` for a power-spectrum input.
    """
    return pysptk.sp2mc(np.maximum(power, floor), order, alpha)


def mcd_frames(ref_mc: np.ndarray, deg_mc: np.ndarray) -> np.ndarray:
    """Per-frame MCD in dB, ``(10 / ln 10) * sqrt(2 * sum_d (dc_d)^2)`` over ``d = 1 .. order`` (``c0`` excluded)."""
    diff = ref_mc[:, 1:] - deg_mc[:, 1:]
    return (10.0 / math.log(10.0)) * np.sqrt(2.0 * np.sum(diff**2, axis=1))


def mcd(ref24: np.ndarray, deg24: np.ndarray, cfg: DictConfig) -> float:
    """Mel-cepstral distortion in dB, averaged over the gt frames within ``energy_range_db`` of the loudest one.

    A gt frame's energy is the sum of its power spectrum; frames more than 40 dB below the loudest gt frame (pauses,
    where both spectra are noise and the cepstra of two noise floors differ by a lot) are left out. The selection
    uses the gt only, so every system is scored on the same frames. NaN when the gt is digital silence.
    """
    ref_power, deg_power = power_spectrogram(ref24, cfg), power_spectrogram(deg24, cfg)
    energy_db = 10.0 * np.log10(np.maximum(ref_power.sum(axis=1), float(cfg.energy_floor)))
    if energy_db.max() <= 10.0 * math.log10(float(cfg.energy_floor)):
        return math.nan
    keep = energy_db >= energy_db.max() - float(cfg.energy_range_db)
    ref_mc = mel_cepstrum(ref_power[keep], int(cfg.order), float(cfg.alpha), float(cfg.power_floor))
    deg_mc = mel_cepstrum(deg_power[keep], int(cfg.order), float(cfg.alpha), float(cfg.power_floor))
    return float(mcd_frames(ref_mc, deg_mc).mean())


def mrstft(ref24: np.ndarray, deg24: np.ndarray, cfg: DictConfig) -> float:
    """Multi-resolution STFT distance (auraloss style): per resolution, spectral convergence plus mean log-magnitude L1.

    ``SC = ||Y - X||_F / ||Y||_F`` and ``L1 = mean |log max(|X|, eps) - log max(|Y|, eps)|`` with ``X`` the system and
    ``Y`` the reference; the result is the mean over the resolutions of ``cfg.resolutions`` (n_fft, hop, window).
    Spectra are computed in float32 with Hann windows and reflect-padded centred frames.
    """
    x, y = torch.from_numpy(np.asarray(deg24, dtype=np.float32)), torch.from_numpy(np.asarray(ref24, dtype=np.float32))
    eps = float(cfg.eps)
    scores = []
    for n_fft, hop, win in cfg.resolutions:
        window = torch.hann_window(int(win))
        kwargs = dict(
            n_fft=int(n_fft), hop_length=int(hop), win_length=int(win), window=window, center=True, return_complex=True
        )
        x_mag, y_mag = torch.stft(x, **kwargs).abs(), torch.stft(y, **kwargs).abs()
        convergence = torch.linalg.norm(y_mag - x_mag) / torch.linalg.norm(y_mag).clamp_min(eps)
        log_distance = (torch.log(x_mag.clamp_min(eps)) - torch.log(y_mag.clamp_min(eps))).abs().mean()
        scores.append(float(convergence + log_distance))
    return float(np.mean(scores))


@functools.lru_cache(maxsize=4)
def _mel_module(params: tuple) -> MelSpectrogramLoss:
    return MelSpectrogramLoss(**dict(params))


def mel_module(cfg: DictConfig) -> MelSpectrogramLoss:
    """The training mel (``MelSpectrogramLoss``) built from ``cfg.mel``; cached per parameter set."""
    params = {k: v for k, v in OmegaConf.to_container(cfg.mel, resolve=True).items()}
    return _mel_module(tuple(sorted(params.items())))


def mel_band_masks(cfg: DictConfig) -> dict[str, np.ndarray]:
    """Boolean masks over the mel bins for each configured band, by bin centre frequency.

    The centres are the HTK mel points ``i + 1`` (of ``n_mels + 2``) spaced evenly on the mel scale between ``f_min``
    and ``f_max``, the same points ``torchaudio.functional.melscale_fbanks`` builds its triangles on.
    """
    mel = cfg.mel
    if str(mel.mel_scale) != "htk":
        raise ValueError(f"band centres are implemented for the htk mel scale, got {mel.mel_scale!r}")

    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    points = np.linspace(hz_to_mel(float(mel.f_min)), hz_to_mel(float(mel.f_max)), int(mel.n_mels) + 2)
    centres = 700.0 * (10.0 ** (points[1:-1] / 2595.0) - 1.0)
    masks = {}
    for name, (low, high) in cfg.mel_bands_hz.items():
        upper = centres <= high if float(high) >= float(mel.f_max) else centres < high
        masks[str(name)] = (centres >= float(low)) & upper
    return masks


def mel_l1(ref24: np.ndarray, deg24: np.ndarray, cfg: DictConfig, deg_sr: int = 24000) -> dict[str, float]:
    """Mean absolute difference of the training log-mel spectrograms, overall and per frequency band.

    ``mel_l1`` is exactly the training mel L1 (``MelSpectrogramLoss``) of the pair. Band columns average the same
    absolute differences over the bins of the band; a band that starts at or above the Nyquist frequency of the
    system (``deg_sr / 2``) is NaN.
    """
    module = mel_module(cfg)
    with torch.inference_mode():
        ref_mel = module.log_mel(torch.from_numpy(np.asarray(ref24, dtype=np.float32))[None])
        deg_mel = module.log_mel(torch.from_numpy(np.asarray(deg24, dtype=np.float32))[None])
        diff = (deg_mel - ref_mel).abs()[0]  # [n_mels, frames]
        out = {"mel_l1": float(diff.mean(dtype=torch.float64))}
        for name, mask in mel_band_masks(cfg).items():
            low = float(cfg.mel_bands_hz[name][0])
            if low >= deg_sr / 2 or not mask.any():
                out[f"mel_l1_{name}"] = math.nan
            else:
                out[f"mel_l1_{name}"] = float(diff[torch.from_numpy(mask)].mean(dtype=torch.float64))
    return out


def signal_metrics(ref24: np.ndarray, deg: np.ndarray, deg_sr: int, cfg: DictConfig) -> dict[str, float]:
    """All signal metrics of one utterance.

    ``ref24`` is the gt, mono float32 at 24 kHz. ``deg`` is the system output at ``deg_sr`` (24000 or 16000) with the
    same duration (``480 T`` samples at 24 kHz, ``320 T`` at 16 kHz); a different length raises ``ValueError``.
    ``cfg`` is ``cfg.eval_metrics.signal``. Returns ``pesq_wb, mcd, mrstft, mel_l1, mel_l1_0_4k, mel_l1_4_8k,
    mel_l1_8_12k``. PESQ failures give NaN for ``pesq_wb``; the 8-12 kHz mel band is NaN for a 16 kHz system.
    """
    rate = int(cfg.sample_rate)
    if deg_sr not in SYSTEM_RATES:
        raise ValueError(f"deg_sr must be one of {SYSTEM_RATES}, got {deg_sr}")
    ref24 = np.asarray(ref24, dtype=np.float32)
    deg = np.asarray(deg, dtype=np.float32)
    if ref24.ndim != 1 or deg.ndim != 1:
        raise ValueError(f"expected mono signals, got shapes {ref24.shape} and {deg.shape}")
    expected = round(len(ref24) * deg_sr / rate)
    if len(deg) != expected:
        raise ValueError(f"degraded signal has {len(deg)} samples at {deg_sr} Hz, expected {expected} to match the gt")

    res_type = str(cfg.resample_type)
    pesq_rate = int(cfg.pesq.sample_rate)
    ref_pesq = resample(ref24, rate, pesq_rate, res_type)
    deg_pesq = resample(deg, deg_sr, pesq_rate, res_type)
    deg24 = resample(deg, deg_sr, rate, res_type)
    if len(deg24) != len(ref24):  # a resampler may round the length of a signal that is not a whole number of frames
        deg24 = np.pad(deg24[: len(ref24)], (0, max(0, len(ref24) - len(deg24))))

    out = {"pesq_wb": pesq_wb(ref_pesq, deg_pesq, cfg)}
    out["mcd"] = mcd(ref24, deg24, cfg.mcd)
    out["mrstft"] = mrstft(ref24, deg24, cfg.mrstft)
    out.update(mel_l1(ref24, deg24, cfg, deg_sr))
    return out
