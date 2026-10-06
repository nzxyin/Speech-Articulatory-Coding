"""Prosody and articulation metrics of one utterance from re-extracted features (EVALUATION.md section 4.5).

Pure numpy, no I/O. ``sys_feats`` is the re-extraction of the system audio, ``ref_feats`` the re-extraction of the
gt audio (both without CREPE dither), so pitch (channel 12) and periodicity (channel 14) are compared in the
extractor's own units. EMA and loudness are compared with the features that drove the vocoder: ``input_ema`` (cached
``feats[:, :12]``) and ``input_loud`` (cached ``loud_raw`` times the eval gain). Streams that differ in length by a
frame or two (the re-extraction of ``480 T`` samples has ``T - 1`` frames) are cut to the shortest and the number of
dropped frames is reported.
"""

import numpy as np

from sparc.vocoders.constants import EMA_NAMES, F0_CHANNEL, LOUDNESS_EPS, N_EMA, PERIODICITY_CHANNEL

WITHIN_CENTS = 50.0
FLOOR_PERCENTILE = 5.0


def prosody_columns() -> list[str]:
    """Names of the metrics returned by :func:`prosody_metrics`, in order."""
    columns = [
        "f0_rmse_cents",
        "f0_med_abs_cents",
        "f0_within50",
        "n_voiced_both",
        "vde",
        "per_mae",
        "per_rmse",
        "loud_db_rmse",
        "loud_db_bias",
    ]
    columns += [f"ema_r_{name}" for name in EMA_NAMES] + [f"ema_rmse_{name}" for name in EMA_NAMES]
    return columns + ["ema_r_mean", "ema_rmse_mean", "n_frames", "n_frames_dropped"]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two vectors; NaN when either is constant or there are fewer than two points."""
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt((x * x).sum() * (y * y).sum())
    if not denominator > 0.0:
        return float("nan")
    return float((x * y).sum() / denominator)


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def prosody_metrics(
    sys_feats: np.ndarray,
    sys_loud: np.ndarray,
    ref_feats: np.ndarray,
    input_ema: np.ndarray,
    input_loud: np.ndarray,
    ema_std: np.ndarray,
    within_cents: float = WITHIN_CENTS,
    floor_percentile: float = FLOOR_PERCENTILE,
    loud_eps: float = LOUDNESS_EPS,
) -> dict[str, float]:
    """Metrics of one utterance (column definitions in EVALUATION.md 4.5).

    ``sys_feats`` and ``ref_feats`` are ``(T, 15)`` re-extractions, ``sys_loud`` is the system's ``loud_raw (T,)``,
    ``input_ema`` is ``(T, 12)``, ``input_loud`` is ``(T,)`` and ``ema_std`` is the training std of the 12 EMA
    channels. A voiced frame has periodicity above 0 (SPARC's thresholding and loudness gate are already applied).

    - ``f0_*``: ``1200 log2(f_sys / f_ref)`` on frames voiced in both streams; ``f0_within50`` is the fraction of those
      frames with ``|error| <= within_cents``; NaN when no frame is voiced in both.
    - ``vde``: fraction of all frames whose voiced/unvoiced decision differs.
    - ``per_mae`` / ``per_rmse``: periodicity (channel 14) error over all frames, so a voicing error counts.
    - ``loud_db_*``: ``20 log10((l_sys + eps) / (l_ref + eps))`` on frames where ``l_ref`` exceeds its
      ``floor_percentile``-th percentile over the utterance; the bias is the signed mean.
    - ``ema_r_<name>``, ``ema_rmse_<name>``: Pearson correlation and RMSE (in units of the training std) over all
      frames; ``ema_r_mean`` / ``ema_rmse_mean`` average the 12 channels (NaN correlations are skipped).
    - ``n_frames`` compared frames and ``n_frames_dropped`` = longest minus shortest of the five time axes.
    """
    arrays = (sys_feats, sys_loud, ref_feats, input_ema, input_loud)
    lengths = [len(a) for a in arrays]
    n = min(lengths)
    dropped = max(lengths) - n
    ema_std = np.asarray(ema_std, dtype=np.float64)
    if sys_feats.shape[1] < PERIODICITY_CHANNEL + 1 or ref_feats.shape[1] < PERIODICITY_CHANNEL + 1:
        raise ValueError("expected re-extracted features with 15 columns")
    if input_ema.shape[1] != N_EMA or ema_std.shape != (N_EMA,):
        raise ValueError(f"expected {N_EMA} EMA channels, got {input_ema.shape[1]} and std shape {ema_std.shape}")

    sys_feats = sys_feats[:n].astype(np.float64)
    ref_feats = ref_feats[:n].astype(np.float64)
    sys_loud = sys_loud[:n].astype(np.float64)
    input_ema = input_ema[:n].astype(np.float64)
    input_loud = input_loud[:n].astype(np.float64)

    out: dict[str, float] = {}
    nan = float("nan")
    sys_per, ref_per = sys_feats[:, PERIODICITY_CHANNEL], ref_feats[:, PERIODICITY_CHANNEL]
    sys_voiced, ref_voiced = sys_per > 0.0, ref_per > 0.0
    both = sys_voiced & ref_voiced
    out["n_voiced_both"] = float(both.sum())
    if both.any():
        cents = 1200.0 * np.log2(sys_feats[both, F0_CHANNEL] / ref_feats[both, F0_CHANNEL])
        out["f0_rmse_cents"] = float(np.sqrt(np.mean(cents**2)))
        out["f0_med_abs_cents"] = float(np.median(np.abs(cents)))
        out["f0_within50"] = float(np.mean(np.abs(cents) <= within_cents))
    else:
        out["f0_rmse_cents"] = out["f0_med_abs_cents"] = out["f0_within50"] = nan
    out["vde"] = float(np.mean(sys_voiced != ref_voiced)) if n else nan
    out["per_mae"] = float(np.mean(np.abs(sys_per - ref_per))) if n else nan
    out["per_rmse"] = float(np.sqrt(np.mean((sys_per - ref_per) ** 2))) if n else nan

    if n:
        keep = input_loud > np.percentile(input_loud, floor_percentile)
    else:
        keep = np.zeros(0, dtype=bool)
    if keep.any():
        db = 20.0 * np.log10((sys_loud[keep] + loud_eps) / (input_loud[keep] + loud_eps))
        out["loud_db_rmse"] = float(np.sqrt(np.mean(db**2)))
        out["loud_db_bias"] = float(np.mean(db))
    else:
        out["loud_db_rmse"] = out["loud_db_bias"] = nan

    sys_ema = sys_feats[:, :N_EMA]
    correlations, errors = [], []
    for c, name in enumerate(EMA_NAMES):
        r = _pearson(sys_ema[:, c], input_ema[:, c])
        rmse = float(np.sqrt(np.mean((sys_ema[:, c] - input_ema[:, c]) ** 2)) / ema_std[c]) if n else nan
        out[f"ema_r_{name}"] = r
        out[f"ema_rmse_{name}"] = rmse
        correlations.append(r)
        errors.append(rmse)
    out["ema_r_mean"] = _nanmean(correlations)
    out["ema_rmse_mean"] = _nanmean(errors)
    out["n_frames"] = float(n)
    out["n_frames_dropped"] = float(dropped)
    return {name: out[name] for name in prosody_columns()}
