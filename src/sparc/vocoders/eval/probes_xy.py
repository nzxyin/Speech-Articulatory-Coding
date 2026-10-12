"""Paired X-Y EMA probes: does a vocoder follow an edit of one sensor moved in two dimensions?

The single-channel probes of :mod:`sparc.vocoders.eval.probes` add a constant to one EMA channel and find a gain of only
about 0.1, although natural EMA dynamics pass through the vocoder and the re-inversion with slope about 1. This module
tests three untested explanations: single coordinates of a sensor are not meaningful alone (paired edits), the vocoder
follows articulatory change over time and not the absolute position (dynamic scaling, local bumps), and the offset also
lands on silence (speech-only edits).

Sensors are the channel pairs ``TD (0, 1)``, ``TB (2, 3)``, ``TT (4, 5)``, ``LI (6, 7)``, ``UL (8, 9)``, ``LL (10, 11)``
as (X, Y). All edit geometry is in *standardized* units: the two channels of a sensor are divided by their training std
(``stats["ema_std"]``), an edit is a time profile ``p(t)`` times a unit vector ``u`` in that 2-D plane, and the raw
change of channel ``c`` is ``ema_std[c] * p(t) * u[c]``. Directions ``u``:

- ``x``: (1, 0); ``y``: (0, 1);
- ``major``: first principal direction of the 2-D covariance of the two z-scored channels on training frames (every
  ``pca.stride``-th frame of ``pca.split``); sign fixed so that its larger component is positive;
- ``minor``: ``major`` rotated by +90 degrees.

Because every ``u`` has unit norm, constant, speech-only and bump edits of amplitude ``norm`` have the same Euclidean
norm for every direction. Profiles ``p(t)`` (``sign`` is -1 or +1):

- ``const``: ``sign * norm`` on every frame (the protocol of the single-channel probes);
- ``speech``: ``sign * norm * w(t)``, with ``w`` the speech mask (loudness above ``speech.db_below_p95`` dB below the
  95th percentile of the utterance, gaps of at most ``speech.gap_frames`` frames closed) smoothed by a Hann ramp of
  ``speech.ramp_frames`` frames;
- ``dyn``: ``alpha * (q(t) - mean(q))`` with ``q(t) = <z(t), u>`` the component of the input along ``u``, so the
  component along ``u`` is scaled by ``1 + alpha`` around its utterance mean and the mean is kept exactly
  (``alpha`` in ``dyn_alphas``; the orthogonal component and all other channels are untouched);
- ``bump``: ``sign * norm`` times a Hann window of ``bump.length_ms`` centred on the speech frames at the fractions
  ``bump.positions`` of the speech frames (three bumps in one edit).

Measurement. ``dt(t)`` is the edited minus the unedited input EMA and ``dm(t)`` the re-inverted EMA of the edited minus
that of the unedited synthesis (same speaker vector and synthesis RNG), both divided by ``ema_std`` and cut to the common
number of frames. With ``a, b`` the channels of the sensor:

- ``gain = sum_t <dm, dt> / sum_t |dt|^2`` (over the 12 channels; ``dt`` is zero outside the sensor), 1 means followed;
- ``gain_perp = sum_t (dt_x dm_y - dt_y dm_x) / sum_t |dt|^2`` (component rotated by +90 degrees);
- ``angle_deg = atan2(gain_perp, gain)``, the mean rotation of the measured 2-D displacement against the target;
- ``leakage = sqrt(sum_{t, c not in (a, b)} dm_c^2 / 10) / sqrt(sum_t |dt|^2)``: RMS over frames and the 10 channels of
  the other sensors, divided by the RMS of the target (both over all frames);
- ``own_rms = sqrt(sum_t (dm_a^2 + dm_b^2)) / sqrt(sum_t |dt|^2)``: total response in the sensor's own two channels.

Every row also stores the sufficient moments ``[N, sxx, sxy, syy, C (12 x 2, row-major), Q (12)]`` (``S = sum dt dt^T``,
``C = sum dm dt^T``, ``Q_c = sum dm_c^2``) separately for speech and non-speech frames (``mom_speech``,
``mom_other``), so that pooled gains, bootstrap intervals and the prediction of a linear regression ``B`` fitted on
natural dynamics (``dm_pred = D^-1 B D dt`` with ``D = diag(ema_std)``) follow from sums (:func:`metrics_from_moments`).

Resumable like the other probes: ``<out_dir>/edits/<edit>.parquet`` per edit (atomic), ``results.parquet`` when all
exist, ``stop()`` is checked between edits.
"""

import fnmatch
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from sparc.vocoders.constants import EMA_NAMES, LOUDNESS_CHANNEL, N_EMA, SAMPLE_RATE

SENSORS: dict[str, tuple[int, int]] = {
    "TD": (0, 1),
    "TB": (2, 3),
    "TT": (4, 5),
    "LI": (6, 7),
    "UL": (8, 9),
    "LL": (10, 11),
}
for _name, (_x, _y) in SENSORS.items():  # the channel order is part of the contract
    assert EMA_NAMES[_x] == _name + "X" and EMA_NAMES[_y] == _name + "Y", (_name, EMA_NAMES)

DIRECTIONS = ("x", "y", "major", "minor")
PROFILES = ("const", "speech", "dyn", "bump")
HOP_SECONDS = 0.02  # 50 Hz feature frames
MOMENT_LEN = 4 + 2 * N_EMA + N_EMA  # N, sxx, sxy, syy, C (12 x 2), Q (12)


# ------------------------------------------------------------------------------------------------------ directions


def sensor_directions(ema: np.ndarray, ema_mean: np.ndarray, ema_std: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Unit vectors ``x, y, major, minor`` (each shape ``(2,)``) of every sensor, from training EMA frames ``ema (n, 12)``.

    The frames are z-scored with ``ema_mean`` and ``ema_std``; ``major`` is the leading eigenvector of the 2 x 2
    covariance of the sensor's two z-scored channels (sign: larger component positive, first component if equal) and
    ``minor`` is ``major`` rotated by +90 degrees.
    """
    z = (np.asarray(ema, dtype=np.float64)[:, :N_EMA] - np.asarray(ema_mean)) / np.asarray(ema_std)
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, (cx, cy) in SENSORS.items():
        cov = np.cov(z[:, [cx, cy]], rowvar=False)
        values, vectors = np.linalg.eigh(cov)
        major = vectors[:, int(np.argmax(values))]
        if abs(major[0]) >= abs(major[1]):
            major = major if major[0] > 0 else -major
        else:
            major = major if major[1] > 0 else -major
        out[name] = {
            "x": np.array([1.0, 0.0]),
            "y": np.array([0.0, 1.0]),
            "major": major,
            "minor": np.array([-major[1], major[0]]),
        }
    return out


def sensor_variances(ema: np.ndarray, ema_mean: np.ndarray, ema_std: np.ndarray) -> dict[str, list[float]]:
    """``[major, minor]`` variance (z-scored units) of every sensor's two channels, the eigenvalues behind :func:`sensor_directions`."""
    z = (np.asarray(ema, dtype=np.float64)[:, :N_EMA] - np.asarray(ema_mean)) / np.asarray(ema_std)
    return {
        name: [float(v) for v in np.linalg.eigvalsh(np.cov(z[:, [cx, cy]], rowvar=False))[::-1]]
        for name, (cx, cy) in SENSORS.items()
    }


def directions_to_json(directions: dict[str, dict[str, np.ndarray]]) -> dict:
    return {s: {d: [float(v) for v in vec] for d, vec in dirs.items()} for s, dirs in directions.items()}


def directions_from_json(data: dict) -> dict[str, dict[str, np.ndarray]]:
    return {s: {d: np.asarray(vec, dtype=np.float64) for d, vec in dirs.items()} for s, dirs in data.items()}


# ------------------------------------------------------------------------------------------------------ masks, bumps


def speech_mask(loudness: np.ndarray, db_below_p95: float = 20.0, gap_frames: int = 5) -> np.ndarray:
    """Boolean speech mask ``[T]``: loudness above ``db_below_p95`` dB under its 95th percentile, short gaps closed."""
    loud = np.asarray(loudness, dtype=np.float64)
    if loud.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = np.percentile(loud, 95.0) * 10.0 ** (-float(db_below_p95) / 20.0)
    mask = loud > threshold
    # close gaps of at most ``gap_frames`` False frames that lie between True frames
    idx = np.flatnonzero(mask)
    if idx.size >= 2:
        gaps = np.diff(idx)
        for left, gap in zip(idx[:-1], gaps):
            if 1 < gap <= gap_frames + 1:
                mask[left + 1 : left + gap] = True
    return mask


def speech_weight(mask: np.ndarray, ramp_frames: int = 5) -> np.ndarray:
    """Float weights in [0, 1]: ``mask`` smoothed by a normalized Hann window of ``ramp_frames`` frames (odd, at least 1)."""
    ramp = max(1, int(ramp_frames))
    ramp += 1 - ramp % 2  # odd
    if ramp == 1:
        return mask.astype(np.float64)
    window = np.hanning(ramp + 2)[1:-1]
    window = window / window.sum()
    return np.clip(np.convolve(mask.astype(np.float64), window, mode="same"), 0.0, 1.0)


def bump_window(length_ms: float) -> np.ndarray:
    """Hann window (odd number of frames, peak 1) of about ``length_ms`` at 50 Hz."""
    n = max(3, int(round(float(length_ms) / 1000.0 / HOP_SECONDS)))
    n += 1 - n % 2
    window = np.hanning(n + 2)[1:-1]
    return window / window.max()


def bump_centers(mask: np.ndarray, positions: Sequence[float], half: int) -> list[int]:
    """Frame indices at the fractions ``positions`` of the speech frames, kept ``half`` frames inside ``[0, T)``.

    Falls back to all frames when fewer than ``len(positions)`` speech frames exist. Short utterances (``T < 2 half + 1``)
    use the middle frame.
    """
    t = len(mask)
    frames = np.flatnonzero(mask)
    if frames.size < len(positions):
        frames = np.arange(t)
    centers = []
    for p in positions:
        c = int(frames[min(len(frames) - 1, int(np.floor(float(p) * len(frames))))])
        centers.append(c if t < 2 * half + 1 else int(np.clip(c, half, t - 1 - half)))
    return centers


# ------------------------------------------------------------------------------------------------------ edits


def _tag(sign: float) -> str:
    return "p" if sign >= 0 else "m"


def xy_edit_name(edit: dict) -> str:
    """``xy_<sensor>_<direction>_<profile>_<p|m>``, for example ``xy_TD_major_dyn_m``."""
    return f"xy_{edit['sensor']}_{edit['direction']}_{edit['profile']}_{_tag(float(edit['sign']))}"


def xy_edits(cfg: DictConfig) -> list[dict]:
    """The edits of ``cfg.eval_features.probes_xy`` in a fixed order (sensor, direction, profile, sign), each with a ``name``.

    An edit is ``{"type": "xy", "sensor", "direction", "profile", "sign", "amp"}``: ``amp`` is ``norm`` for the constant,
    speech-only and bump profiles and the signed ``alpha`` for ``dyn`` (``sign`` is the sign of ``amp``).
    """
    spec = cfg.eval_features.probes_xy
    edits: list[dict] = []
    for sensor in spec.sensors:
        for direction in spec.directions:
            for profile in spec.profiles:
                if profile not in PROFILES:
                    raise ValueError(f"unknown profile {profile!r}")
                amps = [float(a) for a in spec.dyn_alphas] if profile == "dyn" else [float(s) * float(spec.norm) for s in spec.signs]
                for amp in amps:
                    edits.append(
                        {
                            "type": "xy",
                            "sensor": str(sensor),
                            "direction": str(direction),
                            "profile": str(profile),
                            "sign": 1.0 if amp >= 0 else -1.0,
                            "amp": amp,
                        }
                    )
    for edit in edits:
        if edit["sensor"] not in SENSORS or edit["direction"] not in DIRECTIONS:
            raise ValueError(f"bad sensor or direction in {edit}")
        edit["name"] = xy_edit_name(edit)
    names = [e["name"] for e in edits]
    if len(set(names)) != len(names):
        raise ValueError("duplicate edit names")
    only = spec.get("only", None)
    if only:
        edits = [e for e in edits if any(fnmatch.fnmatch(e["name"], pat) for pat in only)]
    return edits


def xy_profile(edit: dict, z_along: np.ndarray, loudness: np.ndarray, spec: DictConfig) -> np.ndarray:
    """Amplitude ``p(t)`` along the edit direction for one utterance (``z_along[t] = <z(t), u>``, ``loudness[t]`` linear)."""
    t = len(loudness)
    profile = edit["profile"]
    amp = float(edit["amp"])
    if profile == "const":
        return np.full(t, amp)
    if profile == "dyn":
        return amp * (z_along - z_along.mean()) if t else np.zeros(0)
    mask = speech_mask(loudness, float(spec.speech.db_below_p95), int(spec.speech.gap_frames))
    if profile == "speech":
        return amp * speech_weight(mask, int(spec.speech.ramp_frames))
    if profile == "bump":
        window = bump_window(float(spec.bump.length_ms))
        half = len(window) // 2
        out = np.zeros(t)
        for c in bump_centers(mask, spec.bump.positions, half):
            lo, hi = c - half, c + half + 1
            wlo, whi = max(0, -lo), len(window) - max(0, hi - t)
            out[max(0, lo) : min(t, hi)] += amp * window[wlo:whi]
        return out
    raise ValueError(f"unknown profile {profile!r}")


def apply_xy_edit(
    features: torch.Tensor, edit: dict, stats: dict, directions: dict[str, dict[str, np.ndarray]], spec: DictConfig
) -> torch.Tensor:
    """Edited copy of raw features ``[B, 15, T]``; only the two channels of the edit's sensor change (``features`` is untouched).

    The change of channel ``c`` is ``ema_std[c] * p(t) * u[c]`` with ``p`` from :func:`xy_profile` and ``u`` the unit vector
    ``directions[sensor][direction]`` in the z-scored sensor plane.
    """
    if features.dim() != 3 or features.shape[1] != 15:
        raise ValueError(f"expected features of shape [B, 15, T], got {tuple(features.shape)}")
    cx, cy = SENSORS[edit["sensor"]]
    std = np.asarray(stats["ema_std"], dtype=np.float64)
    u = np.asarray(directions[edit["sensor"]][edit["direction"]], dtype=np.float64)
    out = features.clone()
    for b in range(features.shape[0]):
        x = features[b, [cx, cy], :].double().cpu().numpy()  # (2, T)
        z_along = (x[0] / std[cx]) * u[0] + (x[1] / std[cy]) * u[1]
        loud = features[b, LOUDNESS_CHANNEL].double().cpu().numpy()
        p = xy_profile(edit, z_along, loud, spec)
        out[b, cx] = features[b, cx] + torch.as_tensor(std[cx] * u[0] * p, dtype=features.dtype, device=features.device)
        out[b, cy] = features[b, cy] + torch.as_tensor(std[cy] * u[1] * p, dtype=features.dtype, device=features.device)
    return out


def target_speech_mask(features: torch.Tensor, spec: DictConfig) -> np.ndarray:
    """Speech mask of the first utterance of a ``[1, 15, T]`` batch (the frames used to split the moments)."""
    loud = features[0, LOUDNESS_CHANNEL].double().cpu().numpy()
    return speech_mask(loud, float(spec.speech.db_below_p95), int(spec.speech.gap_frames))


# ------------------------------------------------------------------------------------------------------ measures


def xy_moments(dt: np.ndarray, dm: np.ndarray, cols: tuple[int, int], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moment vectors (speech frames, other frames) of length :data:`MOMENT_LEN`.

    ``dt`` is the target change of the sensor ``(n, 2)`` and ``dm`` the measured change of all 12 channels ``(n, 12)``,
    both in std units; ``mask`` marks the speech frames. ``cols`` is accepted for symmetry; the target lives in the
    sensor plane by construction.
    """
    out = []
    for sel in (mask, ~mask):
        t, m = dt[sel], dm[sel]
        s = t.T @ t
        c = m.T @ t  # (12, 2)
        q = (m**2).sum(0)
        out.append(np.concatenate([[float(sel.sum()), s[0, 0], s[0, 1], s[1, 1]], c.ravel(), q]))
    return out[0], out[1]


def unpack_moments(mom: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """``(N, S (2, 2), C (12, 2), Q (12,))`` from a moment vector."""
    mom = np.asarray(mom, dtype=np.float64)
    s = np.array([[mom[1], mom[2]], [mom[2], mom[3]]])
    return float(mom[0]), s, mom[4 : 4 + 2 * N_EMA].reshape(N_EMA, 2), mom[4 + 2 * N_EMA : 4 + 3 * N_EMA]


METRIC_NAMES = ("gain", "gain_perp", "angle_deg", "leakage", "own_rms", "target_rms")


def metrics_vec(mom: np.ndarray, cols: tuple[int, int], pred_matrix: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """:func:`metrics_from_moments` for moments of shape ``(..., MOMENT_LEN)`` (for example bootstrap resamples).

    Entries with no target energy (``sxx + syy`` not positive) are NaN.
    """
    mom = np.asarray(mom, dtype=np.float64)
    n = mom[..., 0]
    sxx, sxy, syy = mom[..., 1], mom[..., 2], mom[..., 3]
    c = mom[..., 4 : 4 + 2 * N_EMA].reshape(*mom.shape[:-1], N_EMA, 2)
    q = mom[..., 4 + 2 * N_EMA : 4 + 3 * N_EMA]
    a, b = cols
    if pred_matrix is not None:
        m = np.asarray(pred_matrix, dtype=np.float64)[:, list(cols)]  # (12, 2)
        s2 = np.stack([np.stack([sxx, sxy], -1), np.stack([sxy, syy], -1)], -2)  # (..., 2, 2)
        c = np.einsum("ck,...kj->...cj", m, s2)
        q = np.einsum("ck,...kl,cl->...c", m, s2, m)
    denom = sxx + syy
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = denom > 0
        dot = c[..., a, 0] + c[..., b, 1]
        cross = c[..., b, 0] - c[..., a, 1]
        others = [i for i in range(N_EMA) if i not in (a, b)]
        out = {
            "gain": dot / denom,
            "gain_perp": cross / denom,
            "angle_deg": np.degrees(np.arctan2(cross, dot)),
            "leakage": np.sqrt(q[..., others].sum(-1) / len(others)) / np.sqrt(denom),
            "own_rms": np.sqrt((q[..., a] + q[..., b]) / denom),
            "target_rms": np.sqrt(denom / n),
        }
    return {k: np.where(ok, v, np.nan) for k, v in out.items()}


def metrics_from_moments(mom: np.ndarray, cols: tuple[int, int], pred_matrix: np.ndarray | None = None) -> dict[str, float]:
    """Gain, perpendicular gain, angle, leakage, own RMS and target RMS from (summed) moments; see the module docstring.

    With ``pred_matrix = D^-1 B D`` (12 x 12, std units; :func:`regression_in_std_units`) the moments of the response that
    the linear regression ``B`` would give for the same target replace the measured ones (the prediction).
    """
    return {k: float(v) for k, v in metrics_vec(mom, cols, pred_matrix).items()}


def regression_in_std_units(B: np.ndarray, ema_std: np.ndarray) -> np.ndarray:
    """``D^-1 B D``: the raw-unit regression ``B`` (``y_c' = sum_c B[c', c] x_c``) in std units."""
    std = np.asarray(ema_std, dtype=np.float64)
    return np.asarray(B, dtype=np.float64) * std[None, :] / std[:, None]


def measure_xy(
    base_feats: np.ndarray, edit_feats: np.ndarray, dt_raw: np.ndarray, mask: np.ndarray, edit: dict, ema_std: np.ndarray
) -> dict:
    """Row of metrics and moments for one edit of one utterance (pure numpy).

    ``base_feats`` and ``edit_feats`` are the re-extracted features ``(T', 15)`` of the unedited and edited synthesis,
    ``dt_raw (T, 2)`` the raw change of the sensor's two input channels, ``mask (T,)`` the speech frames.
    """
    cols = SENSORS[edit["sensor"]]
    std = np.asarray(ema_std, dtype=np.float64)
    n = min(len(base_feats), len(edit_feats), len(dt_raw), len(mask))
    dm = (edit_feats[:n, :N_EMA].astype(np.float64) - base_feats[:n, :N_EMA].astype(np.float64)) / std[None, :]
    dt = dt_raw[:n].astype(np.float64) / std[list(cols)][None, :]
    sp, ot = xy_moments(dt, dm, cols, mask[:n])
    row = {"n_frames": float(n), "n_speech": float(sp[0])}
    row.update(metrics_from_moments(sp + ot, cols))
    for tag, mom in (("speech", sp), ("other", ot)):
        row[f"gain_{tag}"] = metrics_from_moments(mom, cols)["gain"]
    row["mom_speech"] = [float(v) for v in sp]
    row["mom_other"] = [float(v) for v in ot]
    return row


# ------------------------------------------------------------------------------------------------------ driver


def _atomic(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def _write_parquet(path: Path, table: pd.DataFrame) -> None:
    _atomic(path, lambda tmp: table.to_parquet(tmp, index=False))


def _write_wav16(path: Path, wav: np.ndarray) -> None:
    import soundfile as sf

    _atomic(path, lambda tmp: sf.write(str(tmp), wav, SAMPLE_RATE, format="WAV", subtype="PCM_16"))


def training_ema(cfg: DictConfig) -> np.ndarray:
    """EMA channels of every ``pca.stride``-th training frame, ``(n, 12)``, from the packed cache."""
    spec = cfg.eval_features.probes_xy
    path = Path(cfg.paths.cache_root) / "packed" / str(spec.pca.split) / "feats.npy"
    return np.asarray(np.load(path, mmap_mode="r")[:: int(spec.pca.stride), :N_EMA], dtype=np.float64)


def run_probes_xy(
    cfg: DictConfig,
    system: str,
    device: torch.device | str,
    out_dir: Path,
    stop: Callable[[], bool] = lambda: False,
    limit: int | None = None,
) -> None:
    """Runs (or resumes) the paired X-Y probes of vocoder ``system`` under ``out_dir`` (see the module docstring).

    The utterances are those of :func:`sparc.vocoders.eval.probes.probe_subset` (``limit``, or
    ``probes_xy.n_utterances``, keeps every k-th of them for smoke runs).
    """
    from sparc.vocoders.eval.probes import _atomic_write_bytes, probe_batches, probe_subset, synthesize
    from sparc.vocoders.eval.reextract import ReExtractor
    from sparc.vocoders.eval.vocoder_loader import checkpoint_info, load_vocoder
    from sparc.vocoders.models.frontend import load_stats

    out_dir = Path(out_dir)
    spec = cfg.eval_features.probes_xy
    edits = xy_edits(cfg)
    edit_dir = out_dir / str(spec.edit_dir)
    edit_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / str(spec.results_name)
    todo = [e for e in edits if not (edit_dir / f"{e['name']}.parquet").exists()]
    if not todo:
        if not results_path.exists():
            _write_parquet(results_path, _concat(edit_dir, edits))
        return
    device = torch.device(device)
    sys_spec = cfg.eval.systems[system]
    module, vcfg = load_vocoder(
        sys_spec.experiment, sys_spec.vocoder, sys_spec.get("ckpt", None), device, list(sys_spec.get("overrides", []) or [])
    )
    stats = load_stats(vcfg.stats_path)
    ema_std = np.asarray(stats["ema_std"], dtype=np.float64)
    train_ema, ema_mean = training_ema(cfg), np.asarray(stats["ema_mean"], dtype=np.float64)
    directions = sensor_directions(train_ema, ema_mean, ema_std)
    ids = probe_subset(cfg, spec.split)
    n_keep = limit if limit is not None else spec.get("n_utterances", None)
    if n_keep is not None and int(n_keep) < len(ids):
        step = max(1, len(ids) // int(n_keep))
        ids = ids[::step][: int(n_keep)]
    batches = probe_batches(vcfg, ids, device, spec.split)
    extractor = ReExtractor(cfg, spec.head, device)
    _atomic_write_bytes(
        out_dir / "probes_xy_meta.json",
        lambda tmp: tmp.write_text(
            json.dumps(
                {
                    "system": system,
                    "checkpoint": checkpoint_info(module),
                    "split": spec.split,
                    "ids": ids,
                    "edits": [e["name"] for e in edits],
                    "ema_std": [float(v) for v in ema_std],
                    "directions": directions_to_json(directions),
                    "direction_variance": sensor_variances(train_ema, ema_mean, ema_std),
                    "probes_xy": OmegaConf.to_container(spec, resolve=True),
                    "reextract": extractor.describe(),
                },
                indent=1,
                default=str,
            )
        ),
    )

    def reextract(wav: np.ndarray) -> np.ndarray:
        return extractor.extract(wav, SAMPLE_RATE)["feats"]

    examples = int(spec.example_utterances)
    base = []
    for i, batch in enumerate(batches):
        wav = synthesize(module, batch["features"], batch["spk_raw"], device)
        base.append(reextract(wav))
        if i < examples:
            _write_wav16(out_dir / "wav" / batch["id"] / "unedited.wav", wav)
    masks = [target_speech_mask(b["features"], spec) for b in batches]
    for edit in todo:
        if stop():
            return
        cx, cy = SENSORS[edit["sensor"]]
        rows = []
        for i, batch in enumerate(batches):
            edited = apply_xy_edit(batch["features"], edit, stats, directions, spec)
            dt_raw = (edited[0, [cx, cy]] - batch["features"][0, [cx, cy]]).double().cpu().numpy().T  # (T, 2)
            wav = synthesize(module, edited, batch["spk_raw"], device)
            row = measure_xy(base[i], reextract(wav), dt_raw, masks[i], edit, ema_std)
            rows.append(
                {
                    "id": batch["id"],
                    "edit": edit["name"],
                    "sensor": edit["sensor"],
                    "direction": edit["direction"],
                    "profile": edit["profile"],
                    "sign": edit["sign"],
                    "amp": edit["amp"],
                    **row,
                }
            )
            if i < examples:
                _write_wav16(out_dir / "wav" / batch["id"] / f"{edit['name']}.wav", wav)
        _write_parquet(edit_dir / f"{edit['name']}.parquet", pd.DataFrame(rows))
    if all((edit_dir / f"{e['name']}.parquet").exists() for e in edits):
        _write_parquet(results_path, _concat(edit_dir, edits))


def _concat(edit_dir: Path, edits: Sequence[dict]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(edit_dir / f"{e['name']}.parquet") for e in edits], ignore_index=True)
