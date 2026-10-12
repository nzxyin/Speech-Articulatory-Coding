"""Prosody metrics on synthetic feature arrays with known answers (CPU, pure numpy)."""

import numpy as np
import pytest

from sparc.vocoders.constants import EMA_NAMES, N_EMA
from sparc.vocoders.eval.prosody import prosody_columns, prosody_metrics

T = 200


def make_features(seed: int = 0, frames: int = T) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic re-extraction: smooth EMA, a voiced F0 track with an unvoiced gap, periodicity and loudness."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames)
    feats = np.zeros((frames, 15), dtype=np.float32)
    for c in range(N_EMA):
        feats[:, c] = np.sin(2 * np.pi * t / (40 + 7 * c) + c) + 0.1 * rng.normal(size=frames)
    feats[:, 12] = 120.0 + 30.0 * np.sin(2 * np.pi * t / 90)
    voiced = np.ones(frames, dtype=bool)
    voiced[60:90] = False
    voiced[150:170] = False
    feats[:, 14] = np.where(voiced, rng.uniform(0.5, 0.99, size=frames), 0.0)
    loud = (0.02 + 0.05 * rng.uniform(size=frames)).astype(np.float32)
    feats[:, 13] = 0.0
    return feats, loud


def run(sys_feats, sys_loud, ref_feats, ref_ema, ref_loud, std=None):
    std = np.ones(N_EMA) if std is None else std
    return prosody_metrics(sys_feats, sys_loud, ref_feats, ref_ema, ref_loud, std)


def test_identity_gives_zero_error_and_perfect_correlation():
    feats, loud = make_features()
    m = run(feats, loud, feats, feats[:, :12], loud)
    assert m["f0_rmse_cents"] == 0.0 and m["f0_med_abs_cents"] == 0.0 and m["f0_within50"] == 1.0
    assert m["vde"] == 0.0 and m["per_mae"] == 0.0 and m["per_rmse"] == 0.0
    assert m["loud_db_rmse"] == 0.0 and m["loud_db_bias"] == 0.0
    assert m["ema_r_mean"] == pytest.approx(1.0) and m["ema_rmse_mean"] == 0.0
    assert all(m[f"ema_r_{name}"] == pytest.approx(1.0) for name in EMA_NAMES)
    assert m["n_frames_dropped"] == 0.0 and m["n_frames"] == T
    assert m["n_voiced_both"] == float((feats[:, 14] > 0).sum())
    assert list(m) == prosody_columns()


def test_known_pitch_shift_of_100_cents():
    feats, loud = make_features()
    shifted = feats.copy()
    shifted[:, 12] *= 2 ** (100 / 1200)
    m = run(shifted, loud, feats, feats[:, :12], loud)
    assert m["f0_rmse_cents"] == pytest.approx(100.0, abs=1e-3)
    assert m["f0_med_abs_cents"] == pytest.approx(100.0, abs=1e-3)
    assert m["f0_within50"] == 0.0
    shifted[:, 12] = feats[:, 12] * 2 ** (30 / 1200)
    assert run(shifted, loud, feats, feats[:, :12], loud)["f0_within50"] == 1.0


def test_f0_is_scored_only_where_both_are_voiced():
    feats, loud = make_features()
    sys = feats.copy()
    sys[60:90, 14] = 0.7  # voiced in sys only: counts as a voicing error, not as an F0 error
    sys[60:90, 12] = 9999.0
    sys[0:10, 14] = 0.0  # unvoiced in sys only
    m = run(sys, loud, feats, feats[:, :12], loud)
    assert m["f0_rmse_cents"] == 0.0
    assert m["vde"] == pytest.approx((30 + 10) / T)
    expected_both = ((feats[:, 14] > 0) & (sys[:, 14] > 0)).sum()
    assert m["n_voiced_both"] == expected_both


def test_no_voiced_frames_in_common_gives_nan_f0_but_finite_other_metrics():
    feats, loud = make_features()
    sys = feats.copy()
    sys[:, 14] = 0.0
    m = run(sys, loud, feats, feats[:, :12], loud)
    assert m["n_voiced_both"] == 0.0 and np.isnan(m["f0_rmse_cents"]) and np.isnan(m["f0_within50"])
    assert np.isfinite(m["vde"]) and np.isfinite(m["per_mae"])


def test_periodicity_error():
    feats, loud = make_features()
    sys = feats.copy()
    sys[:, 14] = np.where(feats[:, 14] > 0, feats[:, 14] - 0.1, 0.0)
    m = run(sys, loud, feats, feats[:, :12], loud)
    voiced = feats[:, 14] > 0
    assert m["per_mae"] == pytest.approx(0.1 * voiced.mean(), rel=1e-5)
    assert m["per_rmse"] == pytest.approx(0.1 * np.sqrt(voiced.mean()), rel=1e-5)
    assert m["vde"] == 0.0


def test_known_6_db_loudness_gain():
    feats, loud = make_features()
    gain = 10 ** (6.0 / 20)
    m = run(feats, loud * gain, feats, feats[:, :12], loud)
    assert m["loud_db_bias"] == pytest.approx(6.0, abs=0.1)
    assert m["loud_db_rmse"] == pytest.approx(6.0, abs=0.1)
    m = run(feats, loud / gain, feats, feats[:, :12], loud)
    assert m["loud_db_bias"] == pytest.approx(-6.0, abs=0.1)


def test_loudness_ignores_frames_below_the_reference_floor():
    feats, loud = make_features()
    sys_loud = loud.copy()
    floor = np.percentile(loud, 5.0)
    quiet = loud <= floor
    assert quiet.sum() > 0
    sys_loud[quiet] *= 100.0  # large error only where the reference is at or below its 5th percentile
    m = run(feats, sys_loud, feats, feats[:, :12], loud)
    assert m["loud_db_rmse"] == 0.0


def test_ema_rmse_is_in_units_of_the_training_std_and_correlation_ignores_scale():
    feats, loud = make_features()
    std = np.linspace(0.5, 2.0, N_EMA)
    sys = feats.copy()
    sys[:, :12] = feats[:, :12] + std[None, :]  # constant offset of one std per channel
    m = run(sys, loud, feats, feats[:, :12], loud, std=std)
    assert m["ema_rmse_mean"] == pytest.approx(1.0, rel=1e-5)
    assert all(m[f"ema_rmse_{name}"] == pytest.approx(1.0, rel=1e-5) for name in EMA_NAMES)
    assert m["ema_r_mean"] == pytest.approx(1.0)
    sys[:, :12] = -feats[:, :12]
    assert run(sys, loud, feats, feats[:, :12], loud)["ema_r_mean"] == pytest.approx(-1.0)


def test_constant_ema_channel_has_nan_correlation_that_the_mean_skips():
    feats, loud = make_features()
    sys = feats.copy()
    sys[:, 3] = 0.25
    m = run(sys, loud, feats, feats[:, :12], loud)
    assert np.isnan(m[f"ema_r_{EMA_NAMES[3]}"])
    assert m["ema_r_mean"] == pytest.approx(
        np.nanmean([m[f"ema_r_{name}"] for name in EMA_NAMES]), rel=1e-12
    )


def test_length_mismatch_is_truncated_and_counted():
    feats, loud = make_features()
    # the re-extraction of 480 T samples has T - 1 frames (and may be one frame longer than that for other lengths)
    m = run(feats[:-1], loud[:-1], feats[: T - 2], feats[:, :12], loud)
    assert m["n_frames"] == T - 2
    assert m["n_frames_dropped"] == 2.0
    assert m["f0_rmse_cents"] == 0.0 and m["ema_r_mean"] == pytest.approx(1.0)
    # truncation keeps the leading frames: a one-frame offset in the tail must not matter
    other = feats.copy()
    other[-1, :12] += 100.0
    assert run(other[:-1], loud[:-1], feats[:-1], feats[:-1, :12], loud[:-1])["ema_rmse_mean"] == 0.0


def test_inputs_are_not_modified_and_dtypes_are_accepted():
    feats, loud = make_features()
    before = (feats.copy(), loud.copy())
    run(feats.astype(np.float32), loud.astype(np.float32), feats, feats[:, :12].astype(np.float32), loud)
    np.testing.assert_array_equal(feats, before[0])
    np.testing.assert_array_equal(loud, before[1])


def test_bad_shapes_are_rejected():
    feats, loud = make_features()
    with pytest.raises(ValueError):
        prosody_metrics(feats[:, :10], loud, feats, feats[:, :12], loud, np.ones(12))
    with pytest.raises(ValueError):
        prosody_metrics(feats, loud, feats, feats[:, :11], loud, np.ones(12))
