import numpy as np
import torch

from sparc.compression import rigor
from sparc.compression.heads import Smooth
from sparc.compression.metrics import ema_metrics


def _arm(rng, n_utt=12, seeds=3, noise=0.5, scale=1.0, offset=0.0):
    Y = [rng.normal(size=(rng.integers(40, 120), 12)) for _ in range(n_utt)]
    stems = [f"u{i}" for i in range(n_utt)]
    return [(stems, [scale * y + offset + noise * rng.normal(size=y.shape) for y in Y], Y) for _ in range(seeds)], Y


def test_metrics_match_reference():
    rng = np.random.default_rng(0)
    runs, _ = _arm(rng, seeds=1)
    _, P, Y = runs[0]
    m = rigor.metrics_from(rigor.utt_stats(P, Y))
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    assert np.isclose(m["rmse_z"], np.sqrt(((Pc - Yc) ** 2).mean(0)).mean())
    assert np.isclose(m["pcc"], ema_metrics(P, Y, n_boot=0)["pcc"])
    # pure offset error: centered rmse ~ 0 and bias share ~ 1
    m2 = rigor.metrics_from(rigor.utt_stats([y + 1.0 for y in Y], Y))
    assert m2["rmse_centered"] < 1e-9 and np.isclose(m2["bias_share"], 1.0) and np.isclose(m2["rmse_z"], 1.0)
    # index subset equals metrics on the subset
    idx = np.array([0, 0, 3])
    sub = rigor.metrics_from(rigor.utt_stats([P[i] for i in idx], [Y[i] for i in idx]))
    assert np.isclose(rigor.metrics_from(rigor.utt_stats(P, Y), idx)["rmse_z"], sub["rmse_z"])


def test_smooth_matches_head_smoother():
    rng = np.random.default_rng(1)
    p = rng.normal(size=(150, 12))
    ref = Smooth()(torch.from_numpy(p).float()[None])[0].double().numpy()
    assert np.allclose(rigor.smooth([p])[0], ref, atol=1e-5)


def test_sign_flip_and_holm():
    assert np.isclose(rigor.sign_flip_p(np.ones(7)), 2 / 128)
    assert rigor.sign_flip_p(np.array([1, -1, 1, -1, 1, -1, 0.01])) > 0.5
    assert rigor.holm([0.01, 0.04, 0.5]) == [True, False, False]


def test_shrink_recovers_gain():
    rng = np.random.default_rng(2)
    lora = {}
    for h in rigor.SPK:
        runs, _ = _arm(rng, noise=0.0, scale=2.0)
        lora[h] = runs
    xs, xc, orc, gains = rigor.shrink_arms(lora)
    assert np.allclose(gains["scalar"], 0.5) and np.allclose(gains["oracle"], 0.5)
    assert rigor.metrics_from(rigor.utt_stats(*xs[rigor.SPK[0]][0][1:]))["rmse_z"] < 1e-9


def test_contrast_detects_better_arm():
    rng = np.random.default_rng(3)
    a, b = {}, {}
    for h in rigor.SPK:
        runs, Y = _arm(rng, noise=0.3)
        a[h] = runs
        b[h] = [(s, [y + 0.8 * rng.normal(size=y.shape) for y in Y], Y) for s, _, _ in runs]
    r = rigor.contrast(a, b, "rmse_z", -1)
    assert r["won"] == 7 and r["mean"] > 0 and r["boot_ci"][0] > 0 and np.isclose(r["sign_flip_p"], 2 / 128)
