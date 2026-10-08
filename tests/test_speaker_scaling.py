"""Sufficient-statistics ridge (sparc.compression.speaker_scaling) matches probe.Ridge on concatenated frames."""

import numpy as np

from sparc.compression.probe import Ridge
from sparc.compression.speaker_scaling import Stats, StatsRidge, sse_from_stats


def _data(rng, n, d=16):
    x = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, d) + rng.normal(size=d)
    y = x @ rng.normal(size=(d, 12)) + rng.normal(size=(n, 12)) + 2.0
    return x, {"z": y, "zca": y @ rng.normal(size=(12, 12))}


def test_stats_ridge_matches_probe_ridge():
    rng = np.random.default_rng(0)
    parts = [_data(rng, n) for n in (40, 55, 70)]
    st = Stats.sum([_stats(x, ys) for x, ys in parts])
    sr = StatsRidge(st)
    for space in ("z", "zca"):
        ref = Ridge([x for x, _ in parts], [ys[space] for _, ys in parts], "cpu")
        for alpha, (W, b) in zip((0.1, 10.0, 1e4), sr.weights(space, (0.1, 10.0, 1e4))):
            Wr, br = ref.weights(alpha)
            assert np.allclose(W, Wr.numpy(), atol=1e-8)
            assert np.allclose(b, br.numpy(), atol=1e-8)


def test_sse_from_stats_matches_direct():
    rng = np.random.default_rng(1)
    x, ys = _data(rng, 80)
    st = _stats(x, ys)
    W, b = rng.normal(size=(16, 12)), rng.normal(size=12)
    for space in ("z", "zca"):
        direct = ((x @ W + b - ys[space]) ** 2).sum(0)
        assert np.allclose(sse_from_stats(st, space, W, b), direct)


def test_stats_copy_is_independent():
    rng = np.random.default_rng(2)
    x, ys = _data(rng, 20)
    st = _stats(x, ys)
    c = st.copy()
    st.add(x, ys)
    assert c.n == 20 and st.n == 40
    assert np.allclose(c.sxx * 2, st.sxx)


def _stats(x, ys):
    st = Stats(x.shape[1], list(ys))
    st.add(x, ys)
    return st
