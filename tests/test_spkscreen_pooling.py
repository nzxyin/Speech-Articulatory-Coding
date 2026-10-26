"""Pooling-operator variants of the speaker-encoder screen on synthetic frames (no cache)."""

import numpy as np
import pytest
import torch

from sparc.vocoders.spkscreen import pooling
from sparc.vocoders.spkscreen.data import FrameBatch, FrameSet
from sparc.vocoders.spkscreen.variants import pooling_ops as P

NAMES = ["pool_asp", "pool_asp_prior", "pool_mhap", "pool_mqap", "pool_netvlad", "pool_ghostvlad", "pool_kmeans",
         "pool_meanstd", "pool_skew", "pool_lowrank_cov", "pool_trimmed", "pool_quantile"]


def synth(n_spk=4, n_utt=3, seed=0):
    rng = np.random.default_rng(seed)
    hid, aux, lens, spk, tg = [], [], [], [], []
    for s in range(n_spk):
        base = rng.normal(size=1024) * 0.5
        for _ in range(n_utt):
            n = int(rng.integers(40, 90))
            hid.append((base + rng.normal(size=(n, 1024))).astype(np.float16))
            aux.append(np.stack([(rng.random(n) > 0.4), rng.normal(size=n), rng.random(n), 100 + rng.random(n)], 1).astype(np.float32))
            lens.append(n)
            spk.append(f"s{s}")
            tg.append(rng.normal(size=12))
    return FrameSet({6: torch.from_numpy(np.concatenate(hid))}, torch.from_numpy(np.concatenate(aux)), None, np.array(lens),
                    np.array(spk), np.array(tg), np.zeros(len(lens)))


@pytest.fixture(scope="module")
def fs():
    return synth()


@pytest.mark.parametrize("name", NAMES)
def test_variant_shape_grad_padding(name, fs):
    torch.manual_seed(0)
    enc = pooling.build(name)
    enc.setup(fs)
    enc.train()
    out = enc(fs.full(np.arange(4), enc.layers))
    assert out.shape == (4, 64) and torch.isfinite(out).all()
    out.square().sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0
    enc.eval()  # extra padding must not change the embedding
    n2, n0 = int(fs.lens[2]), int(fs.lens[0])
    b1 = fs.batch(np.array([2]), np.array([0]), np.array([n2]), enc.layers)
    b2 = fs.batch(np.array([2, 0]), np.array([0, 0]), np.array([n2, max(n2, n0) + 20]), enc.layers)
    torch.testing.assert_close(enc(b1)[0], enc(b2)[0], atol=1e-4, rtol=1e-4)


def test_weighted_quantile_and_trim():
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.normal(size=(1, 50, 1024)).astype(np.float32))
    w = torch.from_numpy(rng.random((1, 50)).astype(np.float32))
    fb = FrameBatch(x[:, None], w, w, w, w, None, torch.ones(1, 50, dtype=torch.bool))
    got = P.Quantiles(qs=(0.5,)).stats(fb)[0, 0].item()
    order = np.argsort(x[0, :, 0].numpy())
    cw = np.cumsum(w[0].numpy()[order]) / w[0].sum().item()
    assert got == pytest.approx(x[0, order[np.searchsorted(cw, 0.5 - 1e-6)], 0].item())
    torch.testing.assert_close(P.TrimmedMean(trim=0.0).stats(fb), pooling.weighted_mean(x, w), atol=1e-5, rtol=1e-5)
    assert P.TrimmedMean(trim=0.1).stats(fb).shape == (1, 1024)


def test_kmeans_codebook_and_ghost(fs):
    enc = pooling.build("pool_kmeans")
    enc.setup(fs)
    assert float(enc.codebook.abs().sum()) > 0
    g = pooling.build("pool_ghostvlad")
    assert g.assign.out_features == 10 and g.ffn.net[0].in_features == 8 * 128
