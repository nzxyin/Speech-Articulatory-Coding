"""Multi-scale variants of the speaker-encoder screen on synthetic frames (no cache)."""

import numpy as np
import pytest
import torch

from sparc.vocoders.spkscreen import pooling
from sparc.vocoders.spkscreen.data import FrameSet

NAMES = ["ms_l1_l6", "ms_layerw", "ms_seg05", "ms_mfa", "ms_tdnn", "ms_xformer", "ms_grl_content", "ms_grl_f0", "ms_grl_both"]


def synth(n_spk=4, n_utt=3, seed=0):
    rng = np.random.default_rng(seed)
    hid = {k: [] for k in (1, 3, 6, 12)}
    aux, lens, spk, tg, lf = [], [], [], [], []
    for s in range(n_spk):
        base = rng.normal(size=1024) * 0.5
        for _ in range(n_utt):
            n = int(rng.integers(40, 90))
            for k in hid:
                hid[k].append((base + rng.normal(size=(n, 1024))).astype(np.float16))
            aux.append(np.stack([rng.random(n), rng.normal(size=n), rng.random(n), 100 + 50 * rng.random(n)], 1).astype(np.float32))
            lens.append(n)
            spk.append(f"s{s}")
            tg.append(rng.normal(size=12))
            lf.append(np.log(120 + 10 * s))
    return FrameSet({k: torch.from_numpy(np.concatenate(v)) for k, v in hid.items()}, torch.from_numpy(np.concatenate(aux)), None,
                    np.array(lens), np.array(spk), np.array(tg), np.array(lf))


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
    loss = out.square().sum() + (enc.extra_loss() if hasattr(enc, "extra_loss") else 0.0)
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0
    enc.eval()  # extra padding must not change the embedding
    n2, n0 = int(fs.lens[2]), int(fs.lens[0])
    b1 = fs.batch(np.array([2]), np.array([0]), np.array([n2]), enc.layers)
    b2 = fs.batch(np.array([2, 0]), np.array([0, 0]), np.array([n2, max(n2, n0) + 20]), enc.layers)
    torch.testing.assert_close(enc(b1)[0], enc(b2)[0], atol=1e-4, rtol=1e-4)


def test_layer_weights_start_uniform():
    enc = pooling.build("ms_layerw")
    torch.testing.assert_close(enc.layer_weights(), torch.full((4,), 0.25))


def test_grl_flips_gradient(fs):
    enc = pooling.build("ms_grl_both")
    enc.setup(fs)
    enc.train()
    b = fs.full(np.arange(6), enc.layers)
    enc(b)
    assert float(enc.extra_loss()) > 0
    w = enc.ffn.net[0].weight
    g_rev = torch.autograd.grad(enc.extra_loss(), w)[0]
    enc.lam = -0.1  # reversal strength with the opposite sign gives the opposite encoder gradient
    enc(b)
    g_fwd = torch.autograd.grad(enc.extra_loss(), w)[0]
    assert float((g_rev * g_fwd).sum()) < 0
    enc.eval()
    enc(b)
    assert enc.extra_loss() == 0.0


def test_segment_pooling_short_crop(fs):
    enc = pooling.build("ms_seg05")
    enc.setup(fs)
    enc.eval()
    b = fs.batch(np.array([1]), np.array([0]), np.array([30]), enc.layers)
    s = enc.stats(b)
    assert s.shape == (1, 1024 + 512) and torch.isfinite(s).all()


def test_multiref_reference_sets_exclude_partner():
    from sparc.vocoders.spkscreen import metrics as M
    from sparc.vocoders.spkscreen.multiref import reference_sets

    spk = np.array(["a"] * 6 + ["b"] * 3)
    partner, _ = M.partners(spk)
    for k in (1, 3, 5):
        for i, r in enumerate(reference_sets(spk, k)):
            assert r[0] == i and partner[i] not in r[1:] and len(set(r)) == len(r) and len(r) <= k
    assert [len(r) for r in reference_sets(spk, 5)][:6] == [5] * 6
