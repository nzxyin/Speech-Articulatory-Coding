"""Segment-aware variants of the speaker-encoder screen on synthetic frames."""

import numpy as np
import pytest
import torch

from sparc.vocoders.spkscreen import pooling
from sparc.vocoders.spkscreen.data import FrameSet
from sparc.vocoders.spkscreen.variants import segment as S

NAMES = ["seg_vuv2", "seg_vus3", "seg_vu_soft", "seg_gate_basic", "seg_gate_feat", "seg_codebook6", "seg_codebook16",
         "seg_mel_on_base", "seg_mel_on_vuv2"]


def synth(n_utt=12, seed=0):
    rng = np.random.default_rng(seed)
    hid, aux, mel, lens, spk, tg = [], [], [], [], [], []
    for u in range(n_utt):
        n = int(rng.integers(80, 160))
        hid.append(rng.normal(size=(n, 1024)).astype(np.float16))
        per = np.where(rng.random(n) > 0.5, 0.9, 0.0)
        zl = np.where(rng.random(n) > 0.3, 0.8, 0.02)
        aux.append(np.stack([per, zl, zl / 10, np.full(n, 120.0)], 1).astype(np.float32))
        mel.append(rng.normal(size=(n, 80)).astype(np.float16))
        lens.append(n)
        spk.append(f"s{u % 4}")
        tg.append(rng.normal(size=6))
    c = lambda a: torch.from_numpy(np.concatenate(a))
    return FrameSet({6: c(hid)}, c(aux), c(mel), np.array(lens), np.array(spk), np.array(tg), np.zeros(n_utt))


def test_masks_partition():
    fs = synth()
    b = fs.full(np.arange(4), (6,))
    w = S.segment_masks(b)
    np.testing.assert_allclose(w.sum(-1).numpy(), b.mask.float().numpy())
    assert float(w[..., 0].sum()) == float(((b.periodicity >= 0.5) & b.mask).sum())


def test_class_means_and_backoff():
    x = torch.randn(2, 10, 5)
    w = torch.zeros(2, 10, 2)
    w[0, :5, 0] = 1
    w[0, 5:, 1] = 1
    w[1, :, 0] = 1  # second item has no class 1
    back = torch.randn(2, 5)
    m, s = S.class_means(x, w, back)
    torch.testing.assert_close(m[0, 0], x[0, :5].mean(0))
    torch.testing.assert_close(m[1, 1], back[1])
    torch.testing.assert_close(m[1, 0], x[1].mean(0))


@pytest.mark.parametrize("name", NAMES)
def test_variant_forward_backward(name):
    fs = synth()
    enc = pooling.build(name)
    enc.setup(fs)
    b = fs.full(np.arange(6), enc.layers, enc.needs_mel)
    enc.train()
    out = enc(b)
    assert out.shape == (6, 64) and torch.isfinite(out).all()
    out.pow(2).sum().backward()
    assert all(p.grad is not None for p in enc.parameters())
    # a short crop (class likely empty) and the padding invariance
    short = fs.batch(np.array([0, 1]), np.array([0, 0]), np.array([2, 40]), enc.layers, enc.needs_mel)
    enc.eval()
    assert torch.isfinite(enc(short)).all()
    solo = enc(fs.batch(np.array([1]), np.array([0]), np.array([40]), enc.layers, enc.needs_mel))
    torch.testing.assert_close(enc(short)[1:], solo, atol=1e-4, rtol=1e-4)


def test_codebook_uses_all_clusters():
    fs = synth()
    enc = pooling.build("seg_codebook6")
    enc.setup(fs)
    assert min(enc.cluster_sizes) > 0
