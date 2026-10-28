"""Tests of the out-of-domain screen pieces that need no cache: augmentation, the frame heads, the report statistics."""

import numpy as np
import pytest
import torch

from sparc.vocoders.spkscreen import augment as A
from sparc.vocoders.spkscreen.data import FrameBatch
from sparc.vocoders.spkscreen.ood_report import combine
from sparc.vocoders.spkscreen.variants.ood import GRUConvFrames, MLPFrames


@pytest.mark.parametrize("seed", range(8))
def test_augment_keeps_length_and_is_deterministic(seed):
    x = np.random.default_rng(0).standard_normal(24000).astype(np.float32) * 0.1
    y = A.augment(x, seed)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all() and np.abs(y).max() <= 0.99 + 1e-6
    assert np.array_equal(y, A.augment(x, seed))


def test_every_augmentation_kind_runs():
    x = np.random.default_rng(1).standard_normal(24000) * 0.1
    for kind, fn in A._FN.items():
        y = fn(x, np.random.default_rng(0))
        assert y.shape == x.shape and np.isfinite(y).all(), kind


def test_frame_heads_start_at_identity_and_ignore_padding():
    torch.manual_seed(0)
    z = torch.randn(2, 12, 1024)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, :12] = True
    mask[1, :7] = True
    for head in (GRUConvFrames(), MLPFrames()):
        head.eval()
        out = head(z * mask[..., None], mask)
        assert torch.allclose(out, z * mask[..., None])  # zero-initialised output layer
        torch.nn.init.normal_(head.conv.weight if hasattr(head, "conv") else head.out.weight, std=0.1)
        a = head(z * mask[..., None], mask)
        z2 = z.clone()
        z2[1, 7:] = 5.0  # different padding content must not change valid frames
        b = head(z2 * mask[..., None], mask)
        assert torch.allclose(a, b, atol=1e-5)


def test_combine_is_mean_of_corpora():
    r = combine({"d": 0.02, "se": 0.01, "n": 30}, {"d": 0.04, "se": 0.01, "n": 400})
    assert r["d"] == pytest.approx(0.03) and r["se"] == pytest.approx(0.5 * np.hypot(0.01, 0.01))
    assert r["lo"] < r["d"] < r["hi"]


def test_framebatch_defaults_are_backward_compatible():
    x = torch.zeros(1, 1, 5, 4)
    w = torch.ones(1, 5)
    fb = FrameBatch(x, w, w, w, w, None, torch.ones(1, 5, dtype=torch.bool))
    assert fb.set_name == "" and fb.utt is None
