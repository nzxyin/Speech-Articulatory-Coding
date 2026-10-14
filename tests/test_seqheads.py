"""Batched heads give each sequence exactly the output it gets alone (padding never leaks into valid frames)."""

import pytest
import torch

from sparc.compression.head_screen import HEADS, make_head


@pytest.mark.parametrize("name", HEADS)
@pytest.mark.parametrize("smooth", [False, True])
def test_batched_equals_unbatched(name, smooth):
    torch.manual_seed(0)
    head = make_head(name, 32, smooth=smooth).eval()
    lens = [50, 31, 12]
    xs = [torch.randn(n, 32) for n in lens]
    T = max(lens)
    xb = torch.full((len(xs), T, 32), 7.0)  # nonzero padding on purpose
    mask = torch.ones(len(xs), T, dtype=torch.bool)
    for i, x in enumerate(xs):
        xb[i, : len(x)] = x
        mask[i, : len(x)] = False
    with torch.no_grad():
        yb = head(xb, mask)
        for i, x in enumerate(xs):
            alone = head(x[None], torch.zeros(1, len(x), dtype=torch.bool))[0]
            assert torch.allclose(yb[i, : len(x)], alone, atol=1e-4), (name, smooth, i)


def test_parallel_scan_matches_sequential():
    from sparc.compression.seqheads import pscan, seq_scan

    torch.manual_seed(1)
    a = torch.rand(2, 37, 5, 3).clamp_min(0.05).requires_grad_(True)
    b = torch.randn(2, 37, 5, 3).requires_grad_(True)
    hp, hs = pscan(a, b), seq_scan(a, b)
    assert torch.allclose(hp, hs, atol=1e-5)
    gp = torch.autograd.grad(hp.square().sum(), (a, b))
    gs = torch.autograd.grad(hs.square().sum(), (a, b))
    assert all(torch.allclose(x, y, atol=1e-4) for x, y in zip(gp, gs))
