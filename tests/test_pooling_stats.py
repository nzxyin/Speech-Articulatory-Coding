"""Pooled linear heads: running input statistics, function-preserving re-standardization, raw pooling."""
import pytest
import torch

from sparc.compression.heads import Head, LayerPool
from sparc.compression.train import load_trainable, trainable_state


def _layers(n=4, B=2, T=11, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, D, generator=g) * (i + 1) + 3 * i for i in range(n)]


@pytest.mark.parametrize("per_articulator", [False, True])
def test_restandardize_preserves_function_and_tracks_stats(per_articulator):
    torch.manual_seed(0)
    head = Head(16, "linear", smooth=False, n_pool_layers=4, pool="attn", per_articulator=per_articulator)
    with torch.no_grad():
        head.pool.v.normal_()
        head.pool.logits.normal_()
    hs = _layers()
    pad = torch.zeros(2, 11, dtype=torch.bool)
    pad[1, 8:] = True
    head.train()
    head(hs, pad_mask=pad)  # accumulate running statistics (valid frames only)
    head.eval()
    with torch.no_grad():
        before = head(hs, pad_mask=pad)
    head.restandardize()
    with torch.no_grad():
        after = head(hs, pad_mask=pad)
        pooled = head.pooled(hs, pad)
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-4)
    p = pooled if pooled.dim() == 4 else pooled[:, None]
    valid = p[0].reshape(p.shape[1], -1, 16)  # utterance 0: all frames valid
    valid = torch.cat([valid, p[1][:, :8]], 1)  # utterance 1: first 8 frames
    torch.testing.assert_close(head.in_mean, valid.mean(1), rtol=1e-4, atol=1e-4)
    assert head._stat_sum is None  # reset after use


def test_trainable_state_carries_input_stats():
    head = Head(16, "linear", smooth=False, n_pool_layers=4, pool="static", per_articulator=True)
    head.set_input_stats(torch.arange(16.0), torch.full((16,), 2.0))
    with torch.no_grad():
        head.in_mean[3] += 5
    model = torch.nn.Linear(2, 2)
    st = trainable_state(model, head)
    assert "head.in_mean" in st and st["head.in_mean"].shape == (6, 16)
    other = Head(16, "linear", smooth=False, n_pool_layers=4, pool="static", per_articulator=True)
    load_trainable(model, other, st)
    torch.testing.assert_close(other.in_mean, head.in_mean)
    old_style = {"head.in_mean": torch.ones(16), "head.in_std": torch.ones(16)}  # (D,) from older runs
    global_head = Head(16, "linear", smooth=False)
    load_trainable(model, global_head, old_style)
    assert global_head.in_mean.shape == (1, 16)


@torch.no_grad()
def test_raw_pooling_of_one_layer_is_that_layer():
    hs = _layers(n=1)
    torch.testing.assert_close(LayerPool(1, 16, "static", norm=False)(hs)[:, 0], hs[0])
    torch.testing.assert_close(LayerPool(1, 16, "attn", norm=False)(hs)[:, 0], hs[0])
