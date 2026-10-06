"""Tests for sparc.compression.components: physical head / FFN-neuron pruning (CPU, tiny random models)."""
import torch

from sparc.compression.components import apply_spec, n_heads, n_neurons, prune_layer, select
from sparc.compression.encoders import subset_model
from test_compression import _tiny, _wav


def _masked_forward(model, x, head_masks, neuron_masks):
    hooks = []
    for layer, mh, mn in zip(model.encoder.layers, head_masks, neuron_masks):
        hd = layer.attention.head_dim

        def pre_attn(mod, inp, mh=mh, hd=hd):
            return (inp[0] * mh.repeat_interleave(hd),)

        def pre_ffn(mod, inp, mn=mn):
            return (inp[0] * mn,)

        hooks.append(layer.attention.out_proj.register_forward_pre_hook(pre_attn))
        hooks.append(layer.feed_forward.output_dense.register_forward_pre_hook(pre_ffn))
    try:
        return model(x).last_hidden_state
    finally:
        for h in hooks:
            h.remove()


def _mask(n, keep):
    m = torch.zeros(n)
    m[list(keep)] = 1.0
    return m


@torch.no_grad()
def test_physical_pruning_equals_masking():
    x = _wav()
    keep = [{"heads": [0, 2], "neurons": list(range(0, 64, 3))},
            {"heads": [], "neurons": [5, 9]},  # every head removed -> constant bias block
            {"heads": [1, 2, 3], "neurons": []}]  # every neuron removed -> constant bias block
    ref = _masked_forward(subset_model(_tiny("wav2vec2"), [1, 2, 3]), x,
                          [_mask(4, k["heads"]) for k in keep], [_mask(64, k["neurons"]) for k in keep])
    pruned = subset_model(_tiny("wav2vec2"), [1, 2, 3])
    apply_spec(pruned, keep)
    assert [n_heads(l) for l in pruned.encoder.layers] == [2, 0, 3]
    assert [n_neurons(l) for l in pruned.encoder.layers] == [22, 2, 0]
    torch.testing.assert_close(pruned(x).last_hidden_state, ref, rtol=1e-5, atol=1e-5)
    again = subset_model(_tiny("wav2vec2"), [1, 2, 3])  # state-dict round trip onto a fresh pruned model
    apply_spec(again, keep)
    again.load_state_dict(pruned.state_dict())
    torch.testing.assert_close(again(x).last_hidden_state, ref, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_iterative_pruning_with_current_indices_matches_one_shot():
    x = _wav()
    one = subset_model(_tiny("wav2vec2"), [1, 2])
    apply_spec(one, [{"heads": [3], "neurons": [7, 40]}, {"heads": [0, 1], "neurons": [63]}])
    two = subset_model(_tiny("wav2vec2"), [1, 2])
    apply_spec(two, [{"heads": [1, 3], "neurons": [2, 7, 40]}, {"heads": [0, 1, 2], "neurons": [10, 63]}])
    prune_layer(two.encoder.layers[0], [1], [1, 2])  # current indices of heads [3], neurons [7, 40]
    prune_layer(two.encoder.layers[1], [0, 1], [1])
    torch.testing.assert_close(two(x).last_hidden_state, one(x).last_hidden_state)


def test_select_keeps_global_top_fraction():
    spec = [{"heads": [0, 1, 2, 3], "neurons": [0, 1, 2, 3]}, {"heads": [0, 1, 2, 3], "neurons": [0, 1, 2, 3]}]
    scores = [(torch.tensor([0.9, 0.1, 0.8, 0.2]), torch.tensor([1.0, 2.0, 3.0, 4.0])),
              (torch.tensor([0.7, 0.6, 0.05, 0.3]), torch.tensor([0.0, 0.0, 5.0, 0.0]))]
    new = select(scores, spec, 0.5, 8, 8)
    assert new[0]["heads"] == [0, 2] and new[1]["heads"] == [0, 1]
    assert new[0]["neurons"] == [1, 2, 3] and new[1]["neurons"] == [2]
