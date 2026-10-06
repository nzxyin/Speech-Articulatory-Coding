"""Tests for sparc.compression: layer-subset encoders, ridge probe, metrics, MNGU0 split/alignment.

Encoder tests use tiny randomly initialized Wav2Vec2/WavLM configs (CPU, no downloads). MNGU0 tests
are skipped when the corpus is not mounted.
"""
import numpy as np
import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model, WavLMConfig, WavLMModel

from sparc.compression import mngu0
from sparc.compression.encoders import subset_model
from sparc.compression.metrics import ema_metrics, phone_class
from sparc.compression.probe import Ridge

TINY = dict(
    hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=64,
    conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 8), num_conv_pos_embeddings=8,
    num_conv_pos_embedding_groups=4, do_stable_layer_norm=True, feat_extract_norm="layer",
    layerdrop=0.0, mask_time_prob=0.0, hidden_dropout=0.0, attention_dropout=0.0,
    activation_dropout=0.0, feat_proj_dropout=0.0,
)


def _tiny(kind):
    torch.manual_seed(0)
    if kind == "wav2vec2":
        return Wav2Vec2Model(Wav2Vec2Config(**TINY)).eval()
    return WavLMModel(WavLMConfig(**TINY, num_buckets=16, max_bucket_distance=32)).eval()


def _wav():
    return torch.randn(2, 4000, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("kind", ["wav2vec2", "wavlm"])
@pytest.mark.parametrize("k", [1, 3, 5])
@torch.no_grad()
def test_prefix_matches_full_hidden_states(kind, k):
    x = _wav()
    full = _tiny(kind)(x, output_hidden_states=True).hidden_states
    sub = subset_model(_tiny(kind), range(1, k + 1))(x).last_hidden_state
    torch.testing.assert_close(sub, full[k])


@torch.no_grad()
def test_noncontiguous_subset_runs_layers_in_order():
    x = _wav()
    ref = _tiny("wav2vec2")
    keep = [1, 3, 4]
    h = ref.feature_projection(ref.feature_extractor(x).transpose(1, 2))[0]
    h = h + ref.encoder.pos_conv_embed(h)
    for i in keep:
        h = ref.encoder.layers[i - 1](h)[0]
    sub = subset_model(_tiny("wav2vec2"), keep)(x).last_hidden_state
    torch.testing.assert_close(sub, h)


def test_subset_validation():
    with pytest.raises(ValueError):
        subset_model(_tiny("wav2vec2"), [3, 2])
    with pytest.raises(ValueError):
        subset_model(_tiny("wav2vec2"), [1, 6])
    with pytest.raises(ValueError):
        subset_model(_tiny("wavlm"), [2, 3])


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(20, 12))
    X = [rng.normal(size=(50, 20)) * 3 + 1 for _ in range(10)]
    Y = [x @ W + 0.5 for x in X]
    r = Ridge(X, Y, "cpu")
    Wh, bh = r.weights(1e-6)
    pred = r.predict(Wh, bh, X[:2])
    np.testing.assert_allclose(pred[0], Y[0], atol=1e-5)


def test_metrics_perfect_and_offset():
    rng = np.random.default_rng(0)
    Y = [np.cumsum(rng.normal(size=(100, 12)), 0) for _ in range(5)]
    m = ema_metrics(Y, Y, n_boot=10)
    assert m["rmse"] == pytest.approx(0) and m["pcc"] == pytest.approx(1)
    m = ema_metrics([y + 2.0 for y in Y], Y, n_boot=0)
    assert m["rmse"] == pytest.approx(2.0) and m["vel_rmse"] == pytest.approx(0, abs=1e-9)
    assert m["per_articulator"]["TT"]["eucl_mean"] == pytest.approx(np.sqrt(8))


def test_phone_classes():
    assert phone_class("p") == "labial" and phone_class("k") == "dorsal"
    assert phone_class("tS") == "coronal" and phone_class("aI") == "vowel"


def test_split_rule():
    assert mngu0.split_of("mngu0_s1_0010") == "valid"
    assert mngu0.split_of("mngu0_s1_0020") == "test"
    assert mngu0.split_of("mngu0_s1_0130") == "valid"
    assert mngu0.split_of("mngu0_s1_0001") == "train"


def test_align():
    f, e = np.arange(10)[:, None], np.arange(5)[:, None]
    a, b = mngu0.align(f, e, 3)
    assert a[:, 0].tolist() == [3, 4, 5, 6, 7] and len(b) == 5
    a, b = mngu0.align(f, e, -2)
    assert a[:, 0].tolist() == [0, 1, 2] and b[:, 0].tolist() == [2, 3, 4]


needs_data = pytest.mark.skipif(not mngu0.EMA_NORM_DIR.exists(), reason="MNGU0 not mounted")


@needs_data
def test_split_matches_corpus_filesets(tmp_path):
    import zipfile

    z = zipfile.ZipFile(mngu0.MNGU0_ROOT.parent / "mngu0_s1_ema_filesets_1.0.0.zip")
    for name, split in [("trainfiles", "train"), ("validationfiles", "valid"), ("testfiles", "test")]:
        ids = z.read(f"mngu0_s1_ema_filesets/{name}.txt").decode().split()
        assert all(mngu0.split_of(s) == split for s in ids), name


@needs_data
def test_utterance_units_and_alignment():
    u = mngu0.load_utterance("mngu0_s1_0001")
    assert u.ema_mm.shape[1] == 12
    assert 40 < u.ema_mm[:, 0].mean() < 70  # tongue dorsum x (back/front) is ~54 mm
    assert u.base_offset == 24  # leading silence ends at 0.488 s
    assert len(mngu0.frame_phones(u, u.base_offset, len(u.ema_mm))) == len(u.ema_mm)


# --- LoRA variants and heads ---------------------------------------------------------------

from sparc.compression.heads import Head, lowpass_kernel  # noqa: E402
from sparc.compression.lora import apply_lora, lora_param_count, merge_lora  # noqa: E402


@pytest.mark.parametrize("kind", ["wav2vec2", "wavlm"])
@pytest.mark.parametrize("variant", ["independent", "shared_a", "shared_gated"])
@torch.no_grad()
def test_lora_zero_init_then_merge(kind, variant):
    x = _wav()
    ref = subset_model(_tiny(kind), [1, 2, 3])(x).last_hidden_state
    m = subset_model(_tiny(kind), [1, 2, 3])
    bank = apply_lora(m, variant, rank=4, alpha=8, targets=("q_proj", "v_proj", "out_proj"), gate="rank")
    torch.testing.assert_close(m(x).last_hidden_state, ref)  # B = 0 at init
    for p in bank.parameters():
        p.normal_(0, 0.1)
    adapted = m(x).last_hidden_state
    assert not torch.allclose(adapted, ref)
    merge_lora(m)
    torch.testing.assert_close(m(x).last_hidden_state, adapted, rtol=1e-4, atol=1e-4)


def test_lora_param_counts():
    d, r, n_layers, n_targets = 32, 4, 3, 2
    counts = {}
    for v in ["independent", "shared_a", "shared_gated"]:
        m = subset_model(_tiny("wav2vec2"), [1, 2, 3])
        counts[v] = lora_param_count(apply_lora(m, v, rank=r, gate="scalar"))
        assert all(not p.requires_grad for n, p in m.named_parameters() if "lora_bank" not in n and ".A" not in n
                   and ".B" not in n and "gate" not in n)
    assert counts["independent"] == n_layers * n_targets * 2 * d * r
    assert counts["shared_a"] == n_targets * (d * r + n_layers * d * r)
    assert counts["shared_gated"] == n_targets * (2 * d * r + n_layers)


def test_lowpass_kernel_matches_filtfilt():
    from sparc.inversion import butter_bandpass_filter

    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    ref = butter_bandpass_filter(x[None, :, None], 10, 50)[0, :, 0]
    k = lowpass_kernel().numpy()
    y = np.convolve(x, k, mode="same")
    np.testing.assert_allclose(y[60:-60], ref[60:-60], atol=1e-3)


@pytest.mark.parametrize("kind", ["conv", "attn"])
@torch.no_grad()
def test_causal_heads_ignore_future(kind):
    torch.manual_seed(0)
    head = Head(16, kind, hidden=16, kernel=5, window=0, causal=True, smooth=False).eval()
    h = torch.randn(1, 30, 16)
    h2 = h.clone()
    h2[:, 20:] = torch.randn(1, 10, 16)
    torch.testing.assert_close(head(h)[:, :20], head(h2)[:, :20])


@torch.no_grad()
def test_linear_head_standardized_init_matches_raw_ridge():
    rng = np.random.default_rng(0)
    W, b = rng.normal(size=(16, 12)) * 1e-3, rng.normal(size=12)
    mean, std = rng.normal(size=16) * 5, rng.uniform(10, 100, size=16)
    head = Head(16, "linear", smooth=False)
    head.set_input_stats(torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32))
    head.init_linear(W, b)
    h = torch.tensor(rng.normal(size=(1, 7, 16)) * std + mean, dtype=torch.float32)
    torch.testing.assert_close(head(h)[0], (h[0].double() @ torch.tensor(W) + torch.tensor(b)).float(),
                               rtol=1e-4, atol=1e-4)


# --- layer pooling ---------------------------------------------------------------------------

from sparc.compression.heads import LayerPool  # noqa: E402


def _layers(n=4, B=2, T=11, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, D, generator=g) * (i + 1) + i for i in range(n)]


@pytest.mark.parametrize("mode", ["static", "attn"])
@torch.no_grad()
def test_layer_pool_starts_uniform_over_normalized_layers(mode):
    hs = _layers()
    out = LayerPool(4, 16, mode, groups=3)(hs)
    ref = torch.stack([torch.nn.functional.layer_norm(h, (16,)) for h in hs]).mean(0)
    assert out.shape == (2, 3, 11, 16)
    for g in range(3):
        torch.testing.assert_close(out[:, g], ref)


def test_attn_pool_is_framewise_and_trainable():
    hs = _layers()
    pool = LayerPool(4, 16, "attn", groups=2)
    with torch.no_grad():
        pool.v.normal_()
    out = pool(hs)
    out.sum().backward()
    assert pool.v.grad.abs().sum() > 0 and pool.proj.weight.grad.abs().sum() > 0
    # weights differ across frames (frame-wise), unlike static pooling
    w = pool.weight_sum
    assert w.shape == (2, 4)
    hs2 = [h.clone() for h in hs]
    hs2[0][:, 5] += 10  # change one frame: only that frame's output may change
    with torch.no_grad():
        o1, o2 = pool(hs), pool(hs2)
    diff = (o1 - o2).abs().sum(dim=(0, 1, 3))
    assert diff[5] > 0 and torch.all(diff[torch.arange(11) != 5] == 0)


@torch.no_grad()
def test_per_articulator_head_maps_groups_to_channel_pairs():
    torch.manual_seed(0)
    head = Head(16, "linear", smooth=False, n_pool_layers=4, pool="static", per_articulator=True)
    hs = _layers()
    y = head(hs)
    assert y.shape == (2, 11, 12)
    head.out.weight[3].zero_()
    head.out.bias[3].fill_(7.0)  # articulator 3 (LI) -> channels 6, 7
    y = head(hs)
    assert torch.all(y[..., 6:8] == 7.0) and not torch.all(y[..., 4:6] == 7.0)


@torch.no_grad()
def test_per_articulator_causal_conv_ignores_future():
    torch.manual_seed(0)
    head = Head(16, "conv", hidden=8, kernel=5, causal=True, smooth=False, n_pool_layers=4, pool="attn",
                per_articulator=True).eval()
    with torch.no_grad():
        head.pool.v.normal_()
    hs = _layers()
    hs2 = [h.clone() for h in hs]
    for h in hs2:
        h[:, 8:] = torch.randn(2, 3, 16)
    torch.testing.assert_close(head(hs)[:, :8], head(hs2)[:, :8])


def test_per_articulator_requires_pooling_and_supported_head():
    with pytest.raises(ValueError):
        Head(16, "linear", per_articulator=True)
    with pytest.raises(ValueError):
        Head(16, "attn", n_pool_layers=3, per_articulator=True)
