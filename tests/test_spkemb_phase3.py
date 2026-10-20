"""Speaker-embedding Phase 3: layer-pool / VAE encoder, multi-reference data, code mixing and the training terms."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from sparc.vocoders.constants import HOP
from sparc.vocoders.data.dataset import CropDataset, FullUtteranceDataset, PackedStore
from sparc.vocoders.features.spk_layers import pooled_stats
from sparc.vocoders.models.speaker import (
    LayerPoolSpeakerEncoder,
    SpeakerFFN,
    gaussian_kl,
    masked_mean,
    total_correlation,
)
from sparc.vocoders.training.speaker_terms import CropSpeakerStats, frozen_code, mix_codes, select_fmaps
from test_data import EVAL_ROWS, build_split, train_rows
from test_training import FakeDataModule, Recorder, StreamBatches, make_cfg

L, D = 2, 1024


def write_layer_stats(root: Path, packed: Path, n_layers: int = 25, seed: int = 0) -> None:
    import pandas as pd

    ids = pd.read_parquet(packed / "index.parquet", columns=["id"])["id"].astype(str).tolist()
    out = root / packed.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.save(out / "mean.npy", rng.standard_normal((len(ids), n_layers, D)).astype(np.float16))
    np.save(out / "std.npy", np.abs(rng.standard_normal((len(ids), n_layers, D))).astype(np.float16))
    (out / "ids.txt").write_text("\n".join(ids) + "\n")


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    build_split(cache, "train.clean.100", train_rows(), seed=1)
    build_split(cache, "dev.clean", EVAL_ROWS, seed=2)
    stats_root = tmp_path_factory.mktemp("layerstats")
    for split in ("train.clean.100", "dev.clean"):
        write_layer_stats(stats_root, cache / "packed" / split)
    return cache, stats_root


def test_pooled_stats_match_weighted_moments():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, 10, 4)).astype(np.float32)
    w = rng.random(12).astype(np.float32)
    mean, std = pooled_stats(h, w)
    ww = w[:10] / w[:10].sum()
    np.testing.assert_allclose(mean, np.einsum("t,ltd->ld", ww, h), rtol=1e-5, atol=1e-6)
    var = np.einsum("t,ltd->ld", ww, (h - mean[:, None]) ** 2)
    np.testing.assert_allclose(std, np.sqrt(var), rtol=1e-5, atol=1e-6)
    m0, _ = pooled_stats(h, np.zeros(10, np.float32))  # zero weights: uniform
    np.testing.assert_allclose(m0, h.mean(1), rtol=1e-5, atol=1e-6)


def test_layer_store_reads_stats_and_checks_ids(roots):
    cache, stats_root = roots
    packed = cache / "packed" / "train.clean.100"
    store = PackedStore([packed], "layers", stats_root)
    v = store.speaker_vector(3)
    assert v.shape == (2, 25, D) and v.dtype == np.float32
    expected = np.load(stats_root / packed.name / "mean.npy")[3].astype(np.float32)
    np.testing.assert_array_equal(v[0], expected)
    bad = stats_root / "bad" / packed.name
    bad.mkdir(parents=True)
    for f in ("mean.npy", "std.npy"):
        (bad / f).write_bytes((stats_root / packed.name / f).read_bytes())
    (bad / "ids.txt").write_text("x\n" * len(store))
    with pytest.raises(ValueError, match="ids.txt"):
        PackedStore([packed], "layers", stats_root / "bad")


def test_crop_dataset_multi_reference(roots):
    cache, stats_root = roots
    packed = cache / "packed" / "train.clean.100"
    ds = CropDataset([packed], crop_frames=8, p_cross=1.0, speaker_layer="layers", n_refs=(1, 3),
                     layer_stats_root=stats_root)  # fmt: skip
    counts = set()
    for pos in range(60):
        s = ds[(0, pos, pos % len(ds))]
        assert s["spk_raw"].shape == (3, 2, 25, D) and s["ref_mask"].shape == (3,)
        k = int(s["ref_mask"].sum())
        counts.add(k)
        assert torch.all(s["spk_raw"][k:] == 0)
        utt = int(s["utt_index"])
        if len(ds.groups.members[ds.groups.group_of[utt]]) > 1:
            assert bool(s["is_cross"])
    assert counts == {1, 2, 3}
    same = ds[(0, 5, 2)]
    again = ds[(0, 5, 2)]
    assert torch.equal(same["spk_raw"], again["spk_raw"])


def test_single_reference_stream_is_unchanged(roots):
    cache, _ = roots
    packed = cache / "packed" / "train.clean.100"
    ds = CropDataset([packed], crop_frames=8, p_cross=0.5)
    s = ds[(1, 4, 3)]
    assert s["spk_raw"].shape == (1024,) and "ref_mask" not in s


def test_full_utterance_multi_ref_t3_holds_the_pool(roots):
    cache, stats_root = roots
    dev = cache / "packed" / "dev.clean"
    t3 = FullUtteranceDataset(dev, condition="T3", speaker_layer="layers", multi_ref=True, layer_stats_root=stats_root)
    t2 = FullUtteranceDataset(dev, condition="T2", speaker_layer="layers", multi_ref=True, layer_stats_root=stats_root)
    for i in range(len(t3)):
        item = t3[i]
        assert item["spk_raw"].shape == (len(t3.pools[i]), 2, 25, D)
        assert bool(item["ref_mask"].all())
    assert t2[0]["spk_raw"].shape == (1, 2, 25, D)


def test_ffn_embed_averages_references():
    ffn = SpeakerFFN().eval()
    x = torch.randn(2, 3, 1024)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    z, info = ffn.embed(x, mask)
    assert info == {}
    torch.testing.assert_close(z[0], ffn(x[0, :2]).mean(0))
    torch.testing.assert_close(z[1], ffn(x[1, 0]))
    torch.testing.assert_close(ffn.embed(x[:, 0])[0], ffn(x[:, 0]))


def test_layer_pool_vae_product_of_gaussians():
    torch.manual_seed(0)
    enc = LayerPoolSpeakerEncoder(n_layers=L, vae=True).eval()
    x = torch.randn(2, 3, 2, L, D)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    mu, logvar = enc.posterior(x, mask)
    # padded reference is ignored
    mu2, logvar2 = enc.posterior(x[:1, :2], None)
    torch.testing.assert_close(mu[:1], mu2)
    torch.testing.assert_close(logvar[:1], logvar2)
    # one reference with the prior: precision 1 + 1 / var_k
    out = enc.per_reference(x[:, 0])
    mu_k, lv_k = out[:, :64], out[:, 64:]
    mu1, lv1 = enc.posterior(x[:, 0])
    torch.testing.assert_close(lv1, -torch.log1p(torch.exp(-lv_k)))
    torch.testing.assert_close(mu1, mu_k * torch.exp(-lv_k) / (1 + torch.exp(-lv_k)))
    # more references: sharper posterior
    assert torch.all(logvar[1] < enc.posterior(x[1:, :1])[1][0])
    z, info = enc.embed(x, mask)
    torch.testing.assert_close(z, info["mu"])
    assert enc.layer_weights().shape == (2, L)
    torch.testing.assert_close(enc.layer_weights().sum(-1), torch.ones(2))


def test_layer_pool_deterministic_and_mean_only():
    enc = LayerPoolSpeakerEncoder(n_layers=L, use_std=False).eval()
    x = torch.randn(4, 2, L, D)
    z, info = enc.embed(x)
    assert z.shape == (4, 64) and info == {}
    x2 = x.clone()
    x2[:, 1] = 100.0  # the std statistic is not used
    torch.testing.assert_close(enc.embed(x2)[0], z)
    torch.testing.assert_close(masked_mean(torch.ones(2, 3, 4), None), torch.ones(2, 4))


def test_kl_and_total_correlation():
    assert float(gaussian_kl(torch.zeros(3, 8), torch.zeros(3, 8)).abs().max()) < 1e-6
    torch.testing.assert_close(gaussian_kl(torch.ones(1, 2), torch.zeros(1, 2)), torch.tensor([1.0]))
    torch.manual_seed(0)
    m, d = 512, 4
    mu = torch.randn(m, d)
    indep = total_correlation(mu + 0.1 * torch.randn(m, d), mu, torch.full((m, d), np.log(0.01)), 10_000)
    shared = torch.randn(m, 1).repeat(1, d)
    dep = total_correlation(shared + 0.1 * torch.randn(m, d), shared, torch.full((m, d), np.log(0.01)), 10_000)
    assert float(dep) > float(indep) + 1.0


def test_mix_codes():
    z = torch.randn(16, 64)
    gen = torch.Generator().manual_seed(3)
    z_used, recon = mix_codes(z, 0.25, 0.5, gen)
    assert int((~recon).sum()) == 4
    torch.testing.assert_close(z_used[recon], z[recon])
    assert not torch.allclose(z_used[~recon], z[~recon])
    z1, r1 = mix_codes(z, 0.25, 0.5, torch.Generator().manual_seed(3))
    torch.testing.assert_close(z1, z_used)
    # pure interpolation stays on the segment between two batch codes
    zi, ri = mix_codes(torch.stack([torch.zeros(4), torch.ones(4)]), 1.0, 0.0, torch.Generator().manual_seed(0))
    assert torch.all((zi >= 0) & (zi <= 1)) and not bool(ri.any())
    assert torch.allclose(zi[:, 0:1].expand(-1, 4), zi)
    z0, r0 = mix_codes(z, 0.0, 0.5, gen)
    assert z0 is z and bool(r0.all())


def test_select_fmaps():
    fm = [[torch.arange(6).reshape(3, 2), torch.arange(3)]]
    out = select_fmaps(fm, torch.tensor([0, 2]))
    assert out[0][0].tolist() == [[0, 1], [4, 5]] and out[0][1].tolist() == [0, 2]


class FakeWavLM(torch.nn.Module):
    def forward(self, x, output_hidden_states=True):
        frames = x.shape[-1] // 320
        h = x[:, : frames * 320].reshape(x.shape[0], frames, 320)[..., :1].expand(-1, -1, D)
        return SimpleNamespace(hidden_states=[h * (i + 1) for i in range(3)])


def test_crop_stats_are_differentiable_and_frozen_code_detaches():
    stats = CropSpeakerStats()
    stats._wavlm.append(FakeWavLM())
    audio = torch.randn(2, 1, HOP * 10, requires_grad=True)
    features = torch.rand(2, 15, 10)
    out = stats(audio, features)
    assert out.shape == (2, 2, 3, D)
    enc = LayerPoolSpeakerEncoder(n_layers=3, vae=True)
    code = frozen_code(enc, out)
    code.sum().backward()
    assert audio.grad is not None and float(audio.grad.abs().sum()) > 0
    assert all(p.grad is None for p in enc.parameters())
    assert enc.training


# ---- training integration: tiny generator, layer-pool VAE with mixing ------------------------------------------


class LayerStream(Dataset):
    def __getitem__(self, i):
        rng = np.random.default_rng([0, int(i)])
        frames = 6
        feats = rng.standard_normal((15, frames)).astype(np.float32)
        feats[12] = 80.0 + 200.0 * rng.random(frames)
        feats[14] = 0.5 + 0.4 * rng.random(frames)
        k = 1 + int(i) % 3
        spk = np.zeros((3, 2, L, D), np.float32)
        spk[:k] = rng.standard_normal((k, 2, L, D))
        return {
            "features": torch.from_numpy(feats),
            "audio": torch.from_numpy(0.1 * rng.standard_normal((1, HOP * frames)).astype(np.float32)),
            "spk_raw": torch.from_numpy(spk),
            "ref_mask": torch.from_numpy(np.arange(3) < k),
        }


class LayerDataModule(FakeDataModule):
    def train_dataloader(self):
        return DataLoader(LayerStream(), batch_sampler=StreamBatches(self), num_workers=0)


def test_layerpool_vae_mix_training_runs(tmp_path, monkeypatch):
    import test_training

    stats_root = tmp_path / "ls"
    stats_root.mkdir()
    np.savez(stats_root / "stats.npz", mean_mean=np.zeros((L, D)), mean_std=np.ones((L, D)),
             std_mean=np.zeros((L, D)), std_std=np.ones((L, D)))  # fmt: skip
    monkeypatch.setattr(test_training, "BATCH", 4)
    cfg = make_cfg(tmp_path / "run", max_g_steps=6, mel_warmup=2)
    cfg.speaker = OmegaConf.create(
        {
            "layer": "layers", "p_cross": 1.0, "dropout": 0.2, "type": "layerpool", "n_refs": [1, 3],
            "layer_stats_root": str(stats_root), "hidden_dim": 32, "use_std": True,
            "vae": {"enabled": True, "capacity_max": 8.0, "capacity_steps": 4, "gamma": 1.0},
            "tc": {"weight": 0.0, "pool_size": 8, "batch": 4},
            "mix": {"p": 0.5, "prior_frac": 0.5, "cycle_weight": 0.0},
        }
    )  # fmt: skip
    from sparc.cli.train_vocoder import run as train_run

    rec = Recorder()
    stats = {
        "ema_mean": [0.0] * 12, "ema_std": [1.0] * 12, "logf0_mean": 5.0, "logf0_std": 0.3,
        "loud_log_mean": -5.0, "loud_log_std": 1.5, "per_mean": 0.5, "per_std": 0.4,
    }  # fmt: skip
    train_run(cfg, datamodule=LayerDataModule(), stats=stats, extra_callbacks=(rec,))
    rows = rec.rows
    assert len(rows) == 6
    assert all(np.isfinite(r["train/kl"]) for r in rows)
    assert rows[-1]["train/kl_capacity"] == pytest.approx(8.0)
    assert "train/d_total" in rows[-1]
