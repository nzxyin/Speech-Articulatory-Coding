"""Speaker-embedding Phase 4: swap-in-training, SV identity loss, speaker-relative F0, per-speaker F0 statistics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from sparc.vocoders.constants import F0_CHANNEL, HOP, N_FEATURES, PERIODICITY_CHANNEL
from sparc.vocoders.data.dataset import (
    CropDataset,
    FullUtteranceDataset,
    PackedStore,
    build_speaker_f0_stats,
    geometric_mean_f0,
    load_speaker_f0,
    main as dataset_main,
    median_voiced_f0,
    save_speaker_f0_stats,
)
from sparc.vocoders.models.frontend import REL_LOGF0_STD, FeatureFrontend
from sparc.vocoders.models.speaker import PitchCode
from sparc.vocoders.training.speaker_terms import (
    CropSpeakerVerifier,
    identity_loss,
    pick_swaps,
    relative_f0_features,
    scale_voiced_f0,
    swap_batch,
    swap_f0_factor,
)
from test_data import EVAL_ROWS, build_split, train_rows
from test_training import CONF, FakeDataModule, Recorder, StreamBatches, TinyGenerator, make_cfg

STATS = {
    "ema_mean": [0.0] * 12, "ema_std": [1.0] * 12, "logf0_mean": 5.0, "logf0_std": 0.3,
    "loud_log_mean": -5.0, "loud_log_std": 1.5, "per_mean": 0.5, "per_std": 0.4,
    "spk_l6_mean": [0.0] * 1024, "spk_l6_std": [1.0] * 1024,
}  # fmt: skip


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    build_split(root, "train.clean.100", train_rows(), seed=1)
    build_split(root, "dev.clean", EVAL_ROWS, seed=2)
    return root


def make_features(b: int = 4, t: int = 10, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(b, N_FEATURES, t, generator=g)
    f[:, F0_CHANNEL] = 100.0 + 100.0 * torch.rand(b, t, generator=g)
    f[:, PERIODICITY_CHANNEL] = torch.rand(b, t, generator=g) - 0.4  # some unvoiced frames
    return f


# ---- swap pairing and F0 scaling ---------------------------------------------------------------------------------


def test_pick_swaps_never_pairs_a_speaker_with_itself():
    ids = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3])
    for seed in range(50):
        picked, partner = pick_swaps(ids, 0.5, torch.Generator().manual_seed(seed))
        assert len(picked) == len(partner) == 4 and len(set(picked.tolist())) == 4
        assert bool((ids[picked] != ids[partner]).all())


def test_pick_swaps_skips_elements_that_have_no_other_speaker():
    picked, partner = pick_swaps(torch.tensor([7, 7, 7, 7]), 0.5, torch.Generator().manual_seed(0))
    assert len(picked) == len(partner) == 0
    picked, _ = pick_swaps(torch.tensor([1, 2]), 0.0, torch.Generator().manual_seed(0))
    assert len(picked) == 0


def test_swap_f0_factor_and_voiced_scaling():
    speaker_f0 = torch.tensor([100.0, 200.0, float("nan"), 150.0])
    picked, partner = torch.tensor([0, 1, 2]), torch.tensor([1, 0, 0])
    factor = swap_f0_factor(speaker_f0, picked, partner)
    torch.testing.assert_close(factor, torch.tensor([2.0, 0.5, 1.0]))  # NaN median: factor 1
    f = make_features()
    out = scale_voiced_f0(f, picked[:2], factor[:2])
    voiced = f[:, PERIODICITY_CHANNEL] > 0
    for k, i in enumerate([0, 1]):
        torch.testing.assert_close(out[i, F0_CHANNEL][voiced[i]], f[i, F0_CHANNEL][voiced[i]] * factor[k])
        torch.testing.assert_close(out[i, F0_CHANNEL][~voiced[i]], f[i, F0_CHANNEL][~voiced[i]])
    assert torch.equal(out[2:], f[2:]) and torch.equal(out[:, :F0_CHANNEL], f[:, :F0_CHANNEL])
    assert not torch.equal(out, f) and torch.equal(scale_voiced_f0(f, picked[:0], factor[:0]), f)


def test_swap_batch_codes_recon_targets_and_features():
    b = 8
    z = torch.randn(b, 64)
    f = make_features(b)
    ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    f0 = torch.tensor([100.0, 100.0, 200.0, 200.0, 150.0, 150.0, 120.0, 120.0])
    f_before = f.clone()
    z_used, features, recon, target, n = swap_batch(z, f, ids, f0, 0.25, torch.Generator().manual_seed(3))
    assert n == 2 and int((~recon).sum()) == 2 and torch.equal(f, f_before)  # the input is not modified
    for i in range(b):
        if recon[i]:
            assert torch.equal(z_used[i], z[i]) and target[i] == i and torch.equal(features[i], f[i])
        else:
            j = int(target[i])
            assert ids[i] != ids[j] and torch.equal(z_used[i], z[j])
            voiced = f[i, PERIODICITY_CHANNEL] > 0
            torch.testing.assert_close(
                features[i, F0_CHANNEL][voiced], f[i, F0_CHANNEL][voiced] * (f0[j] / f0[i])
            )
    _, features, *_ = swap_batch(z, f, ids, f0, 0.25, torch.Generator().manual_seed(3), scale_f0=False)
    assert torch.equal(features, f)


def test_swap_batch_leaves_mixed_elements_alone_and_borrows_original_codes():
    b = 6
    z = torch.randn(b, 8)
    z_mixed = z.clone()
    z_mixed[0] = 5.0
    recon = torch.tensor([False, True, True, True, True, True])
    ids = torch.tensor([0, 0, 1, 1, 2, 2])
    for seed in range(20):
        z_used, _, new_recon, target, _ = swap_batch(
            z_mixed, make_features(b), ids, None, 0.5, torch.Generator().manual_seed(seed), recon=recon, partner_z=z
        )
        assert not bool(new_recon[0]) and torch.equal(z_used[0], z_mixed[0]) and target[0] == 0
        swapped = (~new_recon).nonzero().flatten().tolist()
        for i in swapped:
            if i != 0:
                assert torch.equal(z_used[i], z[int(target[i])])  # never the mixed code of element 0


# ---- speaker-relative F0 -----------------------------------------------------------------------------------------


def test_relative_f0_is_a_ratio_and_invertible():
    f = make_features()
    utt = torch.tensor([150.0, 120.0, float("nan"), 200.0])
    rel = relative_f0_features(f, utt, fallback_hz=148.4)
    median = torch.tensor([150.0, 120.0, 148.4, 200.0])
    torch.testing.assert_close(rel[:, F0_CHANNEL] * median[:, None], f[:, F0_CHANNEL])  # f0 = ratio * median
    assert torch.equal(rel[:, :F0_CHANNEL], f[:, :F0_CHANNEL]) and torch.equal(rel[:, F0_CHANNEL + 1 :], f[:, F0_CHANNEL + 1 :])
    voiced = f[:, PERIODICITY_CHANNEL] > 0
    exact = f.clone()  # an utterance whose voiced median is its median: the voiced ratios have median 1
    exact[0, F0_CHANNEL] = 90.0 + torch.arange(10.0)
    exact[0, PERIODICITY_CHANNEL] = 1.0
    median0 = float(exact[0, F0_CHANNEL].median())
    assert float(relative_f0_features(exact, torch.full((4,), median0), 100.0)[0, F0_CHANNEL].median()) == pytest.approx(1.0)
    assert voiced.any()


def test_frontend_log_relative_input():
    fe = FeatureFrontend(STATS, pitch_mode="log_relative")
    f = make_features(2, 5)
    f[:, F0_CHANNEL] = torch.tensor([1.0, np.e ** REL_LOGF0_STD, 1 / np.e ** REL_LOGF0_STD, 2.0, 0.5])
    out = fe(f).x[:, F0_CHANNEL]
    torch.testing.assert_close(out[0], torch.tensor([0.0, 1.0, -1.0, np.log(2) / REL_LOGF0_STD, np.log(0.5) / REL_LOGF0_STD]).float())
    assert fe(f).f0_hz.shape == (2, 1, 5)
    with pytest.raises(ValueError):
        FeatureFrontend(STATS, pitch_mode="nope")


def test_pitch_code_starts_at_zero_gets_gradient_and_ignores_unknown_medians():
    pc = PitchCode(5.0, 0.3, out_dim=8)
    ref = torch.tensor([100.0, 200.0, float("nan"), 0.0])
    assert float(pc(ref).detach().abs().sum()) == 0.0
    pc(ref).sum().backward()
    assert float(pc.proj.weight.grad.abs().sum()) > 0
    with torch.no_grad():
        pc.proj.weight.fill_(1.0)
    out = pc(ref)[:, 0]
    assert float(out[1]) > float(out[0])  # monotone in the median
    assert float(out[2]) == 0.0 and float(out[3]) == 0.0
    assert float(pc.normalize(torch.tensor([float(np.exp(5.0))])).abs()) < 1e-5


# ---- identity loss -----------------------------------------------------------------------------------------------


class FakeSV(torch.nn.Module):
    """Differentiable stand-in: the embedding is a fixed projection of simple waveform statistics."""

    def forward(self, input_values):
        x = input_values
        emb = torch.stack([x.mean(-1), x.std(-1), x.abs().mean(-1), x.pow(2).mean(-1).sqrt()], dim=-1)
        return SimpleNamespace(embeddings=emb)


def test_identity_loss_values_target_index_and_gradient_flow():
    emb = torch.randn(4, 6, requires_grad=True)
    real = torch.randn(4, 6)
    assert float(identity_loss(real, real, torch.arange(4))) == pytest.approx(0.0, abs=1e-6)
    shifted = identity_loss(real[[1, 0, 3, 2]], real, torch.tensor([1, 0, 3, 2]))  # partner's crop is the target
    assert float(shifted) == pytest.approx(0.0, abs=1e-6)
    loss = identity_loss(emb, real, torch.tensor([0, 0, 2, 2]))
    assert loss.shape == () and 0.0 <= float(loss) <= 2.0
    loss.backward()
    assert emb.grad.shape == emb.shape and float(emb.grad.abs().sum()) > 0


def test_verifier_is_differentiable_frozen_and_outside_the_module_tree():
    sv = CropSpeakerVerifier()
    sv._sv.append(FakeSV())
    assert list(sv.parameters()) == [] and sv.state_dict() == {}
    audio = (0.1 * torch.randn(3, 1, HOP * 8)).requires_grad_(True)
    emb = sv(audio)
    assert emb.shape == (3, 4)
    identity_loss(emb, sv(audio.detach()).detach()[[1, 2, 0]], torch.arange(3)).backward()
    assert audio.grad is not None and float(audio.grad.abs().sum()) > 0


# ---- per-speaker F0 statistics and dataset keys ------------------------------------------------------------------


def manual_speaker_medians(store: PackedStore) -> dict[str, float]:
    pools: dict[str, list[float]] = {}
    for u in range(len(store)):
        f, _ = store.frames(u, 0, int(store.T[u]))
        v = (f[:, PERIODICITY_CHANNEL] > 0) & (f[:, F0_CHANNEL] > 0)
        pools.setdefault(str(store.speakers[store.speaker_codes[u]]), []).extend(f[v, F0_CHANNEL].tolist())
    return {s: float(np.median(np.float32(v))) for s, v in pools.items() if v}


def test_speaker_f0_stats_builder_cli_and_loader(cache, tmp_path):
    packed = cache / "packed" / "train.clean.100"
    stats = build_speaker_f0_stats([packed])
    store = PackedStore([packed])
    expected = manual_speaker_medians(store)
    assert set(stats["median_f0"]) == set(expected) and stats["splits"] == ["train.clean.100"]
    for s, v in expected.items():
        assert stats["median_f0"][s] == pytest.approx(v, rel=1e-6)
    out = tmp_path / "x" / "speaker_f0.json"
    dataset_main(["f0-stats", "--cache-root", str(cache), "--splits", "train.clean.100", "--out", str(out)])
    loaded = load_speaker_f0(out, ["100", "nobody", "103"])
    assert loaded.dtype == np.float32 and np.isnan(loaded[1])
    assert loaded[0] == pytest.approx(expected["100"], rel=1e-6) and loaded[2] == pytest.approx(expected["103"], rel=1e-6)
    save_speaker_f0_stats(stats, tmp_path / "y.json")
    assert (tmp_path / "y.json").exists() and not (tmp_path / "y.json.tmp").exists()


def test_median_and_geometric_mean_helpers():
    f = np.zeros((5, N_FEATURES), np.float32)
    f[:, F0_CHANNEL] = [100, 200, 300, 400, 500]
    f[:, PERIODICITY_CHANNEL] = [1, 0, 1, -1, 1]
    assert median_voiced_f0(f) == 300.0
    f[:, PERIODICITY_CHANNEL] = 0
    assert np.isnan(median_voiced_f0(f))
    assert geometric_mean_f0([100.0, 400.0, float("nan")]) == pytest.approx(200.0)
    assert np.isnan(geometric_mean_f0([float("nan")]))


def test_crop_dataset_phase4_keys_and_unchanged_stream(cache, tmp_path):
    packed = cache / "packed" / "train.clean.100"
    stats_file = tmp_path / "f0.json"
    save_speaker_f0_stats(build_speaker_f0_stats([packed]), stats_file)
    plain = CropDataset([packed], crop_frames=8, p_cross=1.0)
    rich = CropDataset([packed], crop_frames=8, p_cross=1.0, f0_stats=stats_file, f0_conditioning=True)
    assert "speaker_id" not in plain[(0, 0, 0)] and "utt_f0" not in plain[(0, 0, 0)]
    store = rich.store
    for pos in range(25):
        a, b = plain[(0, pos, pos % len(plain))], rich[(0, pos, pos % len(plain))]
        for k, v in a.items():  # the sample itself does not depend on the Phase 4 options
            assert torch.equal(v, b[k]), k
        utt, ref = int(b["utt_index"]), int(b["ref_index"])
        assert int(b["speaker_id"]) == store.speaker_codes[utt]
        assert float(b["speaker_f0"]) == pytest.approx(load_speaker_f0(stats_file, store.speakers)[store.speaker_codes[utt]])
        assert float(b["utt_f0"]) == pytest.approx(store.median_f0(utt), nan_ok=True)
        assert float(b["ref_f0"]) == pytest.approx(store.median_f0(ref), nan_ok=True)  # one reference


def test_crop_dataset_multi_reference_f0_is_the_geometric_mean(cache):
    packed = cache / "packed" / "train.clean.100"
    ds = CropDataset([packed], crop_frames=8, p_cross=1.0, n_refs=(2, 3), f0_conditioning=True)
    found = False
    for pos in range(30):
        s = ds[(0, pos, pos % len(ds))]
        if not np.isfinite(float(s["ref_f0"])):
            continue
        found = True
        assert 0 < float(s["ref_f0"]) < 1e4 and s["ref_mask"].shape == (3,)
    assert found


def test_full_utterance_f0_conditioning_per_condition(cache):
    dev = cache / "packed" / "dev.clean"
    for condition in ("T1", "T2", "T3"):
        ds = FullUtteranceDataset(dev, condition=condition, f0_conditioning=True)
        plain = FullUtteranceDataset(dev, condition=condition)
        store = ds.store
        assert "utt_f0" not in plain[0]
        for i in range(len(ds)):
            item = ds[i]
            utt = ds.items[i]
            assert float(item["utt_f0"]) == pytest.approx(store.median_f0(utt), nan_ok=True)
            refs = ds.pools[i] if condition == "T3" else [ds.references[i]]
            want = geometric_mean_f0([store.median_f0(int(r)) for r in refs])
            assert float(item["ref_f0"]) == pytest.approx(want, nan_ok=True)
            if condition == "T1":
                assert float(item["ref_f0"]) == pytest.approx(float(item["utt_f0"]), nan_ok=True)


# ---- training integration (tiny generator) -----------------------------------------------------------------------

SEEN_F0: list[torch.Tensor] = []


class RelGenerator(TinyGenerator):
    """Tiny generator that accepts ``pitch_mode`` and records the F0 channel it is given."""

    def __init__(self, stats, channels: int = 8, pitch_mode: str = "log"):
        super().__init__(stats, channels)
        self.pitch_mode = pitch_mode

    def forward(self, features, spk):
        SEEN_F0.append(features[:, F0_CHANNEL].detach().clone())
        return super().forward(features, spk)


class Phase4Stream(Dataset):
    """Items 0, 1 are speaker 0 and items 2, 3 speaker 1 (batches of 4 start at multiples of 4)."""

    def __getitem__(self, i):
        rng = np.random.default_rng([0, int(i)])
        frames = 6
        feats = rng.standard_normal((15, frames)).astype(np.float32)
        feats[12] = 150.0 + 100.0 * (int(i) // 2 % 2) + 20.0 * rng.random(frames)
        feats[14] = 0.5 + 0.4 * rng.random(frames)
        speaker = int(i) // 2 % 2
        return {
            "features": torch.from_numpy(feats),
            "audio": torch.from_numpy(0.1 * rng.standard_normal((1, HOP * frames)).astype(np.float32)),
            "spk_raw": torch.from_numpy(rng.standard_normal(1024).astype(np.float32)),
            "speaker_id": torch.tensor(speaker),
            "speaker_f0": torch.tensor(170.0 + 100.0 * speaker),
            "utt_f0": torch.tensor(float(np.median(feats[12]))),
            "ref_f0": torch.tensor(170.0 + 100.0 * speaker),
        }


class Phase4Eval(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, i):
        item = Phase4Stream()[i]
        return {**item, "id": f"u{i}", "condition": "T1", "ref_id": f"u{i}"}


class Phase4DataModule(FakeDataModule):
    def train_dataloader(self):
        return DataLoader(Phase4Stream(), batch_sampler=StreamBatches(self), num_workers=0)

    def val_dataloader(self):
        return DataLoader(Phase4Eval(), batch_size=1)

    def predict_dataloader(self):
        return DataLoader(Phase4Eval(), batch_size=1)


def phase4_cfg(tmp_path, monkeypatch, steps=6, warmup=2, **speaker):
    import test_training

    monkeypatch.setattr(test_training, "BATCH", 4)
    cfg = make_cfg(tmp_path / "run", max_g_steps=steps, mel_warmup=warmup)
    cfg.speaker = OmegaConf.merge(cfg.speaker, OmegaConf.create(speaker))
    cfg.vocoder.generator._target_ = f"{__name__}.RelGenerator"
    return cfg


def fit(cfg, monkeypatch):
    from sparc.cli.train_vocoder import run as train_run

    monkeypatch.setattr(CropSpeakerVerifier, "model", lambda self, device: FakeSV())
    rec = Recorder()
    _, module = train_run(cfg, datamodule=Phase4DataModule(), stats=STATS, extra_callbacks=(rec,))
    return rec.rows, module


def test_swap_and_identity_training_logs_and_gradients(tmp_path, monkeypatch):
    cfg = phase4_cfg(tmp_path, monkeypatch, swap={"p": 0.5, "f0_stats": None}, identity={"weight": 1.0})
    rows, module = fit(cfg, monkeypatch)
    assert len(rows) == 6 and all(np.isfinite(v) for r in rows for v in r.values())
    assert [r["train/swap_frac"] for r in rows] == [0.0, 0.0, 0.5, 0.5, 0.5, 0.5]  # none during the mel warm-up
    assert all("train/identity" not in r for r in rows[:2]) and all(0.0 <= r["train/identity"] <= 2.0 for r in rows[2:])
    assert module.verifier is not None and not any("verifier" in k for k in module.state_dict())


def test_identity_only_uses_the_own_crop(tmp_path, monkeypatch):
    cfg = phase4_cfg(tmp_path, monkeypatch, identity={"weight": 0.5})
    rows, module = fit(cfg, monkeypatch)
    assert all("train/swap_frac" not in r for r in rows) and "train/identity" in rows[-1]
    assert module.swap_p == 0.0 and module.identity_weight == 0.5


def test_default_speaker_config_leaves_training_unchanged(tmp_path, monkeypatch):
    cfg = phase4_cfg(tmp_path, monkeypatch)
    rows, module = fit(cfg, monkeypatch)
    assert module.verifier is None and module.pitch_code is None and not module.rel_f0
    assert not any(k.startswith("train/swap") or k == "train/identity" for r in rows for k in r)


def test_relative_f0_generator_input_code_injection_and_validation(tmp_path, monkeypatch):
    SEEN_F0.clear()
    cfg = phase4_cfg(tmp_path, monkeypatch, rel_f0={"enabled": True})
    cfg.vocoder.generator.pitch_mode = "log_relative"
    rows, module = fit(cfg, monkeypatch)
    assert module.pitch_code is not None and any(p is module.pitch_code.proj.weight for p in module.generator_parameters)
    assert all(np.isfinite(v) for r in rows for v in r.values())
    seen = torch.cat([f.flatten() for f in SEEN_F0])
    assert float(seen.max()) < 5.0 and float(seen.min()) > 0.2  # ratios, not Hz (raw values are 150-370)
    assert float(module.pitch_code.proj.weight.abs().sum()) > 0  # trained: the reference pitch level reaches the code
    # synthesis through predict/validation paths uses the same transform and code
    batch = next(iter(Phase4DataModule().val_dataloader()))
    wav = module.synthesize(batch)
    assert wav.shape == (1, 1, 6 * HOP)
    z0, _ = module.speaker_code(batch)
    z1, _ = module.speaker_code({**batch, "ref_f0": batch["ref_f0"] * 1.5})
    assert not torch.allclose(z0, z1)  # the target's pitch level changes the code
    bad = phase4_cfg(tmp_path / "bad", monkeypatch, rel_f0={"enabled": True})
    from sparc.vocoders.training.module import VocoderGANModule

    with pytest.raises(ValueError, match="log_relative"):
        VocoderGANModule(bad, stats=STATS)


def test_relative_f0_with_swap_does_not_scale_f0(tmp_path, monkeypatch):
    SEEN_F0.clear()
    cfg = phase4_cfg(tmp_path, monkeypatch, rel_f0={"enabled": True}, swap={"p": 0.5, "f0_stats": None})
    cfg.vocoder.generator.pitch_mode = "log_relative"
    rows, _ = fit(cfg, monkeypatch)
    seen = torch.cat([f.flatten() for f in SEEN_F0])
    assert rows[-1]["train/swap_frac"] == 0.5 and float(seen.max()) < 5.0  # ratios stay near 1 for swapped crops too


# ---- configs -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["s", "v", "sv", "rf0", "m"])
def test_phase4_experiments_compose_over_b0(arm, monkeypatch):
    from hydra import compose, initialize_config_dir

    for var in ("SPARC_VOC_CACHE", "SPARC_VOC_RUNS", "LIBRITTSR_RAW", "SPARC_REFIT_NPZ"):
        monkeypatch.setenv(var, f"/env/{var}")
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        b0 = compose("vocoder_config", overrides=["experiment=ablation_100k"])
        cfg = compose("vocoder_config", overrides=[f"experiment=p4_{arm}"])
    assert cfg.experiment_name == f"p4_{arm}" and cfg.run_dir == f"/env/SPARC_VOC_RUNS/p4_{arm}/hifigan"
    for key in ("max_g_steps", "mel_warmup_steps", "val_every_g_steps"):
        assert cfg.train[key] == b0.train[key]
    assert cfg.data.batch_size == 16 and cfg.data.crop_frames == 64
    assert cfg.speaker.layer == "l6" and cfg.speaker.p_cross == 0.5 and cfg.speaker.type == "ffn"
    swap_on, ident_on = arm in ("s", "sv"), arm in ("v", "sv")
    assert cfg.speaker.swap.p == (0.25 if swap_on else 0.0)
    assert cfg.speaker.identity.weight == (20.0 if ident_on else 0.0)
    assert cfg.speaker.rel_f0.enabled == (arm == "rf0")
    assert cfg.vocoder.generator.pitch_mode == ("log_relative" if arm == "rf0" else "log")
    assert list(cfg.speaker.n_refs) == [1, 8] if arm == "m" else cfg.speaker.n_refs is None
    assert b0.speaker.swap.p == 0.0 and b0.speaker.identity.weight == 0.0 and not b0.speaker.rel_f0.enabled
    assert OmegaConf.to_container(cfg.loss) == OmegaConf.to_container(b0.loss)


def test_datamodule_passes_phase4_options_to_the_datasets(cache, tmp_path):
    from sparc.vocoders.data.datamodule import VocoderDataModule

    stats_file = tmp_path / "f0.json"
    save_speaker_f0_stats(build_speaker_f0_stats([cache / "packed" / "train.clean.100"]), stats_file)
    base = {
        "paths": {"cache_root": str(cache)}, "stats_path": str(tmp_path / "none.json"), "seed": 0,
        "data": {
            "train_splits": ["train.clean.100"], "val_split": "dev.clean", "val_subset_size": 3, "logging_subset_size": 1,
            "predict_split": "dev.clean", "predict_conditions": ["T1"], "crop_frames": 8, "batch_size": 2,
            "num_workers": 0, "gain_db_range": [-6.0, -1.0], "eval_gain_db": -3.0, "ref_min_dur": 3.0,
        },
    }  # fmt: skip
    speaker = {"layer": "l6", "p_cross": 0.5, "n_refs": None}
    plain = VocoderDataModule(OmegaConf.create({**base, "speaker": speaker}))
    assert "speaker_id" not in plain.train_dataset()[(0, 0, 0)]
    cfg = OmegaConf.create(
        {**base, "speaker": {**speaker, "swap": {"p": 0.25, "f0_stats": str(stats_file)}, "rel_f0": {"enabled": True}}}
    )
    dm = VocoderDataModule(cfg)
    sample = dm.train_dataset()[(0, 0, 0)]
    assert {"speaker_id", "speaker_f0", "utt_f0", "ref_f0"} <= set(sample)
    val = next(iter(dm.val_dataloader()))
    assert {"utt_f0", "ref_f0"} <= set(val) and "speaker_id" not in val


# ---- review fixes ------------------------------------------------------------------------------------------------


def test_identity_grad_ratio_is_logged_on_probe_steps(tmp_path, monkeypatch):
    cfg = phase4_cfg(tmp_path, monkeypatch, identity={"weight": 5.0, "grad_probe_every": 2})
    rows, _ = fit(cfg, monkeypatch)
    probed = [i for i, r in enumerate(rows) if "train/identity_grad_ratio" in r]
    # the probe fires on the update after which g_step (the value logging tests) is a multiple of 2
    assert probed and all(rows[i]["g_step"] % 2 == 0 for i in probed)
    assert probed == [3, 5]  # post-warm-up updates
    assert all(np.isfinite(rows[i]["train/identity_grad_ratio"]) and rows[i]["train/identity_grad_ratio"] > 0 for i in probed)
    assert all("train/identity_grad_ratio" not in r for r in fit(phase4_cfg(tmp_path / "b", monkeypatch, identity={"weight": 5.0}), monkeypatch)[0])


def test_relative_f0_synthesize_without_utt_f0_fails_loudly(tmp_path, monkeypatch):
    cfg = phase4_cfg(tmp_path, monkeypatch, rel_f0={"enabled": True})
    cfg.vocoder.generator.pitch_mode = "log_relative"
    _, module = fit(cfg, monkeypatch)
    batch = dict(next(iter(Phase4DataModule().val_dataloader())))
    del batch["utt_f0"]
    with pytest.raises(KeyError, match="utt_f0"):
        module.synthesize(batch)


def test_crop_dataset_rejects_a_partial_f0_file(cache, tmp_path):
    packed = cache / "packed" / "train.clean.100"
    stats = build_speaker_f0_stats([packed])
    ok = tmp_path / "ok.json"
    save_speaker_f0_stats(stats, ok)
    bad = tmp_path / "bad.json"
    save_speaker_f0_stats({**stats, "median_f0": {}}, bad)
    CropDataset([packed], crop_frames=8, f0_stats=ok)
    with pytest.raises(ValueError, match="no median F0"):
        CropDataset([packed], crop_frames=8, f0_stats=bad)


def test_verifier_takes_its_directory_from_the_argument():
    assert CropSpeakerVerifier("/x/y").sv_dir == Path("/x/y")
