"""Phase 6 speaker-encoder inputs: composition, the screen's pooling, side files, the packed store, the speaker FFN,
the cloning-evaluation path and the nine composed configs."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sparc.vocoders.data.dataset import CropDataset, FullUtteranceDataset, PackedStore
from sparc.vocoders.eval.clone_pools import speaker_vector
from sparc.vocoders.features import spk_phase6 as P6
from sparc.vocoders.models.speaker import SpeakerFFN
from sparc.vocoders.training.speaker_terms import build_speaker
from test_data import EVAL_ROWS, build_split, train_rows

CONF = Path(P6.__file__).resolve().parents[2] / "conf"
DIMS = {"l6": 1024, "l1l6": 2048, "vus3_l6": 3072, "l6_rep3": 3072}


def synthetic_utterance(seed: int, frames: int = 120, silent: bool = False):
    rng = np.random.default_rng(seed)
    h1, h6 = rng.standard_normal((2, frames, 1024)).astype(np.float32)
    per = (rng.random(frames) > 0.5).astype(np.float32) * rng.uniform(0.5, 1.0, frames).astype(np.float32)
    zloud = rng.standard_normal(frames).astype(np.float32)
    if silent:  # no silence frames: the class must back off
        zloud = np.abs(zloud) + 1.0
    return h1, h6, per, zloud


def test_compose_input_widths_and_layout():
    l6 = np.arange(1024, dtype=np.float32)
    l1 = -np.arange(1024, dtype=np.float32)
    vus3 = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024)
    assert P6.compose_input("l6", l6) is l6
    out = P6.compose_input("l1l6", l6, l1=l1)
    assert out.shape == (2048,) and np.array_equal(out[:1024], l1) and np.array_equal(out[1024:], l6)
    out = P6.compose_input("vus3_l6", l6, vus3=vus3)
    assert out.shape == (3072,) and np.array_equal(out[1024:2048], vus3[1])
    out = P6.compose_input("l6_rep3", l6)
    assert out.shape == (3072,) and all(np.array_equal(out[i * 1024 : (i + 1) * 1024], l6) for i in range(3))
    batch = P6.compose_input("l6_rep3", torch.randn(5, 4, 1024))  # torch with leading dims
    assert batch.shape == (5, 4, 3072)
    assert P6.compose_input("vus3_l6", torch.zeros(2, 1024), vus3=torch.zeros(2, 3, 1024)).shape == (2, 3072)
    with pytest.raises(ValueError):
        P6.compose_input("l9", l6)


@pytest.mark.parametrize("silent", [False, True])
def test_pool_inputs_match_the_screen_module_and_numpy(silent):
    from sparc.vocoders.spkscreen.data import FrameBatch
    from sparc.vocoders.spkscreen.pooling import build

    h1, h6, per, zloud = synthetic_utterance(3, silent=silent)
    out = P6.pool_inputs(h1, h6, per, zloud)
    t = h6.shape[0]
    batch = FrameBatch(
        torch.from_numpy(h6)[None, None], torch.from_numpy(per)[None], torch.from_numpy(zloud)[None],
        torch.zeros(1, t), torch.zeros(1, t), None, torch.ones(1, t, dtype=torch.bool),
    )  # fmt: skip
    module = build("seg_vus3")
    assert module.layers == (6,)
    assert np.allclose(module.features(batch)[0].numpy(), out["vus3"].reshape(-1), atol=1e-6, rtol=0)
    w = per.astype(np.float64)
    assert np.allclose(out["l6"], (w[:, None] * h6).sum(0) / w.sum(), atol=1e-5)
    assert np.allclose(out["l1"], (w[:, None] * h1).sum(0) / w.sum(), atol=1e-5)
    voiced = per >= 0.5
    assert np.allclose(out["vus3"][0], h6[voiced].mean(0), atol=1e-5)
    if silent:  # silence class has fewer than 3 frames: back-off to the periodicity-weighted mean
        assert np.array_equal(out["vus3"][2], out["l6"])
    else:
        sil = (per < 0.5) & (zloud < 0.1)
        assert np.allclose(out["vus3"][2], h6[sil].mean(0), atol=1e-5)


def test_pool_inputs_truncates_to_the_shorter_stream():
    h1, h6, per, zloud = synthetic_utterance(5, frames=100)
    a = P6.pool_inputs(h1, h6, per[:90], zloud[:90])
    b = P6.pool_inputs(h1[:90], h6[:90], per, zloud)
    for k in a:
        assert np.array_equal(a[k], b[k])


def write_side(root: Path, tag: str, ids: list[str], seed: int):
    rng = np.random.default_rng(seed)
    d = root / tag
    d.mkdir(parents=True)
    l1 = rng.standard_normal((len(ids), 1024)).astype(np.float32)
    vus3 = rng.standard_normal((len(ids), 3, 1024)).astype(np.float32)
    np.save(d / "l1.npy", l1)
    np.save(d / "vus3.npy", vus3)
    (d / "ids.txt").write_text("\n".join(ids) + "\n")
    return l1, vus3


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("p6")
    train = build_split(root, "train.clean.100", train_rows(), seed=1)
    dev = build_split(root, "dev.clean", EVAL_ROWS, seed=2)
    import pandas as pd

    ids_train = pd.read_parquet(train / "index.parquet")["id"].astype(str).tolist()
    ids_dev = pd.read_parquet(dev / "index.parquet")["id"].astype(str).tolist()
    side = root / "side"
    # the side tags hold the rows in a different order than the packed dirs use for subsets
    l1_t, v_t = write_side(side, "train.clean.100", ids_train, 11)
    l1_d, v_d = write_side(side, "dev.clean", ids_dev, 12)
    return dict(root=root, train=train, dev=dev, side=side, ids_train=ids_train, ids_dev=ids_dev, l1=l1_t, vus3=v_t, l1_d=l1_d, vus3_d=v_d)


def test_default_l6_store_is_bit_identical_to_the_packed_array(world):
    store = PackedStore([world["train"]], "l6")
    raw = np.load(world["train"] / "spk_l6.npy")
    assert store.side is None
    for u in range(len(store)):
        v = store.speaker_vector(u)
        assert v.dtype == np.float32 and np.array_equal(v, raw[u])
    refs, mask = store.speaker_references([0, 3, 3], 8)
    assert refs.shape == (8, 1024) and np.array_equal(refs[1], raw[3]) and mask.tolist() == [True] * 3 + [False] * 5


@pytest.mark.parametrize("layer", ["l1l6", "vus3_l6", "l6_rep3"])
def test_store_phase6_vectors(world, layer):
    store = PackedStore([world["train"], world["dev"]], layer, side_root=world["side"])
    l6 = np.concatenate([np.load(world["train"] / "spk_l6.npy"), np.load(world["dev"] / "spk_l6.npy")])
    l1 = np.concatenate([world["l1"], world["l1_d"]])
    vus3 = np.concatenate([world["vus3"], world["vus3_d"]])
    for u in range(len(store)):
        v = store.speaker_vector(u)
        assert v.shape == (DIMS[layer],) and v.dtype == np.float32
        assert np.array_equal(v, P6.compose_input(layer, l6[u], l1[u], vus3[u]))
    refs, mask = store.speaker_references([1, 2], 8)
    assert refs.shape == (8, DIMS[layer]) and mask.sum() == 2 and not refs[2:].any()


def test_store_finds_side_rows_by_id_for_subset_splits_and_rejects_missing(world, tmp_path):
    import pandas as pd

    store_full = PackedStore([world["train"]], "l1l6", side_root=world["side"])
    keep = world["ids_train"][3:9]
    # a store over a directory with another row order: write a reordered copy of the index
    sub = tmp_path / "sub"
    sub.mkdir()
    index = pd.read_parquet(world["train"] / "index.parquet")
    for name in ("feats.npy", "loud_raw.npy", "spk_l0.npy", "spk_l6.npy", "spk_enplus64.npy"):
        (sub / name).symlink_to(world["train"] / name)
    index.iloc[::-1].reset_index(drop=True).to_parquet(sub / "index.parquet")
    spk = np.load(world["train"] / "spk_l6.npy")[::-1].copy()
    (sub / "spk_l6.npy").unlink()
    np.save(sub / "spk_l6.npy", spk)
    store_rev = PackedStore([sub], "l1l6", side_root=world["side"])
    for uid in keep:
        assert np.array_equal(store_rev.speaker_vector(store_rev.index_of(uid)), store_full.speaker_vector(store_full.index_of(uid)))
    other = tmp_path / "side_missing"
    write_side(other, "train.clean.100", world["ids_train"][:5], 1)
    with pytest.raises(KeyError, match="no Phase 6 vectors"):
        PackedStore([world["train"]], "vus3_l6", side_root=other)
    PackedStore([world["train"]], "l6_rep3", side_root=other)  # needs no side file


def test_crop_dataset_and_full_utterance_dataset_widths(world):
    for layer, dim in DIMS.items():
        ds = CropDataset([world["train"]], crop_frames=20, speaker_layer="l6", n_refs=(1, 8), seed=3)
        if layer != "l6":
            ds.store = PackedStore([world["train"]], layer, side_root=world["side"])
        item = ds[(0, 0, 0)]
        assert item["spk_raw"].shape == (8, dim) and item["ref_mask"].dtype == torch.bool
        full = FullUtteranceDataset(world["dev"], condition="T2", speaker_layer="l6", multi_ref=True)
        if layer != "l6":
            full.store = PackedStore([world["dev"]], layer, side_root=world["side"])
        assert full[0]["spk_raw"].shape[-1] == dim


def test_l6_sample_is_unchanged_by_the_option(world):
    a = CropDataset([world["train"]], crop_frames=20, speaker_layer="l6", n_refs=(1, 8), seed=3)
    raw = np.load(world["train"] / "spk_l6.npy")
    for item in range(len(a)):
        s = a[(1, item, item)]
        k = int(s["ref_mask"].sum())
        assert k >= 1 and np.array_equal(s["spk_raw"][0].numpy(), raw[int(s["ref_index"])])


STATS = {"spk_l6_mean": np.linspace(-1, 1, 1024).tolist(), "spk_l6_std": np.linspace(0.5, 2, 1024).tolist()}


def p6_stats_file(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "stats.npz"
    np.savez(
        path, l1_mean=rng.standard_normal(1024), l1_std=rng.uniform(0.5, 2, 1024),
        vus3_mean=rng.standard_normal(3072), vus3_std=rng.uniform(0.5, 2, 3072),
    )  # fmt: skip
    return path


@pytest.mark.parametrize("layer", list(DIMS))
def test_build_speaker_widths_and_z_score(layer, tmp_path):
    cfg = OmegaConf.create({"speaker": {"type": "ffn", "layer": layer, "dropout": 0.2, "input_stats": str(p6_stats_file(tmp_path))}})
    ffn = build_speaker(cfg, STATS)
    assert isinstance(ffn, SpeakerFFN)
    assert ffn.net[0].in_features == DIMS[layer] and ffn.net[0].out_features == 1024 and ffn.net[-1].out_features == 64
    assert ffn.mean.shape == (DIMS[layer],) and (ffn.std > 0).all()
    z, _ = ffn.embed(torch.randn(3, DIMS[layer]))
    assert z.shape == (3, 64)
    z, _ = ffn.embed(torch.randn(3, 8, DIMS[layer]), torch.ones(3, 8, dtype=torch.bool))
    assert z.shape == (3, 64)
    if layer == "l6_rep3":  # the repeat of E3 normalizes as the layer 6 mean does, three times
        m6 = np.asarray(STATS["spk_l6_mean"], np.float32)
        assert np.allclose(ffn.mean.numpy(), np.tile(m6, 3))


def test_default_speaker_is_the_unchanged_ffn():
    cfg = OmegaConf.create({"speaker": {"type": "ffn", "layer": "l6", "dropout": 0.2}})
    torch.manual_seed(0)
    built = build_speaker(cfg, STATS)
    torch.manual_seed(0)
    direct = SpeakerFFN(dropout=0.2, mean=STATS["spk_l6_mean"], std=STATS["spk_l6_std"])
    assert built.state_dict().keys() == direct.state_dict().keys()
    for k, v in built.state_dict().items():
        assert torch.equal(v, direct.state_dict()[k]), k


def test_clone_path_uses_the_same_composition():
    h1, h6, per, zloud = synthetic_utterance(9)
    out = P6.pool_inputs(h1, h6, per, zloud)
    enc = {"spk_l6": out["l6"], "spk_l1": out["l1"], "spk_vus3": out["vus3"]}
    for layer in ("l1l6", "vus3_l6", "l6_rep3"):
        v = speaker_vector(enc, layer)
        assert v.shape == (DIMS[layer],) and v.dtype == np.float32
        assert np.array_equal(v, P6.compose_input(layer, out["l6"], out["l1"], out["vus3"]))
    assert np.array_equal(speaker_vector(enc, "l6"), out["l6"])


def test_input_stats_concatenation(tmp_path):
    path = p6_stats_file(tmp_path)
    z = np.load(path)
    m, s = P6.input_stats("l1l6", STATS, path)
    assert m.shape == (2048,) and np.allclose(m[:1024], z["l1_mean"]) and np.allclose(s[1024:], STATS["spk_l6_std"])
    m, s = P6.input_stats("vus3_l6", STATS, path)
    assert m.shape == (3072,) and np.allclose(s, z["vus3_std"])
    m, s = P6.input_stats("l6", STATS)
    assert np.allclose(m, STATS["spk_l6_mean"])


# ------------------------------------------------------------------------------------------------ configs


ENV = {
    "SPARC_VOC_CACHE": "/tmp/p6cache", "SPARC_VOC_RUNS": "/tmp/p6runs", "SPARC_REFIT_NPZ": "/tmp/refit.npz",
    "LIBRITTSR_RAW": "/tmp/raw", "LIBRITTSR_FILTERED": "/tmp/filtered", "REFIT_LINEAR": "/tmp/refit.pkl",
}  # fmt: skip


def compose(name: str, monkeypatch):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return OmegaConf.to_container(
            hydra_compose(config_name="vocoder_config", overrides=["vocoder=hifigan", f"experiment={name}"]), resolve=True
        )


@pytest.mark.parametrize("p5", ["r", "f", "fv"])
@pytest.mark.parametrize("arm,layer", [("e1", "l1l6"), ("e2", "vus3_l6"), ("e3", "l6_rep3")])
def test_p6_config_equals_its_p5_arm_except_the_speaker_input(p5, arm, layer, monkeypatch):
    base = compose(f"p5_{p5}", monkeypatch)
    cfg = compose(f"p6_{arm}_p5_{p5}", monkeypatch)
    assert cfg["experiment_name"] == f"p6_{arm}_p5_{p5}" and cfg["speaker"]["layer"] == layer
    assert cfg["seed"] == 0 and cfg["train"]["max_g_steps"] == 100000 and cfg["vocoder"]["name"] == "hifigan"
    assert cfg["run_dir"].split("/")[-2:] == [f"p6_{arm}_p5_{p5}", "hifigan"]
    for c in (base, cfg):
        c["experiment_name"] = ""
        c["run_dir"] = ""
        c["predict"]["output_dir"] = ""
        c["speaker"]["layer"] = ""
    assert base == cfg
