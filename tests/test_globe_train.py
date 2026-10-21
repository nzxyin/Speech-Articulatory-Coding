"""Phase 4 arm G: GLOBE v3 training cache (listing with the ECAPA leak filter, packing), mixed packed directories."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from sparc.vocoders.constants import N_FEATURES, SAMPLE_RATE, feature_length
from sparc.vocoders.data.datamodule import VocoderDataModule
from sparc.vocoders.data.dataset import CropDataset
from sparc.vocoders.features import globe_train as gt
from sparc.vocoders.features.cache import ENPLUS_DIM, file_sha256, utt_path, utterance_seed, write_utt
from sparc.vocoders.features.pack import INDEX_COLUMNS
from test_data import build_split, make_cfg, train_rows
from test_training import CONF

SHARD = "train-00000-of-00181"
REAL_INDEX = Path("/data/user_data/xoy/sparc-vocoders/cache/packed/train.clean.100/index.parquet")


def unit(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    return np.array([np.cos(a), np.sin(a)], dtype=np.float32)


# ---- list ---------------------------------------------------------------------------------------------------------


def test_max_cosine_normalizes_and_takes_the_largest_pool_similarity():
    train = np.stack([3.0 * unit(0), 0.5 * unit(90), unit(45)])
    pool = np.stack([7.0 * unit(0), 2.0 * unit(180)])
    got = gt.max_cosine(train, pool, chunk=2)
    assert np.allclose(got, [1.0, 0.0, np.cos(np.pi / 4)], atol=1e-6)


def test_leak_filter_drops_at_or_above_the_threshold_and_reports_per_accent():
    table = pd.DataFrame({"stem": list("abcd"), "accent": ["X", "X", "Y", "Y"]})
    kept, report = gt.leak_filter(table, np.array([0.69, 0.70, 0.99, 0.10]), 0.70)
    assert kept["stem"].tolist() == ["a", "d"]
    assert report["dropped"] == 2 and report["kept"] == 2 and report["total"] == 4
    assert report["per_accent"] == {
        "X": {"total": 2, "dropped": 1, "kept": 1},
        "Y": {"total": 2, "dropped": 1, "kept": 1},
    }


def write_globe_fixture(tmp_path: Path):
    """Four train utterances in two parquet shards and two pool utterances; ECAPA angles decide who leaks."""
    root = tmp_path / "globe"
    (root / "ecapa").mkdir(parents=True)
    train_angle = {
        f"{SHARD}-000001": 0.0,
        f"{SHARD}-000003": 60.0,
        "train-00001-of-00181-000000": 85.0,
        "train-00001-of-00181-000002": 120.0,
    }
    pool_angle = {"dev-00000-of-00003-000000": 5.0, "dev-00000-of-00003-000001": 200.0}
    angle = {**train_angle, **pool_angle}
    names = list(angle)
    for file, part, scale in (("a.npz", names[:3], 2.0), ("b.npz", names[3:], 0.3)):  # unnormalized on purpose
        np.savez(root / "ecapa" / file, stems=np.array(part), emb=np.stack([scale * unit(angle[k]) for k in part]))
    accents = ["American", "American", "Irish", "Irish"]
    pd.DataFrame(
        {
            "stem": list(train_angle),
            "split": "train",
            "shard": [s.rsplit("-", 1)[0] for s in train_angle],
            "row": [str(int(s.rsplit("-", 1)[1])) for s in train_angle],
            "snr": "100.0",
            "accent": accents,
            "transcript": 'say "hi"',
        }
    ).to_csv(root / "manifest.tsv", sep="\t", index=False)
    train = tmp_path / "train.tsv"
    pd.DataFrame({"stem": list(train_angle), "source": "globe_v3", "speaker": "unknown", "accent": accents}).to_csv(
        train, sep="\t", index=False
    )
    pool_table = tmp_path / "pool.parquet"
    pd.DataFrame({"id": list(pool_angle)}).to_parquet(pool_table)
    return root, train, pool_table


def test_build_list_excludes_leaking_utterances(tmp_path):
    root, train, pool_table = write_globe_fixture(tmp_path)
    table, report = gt.build_list(train, root, "/raw", pool_table, threshold=0.70)
    # best cosine to the pool (5 and 200 degrees): 0 deg 0.996 (leaks), 60 deg 0.57, 85 deg 0.17, 120 deg 0.17
    assert table["stem"].tolist() == [f"{SHARD}-000003", "train-00001-of-00181-000000", "train-00001-of-00181-000002"]
    assert list(table.columns) == list(gt.LIST_COLUMNS) and (table["max_cos_pool"] < 0.70).all()
    assert report["dropped"] == 1 and report["kept"] == 3
    assert report["per_accent"]["American"] == {"total": 2, "dropped": 1, "kept": 1}
    assert table["parquet"].iloc[0] == f"/raw/{SHARD}.parquet" and table["row"].tolist() == [3, 0, 2]
    assert not table["stem"].str.startswith("dev-").any()  # the pool is never part of the training list
    strict, _ = gt.build_list(train, root, "/raw", pool_table, threshold=0.5)  # the threshold is a parameter
    assert strict["stem"].tolist() == ["train-00001-of-00181-000000", "train-00001-of-00181-000002"]


def test_build_list_requires_ecapa_for_every_utterance(tmp_path):
    root, train, pool_table = write_globe_fixture(tmp_path)
    pd.DataFrame({"id": ["dev-99999-of-00003-000000"]}).to_parquet(pool_table)
    with pytest.raises(ValueError, match="no ECAPA embedding"):
        gt.build_list(train, root, "/raw", pool_table)


def test_shard_assignment_is_round_robin_over_sorted_parquet_shards():
    shards = ["b", "a", "a", "d", "c", "e"]
    assert gt.shard_assignment(shards, 0, 2) == ["a", "c", "e"]
    assert gt.shard_assignment(shards, 1, 2) == ["b", "d"]
    with pytest.raises(ValueError):
        gt.shard_assignment(shards, 2, 2)


# ---- pack ---------------------------------------------------------------------------------------------------------


def fake_cache(tmp_path: Path, seconds: dict[str, float], seed: int = 0):
    """The list, cache files that pass ``check_arrays`` and a meta.json, as ``extract`` leaves them."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "globe_train"
    paths = gt.out_paths(root)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"fake": 1}))
    meta_hash = file_sha256(meta)
    table = pd.DataFrame(
        {
            "stem": list(seconds),
            "shard": [s.rsplit("-", 1)[0] for s in seconds],
            "row": [int(s.rsplit("-", 1)[1]) for s in seconds],
            "accent": "A",
            "source": "globe_v3",
            "speaker_orig": "unknown",
            "parquet": "/raw/x.parquet",
            "max_cos_pool": 0.1,
        }
    )
    paths["list"].parent.mkdir(parents=True)
    table.to_parquet(paths["list"], index=False)
    truth = {}
    for stem, sec in seconds.items():
        n24 = int(round(sec * SAMPLE_RATE))
        T = feature_length(n24)
        arrays = {
            "feats": rng.standard_normal((T, N_FEATURES)).astype(np.float32),
            "loud_raw": rng.random(T).astype(np.float32),
            "spk_l0": rng.standard_normal(1024).astype(np.float32),
            "spk_l6": rng.standard_normal(1024).astype(np.float32),
            "spk_enplus64": rng.standard_normal(ENPLUS_DIM).astype(np.float32),
            "spk_wsum": np.float64(12.5),
            "spk_fallback": np.bool_(False),
            "n24": n24,
            "T": T,
            "peak24": 0.8,
            "seed": utterance_seed(stem),
            "meta_hash": meta_hash,
            "gpu": "cpu",
        }
        write_utt(utt_path(paths["utt"], stem.rsplit("-", 1)[0], stem), arrays)
        wav = paths["wav"] / stem.rsplit("-", 1)[0] / f"{stem}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav, rng.uniform(-0.02, 0.02, n24), SAMPLE_RATE, subtype="PCM_16")
        truth[stem] = arrays
    return root, meta, truth


def test_pack_matches_the_librittsr_layout_and_content(tmp_path):
    seconds = {f"{SHARD}-000004": 2.0, f"{SHARD}-000002": 3.5, "train-00001-of-00181-000001": 1.0}
    root, meta, truth = fake_cache(tmp_path, seconds)
    out = gt.run_pack(root, meta_path=meta)
    libri = build_split(tmp_path / "libri", "train.clean.100", train_rows(), seed=1)
    assert out == root / "packed" / "globe.train"
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in libri.iterdir())
    for name in ("feats", "loud_raw", "spk_l0", "spk_l6", "spk_enplus64"):
        a, b = np.load(out / f"{name}.npy"), np.load(libri / f"{name}.npy")
        assert a.dtype == b.dtype and a.shape[1:] == b.shape[1:], name
    index, ref = pd.read_parquet(out / "index.parquet"), pd.read_parquet(libri / "index.parquet")
    assert list(index.columns) == list(INDEX_COLUMNS) and set(ref.columns) <= set(INDEX_COLUMNS)  # fixture lacks split, seed...
    # build_split writes T / duration as int64 / float64; the real LibriTTS-R index (manifest dtypes) has int32 / float32
    for col in set(ref.columns) - {"T", "duration"}:
        assert index[col].dtype == ref[col].dtype, col
    assert index["T"].dtype == np.int32 and index["duration"].dtype == np.float32
    if REAL_INDEX.is_file():
        assert index.dtypes.astype(str).to_dict() == pd.read_parquet(REAL_INDEX).dtypes.astype(str).to_dict()
    assert index["id"].tolist() == sorted(truth)
    assert (index["speaker"] == "globe_" + index["id"]).all() and (index["chapter"] == index["id"]).all()
    assert (index["split"] == "globe.train").all() and index["speaker"].is_unique
    feats = np.load(out / "feats.npy")
    for row in index.itertuples():
        t = truth[row.id]
        assert np.array_equal(feats[row.offset : row.offset + row.T], t["feats"]) and row.n24 == t["n24"]
        assert row.wav_path == str(root / "wav24" / row.id.rsplit("-", 1)[0] / f"{row.id}.wav")
    assert (index["offset"].to_numpy() == np.concatenate([[0], np.cumsum(index["T"])[:-1]])).all()


def test_pack_skips_not_encodable_and_rejects_other_missing_files(tmp_path):
    root, meta, truth = fake_cache(tmp_path, {f"{SHARD}-000001": 2.0, f"{SHARD}-000002": 2.0})
    paths = gt.out_paths(root)
    table = pd.read_parquet(paths["list"])
    short = table.iloc[:1].assign(stem=f"{SHARD}-000009", row=9)
    absent = table.iloc[:1].assign(stem=f"{SHARD}-000010", row=10)
    pd.concat([table, short, absent]).to_parquet(paths["list"], index=False)
    gt.write_json_atomic(paths["errors"] / f"not_encodable_{SHARD}.json", {f"{SHARD}-000009": "too short"})
    with pytest.raises(FileNotFoundError, match="1 cache files are missing"):
        gt.run_pack(root, meta_path=meta)
    out = gt.run_pack(root, allow_missing=True, meta_path=meta)
    assert pd.read_parquet(out / "index.parquet")["id"].tolist() == sorted(truth)
    assert (paths["errors"] / "pack_not_encodable.txt").read_text() == f"{SHARD}-000009\n"
    assert (paths["errors"] / "pack_missing.txt").read_text() == f"{SHARD}-000010\n"


def test_pack_refuses_cache_files_from_another_meta(tmp_path):
    root, meta, _ = fake_cache(tmp_path, {f"{SHARD}-000001": 2.0})
    meta.write_text(json.dumps({"fake": 2}))
    with pytest.raises(ValueError, match="different meta.json"):
        gt.run_pack(root, meta_path=meta)


# ---- mixed packed directories -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    base = tmp_path_factory.mktemp("mixed")
    root, meta, _ = fake_cache(base, {f"{SHARD}-000001": 4.0, f"{SHARD}-000002": 5.0, f"{SHARD}-000003": 4.5})
    globe = gt.run_pack(root, meta_path=meta)
    libri = build_split(base / "libri", "train.clean.100", train_rows(), seed=1)
    dev = [("200_1_000000_000000", "200", "1", 4.0), ("200_1_000001_000000", "200", "1", 4.0)]
    build_split(base / "libri", "dev.clean", dev, seed=2)
    return base / "libri", libri, globe


def test_datamodule_accepts_absolute_split_paths(mixed):
    cache_root, libri, globe = mixed
    dm = VocoderDataModule(make_cfg(cache_root, train_splits=["train.clean.100", str(globe)]))
    assert dm._packed("train.clean.100") == libri and dm._packed(str(globe)) == globe
    dataset = dm.train_dataset()
    n_libri = len(pd.read_parquet(libri / "index.parquet"))
    assert len(dataset.store) == n_libri + 3 and dataset.store.packed_dirs == [libri, globe]
    assert next(iter(dm.val_dataloader()))["features"].shape[0] == 1  # relative val_split still resolves


def test_singleton_speakers_reference_themselves_over_mixed_directories(mixed):
    _, libri, globe = mixed
    for p_cross in (0.0, 1.0):
        dataset = CropDataset([libri, globe], stats=None, crop_frames=8, p_cross=p_cross, ref_min_dur=3.0, seed=3)
        store, seen = dataset.store, 0
        for item, utt in enumerate(dataset.items):
            if not str(store.ids[utt]).startswith("train-"):
                continue
            for epoch in range(4):
                sample = dataset[(epoch, 0, item)]
                assert int(sample["ref_index"]) == int(utt) and not bool(sample["is_cross"])
                assert np.array_equal(sample["spk_raw"].numpy(), np.asarray(store.speaker_vector(int(utt))))
            seen += 1
        assert seen == 3
    # LibriTTS speakers with several utterances still draw other utterances at p_cross = 1
    crossed = [
        bool(dataset[(e, 0, i)]["is_cross"]) for i, u in enumerate(dataset.items) if "_" in str(store.ids[u]) for e in range(4)
    ]
    assert any(crossed)


def test_p4_g_composes_over_the_b0_recipe(monkeypatch):
    from hydra import compose, initialize_config_dir

    for var in ("SPARC_VOC_CACHE", "SPARC_VOC_RUNS", "LIBRITTSR_RAW", "SPARC_REFIT_NPZ"):
        monkeypatch.setenv(var, f"/env/{var}")
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        b0 = compose("vocoder_config", overrides=["experiment=ablation_100k"])
        g = compose("vocoder_config", overrides=["experiment=p4_g"])
    assert g.experiment_name == "p4_g" and g.run_dir == "/env/SPARC_VOC_RUNS/p4_g/hifigan"
    assert list(g.data.train_splits[:3]) == list(b0.data.train_splits) and len(g.data.train_splits) == 4
    assert g.data.train_splits[3] == "/data/user_data/xoy/spkemb/phase4/globe_train/packed/globe.train"
    assert g.stats_path == b0.stats_path == "/env/SPARC_VOC_CACHE/stats/train_stats.json"
    assert OmegaConf.to_container(g.train) == OmegaConf.to_container(b0.train)
    assert OmegaConf.to_container(g.speaker) == OmegaConf.to_container(b0.speaker)
    assert g.data.batch_size == 16 and g.data.crop_frames == 64 and g.seed == b0.seed
    assert OmegaConf.to_container(g.loss) == OmegaConf.to_container(b0.loss)
