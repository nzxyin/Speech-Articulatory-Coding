"""Tests for the vocoder data pipeline: gain, resumable sampler, crop and full-utterance datasets, data module."""

import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import sparc
from sparc.vocoders.constants import HOP, LOUDNESS_CHANNEL, N_FEATURES, SAMPLE_RATE, feature_length
from sparc.vocoders.data.datamodule import (
    VocoderDataModule,
    select_logging_ids,
    select_validation_ids,
)
from sparc.vocoders.data.dataset import (
    CropDataset,
    FullUtteranceDataset,
    PackedStore,
    evaluation_pool,
    reference_manifest,
)
from sparc.vocoders.data.gain import apply_gain, gain_factor, sample_gain_db
from sparc.vocoders.data.sampler import ResumableSampler

CROP = 20
BURST_AMPLITUDE = 0.5
BURST_SAMPLES = 120
NOISE_AMPLITUDE = 0.02
CONF_DATA = Path(sparc.__file__).parent / "conf" / "data" / "librittsr_filtered.yaml"


def utterance_rows(speaker: int, chapters: dict[int, list[float]]) -> list[tuple[str, str, str, float]]:
    rows = []
    for chapter, durations in chapters.items():
        for k, seconds in enumerate(durations):
            rows.append((f"{speaker}_{chapter}_{k:06d}_000000", str(speaker), str(chapter), seconds))
    return rows


def build_split(root: Path, split: str, rows: list[tuple[str, str, str, float]], seed: int = 0) -> Path:
    """Writes a packed split (contract 1.5) with real wav files whose bursts mark feature channel 0."""
    rng = np.random.default_rng(seed)
    out = root / "packed" / split
    wav_dir = root / "wav" / split
    out.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    feats, loud, records = [], [], []
    offset = 0
    for uid, speaker, chapter, seconds in rows:
        n24 = int(round(seconds * SAMPLE_RATE))
        T = feature_length(n24)
        audio = rng.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE, n24)
        marker = np.zeros(T, dtype=np.float32)
        if T >= 6:
            frame = int(rng.integers(2, T - 2))
            audio[HOP * frame : HOP * frame + BURST_SAMPLES] = BURST_AMPLITUDE
            marker[frame] = 1.0
        path = wav_dir / f"{uid}.wav"
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        audio = sf.read(path, dtype="float32")[0]
        f = rng.standard_normal((T, N_FEATURES)).astype(np.float32)
        f[:, 0] = marker
        feats.append(f)
        loud.append(np.abs(audio[: HOP * T]).reshape(T, HOP).mean(axis=1).astype(np.float32))
        records.append(
            dict(
                id=uid, speaker=speaker, chapter=chapter, wav_path=str(path), n24=n24, T=T, offset=offset,
                peak24=float(np.abs(audio).max()), seed=0, duration=n24 / SAMPLE_RATE,
                spk_wsum=float(rng.uniform(5.0, 50.0)), spk_fallback=False,
            )
        )  # fmt: skip
        offset += T
    n = len(rows)
    np.save(out / "feats.npy", np.concatenate(feats))
    np.save(out / "loud_raw.npy", np.concatenate(loud))
    np.save(out / "spk_l0.npy", rng.standard_normal((n, 1024)).astype(np.float32))
    np.save(out / "spk_l6.npy", rng.standard_normal((n, 1024)).astype(np.float32))
    np.save(out / "spk_enplus64.npy", rng.standard_normal((n, 64)).astype(np.float32))
    pd.DataFrame(records).to_parquet(out / "index.parquet")
    return out


def train_rows() -> list[tuple[str, str, str, float]]:
    rows = []
    rows += utterance_rows(100, {1: [1.0, 3.5, 4.5, 2.0], 2: [3.2, 6.0, 1.5]})
    rows += utterance_rows(101, {1: [3.5, 4.0, 5.0, 2.5]})
    rows += utterance_rows(102, {5: [4.0], 6: [1.2, 0.3], 7: [2.0, 2.2]})
    rows += utterance_rows(103, {9: [6.0]})
    rows += utterance_rows(104, {3: [1.5, 2.0], 4: [5.0, 3.0, 3.1]})
    return rows


EVAL_ROWS = (
    utterance_rows(200, {1: [4.0, 5.0, 3.5, 1.0], 2: [3.0, 6.0, 2.0]})  # two chapters, mixed durations
    + utterance_rows(201, {1: [4.0, 5.0, 3.5, 6.0]})  # single chapter: same-chapter fallback
    + utterance_rows(202, {1: [5.0]})  # single utterance: no reference
    + utterance_rows(203, {1: [1.0, 2.0], 2: [1.5, 2.5]})  # nothing >= 3 s: relaxed
    + utterance_rows(204, {1: [1.0, 1.0], 2: [0.9]})  # relaxed with ties
)


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("cache")
    build_split(root, "train.clean.100", train_rows(), seed=1)
    build_split(root, "dev.clean", EVAL_ROWS, seed=2)
    return root


@pytest.fixture(scope="module")
def train_dir(cache_root) -> Path:
    return cache_root / "packed" / "train.clean.100"


@pytest.fixture(scope="module")
def dev_dir(cache_root) -> Path:
    return cache_root / "packed" / "dev.clean"


def make_cfg(cache_root: Path, **data) -> OmegaConf:
    base = OmegaConf.load(CONF_DATA)
    base.update(
        train_splits=["train.clean.100"], val_split="dev.clean", predict_split="dev.clean", crop_frames=CROP,
        batch_size=4, num_workers=0, eval_num_workers=0, val_subset_size=12, logging_subset_size=3,
    )  # fmt: skip
    base.update(data)
    return OmegaConf.create(
        {
            "data": base,
            "paths": {"cache_root": str(cache_root)},
            "speaker": {"layer": "l6", "p_cross": 0.5},
            "seed": 7,
            "stats_path": str(cache_root / "stats" / "train_stats.json"),
        }
    )


def oracle_reference(rows, target_id: str, min_dur: float = 3.0):
    """Plain-Python statement of the T2 rule from the speaker notes (section 4.3): returns (pool ids, flags)."""
    table = {uid: (spk, chap, round(sec * SAMPLE_RATE)) for uid, spk, chap, sec in rows}
    spk, chap, _ = table[target_id]
    same = sorted(u for u, v in table.items() if v[0] == spk and u != target_id)
    if not same:
        return None
    pool = [u for u in same if table[u][2] >= round(min_dur * SAMPLE_RATE)]
    relaxed = not pool
    if relaxed:
        other = [u for u in same if table[u][1] != chap]
        source = other or same
        best = max(table[u][2] for u in source)
        pool = [next(u for u in sorted(source) if table[u][2] == best)]
    other_chapter = [u for u in pool if table[u][1] != chap]
    if other_chapter:
        return other_chapter, False, relaxed
    return pool, True, relaxed


def oracle_draw(pool: list[str], target_id: str, relaxed: bool) -> str:
    if relaxed:
        return pool[0]
    key = int.from_bytes(hashlib.sha1(target_id.encode()).digest()[:8], "big")
    return pool[int(np.random.default_rng([0, key]).integers(len(pool)))]


# ----------------------------------------------------------------------------------------------- gain


def test_sample_gain_db_range_and_rounding():
    rng = np.random.default_rng(0)
    values = np.array([sample_gain_db(rng) for _ in range(2000)])
    assert values.min() >= -6.0 and values.max() <= -1.0
    assert np.allclose(values, np.round(values, 2))
    assert values.min() < -5.5 and values.max() > -1.5
    assert sample_gain_db(np.random.default_rng(5)) == sample_gain_db(np.random.default_rng(5))


def test_gain_factor_peak_normalization_is_exact():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(24000) * 0.1).astype(np.float32)
    peak = float(np.abs(x).max())
    for db in (-6.0, -3.0, -1.0):
        y = apply_gain(x, gain_factor(peak, db))
        assert y.dtype == np.float32
        assert float(np.abs(y).max()) == pytest.approx(10 ** (db / 20), rel=1e-6)
    with pytest.raises(ValueError):
        gain_factor(0.0, -3.0)


def test_crop_gain_and_loudness_match_full_file_recomputation(train_dir):
    dataset = CropDataset([train_dir], crop_frames=CROP, seed=3)
    store = dataset.store
    for key in [(0, 5, 0), (1, 9, 4), (2, 13, 7)]:
        sample = dataset[key]
        utt = int(dataset.items[key[2]])
        db = float(sample["gain_db"])
        full, _ = sf.read(store.wav_paths[utt], dtype="float64")
        normalized = full / np.abs(full).max() * 10 ** (round(db, 2) / 20)
        frames_all = np.load(train_dir / "loud_raw.npy")[store.offset[utt] : store.offset[utt] + store.T[utt]]
        burst = sample["features"][0].numpy()
        assert burst.sum() in (0.0, 1.0)
        # locate the crop: the audio crop must be a slice of the normalized file
        crop = sample["audio"][0].numpy().astype(np.float64)
        starts = [
            s
            for s in range(0, len(normalized) - len(crop) + 1, HOP)
            if np.allclose(normalized[s : s + len(crop)], crop, atol=2e-7)
        ]
        assert len(starts) == 1
        t = starts[0] // HOP
        g = gain_factor(store.peak24[utt], round(db, 2))
        expected = (frames_all[t : t + CROP].astype(np.float64) * g).astype(np.float32)
        assert np.array_equal(sample["features"][LOUDNESS_CHANNEL].numpy(), expected)


# --------------------------------------------------------------------------------------------- sampler


def take(sampler: ResumableSampler, n: int) -> list:
    it = iter(sampler)
    return [next(it) for _ in range(n)]


def test_sampler_epochs_are_permutations_and_keys_are_stable():
    s = ResumableSampler(10, seed=4)
    keys = take(s, 25)
    for epoch in range(2):
        chunk = keys[10 * epoch : 10 * (epoch + 1)]
        assert sorted(k[2] for k in chunk) == list(range(10))
        assert [k[1] for k in chunk] == list(range(10))
        assert {k[0] for k in chunk} == {epoch}
    assert keys == take(ResumableSampler(10, seed=4), 25)
    assert keys != take(ResumableSampler(10, seed=5), 25)


@pytest.mark.parametrize("k", [0, 1, 9, 10, 17, 23])
def test_sampler_resume_equals_uninterrupted_stream(k):
    full = take(ResumableSampler(10, seed=4), 40)
    first = ResumableSampler(10, seed=4)
    assert take(first, k) == full[:k]
    first.set_samples_consumed(k)
    state = first.state_dict()
    resumed = ResumableSampler(10, seed=4)
    resumed.load_state_dict(state)
    assert resumed.samples_consumed == k
    assert take(resumed, 40 - k) == full[k:]
    other = ResumableSampler(10, seed=4)
    other.set_samples_consumed(k)
    assert take(other, 40 - k) == full[k:]


def test_sampler_iterator_captures_start_when_created():
    s = ResumableSampler(10, seed=4)
    full = take(s, 12)
    it = iter(s)
    s.set_samples_consumed(5)
    assert next(it) == full[0]
    assert next(iter(s)) == full[5]


def test_sampler_ddp_ranks_stride_one_permutation_without_overlap():
    world, n = 4, 22
    samplers = [ResumableSampler(n, seed=1, rank=r, world_size=world) for r in range(world)]
    assert all(s.epoch_length == n // world for s in samplers)
    perm = np.random.default_rng([1, 0]).permutation(n)
    per_rank = [take(s, 5) for s in samplers]
    items = [k[2] for keys in per_rank for k in keys]
    assert len(set(items)) == 20 and set(items) == set(perm[:20].tolist())
    for r, keys in enumerate(per_rank):
        assert [k[1] for k in keys] == [j * world + r for j in range(5)]
        assert [k[2] for k in keys] == [int(perm[j * world + r]) for j in range(5)]
    resumed = ResumableSampler(n, seed=1, rank=2, world_size=world)
    resumed.set_samples_consumed(12)
    assert take(resumed, 8) == take(samplers[2], 11)[3:]


def test_sampler_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ResumableSampler(10, 0, rank=2, world_size=2)
    with pytest.raises(ValueError):
        ResumableSampler(3, 0, rank=0, world_size=4)
    s = ResumableSampler(10, 0, world_size=2)
    with pytest.raises(ValueError):
        s.set_samples_consumed(3)


# ------------------------------------------------------------------------------------------ crop data


def test_packed_store_and_item_filter(train_dir):
    store = PackedStore([train_dir])
    assert len(store) == len(train_rows())
    assert store.index_of("101_1_000000_000000") == [r[0] for r in train_rows()].index("101_1_000000_000000")
    short = store.T < CROP
    dataset = CropDataset([train_dir], crop_frames=CROP)
    assert short.any() and len(dataset) == int((~short).sum())
    assert all(store.T[u] >= CROP for u in dataset.items)
    with pytest.raises(ValueError):
        PackedStore([train_dir], speaker_layer="l3")


def test_crop_features_and_audio_are_aligned(train_dir):
    dataset = CropDataset([train_dir], crop_frames=CROP, seed=0, p_cross=0.0)
    sampler = ResumableSampler(len(dataset), seed=0)
    starts_seen, checked = {}, 0
    for key in take(sampler, 400):
        sample = dataset[key]
        utt = int(dataset.items[key[2]])
        assert sample["features"].shape == (N_FEATURES, CROP) and sample["features"].dtype == torch.float32
        assert sample["audio"].shape == (1, HOP * CROP) and sample["audio"].dtype == torch.float32
        level = 10 ** (float(sample["gain_db"]) / 20)
        burst = np.flatnonzero(np.abs(sample["audio"][0].numpy()) > 0.5 * level)
        marker = np.flatnonzero(sample["features"][0].numpy() == 1.0)
        if len(burst):
            assert len(marker) == 1
            assert burst[0] // HOP == marker[0] == burst[-1] // HOP
            assert burst[0] % HOP == 0
            checked += 1
        else:
            assert len(marker) == 0
        t = int(dataset.store.T[utt])
        starts_seen.setdefault(utt, set())
        if len(marker):
            frame = int(np.flatnonzero(dataset.store.frames(utt, 0, t)[0][:, 0] == 1.0)[0])
            start = frame - int(marker[0])
            assert 0 <= start <= t - CROP
            starts_seen[utt].add(start)
    assert checked > 30
    assert max(len(v) for v in starts_seen.values()) >= 4


def test_crop_is_a_pure_function_of_the_key(train_dir):
    a = CropDataset([train_dir], crop_frames=CROP, seed=3)
    b = CropDataset([train_dir], crop_frames=CROP, seed=3)
    c = CropDataset([train_dir], crop_frames=CROP, seed=4)
    key = (2, 11, 5)
    x, y, z = a[key], b[key], c[key]
    for name in x:
        assert torch.equal(x[name], y[name]), name
    assert not torch.equal(x["audio"], z["audio"]) or not torch.equal(x["gain_db"], z["gain_db"])
    assert not torch.equal(x["audio"], a[(3, 11, 5)]["audio"])
    assert int(x["position"]) == 11


def test_cross_probability_and_reference_rule(train_dir):
    for p_cross in (0.0, 0.3, 0.5, 1.0):
        dataset = CropDataset([train_dir], crop_frames=CROP, seed=1, p_cross=p_cross)
        store = dataset.store
        eligible = np.array([(store.speaker_codes == store.speaker_codes[u]).sum() > 1 for u in dataset.items])
        keys = [(e, p, i) for e in range(40) for p, i in enumerate(range(len(dataset)))]
        cross = []
        for key in keys:
            utt = int(dataset.items[key[2]])
            sample = dataset[key]
            ref = int(sample["ref_index"])
            cross.append(bool(sample["is_cross"]))
            assert bool(sample["is_cross"]) == (ref != utt)
            assert np.array_equal(sample["spk_raw"].numpy(), store.speaker_vector(ref))
            if ref != utt:
                assert store.speaker_codes[ref] == store.speaker_codes[utt]
                same = np.flatnonzero(store.speaker_codes == store.speaker_codes[utt])
                others = [u for u in same if u != utt]
                tier1 = [
                    u
                    for u in others
                    if store.chapter_codes[u] != store.chapter_codes[utt] and store.n24[u] >= 3 * SAMPLE_RATE
                ]
                tier2 = [u for u in others if store.n24[u] >= 3 * SAMPLE_RATE]
                allowed = tier1 or tier2 or others
                assert ref in allowed
        cross = np.array(cross)
        rows_eligible = np.tile(eligible, 40)
        if p_cross == 0.0:
            assert not cross.any()
        elif p_cross == 1.0:
            assert cross[rows_eligible].all() and not cross[~rows_eligible].any()
        else:
            n = int(rows_eligible.sum())
            rate = cross[rows_eligible].mean()
            assert abs(rate - p_cross) < 5 * np.sqrt(p_cross * (1 - p_cross) / n)
            assert not cross[~rows_eligible].any()


def test_speaker_with_one_utterance_references_itself(train_dir):
    dataset = CropDataset([train_dir], crop_frames=CROP, p_cross=1.0)
    ids = dataset.store.ids
    item = [i for i, u in enumerate(dataset.items) if str(ids[u]).startswith("103_")][0]
    sample = dataset[(0, 0, item)]
    assert not bool(sample["is_cross"]) and int(sample["ref_index"]) == int(sample["utt_index"])


def test_fixed_crops_and_subset(train_dir):
    dataset = CropDataset([train_dir], crop_frames=CROP, seed=2, p_cross=0.5, fixed_crops=True, subset_size=5)
    assert len(dataset) == 5
    assert np.array_equal(dataset.items, CropDataset([train_dir], crop_frames=CROP, subset_size=5).items)
    for item in range(5):
        first, again = dataset[(0, item, item)], dataset[(9, 100 + item, item)]
        for name in first:
            if name != "position":
                assert torch.equal(first[name], again[name]), name


# ----------------------------------------------------------------------------------- full utterances


def test_full_utterance_t1(dev_dir):
    dataset = FullUtteranceDataset(dev_dir, gain_db=-3.0, condition="T1")
    assert len(dataset) == len(EVAL_ROWS)
    sample = dataset[3]
    utt = dataset.items[3]
    T = int(dataset.store.T[utt])
    assert sample["features"].shape == (N_FEATURES, T) and sample["audio"].shape == (1, HOP * T)
    assert float(sample["audio"].abs().max()) <= 10 ** (-3 / 20) * (1 + 1e-6)
    assert sample["id"] == EVAL_ROWS[3][0] and sample["ref_id"] == sample["id"] and sample["condition"] == "T1"
    assert np.array_equal(sample["spk_raw"].numpy(), dataset.store.speaker_vector(utt))
    g = gain_factor(dataset.store.peak24[utt], -3.0)
    loud = np.load(dev_dir / "loud_raw.npy")[dataset.store.offset[utt] : dataset.store.offset[utt] + T]
    assert np.array_equal(sample["features"][LOUDNESS_CHANNEL].numpy(), apply_gain(loud, g))
    ids = [EVAL_ROWS[5][0], EVAL_ROWS[0][0]]
    subset = FullUtteranceDataset(dev_dir, ids=ids)
    assert [subset[i]["id"] for i in range(2)] == ids


def test_t2_follows_the_reference_rule(dev_dir):
    dataset = FullUtteranceDataset(dev_dir, condition="T2")
    assert dataset.skipped_ids == ["202_1_000000_000000"]
    assert len(dataset) == len(EVAL_ROWS) - 1
    seen_same, seen_relaxed = 0, 0
    for i in range(len(dataset)):
        sample = dataset[i]
        pool, same_chapter, relaxed = oracle_reference(EVAL_ROWS, sample["id"])
        assert sample["ref_id"] == oracle_draw(pool, sample["id"], relaxed), sample["id"]
        assert dataset.flags[i] == (same_chapter, relaxed, len(pool))
        assert sample["ref_id"] != sample["id"] and sample["ref_id"].split("_")[0] == sample["id"].split("_")[0]
        ref = dataset.store.index_of(sample["ref_id"])
        assert np.array_equal(sample["spk_raw"].numpy(), dataset.store.speaker_vector(ref))
        seen_same += same_chapter
        seen_relaxed += relaxed
    assert seen_same > 0 and seen_relaxed > 0


def test_t2_t3_are_deterministic_and_order_independent(cache_root, dev_dir):
    first = FullUtteranceDataset(dev_dir, condition="T2")
    second = FullUtteranceDataset(dev_dir, condition="T2")
    assert [first[i]["ref_id"] for i in range(len(first))] == [second[i]["ref_id"] for i in range(len(second))]
    order = np.random.default_rng(0).permutation(len(EVAL_ROWS))
    shuffled_rows = [EVAL_ROWS[i] for i in order]
    shuffled_dir = build_split(cache_root, "dev.shuffled", shuffled_rows, seed=2)
    pairs = {}
    for name, directory in (("a", dev_dir), ("b", shuffled_dir)):
        dataset = FullUtteranceDataset(directory, condition="T2")
        pairs[name] = {dataset[i]["id"]: dataset[i]["ref_id"] for i in range(len(dataset))}
    assert pairs["a"] == pairs["b"]


def test_t3_is_the_weighted_pool_mean(dev_dir):
    dataset = FullUtteranceDataset(dev_dir, condition="T3")
    again = FullUtteranceDataset(dev_dir, condition="T3")
    store = dataset.store
    for i in range(len(dataset)):
        sample = dataset[i]
        pool, _, _ = oracle_reference(EVAL_ROWS, sample["id"])
        idx = [store.index_of(u) for u in pool]
        vectors = np.stack([store.speaker_vector(u) for u in idx]).astype(np.float64)
        w = store.spk_wsum[idx]
        expected = (w @ vectors / w.sum()).astype(np.float32)
        assert np.array_equal(sample["spk_raw"].numpy(), expected)
        assert np.array_equal(sample["spk_raw"].numpy(), again[i]["spk_raw"].numpy())
        assert sample["ref_id"] == "speaker_mean" and sample["condition"] == "T3"
    assert dataset.skipped_ids == ["202_1_000000_000000"]


def test_evaluation_pool_cases():
    chapters = np.array(["1", "1", "2", "2"])
    long_, short = 72000, 1000
    pool = evaluation_pool(chapters, np.array([long_] * 4), 0, 72000)
    assert pool.members.tolist() == [2, 3] and not pool.same_chapter and not pool.relaxed
    pool = evaluation_pool(chapters, np.array([long_, long_, short, short]), 0, 72000)
    assert pool.members.tolist() == [1] and pool.same_chapter and not pool.relaxed
    pool = evaluation_pool(chapters, np.array([short, 5, 7, 7]), 0, 72000)
    assert pool.members.tolist() == [2] and pool.relaxed and not pool.same_chapter
    assert evaluation_pool(np.array(["1"]), np.array([long_]), 0, 72000) is None


@pytest.mark.skipif("SPARC_VOC_CACHE" not in os.environ, reason="Phase 1 manifests not available")
@pytest.mark.parametrize("split", ["dev.clean", "test.clean"])
def test_t2_rule_reproduces_phase1_reference_manifest(split):
    sources = Path(os.environ["SPARC_VOC_CACHE"]) / "sources"
    if not (sources / f"ref_manifest_{split}.jsonl").is_file():
        pytest.skip("reference manifest missing")
    with open(sources / "manifest_clean.tsv", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = [r for r in reader if r["split"] == split]
    table = pd.DataFrame(
        {
            "id": [r["id"] for r in rows],
            "speaker": [r["speaker"] for r in rows],
            "chapter": [r["chapter"] for r in rows],
            "n24": [int(r["num_samples_24k"]) for r in rows],
        }
    )
    expected = [json.loads(line) for line in open(sources / f"ref_manifest_{split}.jsonl")]
    got = reference_manifest(table.sample(frac=1.0, random_state=1))
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g["id"] == e["id"] and g["skipped"] == e["skipped"]
        if not e["skipped"]:
            assert g["ref_id"] == e["ref_id"], e["id"]
            assert (g["ref_same_chapter"], g["ref_relaxed"], g["n_pool"]) == (
                e["ref_same_chapter"], e["ref_relaxed"], e["n_pool"],
            )  # fmt: skip


# --------------------------------------------------------------------------------------- data module


def test_selection_helpers_are_deterministic(dev_dir):
    index = pd.read_parquet(dev_dir / "index.parquet", columns=["id", "speaker", "n24"])
    logging = select_logging_ids(index, 3, seed=0)
    assert logging == select_logging_ids(index.sample(frac=1.0, random_state=0), 3, seed=0)
    assert len(logging) == len(set(logging)) <= 3
    assert len({i.split("_")[0] for i in logging}) == len(logging)
    n24 = index.set_index("id")["n24"]
    assert all(4 * SAMPLE_RATE <= n24[i] <= 8 * SAMPLE_RATE for i in logging)
    val = select_validation_ids(index, 12, logging, seed=0)
    assert val[: len(logging)] == logging and val[len(logging) :] == sorted(val[len(logging) :])
    assert len(val) == len(set(val)) == 12
    assert val == select_validation_ids(index, 12, logging, seed=0)
    assert len(select_validation_ids(index, 1000, logging, seed=0)) == len(index)


def stream(dm: VocoderDataModule, batches: int) -> list[dict]:
    loader = dm.train_dataloader()
    it = iter(loader)
    return [next(it) for _ in range(batches)]


def test_datamodule_train_batches(cache_root):
    dm = VocoderDataModule(make_cfg(cache_root))
    batch = stream(dm, 1)[0]
    B = 4
    assert set(batch) == {"features", "audio", "spk_raw", "gain_db", "utt_index", "ref_index", "is_cross", "position"}
    assert batch["features"].shape == (B, N_FEATURES, CROP) and batch["features"].dtype == torch.float32
    assert batch["audio"].shape == (B, 1, HOP * CROP) and batch["audio"].dtype == torch.float32
    assert batch["spk_raw"].shape == (B, 1024) and batch["spk_raw"].dtype == torch.float32
    for name in ("gain_db", "utt_index", "ref_index", "is_cross", "position"):
        assert batch[name].shape == (B,), name
    assert batch["is_cross"].dtype == torch.bool and batch["utt_index"].dtype == torch.int64
    assert batch["position"].tolist() == [0, 1, 2, 3]
    assert dm.stats is None


@pytest.mark.parametrize("workers", [0, 2])
def test_datamodule_resume_is_bit_exact(cache_root, workers):
    cfg = make_cfg(cache_root, num_workers=workers)
    full = stream(VocoderDataModule(cfg), 9)
    k = 4
    dm = VocoderDataModule(cfg)
    dm.set_samples_consumed(k * 4)
    assert dm.samples_consumed == k * 4
    resumed = stream(dm, 9 - k)
    for a, b in zip(full[k:], resumed):
        for name in a:
            assert torch.equal(a[name], b[name]), name
    late = VocoderDataModule(cfg)
    loader = late.train_dataloader()
    late.set_samples_consumed(k * 4)
    for a, b in zip(full[k:], [next(iter(loader)) for _ in range(1)]):
        for name in a:
            assert torch.equal(a[name], b[name]), name


def test_datamodule_ranks_draw_disjoint_samples(cache_root):
    cfg = make_cfg(cache_root, batch_size=2)
    r0 = stream(VocoderDataModule(cfg, rank=0, world_size=2), 2)
    r1 = stream(VocoderDataModule(cfg, rank=1, world_size=2), 2)
    p0 = torch.cat([b["position"] for b in r0]).tolist()
    p1 = torch.cat([b["position"] for b in r1]).tolist()
    assert p0 == [0, 2, 4, 6] and p1 == [1, 3, 5, 7]
    resumed = VocoderDataModule(cfg, rank=1, world_size=2)
    resumed.set_samples_consumed(4)
    assert torch.cat([b["position"] for b in stream(resumed, 1)]).tolist() == [5, 7]


def test_datamodule_eval_loaders(cache_root):
    dm = VocoderDataModule(make_cfg(cache_root))
    assert dm.val_ids[: len(dm.logging_ids)] == dm.logging_ids and len(dm.val_ids) == 12
    assert len(dm.logging_ids) == 3
    val = list(dm.val_dataloader())
    assert [b["id"][0] for b in val] == dm.val_ids
    batch = val[0]
    assert batch["features"].shape[:2] == (1, N_FEATURES) and batch["audio"].shape[:2] == (1, 1)
    assert batch["condition"] == ["T1"] and isinstance(batch["id"], list) and isinstance(batch["ref_id"], list)
    assert float(batch["gain_db"]) == -3.0
    predict = dm.predict_dataloader()
    assert [next(iter(loader))["condition"][0] for loader in predict] == ["T1", "T2", "T3"]
    assert [len(loader) for loader in predict] == [len(EVAL_ROWS), len(EVAL_ROWS) - 1, len(EVAL_ROWS) - 1]
    limited = VocoderDataModule(make_cfg(cache_root, predict_limit=6, predict_conditions=["T2"]))
    ids = limited.predict_ids()
    assert len(ids) == 6 and ids == limited.predict_ids()
    assert len(limited.predict_dataloader()) == 1


def test_datamodule_loads_stats_when_present(cache_root, tmp_path):
    cfg = make_cfg(cache_root)
    cfg.stats_path = str(tmp_path / "train_stats.json")
    (tmp_path / "train_stats.json").write_text(json.dumps({"ema_mean": [0.0]}))
    assert VocoderDataModule(cfg).stats == {"ema_mean": [0.0]}


def test_datamodule_fixed_crops_option(cache_root):
    cfg = make_cfg(cache_root, fixed_crops=True, train_subset_size=3, batch_size=3)
    a, b = stream(VocoderDataModule(cfg), 2)
    assert len(VocoderDataModule(cfg).train_dataset()) == 3
    assert torch.equal(a["audio"].sort(dim=0).values, b["audio"].sort(dim=0).values)


# ------------------------------------------------------------------------------------ worker signals


def started_workers(cache_root, workers: int = 2):
    """Iterator over a training loader whose workers have all produced a batch (so their init has finished)."""
    dm = VocoderDataModule(make_cfg(cache_root, num_workers=workers, prefetch_factor=2))
    loader = dm.train_dataloader()
    assert isinstance(loader, DataLoader)
    it = iter(loader)
    for _ in range(2 * workers):
        next(it)
    return loader, it, list(it._workers)


def test_workers_survive_sigterm_and_sigusr1_from_other_processes(cache_root):
    loader, it, workers = started_workers(cache_root)
    for worker in workers:
        for name in ("USR1", "TERM"):
            subprocess.run(["kill", f"-{name}", str(worker.pid)], check=True)
    time.sleep(0.5)
    assert all(worker.is_alive() for worker in workers)
    for _ in range(4):
        assert next(it)["features"].shape[0] == 4


def test_workers_exit_when_the_parent_terminates_them(cache_root):
    loader, it, workers = started_workers(cache_root)
    for worker in workers:
        os.kill(worker.pid, signal.SIGTERM)
    for worker in workers:
        worker.join(timeout=20)
    assert [worker.exitcode for worker in workers] == [0, 0]
    it._shutdown_workers()


CRASH_SCRIPT = """
import sys
from omegaconf import OmegaConf
from sparc.vocoders.data.datamodule import VocoderDataModule
cfg = OmegaConf.load(sys.argv[1])
it = iter(VocoderDataModule(cfg).train_dataloader())
next(it)
raise RuntimeError("crash with live workers")
"""


def test_process_with_live_workers_exits_after_an_exception(cache_root, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(make_cfg(cache_root, num_workers=2), cfg_path)
    start = time.monotonic()
    command = [sys.executable, "-c", CRASH_SCRIPT, str(cfg_path)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 1 and "crash with live workers" in done.stderr
    assert "killed by signal" not in done.stderr
    assert time.monotonic() - start < 100


def test_train_loader_leaves_the_global_torch_rng_alone(cache_root):
    dm = VocoderDataModule(make_cfg(cache_root, num_workers=2))
    before = torch.get_rng_state()
    it = iter(dm.train_dataloader())
    next(it)
    assert torch.equal(before, torch.get_rng_state())


def test_train_loader_refuses_a_distributed_sampler_wrapper(cache_root):
    class Connector:
        use_distributed_sampler = True

    class FakeTrainer:
        global_rank, world_size, _accelerator_connector = 0, 2, Connector()

    dm = VocoderDataModule(make_cfg(cache_root))
    dm.trainer = FakeTrainer()
    with pytest.raises(RuntimeError, match="use_distributed_sampler"):
        dm.train_dataloader()
    Connector.use_distributed_sampler = False
    assert dm.train_dataloader() is not None


def test_frames_outside_the_utterance_are_rejected(train_dir):
    store = PackedStore([train_dir])
    T = int(store.T[0])
    assert store.frames(0, 0, T)[0].shape == (T, N_FEATURES)
    for start, length in ((0, T + 1), (T - 1, 2), (-1, 3)):
        with pytest.raises(IndexError):
            store.frames(0, start, length)


# ------------------------------------------------------------------------------- Lightning integration


class StreamProbe(L.LightningModule):
    """Manual-optimization module that records batch positions and restores the data stream like the real module."""

    def __init__(self, steps: int):
        super().__init__()
        self.automatic_optimization = False
        self.layer = torch.nn.Linear(1, 1)
        self.steps, self.samples_consumed, self.positions, self.val_ids, self.predicted = steps, 0, [], [], []

    def training_step(self, batch, batch_idx):
        self.positions.append(batch["position"].tolist())
        self.samples_consumed += batch["position"].numel() * self.trainer.world_size
        if len(self.positions) >= self.steps:
            self.trainer.should_stop = True

    def validation_step(self, batch, batch_idx):
        self.val_ids.append(batch["id"][0])

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.predicted.append((dataloader_idx, batch["condition"][0], batch["id"][0], batch["ref_id"][0]))

    def configure_optimizers(self):
        return None

    def on_save_checkpoint(self, checkpoint):
        checkpoint["samples_consumed"] = self.samples_consumed

    def on_load_checkpoint(self, checkpoint):
        self.samples_consumed = int(checkpoint["samples_consumed"])
        self.trainer.datamodule.set_samples_consumed(self.samples_consumed)


def lightning_trainer(**kwargs) -> L.Trainer:
    return L.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, use_distributed_sampler=False, max_epochs=-1, **kwargs,
    )  # fmt: skip


@pytest.mark.parametrize("workers", [0, 2])
def test_lightning_fit_resume_validate_predict(cache_root, tmp_path, workers):
    cfg = make_cfg(cache_root, num_workers=workers, eval_num_workers=workers, batch_size=2)
    straight = StreamProbe(steps=6)
    lightning_trainer().fit(straight, datamodule=VocoderDataModule(cfg))
    assert len(straight.positions) == 6
    first = StreamProbe(steps=2)
    trainer = lightning_trainer()
    trainer.fit(first, datamodule=VocoderDataModule(cfg))
    trainer.save_checkpoint(tmp_path / "step.ckpt")
    assert first.positions == straight.positions[:2]
    resumed = StreamProbe(steps=4)
    lightning_trainer().fit(resumed, datamodule=VocoderDataModule(cfg), ckpt_path=str(tmp_path / "step.ckpt"))
    assert resumed.positions == straight.positions[2:]
    dm = VocoderDataModule(cfg)
    val = StreamProbe(steps=1)
    lightning_trainer().validate(val, datamodule=dm)
    assert val.val_ids == dm.val_ids
    predict = StreamProbe(steps=1)
    lightning_trainer().predict(predict, datamodule=dm)
    by_loader = {i: [p for p in predict.predicted if p[0] == i] for i in range(3)}
    assert [len(by_loader[i]) for i in range(3)] == [len(EVAL_ROWS), len(EVAL_ROWS) - 1, len(EVAL_ROWS) - 1]
    assert {p[1] for p in by_loader[1]} == {"T2"} and {p[3] for p in by_loader[2]} == {"speaker_mean"}
