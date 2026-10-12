"""``predict.skip_existing``: resumable synthesis through the existing predict path (CPU, seconds)."""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf

import sparc
from sparc.cli import predict_vocoder
from sparc.vocoders.constants import HOP
from sparc.vocoders.data.datamodule import VocoderDataModule
from sparc.vocoders.eval import io
from toy_split import SPLIT, build_toy_cache

CONF = Path(sparc.__file__).parent / "conf"


class FakeVocoder(LightningModule):
    """Writes a constant waveform of the right length; records which utterances it was asked to synthesize."""

    calls: list[tuple[str, str]] = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        n = batch["features"].shape[-1]
        FakeVocoder.calls.append((batch["id"][0], batch["condition"][0]))
        return {"id": batch["id"], "condition": batch["condition"], "wav": torch.full((1, 1, HOP * n), 0.25)}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    build_toy_cache(tmp_path)
    FakeVocoder.calls = []
    monkeypatch.setattr(predict_vocoder, "VocoderGANModule", lambda cfg, stats=None: FakeVocoder())
    ckpt = tmp_path / "step.ckpt"
    torch.save({"state_dict": {}}, ckpt)
    data = OmegaConf.load(CONF / "data" / "librittsr_filtered.yaml")
    data.update(predict_split=SPLIT, eval_num_workers=0)

    def make(skip: bool, conditions=("T1", "T2", "T3"), limit=None):
        cfg = OmegaConf.create(
            {
                "data": {**OmegaConf.to_container(data), "predict_conditions": list(conditions), "predict_limit": limit},
                "paths": {"cache_root": str(tmp_path / "cache")},
                "speaker": {"layer": "l6"},
                "seed": 0,
                "run_dir": str(tmp_path / "run"),
                "predict": {"ckpt": str(ckpt), "output_dir": str(tmp_path / "out"), "precision": "32-true", "skip_existing": skip},
                "trainer": {"trainer": {"accelerator": "cpu"}},
            }
        )
        return cfg

    return tmp_path, make


def dataset_ids(loader) -> list[str]:
    return [loader.dataset[i]["id"] for i in range(len(loader.dataset))]


def test_default_config_keeps_todays_behaviour():
    assert OmegaConf.load(CONF / "vocoder_config.yaml").predict.skip_existing is False


def test_without_skip_every_utterance_is_predicted_even_if_a_wav_exists(setup):
    root, make = setup
    cfg = make(skip=False, conditions=("T1",))
    io.write_wav(root / "out" / SPLIT / "T1" / "300_1_000000_000000.wav", np.zeros(5, np.float32), 24000)
    dm = VocoderDataModule(cfg)
    (loader,) = dm.predict_dataloader()
    assert len(dataset_ids(loader)) == 20
    assert dm.predict_output_dir() is None


def test_skip_existing_drops_exactly_the_existing_ids(setup):
    root, make = setup
    out = root / "out" / SPLIT
    existing = {"T1": ["300_1_000000_000000", "301_1_000001_000000"], "T2": ["300_1_000000_000000"], "T3": []}
    for condition, ids in existing.items():
        for uid in ids:
            io.write_wav(out / condition / f"{uid}.wav", np.zeros(5, np.float32), 24000)
    (out / "T1" / "302_1_000000_000000.wav.tmp").write_bytes(b"half")  # a leftover temporary file is not a finished WAV
    full = VocoderDataModule(make(skip=False)).predict_dataloader()
    partial = VocoderDataModule(make(skip=True)).predict_dataloader()
    for condition, loader_full, loader_part in zip(("T1", "T2", "T3"), full, partial):
        all_ids, kept = dataset_ids(loader_full), dataset_ids(loader_part)
        assert kept == [i for i in all_ids if i not in existing[condition]]
        assert len(all_ids) - len(kept) == len(existing[condition])
    # the references of the remaining items do not change when others are dropped
    for loader_full, loader_part in zip(full[1:2], partial[1:2]):
        refs = {loader_full.dataset[i]["id"]: loader_full.dataset[i]["ref_id"] for i in range(len(loader_full.dataset))}
        for i in range(len(loader_part.dataset)):
            item = loader_part.dataset[i]
            assert refs[item["id"]] == item["ref_id"]


def test_skip_existing_composes_with_the_limit_subset(setup):
    root, make = setup
    dm = VocoderDataModule(make(skip=True, conditions=("T1",), limit=6))
    subset = dm.predict_ids()
    assert len(subset) == 6
    io.write_wav(root / "out" / SPLIT / "T1" / f"{subset[0]}.wav", np.zeros(5, np.float32), 24000)
    (loader,) = dm.predict_dataloader()
    assert dataset_ids(loader) == subset[1:]


def test_finished_conditions_get_no_loader_and_nothing_to_do(setup):
    root, make = setup
    table = io.build_utterance_table(root / "cache", SPLIT)
    for condition in ("T1", "T2", "T3"):
        for uid in io.condition_ids(table, condition):
            io.write_wav(root / "out" / SPLIT / condition / f"{uid}.wav", np.zeros(5, np.float32), 24000)
    dm = VocoderDataModule(make(skip=True))
    assert dm.predict_dataloader() == [] and not dm.has_pending_prediction()
    # one missing T3 file: only that condition is predicted, the others have no loader
    (root / "out" / SPLIT / "T3" / "303_7_000001_000000.wav").unlink()
    loaders = dm.predict_dataloader()
    assert len(loaders) == 1 and dataset_ids(loaders[0]) == ["303_7_000001_000000"] and dm.has_pending_prediction()


def test_run_finishes_what_is_missing_and_a_second_run_does_nothing(setup):
    root, make = setup
    cfg = make(skip=True)
    out = predict_vocoder.run(cfg)
    table = io.build_utterance_table(root / "cache", SPLIT)
    for condition in ("T1", "T2", "T3"):
        ids = io.condition_ids(table, condition)
        assert sorted(p.stem for p in (out / condition).glob("*.wav")) == ids
    wav, rate = sf.read(out / "T1" / "300_1_000000_000000.wav", dtype="float32")
    assert rate == 24000 and wav.shape == (HOP * int(table.set_index("id").loc["300_1_000000_000000", "T"]),)
    first_calls = len(FakeVocoder.calls)
    assert first_calls == 20 + 19 + 19

    stamp = {p: p.stat().st_mtime_ns for p in out.rglob("*.wav")}
    (out / "T2" / "301_1_000001_000000.wav").unlink()
    (out / "T3" / "300_2_000001_000000.wav").unlink()
    FakeVocoder.calls = []
    predict_vocoder.run(cfg)
    assert sorted(FakeVocoder.calls) == [("300_2_000001_000000", "T3"), ("301_1_000001_000000", "T2")]
    for path, ns in stamp.items():
        if path.exists() and path.name not in ("301_1_000001_000000.wav", "300_2_000001_000000.wav"):
            assert path.stat().st_mtime_ns == ns  # untouched

    FakeVocoder.calls = []
    os.remove(root / "step.ckpt")  # nothing pending: the checkpoint is not even loaded
    assert predict_vocoder.run(cfg) == out and FakeVocoder.calls == []
    assert not list(out.rglob("*.tmp"))


def test_run_without_skip_rewrites_everything(setup):
    root, make = setup
    cfg = make(skip=False, conditions=("T1",))
    predict_vocoder.run(cfg)
    FakeVocoder.calls = []
    predict_vocoder.run(cfg)
    assert len(FakeVocoder.calls) == 20
