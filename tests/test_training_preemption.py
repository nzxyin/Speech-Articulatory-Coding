"""Regression tests for #15: step counting, RNG checkpointing, resume, requeue plumbing, toggle_optimizer."""
import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from sparc.cli import train as train_cli
from sparc.training.lightning_module import (
    OPTIMIZERS_PER_BATCH, SparcVocoderTraining, batches_to_global_steps,
)

TINY = dict(channels=16, resblock_kernel_sizes=[3], resblock_dilations=[[1, 3, 5]])


class FakeData(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {
            "art": torch.randn(16, 14, generator=g),
            "spk_raw": torch.randn(1024, generator=g),
            "audio": torch.randn(16 * 320, generator=g).clamp(-1, 1),
        }


def _model():
    return SparcVocoderTraining(generator_config=TINY, log_audio_every_n_steps=10**9)


def _trainer(root, max_batches, every=None):
    cb = ModelCheckpoint(
        dirpath=str(root), save_last=True,
        every_n_train_steps=batches_to_global_steps(every) if every else None,
        save_on_exception=True,
    )
    return pl.Trainer(
        max_steps=batches_to_global_steps(max_batches), accelerator="cpu", devices=1,
        default_root_dir=str(root), logger=False, callbacks=[cb],
        enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1,
    )


def _loader():
    return DataLoader(FakeData(), batch_size=2, shuffle=True, num_workers=0)


def test_step_conversion():
    assert OPTIMIZERS_PER_BATCH == 2
    assert batches_to_global_steps(1500000) == 3000000
    assert batches_to_global_steps(None) is None


def test_steps_rng_and_resume():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        pl.seed_everything(0)
        m = _model()
        t = _trainer(root, max_batches=3, every=1)
        t.fit(m, _loader())
        # 3 batches == 6 optimizer steps; batch_step reports batches
        assert t.global_step == 6 and m.batch_step == 3
        assert t.fit_loop.epoch_loop._batches_that_stepped == 3
        ck = torch.load(root / "last.ckpt", weights_only=False)
        assert ck["global_step"] == 6
        assert set(ck["sparc_rng_state"]) >= {"python", "numpy", "torch"}
        # every_n_train_steps=1 batch fired at the final batch (global step 6)
        assert [p.name for p in root.glob("epoch=*-step=*.ckpt")] == ["epoch=0-step=6.ckpt"]  # save_top_k=1

        # resume: continues from step 6 to 5 batches (10 optimizer steps)
        m2 = _model()
        t2 = _trainer(root, max_batches=5)
        t2.fit(m2, _loader(), ckpt_path=str(root / "last.ckpt"))
        assert t2.global_step == 10 and m2.batch_step == 5


def test_rng_restore():
    m = _model()
    m.trainer = SimpleNamespace(world_size=1)  # on_load_checkpoint only reads world_size
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    ck = {}
    m.on_save_checkpoint(ck)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    m.on_load_checkpoint(ck)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


def test_discriminator_gets_no_grad_in_generator_step():
    m = _model()
    seen = {}
    orig = m.untoggle_optimizer

    def spy(opt):
        # right before untoggling the generator optimizer, discriminator params must be frozen
        if any(p is next(m.generator.parameters()) for g in opt.param_groups for p in g["params"]):
            seen["d_frozen"] = all(not p.requires_grad for p in m.mpd.parameters())
            seen["g_trainable"] = all(p.requires_grad for p in m.generator.parameters())
        return orig(opt)

    m.untoggle_optimizer = spy
    t = _trainer(Path(tempfile.mkdtemp()), max_batches=1)
    t.fit(m, _loader())
    assert seen == {"d_frozen": True, "g_trainable": True}, seen
    assert all(p.requires_grad for p in m.mpd.parameters())  # restored afterwards


def test_resume_discovery_and_run_name():
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d)
        cfg = SimpleNamespace(resume_from_checkpoint=None, run_name=None)
        assert train_cli._find_resume_ckpt(cfg, ck, None) is None  # unnamed run: no auto-resume
        assert train_cli._find_resume_ckpt(cfg, ck, "r") is None   # nothing saved yet
        (ck / "last.ckpt").write_bytes(b"x")
        hpc = ck / "hpc_ckpt_1.ckpt"
        hpc.write_bytes(b"y")
        os.utime(ck / "last.ckpt", (1, 1))
        assert train_cli._find_resume_ckpt(cfg, ck, "r") == str(hpc)  # newest wins
        cfg.resume_from_checkpoint = "/explicit.ckpt"
        assert train_cli._find_resume_ckpt(cfg, ck, "r") == "/explicit.ckpt"

    saved = dict(os.environ)
    try:
        for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME"):
            os.environ.pop(k, None)
        cfg = SimpleNamespace(run_name=None)
        assert train_cli._resolve_run_name(cfg) is None
        os.environ.update(SLURM_JOB_ID="123", SLURM_JOB_NAME="sparc_train")
        assert train_cli._resolve_run_name(cfg) == "slurm_123"
        os.environ["SLURM_JOB_NAME"] = "interactive"
        assert train_cli._resolve_run_name(cfg) is None
        assert train_cli._resolve_run_name(SimpleNamespace(run_name="x")) == "x"
        # importing the CLI must not overwrite SLURM_JOB_NAME (that disabled Lightning's requeue plugin)
        assert os.environ["SLURM_JOB_NAME"] == "interactive"
    finally:
        os.environ.clear()
        os.environ.update(saved)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
