"""Tests of the vocoder training module, callbacks, schedules, dispatcher and CLI helpers (CPU, synthetic data)."""

import os
import signal
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from lightning.pytorch import Callback, LightningDataModule
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sparc.cli.predict_vocoder import run as predict_run
from sparc.cli.train import routes_to_vocoder
from sparc.cli.train_vocoder import run as train_run
from sparc.vocoders.constants import HOP
from sparc.vocoders.models.base import Vocoder
from sparc.vocoders.training.callbacks import (
    capture_rng_state,
    checkpoint_path,
    find_resume_checkpoint,
    list_step_checkpoints,
    restore_rng_state,
)
from sparc.vocoders.training.schedules import WarmupCosine

CONF = Path(__file__).resolve().parents[1] / "src" / "sparc" / "conf"
FRAMES = 6
BATCH = 2


@pytest.fixture(scope="module")
def stats() -> dict:
    return {
        "ema_mean": [0.0] * 12,
        "ema_std": [1.0] * 12,
        "logf0_mean": 5.0,
        "logf0_std": 0.3,
        "loud_log_mean": -5.0,
        "loud_log_std": 1.5,
        "per_mean": 0.5,
        "per_std": 0.4,
        "spk_l0_mean": [0.0] * 1024,
        "spk_l0_std": [1.0] * 1024,
        "spk_l6_mean": [0.0] * 1024,
        "spk_l6_std": [1.0] * 1024,
    }


class TinyGenerator(Vocoder):
    """Conv + transposed conv generator with dropout and a speaker bias."""

    def __init__(self, stats, channels: int = 8):
        super().__init__()
        self.encode = nn.Conv1d(15, channels, 3, padding=1)
        self.drop = nn.Dropout(0.2)
        self.speaker_bias = nn.Linear(64, channels)
        self.decode = nn.ConvTranspose1d(channels, 1, 2 * HOP, HOP, padding=HOP // 2)

    def forward(self, features, spk):
        h = self.drop(torch.relu(self.encode(features / 100.0) + self.speaker_bias(spk)[:, :, None]))
        return torch.tanh(self.decode(h))[..., : features.shape[-1] * HOP]


class TinyDisc(nn.Module):
    """Two-layer convolutional discriminator that records how it is called."""

    calls: list = []

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(1, 4, 41, 20, padding=20)
        self.c2 = nn.Conv1d(4, 1, 3, padding=1)

    def run(self, x):
        h = torch.relu(self.c1(x))
        return self.c2(h), [h]

    def forward(self, y, y_hat):
        if self.training:
            TinyDisc.calls.append(
                {"params_trainable": all(p.requires_grad for p in self.parameters()), "fake_has_grad": y_hat.requires_grad}
            )
        real, real_maps = self.run(y)
        fake, fake_maps = self.run(y_hat)
        return [real], [fake], [real_maps], [fake_maps]


class StreamDataset(Dataset):
    """Item ``i`` is a pure function of the sample counter ``i``."""

    def __getitem__(self, i):
        rng = np.random.default_rng([0, int(i)])
        feats = rng.standard_normal((15, FRAMES)).astype(np.float32)
        feats[12] = 80.0 + 200.0 * rng.random(FRAMES)
        feats[13] = 0.2 * rng.random(FRAMES)
        feats[14] = 0.5 + 0.4 * rng.random(FRAMES)
        return {
            "features": torch.from_numpy(feats),
            "audio": torch.from_numpy(0.1 * rng.standard_normal((1, HOP * FRAMES)).astype(np.float32)),
            "spk_raw": torch.from_numpy(rng.standard_normal(1024).astype(np.float32)),
        }


class StreamBatches:
    """Infinite batch sampler that starts at the data module's sample counter when an iterator is created."""

    def __init__(self, module):
        self.module = module

    def __iter__(self):
        start = self.module.samples_consumed
        while True:
            yield list(range(start, start + BATCH))
            start += BATCH


class EvalDataset(Dataset):
    def __init__(self, frames=(12, 9, 15)):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        rng = np.random.default_rng([1, i])
        n = self.frames[i]
        feats = rng.standard_normal((15, n)).astype(np.float32)
        feats[12] = 100.0 + 50.0 * rng.random(n)
        feats[13] = 0.1 * rng.random(n)
        feats[14] = 0.6
        return {
            "features": torch.from_numpy(feats),
            "audio": torch.from_numpy(0.1 * rng.standard_normal((1, HOP * n)).astype(np.float32)),
            "spk_raw": torch.from_numpy(rng.standard_normal(1024).astype(np.float32)),
            "id": f"utt{i}",
            "condition": "T1",
            "ref_id": f"utt{i}",
        }


class FakeDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.samples_consumed = 0

    def set_samples_consumed(self, n: int) -> None:
        self.samples_consumed = n

    def train_dataloader(self):
        return DataLoader(StreamDataset(), batch_sampler=StreamBatches(self), num_workers=0)

    def val_dataloader(self):
        return DataLoader(EvalDataset(), batch_size=1)

    def predict_dataloader(self):
        return DataLoader(EvalDataset(), batch_size=1)


def small_losses() -> dict:
    return {
        "mel": {
            "_target_": "sparc.vocoders.losses.losses.MelSpectrogramLoss",
            "sample_rate": 24000, "n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 16,
            "f_min": 0.0, "f_max": 12000.0, "power": 1.0, "center": True, "mel_scale": "htk", "clamp": 1.0e-5,
        },
        "mss": {
            "_target_": "sparc.vocoders.losses.losses.MultiScaleSpectralLoss",
            "fft_sizes": [256, 128, 64], "hop_ratio": 0.25, "alpha": 1.0,
        },
    }


def make_cfg(run_dir, max_g_steps=8, mel_warmup=3, backends=(), **train) -> "OmegaConf":
    return OmegaConf.create(
        {
            "experiment_name": "t",
            "seed": 0,
            "stats_path": None,
            "run_dir": str(run_dir),
            "resume": "auto",
            "vocoder": {"name": "tiny", "generator": {"_target_": f"{__name__}.TinyGenerator", "channels": 8}},
            "data": {"logging_subset_size": 2, "predict_split": "test.clean"},
            "loss": {
                "name": "tiny", "adversarial": "hinge", "mel_weight": 45.0, "mss_weight": 0.0, **small_losses(),
                "discriminators": {"tiny": {"module": {"_target_": f"{__name__}.TinyDisc"}, "adv_weight": 1.0, "fm_weight": 1.0}},
            },
            "trainer": {
                "trainer": {
                    "accelerator": "cpu", "devices": 1, "num_nodes": 1, "strategy": "auto", "precision": "32-true",
                    "num_sanity_val_steps": 0, "limit_val_batches": 0, "enable_progress_bar": False,
                    "enable_model_summary": False,
                },
                "logging": {"backends": list(backends), "wandb_mode": "offline", "wandb_project": "t"},
            },
            "optim": {"lr": 1.0e-3, "betas": [0.8, 0.9], "weight_decay": 0.01, "warmup_steps": 2},
            "train": {
                "max_g_steps": max_g_steps, "mel_warmup_steps": mel_warmup, "val_every_g_steps": 1000,
                "log_every_g_steps": 1, "milestone_every_g_steps": 0, "checkpoint_every_minutes": 0,
                "keep_last_checkpoints": 3, **train,
            },
            "speaker": {"layer": "l6", "p_cross": 0.5, "dropout": 0.2},
            "predict": {"ckpt": None, "output_dir": str(Path(run_dir) / "predictions"), "precision": "32-true"},
        }
    )


class Recorder(Callback):
    """Collects the per-step losses of the module."""

    def __init__(self):
        self.rows = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        row = {k: float(v) for k, v in pl_module.last_losses.items()}
        row["g_step"] = pl_module.g_step
        self.rows.append(row)


class SignalAt(Callback):
    """Sends SIGUSR1 to this process while the batch that starts at ``g_step == step`` is being processed."""

    def __init__(self, step: int):
        self.step = step

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if pl_module.g_step == self.step:
            os.kill(os.getpid(), signal.SIGUSR1)


def train(cfg, stats, *callbacks):
    return train_run(cfg, datamodule=FakeDataModule(), stats=stats, extra_callbacks=callbacks)


def test_schedule_warmup_then_cosine():
    sched = WarmupCosine(base_lr=2.0, warmup_steps=4, total_steps=14)
    values = [sched(i) for i in range(14)]
    assert values[0] == pytest.approx(2.0 / 5)
    assert values[3] == pytest.approx(2.0 * 4 / 5)
    assert values[4] == pytest.approx(2.0)
    assert values[9] == pytest.approx(1.0)
    assert values[13] < values[12] < values[5] < 2.0
    assert sched(14) == pytest.approx(0.0, abs=1e-12)
    assert sched(1000) == pytest.approx(0.0, abs=1e-12)


def test_warmup_and_adversarial_steps_advance_counters(tmp_path, stats):
    TinyDisc.calls.clear()
    recorder = Recorder()
    trainer, module = train(make_cfg(tmp_path, max_g_steps=5, mel_warmup=3), stats, recorder)
    assert (module.g_step, module.d_step, module.samples_consumed) == (5, 2, 5 * BATCH)
    assert trainer.global_step == 3 + 2 * 2
    assert [r["g_step"] for r in recorder.rows] == [1, 2, 3, 4, 5]
    assert all("train/d_total" not in r for r in recorder.rows[:3])
    assert all("train/d_total" in r and "train/g_adv/tiny" in r for r in recorder.rows[3:])
    assert all(np.isfinite(v) for r in recorder.rows for v in r.values())
    opt_g, opt_d = trainer.optimizers
    assert opt_g.param_groups[0]["lr"] == pytest.approx(module.sched_g(4))
    assert opt_d.param_groups[0]["lr"] == pytest.approx(module.sched_d(1))
    assert (tmp_path / "status.json").read_text() == '{"finished": true, "g_step": 5}'
    assert checkpoint_path(tmp_path / "ckpt", 5).exists()
    counters = torch.load(checkpoint_path(tmp_path / "ckpt", 5), map_location="cpu", weights_only=False)["vocoder_counters"]
    assert counters == {"g_step": 5, "d_step": 2, "samples_consumed": 10, "last_validated_g_step": 0}


def test_bf16_mixed_precision_steps_are_finite(tmp_path, stats):
    cfg = make_cfg(tmp_path, max_g_steps=4, mel_warmup=2)
    cfg.trainer.trainer.precision = "bf16-mixed"
    recorder = Recorder()
    _, module = train(cfg, stats, recorder)
    assert (module.g_step, module.d_step) == (4, 2)
    assert all(np.isfinite(v) for r in recorder.rows for v in r.values())


def test_multi_device_trainer_uses_ddp_with_unused_parameters(tmp_path):
    from lightning.pytorch.plugins.environments import LightningEnvironment
    from lightning.pytorch.strategies import DDPStrategy

    from sparc.cli.train_vocoder import build_trainer

    cfg = make_cfg(tmp_path)
    cfg.trainer.trainer.devices = 2
    trainer = build_trainer(cfg, tmp_path)
    assert isinstance(trainer.strategy, DDPStrategy) and trainer.strategy._ddp_kwargs["find_unused_parameters"]
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)
    assert trainer.max_steps == -1 and trainer.checkpoint_callbacks == []


def test_discriminator_sees_detached_fake_and_is_frozen_in_generator_step(tmp_path, stats):
    TinyDisc.calls.clear()
    train(make_cfg(tmp_path, max_g_steps=5, mel_warmup=3), stats)
    assert TinyDisc.calls == [
        {"params_trainable": True, "fake_has_grad": False},
        {"params_trainable": False, "fake_has_grad": True},
    ] * 2


def test_sigusr1_checkpoint_and_bit_exact_resume(tmp_path, stats, monkeypatch):
    monkeypatch.delenv("REQUEUE_CMD", raising=False)
    reference = Recorder()
    _, ref_module = train(make_cfg(tmp_path / "a"), stats, reference)

    run_dir = tmp_path / "b"
    marker = tmp_path / "requeued"
    first = Recorder()
    _, interrupted = train(make_cfg(run_dir), stats, first, SignalAt(4))
    assert interrupted.g_step == 5
    assert [s for s, _ in list_step_checkpoints(run_dir / "ckpt")] == [5]
    assert not (run_dir / "ckpt" / "step000000008.ckpt").exists()
    assert (run_dir / "status.json").read_text() == '{"finished": false, "g_step": 5}'
    assert not marker.exists()

    second = Recorder()
    _, resumed = train(make_cfg(run_dir), stats, second)
    assert resumed.g_step == 8 and resumed.d_step == ref_module.d_step
    assert resumed.samples_consumed == ref_module.samples_consumed
    assert first.rows + second.rows == reference.rows
    for (name, a), (_, b) in zip(ref_module.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), name
    assert (run_dir / "status.json").read_text() == '{"finished": true, "g_step": 8}'


def test_requeue_command_runs_only_when_set(tmp_path, stats, monkeypatch):
    marker = tmp_path / "requeued"
    monkeypatch.setenv("REQUEUE_CMD", f"echo $SLURM_JOB_ID > {marker}")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    train(make_cfg(tmp_path / "run"), stats, SignalAt(2))
    assert marker.read_text().strip() == "12345"
    assert [s for s, _ in list_step_checkpoints(tmp_path / "run" / "ckpt")] == [3]


def test_sigterm_also_checkpoints(tmp_path, stats, monkeypatch):
    monkeypatch.delenv("REQUEUE_CMD", raising=False)

    class TermAt(SignalAt):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if pl_module.g_step == self.step:
                os.kill(os.getpid(), signal.SIGTERM)

    _, module = train(make_cfg(tmp_path), stats, TermAt(1))
    assert module.g_step == 2
    assert checkpoint_path(tmp_path / "ckpt", 2).exists()


def test_signal_handlers_are_restored(tmp_path, stats):
    before = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
    train(make_cfg(tmp_path, max_g_steps=2, mel_warmup=1), stats)
    assert {s: signal.getsignal(s) for s in before} == before


def test_periodic_checkpoints_keep_milestones_and_newest(tmp_path, stats):
    cfg = make_cfg(tmp_path, milestone_every_g_steps=4, checkpoint_every_minutes=1e-9, keep_last_checkpoints=2)
    train(cfg, stats)
    assert [s for s, _ in list_step_checkpoints(tmp_path / "ckpt")] == [4, 6, 7, 8]


def test_finished_run_is_not_redone(tmp_path, stats):
    cfg = make_cfg(tmp_path, max_g_steps=3, mel_warmup=1)
    train(cfg, stats)
    recorder = Recorder()
    _, module = train(cfg, stats, recorder)
    assert recorder.rows == [] and module.g_step == 0
    assert (tmp_path / "status.json").read_text() == '{"finished": true, "g_step": 3}'


def test_resume_picks_newest_loadable_by_step(tmp_path):
    for step in (5, 10, 20):
        torch.save({"step": step}, checkpoint_path(tmp_path, step))
    checkpoint_path(tmp_path, 20).write_bytes(b"not a checkpoint")
    now = os.stat(checkpoint_path(tmp_path, 5)).st_mtime
    os.utime(checkpoint_path(tmp_path, 5), (now + 100, now + 100))
    (tmp_path / "step000000030.ckpt.tmp").write_bytes(b"partial")
    step, path = find_resume_checkpoint(tmp_path)
    assert step == 10 and path == checkpoint_path(tmp_path, 10)
    assert find_resume_checkpoint(tmp_path / "missing") is None


def test_rng_state_round_trip():
    state = capture_rng_state()
    expected = (torch.rand(3), np.random.rand(3).tolist(), __import__("random").random())
    torch.rand(5), np.random.rand(5), __import__("random").random()
    restore_rng_state(state)
    again = (torch.rand(3), np.random.rand(3).tolist(), __import__("random").random())
    assert torch.equal(expected[0], again[0]) and expected[1:] == again[1:]


def test_validation_logs_scalars_and_media_and_predict_writes_wavs(tmp_path, stats):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cfg = make_cfg(tmp_path, max_g_steps=4, mel_warmup=2, val_every_g_steps=2, backends=["tensorboard"])
    _, module = train(cfg, stats)
    assert set(module.last_validation) == {"val/mel_l1", "val/mr_stft"}
    assert all(np.isfinite(v) for v in module.last_validation.values())
    acc = EventAccumulator(str(tmp_path / "tb" / "version_0"), size_guidance={"images": 0, "audio": 0})
    acc.Reload()
    tags = acc.Tags()
    assert {"train/mel", "val/mel_l1", "val/mr_stft"} <= set(tags["scalars"])
    assert [e.step for e in acc.Scalars("val/mel_l1")] == [2, 4]
    assert "val/utt0/mel" in tags["images"] and "ref/utt1/mel" in tags["images"]
    assert "val/utt0/audio" in tags["audio"]
    assert "val/utt2/mel" not in tags["images"]

    out = predict_run(cfg, datamodule=FakeDataModule(), stats=stats)
    for i, frames in enumerate(EvalDataset().frames):
        wav, rate = sf.read(out / "T1" / f"utt{i}.wav", dtype="float32")
        assert rate == 24000 and wav.shape == (frames * HOP,) and np.isfinite(wav).all()


@pytest.mark.parametrize(
    "argv",
    [
        ["vocoder=vocos", "experiment=main"],
        ["experiment=smoke", "trainer=smoke", "vocoder.generator.channels=32"],
        ["+paths.cache_root=/x"],
        ["~loss.mss"],
        ["train.max_g_steps=10"],
        ["optim.lr=1e-4", "speaker.layer=l0", "data.batch_size=4", "run_dir=/x"],
        ["--config-name", "vocoder_config"],
        ["--config-name=vocoder_config", "seed=1"],
        ["-cn", "vocoder_config"],
        ["seed=1", "vocoder=hifigan", "hydra.run.dir=\"/x\"", "hydra.job.name=train_ddp_process_1", "hydra.output_subdir=null"],
    ],
)
def test_dispatcher_routes_vocoder_overrides(argv):
    assert routes_to_vocoder(argv)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["dataset=ljspeech"],
        ["dataset=librittsr_dev_clean", "max_steps=100", "batch_size=8"],
        ["--config-name", "train_config", "dataset=vctk"],
        ["lr=1e-4", "device=cpu", "hydra.job.name=train_ddp_process_1"],
        ["--multirun"],
    ],
)
def test_dispatcher_keeps_legacy_overrides(argv):
    assert not routes_to_vocoder(argv)


def test_main_calls_the_selected_path(monkeypatch):
    import sparc.cli.train as cli
    import sparc.cli.train_vocoder as vocoder_cli

    calls = []
    monkeypatch.setattr(cli, "legacy_main", lambda: calls.append("legacy"))
    monkeypatch.setattr(vocoder_cli, "main", lambda: calls.append("vocoder"))
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    monkeypatch.setattr(sys, "argv", ["sparc-train", "vocoder=vocos", "experiment=main"])
    cli.main()
    assert calls == ["vocoder"] and "SLURM_JOB_NAME" not in os.environ
    monkeypatch.setattr(sys, "argv", ["sparc-train", "dataset=ljspeech"])
    cli.main()
    assert calls == ["vocoder", "legacy"] and os.environ["SLURM_JOB_NAME"] == "interactive"
    monkeypatch.delenv("SLURM_JOB_NAME")


def test_vocoder_config_composes_experiments_over_the_root(monkeypatch):
    from hydra import compose, initialize_config_dir

    for var in ("SPARC_VOC_CACHE", "SPARC_VOC_RUNS", "LIBRITTSR_RAW", "SPARC_REFIT_NPZ"):
        monkeypatch.setenv(var, f"/env/{var}")
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        main = compose("vocoder_config", overrides=["experiment=main"])
        pilot = compose(
            "vocoder_config", overrides=["vocoder=vocos", "experiment=precision_pilot", "trainer.trainer.precision=bf16-mixed"]
        )
        ab = compose("vocoder_config", overrides=["vocoder=vocos_a", "experiment=vocos_ab_pilot"])
        smoke = compose("vocoder_config", overrides=["experiment=smoke", "trainer=smoke"])
        overfit = compose("vocoder_config", overrides=["experiment=overfit", "trainer=overfit"])
        default = compose("vocoder_config")
    assert main.train.max_g_steps == 400000 and main.run_dir == "/env/SPARC_VOC_RUNS/main/hifigan"
    assert default.experiment_name == "default" and default.train.max_g_steps == 400000
    assert pilot.train.max_g_steps == 30000 and pilot.run_dir.endswith("precision_pilot_bf16-mixed/vocos")
    assert ab.train.max_g_steps == 25000 and ab.run_dir.endswith("vocos_ab_pilot/" + ab.vocoder.name)
    assert smoke.trainer.trainer.accelerator == "cpu" and smoke.train.max_g_steps == 6
    assert overfit.train.mel_warmup_steps == 500 and overfit.trainer.trainer.enable_progress_bar
    assert main.trainer.trainer.devices == 1 and list(main.trainer.logging.backends) == ["tensorboard"]


def test_real_hifigan_with_shared_loss_config(tmp_path, stats):
    cfg = make_cfg(tmp_path, max_g_steps=3, mel_warmup=1, val_every_g_steps=3, backends=["tensorboard"])
    cfg.vocoder = OmegaConf.load(CONF / "vocoder" / "hifigan.yaml")
    cfg.vocoder.generator.channels = 32
    cfg.loss = OmegaConf.load(CONF / "loss" / "shared.yaml")
    recorder = Recorder()
    _, module = train(cfg, stats, recorder)
    assert (module.g_step, module.d_step) == (3, 2)
    keys = set(recorder.rows[-1])
    assert {"train/mel", "train/g_adv/mpd", "train/g_adv/mrd", "train/g_fm/mpd", "train/g_fm/mrd", "train/d_total"} <= keys
    assert all(np.isfinite(v) for r in recorder.rows for v in r.values())
    assert recorder.rows[0]["train/grad_norm_g"] > 0
    assert np.isfinite(module.last_validation["val/mr_stft"])


def validation_steps(run_dir) -> list[int]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(Path(run_dir) / "tb" / "version_0"))
    acc.Reload()
    return [e.step for e in acc.Scalars("val/mel_l1")]


def test_resume_at_a_validation_boundary_is_bit_exact_and_validates_once(tmp_path, stats, monkeypatch):
    monkeypatch.delenv("REQUEUE_CMD", raising=False)
    kwargs = {"val_every_g_steps": 2, "backends": ["tensorboard"]}
    reference = Recorder()
    _, ref_module = train(make_cfg(tmp_path / "a", **kwargs), stats, reference)
    run_dir = tmp_path / "b"
    first = Recorder()
    _, interrupted = train(make_cfg(run_dir, **kwargs), stats, first, SignalAt(3))
    assert interrupted.g_step == 4
    second = Recorder()
    _, resumed = train(make_cfg(run_dir, **kwargs), stats, second)
    assert first.rows + second.rows == reference.rows
    for (name, a), (_, b) in zip(ref_module.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), name
    assert validation_steps(tmp_path / "a") == [2, 4, 6, 8]
    assert validation_steps(run_dir) == [2, 4, 6, 8]


def test_checkpoint_is_written_through_a_temporary_file_once_per_step(tmp_path):
    from sparc.vocoders.training.callbacks import PreemptionCheckpoint

    class FakeTrainer:
        is_global_zero = True

        def __init__(self, fail=False):
            self.fail, self.calls = fail, []

        def save_checkpoint(self, path):
            self.calls.append(Path(path).name)
            Path(path).write_bytes(b"partial")
            if self.fail:
                raise KeyboardInterrupt

    callback = PreemptionCheckpoint(tmp_path, every_minutes=0, keep_last=3)
    crashing = FakeTrainer(fail=True)
    with pytest.raises(KeyboardInterrupt):
        callback._save(crashing, 5)
    assert list_step_checkpoints(tmp_path) == []

    trainer = FakeTrainer()
    callback._save(trainer, 5)
    callback._save(trainer, 5)
    assert trainer.calls == ["step000000005.ckpt.tmp"]
    assert [s for s, _ in list_step_checkpoints(tmp_path)] == [5]
    assert list(tmp_path.glob("*.tmp")) == []


def test_final_step_signal_does_not_requeue(tmp_path, stats, monkeypatch):
    marker = tmp_path / "requeued"
    monkeypatch.setenv("REQUEUE_CMD", f"touch {marker}")
    _, module = train(make_cfg(tmp_path / "run", max_g_steps=4, mel_warmup=2), stats, SignalAt(3))
    assert module.g_step == 4 and not marker.exists()
    assert (tmp_path / "run" / "status.json").read_text() == '{"finished": true, "g_step": 4}'


class NoisyGenerator(TinyGenerator):
    """TinyGenerator plus noise drawn from the global generator, like the DDSP noise branch."""

    def forward(self, features, spk):
        wav = super().forward(features, spk)
        return wav + 0.1 * torch.randn_like(wav)


def test_fixed_torch_rng_is_reproducible_and_restores_the_global_state():
    from sparc.vocoders.training.callbacks import fixed_torch_rng

    torch.manual_seed(1)
    before = torch.get_rng_state()
    with fixed_torch_rng(torch.device("cpu"), 5):
        a = torch.randn(4)
    assert torch.equal(torch.get_rng_state(), before)
    torch.manual_seed(2)
    with fixed_torch_rng(torch.device("cpu"), 5):
        b = torch.randn(4)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fixed_torch_rng_covers_the_cuda_generator():
    from sparc.vocoders.training.callbacks import fixed_torch_rng

    device = torch.device("cuda")
    torch.cuda.manual_seed(1)
    before = torch.cuda.get_rng_state()
    with fixed_torch_rng(device, 5):
        a = torch.randn(4, device=device)
    assert torch.equal(torch.cuda.get_rng_state(), before)
    torch.cuda.manual_seed(2)
    with fixed_torch_rng(device, 5):
        b = torch.randn(4, device=device)
    assert torch.equal(a, b)


def test_validation_and_predict_do_not_depend_on_the_global_rng(tmp_path, stats):
    cfg = make_cfg(tmp_path, max_g_steps=2, mel_warmup=1, val_every_g_steps=1000)
    cfg.vocoder.generator._target_ = f"{__name__}.NoisyGenerator"
    _, module = train(cfg, stats)
    torch.manual_seed(1)
    first = dict(module.run_validation())
    torch.manual_seed(2)
    second = dict(module.run_validation())
    assert first == second
    module.eval()
    batch = next(iter(FakeDataModule().predict_dataloader()))
    torch.manual_seed(3)
    state = torch.get_rng_state()
    wav_a = module.predict_step(batch, 0)["wav"]
    assert torch.equal(torch.get_rng_state(), state)
    torch.manual_seed(4)
    assert torch.equal(wav_a, module.predict_step(batch, 0)["wav"])


def test_module_uses_the_shared_adversarial_loss_table():
    from sparc.vocoders.losses import losses
    from sparc.vocoders.training import module

    assert module.ADVERSARIAL_LOSSES is losses.ADVERSARIAL_LOSSES


def test_deterministic_run_sets_the_cublas_workspace_config(tmp_path, stats, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    cfg = make_cfg(tmp_path, max_g_steps=1, mel_warmup=1)
    cfg.trainer.trainer.deterministic = True
    try:
        train(cfg, stats)
    finally:
        torch.use_deterministic_algorithms(False)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")


def test_train_script_detects_deterministic_in_the_resolved_config(monkeypatch):
    """scripts/slurm/train_vocoder.sh greps the output of ``--cfg job --resolve`` (``OmegaConf.to_yaml``)."""
    import re

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    for var in ("SPARC_VOC_CACHE", "SPARC_VOC_RUNS", "LIBRITTSR_RAW", "SPARC_REFIT_NPZ"):
        monkeypatch.setenv(var, f"/env/{var}")
    script = (CONF.parents[2] / "scripts" / "slurm" / "train_vocoder.sh").read_text()
    pattern = re.search(r"grep -qE '([^']*deterministic[^']*)'", script).group(1)
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        on = compose("vocoder_config", overrides=["experiment=main", "trainer.trainer.deterministic=true"])
        off = compose("vocoder_config", overrides=["experiment=main"])
    assert re.search(pattern, OmegaConf.to_yaml(on, resolve=True), re.MULTILINE)
    assert not re.search(pattern, OmegaConf.to_yaml(off, resolve=True), re.MULTILINE)
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script


@pytest.mark.parametrize("name", ["train_vocoder.sh", "cache_features.sh"])
def test_sbatch_output_default_is_a_relative_path(name):
    scripts = CONF.parents[2] / "scripts" / "slurm"
    lines = [line for line in (scripts / name).read_text().splitlines() if line.startswith("#SBATCH --output=")]
    assert len(lines) == 1 and not lines[0].split("=", 1)[1].startswith(("/", "~", "$"))
    for script in scripts.glob("*.sh"):
        assert "/home/" not in script.read_text() and "/data/user_data/" not in script.read_text(), script.name
