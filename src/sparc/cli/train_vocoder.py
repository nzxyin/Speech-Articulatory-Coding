"""Training of the 24 kHz articulatory vocoders: ``sparc-train vocoder=<name> experiment=<name>`` (contract 6)."""

import hashlib
import json
import os
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig, OmegaConf

from sparc.vocoders.training.callbacks import PreemptionCheckpoint, RNGStateCallback, find_resume_checkpoint
from sparc.vocoders.training.module import VocoderGANModule

STATUS_FILE = "status.json"
MULTI_DEVICE_STRATEGY = "ddp_find_unused_parameters_true"
CUBLAS_DETERMINISTIC_CONFIG = ":4096:8"


def build_loggers(cfg: DictConfig, run_dir: Path) -> list[Logger]:
    """Loggers named in ``cfg.trainer.logging.backends``; their directories are fixed so a requeued run appends."""
    loggers: list[Logger] = []
    backends = list(cfg.trainer.logging.backends)
    if "tensorboard" in backends:
        loggers.append(TensorBoardLogger(save_dir=str(run_dir), name="tb", version=0))
    if "wandb" in backends:
        wandb_dir = run_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        loggers.append(
            WandbLogger(
                project=cfg.trainer.logging.wandb_project,
                name=f"{cfg.experiment_name}/{cfg.vocoder.name}",
                id=hashlib.sha1(str(run_dir).encode()).hexdigest()[:16],
                resume="allow",
                save_dir=str(wandb_dir),
                mode=cfg.trainer.logging.wandb_mode,
            )
        )
    return loggers


def build_trainer(cfg: DictConfig, run_dir: Path, extra_callbacks=()) -> pl.Trainer:
    """Trainer with explicit ``LightningEnvironment``, our own checkpointing and no Lightning SLURM handling."""
    kwargs = OmegaConf.to_container(cfg.trainer.trainer, resolve=True)
    devices, num_nodes = kwargs.get("devices", 1), kwargs.get("num_nodes", 1)
    if isinstance(devices, int) and devices * num_nodes > 1 and kwargs.get("strategy", "auto") == "auto":
        kwargs["strategy"] = MULTI_DEVICE_STRATEGY
    loggers = build_loggers(cfg, run_dir)
    callbacks = [
        PreemptionCheckpoint(
            run_dir / "ckpt",
            every_minutes=cfg.train.checkpoint_every_minutes,
            milestone_every=cfg.train.milestone_every_g_steps,
            keep_last=cfg.train.keep_last_checkpoints,
            max_g_steps=cfg.train.max_g_steps,
        ),
        RNGStateCallback(cfg.seed),
        *extra_callbacks,
    ]
    return pl.Trainer(
        **kwargs,
        logger=loggers or False,
        callbacks=callbacks,
        plugins=[LightningEnvironment()],
        default_root_dir=str(run_dir),
        enable_checkpointing=False,
        max_steps=-1,
        max_epochs=-1,
        use_distributed_sampler=False,
    )


def write_status(run_dir: Path, g_step: int, finished: bool) -> None:
    """Atomically writes ``status.json``; the sbatch script writes ``DONE`` only when ``finished`` is true."""
    path = run_dir / STATUS_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"finished": finished, "g_step": g_step}))
    os.replace(tmp, path)


def resolve_resume(cfg: DictConfig, ckpt_dir: Path) -> tuple[int, Path] | None:
    """``(step, path)`` of the checkpoint to resume from according to ``cfg.resume``, or ``None`` for a fresh start."""
    if cfg.resume == "none":
        return None
    if cfg.resume == "auto":
        return find_resume_checkpoint(ckpt_dir)
    counters = torch.load(cfg.resume, map_location="cpu", weights_only=False)["vocoder_counters"]
    return int(counters["g_step"]), Path(cfg.resume)


def run(
    cfg: DictConfig, datamodule: pl.LightningDataModule | None = None, stats: dict | None = None, extra_callbacks=()
):
    """Trains (or resumes) the run described by ``cfg``; returns ``(trainer, module)``.

    ``datamodule`` and ``stats`` default to ``VocoderDataModule(cfg)`` and the JSON file at ``cfg.stats_path``.
    """
    run_dir = Path(cfg.run_dir)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    if cfg.trainer.trainer.get("deterministic"):
        # the sbatch script exports it too; this covers other launchers as long as no CUDA work has happened yet
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_DETERMINISTIC_CONFIG)
    pl.seed_everything(cfg.seed)
    module = VocoderGANModule(cfg, stats=stats)
    if datamodule is None:
        from sparc.vocoders.data.datamodule import VocoderDataModule

        datamodule = VocoderDataModule(cfg)
    trainer = build_trainer(cfg, run_dir, extra_callbacks)
    resume = resolve_resume(cfg, run_dir / "ckpt")
    rank_zero_info(
        f"run_dir={run_dir} resume={resume[1] if resume else None} precision={cfg.trainer.trainer.precision} "
        f"cudnn_tf32={torch.backends.cudnn.allow_tf32} matmul_tf32={torch.backends.cuda.matmul.allow_tf32}"
    )
    if resume is not None and resume[0] >= cfg.train.max_g_steps:
        rank_zero_info(f"checkpoint at g_step {resume[0]} already reaches max_g_steps: nothing to do")
        write_status(run_dir, resume[0], finished=True)
        return trainer, module
    trainer.fit(module, datamodule=datamodule, ckpt_path=str(resume[1]) if resume else None)
    if trainer.is_global_zero:
        write_status(run_dir, module.g_step, finished=module.g_step >= cfg.train.max_g_steps)
    return trainer, module


@hydra.main(version_base=None, config_path="../conf", config_name="vocoder_config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
