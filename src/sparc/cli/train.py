import os
import signal
from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment, SLURMEnvironment
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from sparc.training.dataset import VocoderDataset, collate
from sparc.training.lightning_module import SparcVocoderTraining, batches_to_global_steps


def _under_slurm_batch() -> bool:
    """True inside an sbatch job; False in salloc/srun-interactive shells and outside SLURM."""
    return "SLURM_JOB_ID" in os.environ and os.environ.get("SLURM_JOB_NAME") not in ("bash", "interactive")


def _resolve_run_name(cfg) -> str | None:
    """Stable per-run name. A requeued SLURM job keeps its SLURM_JOB_ID, so keying on it makes a
    requeue find its own checkpoints while a fresh submission starts a new run. Outside SLURM the
    name is only set if configured (cfg.run_name), otherwise the legacy flat layout is kept."""
    if cfg.run_name:
        return str(cfg.run_name)
    if _under_slurm_batch():
        return f"slurm_{os.environ['SLURM_JOB_ID']}"
    return None


def _find_resume_ckpt(cfg, ckpt_dir: Path, run_name: str | None) -> str | None:
    """Explicit cfg.resume_from_checkpoint wins. Otherwise, for a named run, pick the newest of
    last.ckpt (periodic/exception saves) and Lightning's hpc_ckpt_*.ckpt (written by the SIGUSR1
    requeue handler) in the run's checkpoint dir."""
    if cfg.resume_from_checkpoint:
        return str(cfg.resume_from_checkpoint)
    if run_name is None:
        return None
    candidates = [ckpt_dir / "last.ckpt", *ckpt_dir.glob("hpc_ckpt_*.ckpt")]
    candidates = [p for p in candidates if p.is_file()]
    return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None


def _build_loggers(cfg, save_dir: Path, run_name: str | None = None):
    """Builds every logger named in cfg.logger_backends. Multiple backends
    can run simultaneously -- Lightning dispatches self.log/self.log_dict
    scalars to all of them automatically; SparcVocoderTraining._log_media
    handles audio/image logging per-backend since there's no common API
    for that across loggers.
    """
    loggers = []
    backends = list(cfg.logger_backends)
    if not backends:
        raise ValueError("cfg.logger_backends is empty -- need at least one logger")

    if "tensorboard" in backends:
        tb_dir = Path(cfg.tb_log_dir) if cfg.tb_log_dir else save_dir / "tb_logs"
        tb_dir.mkdir(parents=True, exist_ok=True)
        # fixed version for named runs so a requeued job appends to the same event dir
        loggers.append(TensorBoardLogger(save_dir=str(tb_dir.parent), name=tb_dir.name, version=run_name))

    if "wandb" in backends:
        wandb_dir = Path(cfg.wandb_dir) if cfg.wandb_dir else save_dir / "wandb_logs"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        loggers.append(
            WandbLogger(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                id=run_name,  # same wandb run across requeues
                resume="allow" if run_name else None,
                save_dir=str(wandb_dir),
                # Compute nodes have internet access, but wandb "online" mode
                # needs an API key (`wandb login` / WANDB_API_KEY) that isn't
                # configured for this user by default. "offline" writes logs
                # locally with no auth required; run `wandb sync <run_dir>`
                # later to upload, or set wandb_mode=online once logged in.
                mode=cfg.wandb_mode,
            )
        )

    return loggers


@hydra.main(version_base=None, config_path="../conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    save_dir = Path(cfg.dataset.save_dir)
    run_name = _resolve_run_name(cfg)
    ckpt_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else save_dir / "vocoder_ckpt"
    if run_name and not cfg.checkpoint_dir:
        ckpt_dir = ckpt_dir / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = VocoderDataset(
        wav_dir=cfg.dataset.wav_dir,
        sparc_dir=cfg.dataset.save_dir,
        segment_frames=cfg.segment_frames,
    )
    print(f"Training set: {len(dataset)} utterances with cached emasrc + spk_raw features")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    model = SparcVocoderTraining(
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        lr_halve_every=cfg.lr_halve_every,
        lr_static_after=cfg.lr_static_after,
        mel_weight=cfg.mel_weight,
        fm_weight=cfg.fm_weight,
        gan_weight=cfg.gan_weight,
        log_audio_every_n_steps=cfg.log_audio_every_n_steps,
    )

    loggers = _build_loggers(cfg, save_dir, run_name)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_last=True,
        # config counts batches; Lightning's global_step counts optimizer steps (2 per batch)
        every_n_train_steps=batches_to_global_steps(cfg.checkpoint_every_n_steps),
        save_on_exception=True,  # SIGTERM -> checkpoint at the next batch end
        save_top_k=cfg.keep_last_n_checkpoints,
        # no validation loss to rank by -- rank by recency instead (see
        # SparcVocoderTraining.training_step's step_metric log call).
        monitor="step_metric" if cfg.keep_last_n_checkpoints not in (-1, 1) else None,
        mode="max",
    )

    if _under_slurm_batch() and cfg.slurm_requeue:
        # SIGUSR1 (sbatch --signal=B:USR1@120) -> save hpc checkpoint and `scontrol requeue`
        plugins = [SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR1)]
    else:
        plugins = [LightningEnvironment()]
    devices = cfg.devices
    strategy = cfg.strategy
    if strategy == "auto" and devices != 1:
        # two optimizers stepped manually: some params get no grad in each backward
        strategy = "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        max_steps=batches_to_global_steps(cfg.max_steps),
        accelerator="gpu" if cfg.device.startswith("cuda") else "cpu",
        devices=devices,
        strategy=strategy,
        plugins=plugins,
        default_root_dir=str(ckpt_dir),  # hpc_ckpt_*.ckpt (requeue) lands next to last.ckpt
        logger=loggers,
        callbacks=[checkpoint_cb],
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=True,
    )
    resume_path = _find_resume_ckpt(cfg, ckpt_dir, run_name)
    if resume_path:
        print(f"Resuming from {resume_path}")
    trainer.fit(model, train_dataloaders=loader, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
