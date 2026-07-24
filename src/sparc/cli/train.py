import os
from pathlib import Path

# This CLI always runs as a single task/single GPU inside one srun/sbatch
# allocation, never as an elastic `srun python train.py` multi-node launch.
# Lightning's SLURMEnvironment.detect() auto-activates whenever SLURM_NTASKS
# is set and SLURM_JOB_NAME isn't "bash"/"interactive" (see
# lightning/fabric/plugins/environments/slurm.py:_is_slurm_interactive_mode),
# which misreads our sbatch job's SLURM_* vars and tries to bind a CUDA
# device that doesn't exist in this process's CUDA_VISIBLE_DEVICES, raising
# "CUDA-capable device(s) is/are busy or unavailable". Spoofing the
# interactive-mode job name (before Trainer construction reads it) is the
# documented escape hatch and is more robust than passing an explicit
# `plugins=` override, which does not fully suppress the auto-detection.
os.environ["SLURM_JOB_NAME"] = "interactive"

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from sparc.training.dataset import VocoderDataset, collate
from sparc.training.lightning_module import SparcVocoderTraining


@hydra.main(version_base=None, config_path="../conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    save_dir = Path(cfg.dataset.save_dir)
    ckpt_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else save_dir / "vocoder_ckpt"
    tb_dir = Path(cfg.tb_log_dir) if cfg.tb_log_dir else save_dir / "tb_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

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

    logger = TensorBoardLogger(save_dir=str(tb_dir.parent), name=tb_dir.name)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_last=True,
        every_n_train_steps=cfg.checkpoint_every_n_steps,
        save_top_k=cfg.keep_last_n_checkpoints,
        # no validation loss to rank by -- rank by recency instead (see
        # SparcVocoderTraining.training_step's step_metric log call).
        monitor="step_metric" if cfg.keep_last_n_checkpoints not in (-1, 1) else None,
        mode="max",
    )

    trainer = pl.Trainer(
        max_steps=cfg.max_steps,
        accelerator="gpu" if cfg.device.startswith("cuda") else "cpu",
        devices=1,
        # This CLI runs as a single task/single GPU inside one srun/sbatch
        # allocation, not as an elastic `srun python train.py` multi-node
        # launch -- Lightning's SLURM auto-detection otherwise misreads the
        # SLURM_* env vars srun sets and tries to bind a CUDA device that
        # doesn't correspond to this process, raising
        # "CUDA-capable device(s) is/are busy or unavailable".
        plugins=[LightningEnvironment()],
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=loader, ckpt_path=cfg.resume_from_checkpoint)


if __name__ == "__main__":
    main()
