"""Synthesis with a trained vocoder: ``sparc-predict vocoder=<name> experiment=<name> data.predict_split=test.clean``.

Pass the same ``vocoder`` and ``experiment`` overrides as for training. The split and conditions come from
``data.predict_split`` and ``data.predict_conditions``; the checkpoint is ``predict.ckpt`` or, if null, the newest
loadable one of the run. WAVs go to ``${predict.output_dir}/<split>/<condition>/<id>.wav``, written atomically. With
``predict.skip_existing=true`` utterances whose WAV exists are not synthesized again (the evaluation sets this).
"""

from collections.abc import Sequence
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.plugins.environments import LightningEnvironment
from omegaconf import DictConfig

from sparc.vocoders.training.callbacks import WavWriter, find_resume_checkpoint
from sparc.vocoders.training.module import VocoderGANModule


def run(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule | None = None,
    stats: dict | None = None,
    callbacks: Sequence[Callback] = (),
) -> Path:
    """Writes the predictions of ``cfg.data.predict_split`` and returns the output directory.

    ``callbacks`` are extra Lightning callbacks, called after the WAV writer (the evaluation uses one to stop between
    utterances on SIGUSR1/SIGTERM).
    """
    if datamodule is None:
        from sparc.vocoders.data.datamodule import VocoderDataModule

        datamodule = VocoderDataModule(cfg)
    if cfg.predict.get("skip_existing", False) and not datamodule.has_pending_prediction():
        return Path(cfg.predict.output_dir) / cfg.data.predict_split
    ckpt = cfg.predict.ckpt
    if ckpt is None:
        found = find_resume_checkpoint(Path(cfg.run_dir) / "ckpt")
        if found is None:
            raise FileNotFoundError(f"no loadable step*.ckpt in {Path(cfg.run_dir) / 'ckpt'}")
        ckpt = found[1]
    module = VocoderGANModule(cfg, stats=stats)
    module.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    writer = WavWriter(cfg.predict.output_dir, cfg.data.predict_split)
    trainer = pl.Trainer(
        accelerator=cfg.trainer.trainer.accelerator,
        devices=1,
        precision=cfg.predict.precision,
        logger=False,
        callbacks=[writer, *callbacks],
        plugins=[LightningEnvironment()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
    )
    trainer.predict(module, datamodule=datamodule, return_predictions=False)
    return Path(cfg.predict.output_dir) / cfg.data.predict_split


@hydra.main(version_base=None, config_path="../conf", config_name="vocoder_config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
