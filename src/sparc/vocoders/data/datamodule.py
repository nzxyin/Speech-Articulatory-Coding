"""Lightning data module of the vocoder comparison: resumable training crops and full-utterance evaluation sets."""

import json
import os
import signal
import threading
from collections.abc import Sequence
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from sparc.vocoders.constants import SAMPLE_RATE
from sparc.vocoders.data.dataset import CropDataset, FullUtteranceDataset
from sparc.vocoders.data.sampler import ResumableSampler

LOGGING_STREAM = 0
VAL_FILL_STREAM = 1
PREDICT_STREAM = 2
LOGGING_DURATION_RANGE_S = (4.0, 8.0)


def _exit_on_parent_terminate(parent_pid: int) -> None:
    """Waits for SIGTERM and exits cleanly when the parent process sent it; SIGTERM from anyone else is dropped."""
    while True:
        if signal.sigwaitinfo({signal.SIGTERM}).si_pid == parent_pid:
            os._exit(0)


def ignore_stop_signals(worker_id: int) -> None:
    """``worker_init_fn``: DataLoader workers ignore the SIGTERM and SIGUSR1 that SLURM sends to every process.

    Only the main process reacts to them (it checkpoints at a batch boundary); a worker killed by the signal would
    otherwise abort the step with a "worker killed by signal" error. SIGTERM is not ignored outright: a thread of
    the worker receives it synchronously and exits the worker (status 0, like PyTorch's own handler) when the
    sender is the parent, which is how ``multiprocessing`` stops daemon workers at interpreter exit. With a plain
    ``SIG_IGN`` a main process that exits on an exception waits for its workers forever.
    """
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    threading.Thread(target=_exit_on_parent_terminate, args=(os.getppid(),), daemon=True).start()


def select_logging_ids(
    index: pd.DataFrame, size: int, seed: int, duration_range_s: Sequence[float] = LOGGING_DURATION_RANGE_S
) -> list[str]:
    """Fixed audio-logging subset: one utterance within ``duration_range_s`` for each of ``size`` speakers.

    ``index`` has columns ``id, speaker, n24``. Speakers and utterances are drawn in id order, so the result does
    not depend on the row order of ``index``.
    """
    low, high = (int(round(float(d) * SAMPLE_RATE)) for d in duration_range_s)
    index = index.assign(id=index["id"].astype(str), speaker=index["speaker"].astype(str)).sort_values("id")
    index = index[(index["n24"] >= low) & (index["n24"] <= high)]
    candidates = {speaker: group["id"].tolist() for speaker, group in index.groupby("speaker", sort=True)}
    rng = np.random.default_rng([int(seed), LOGGING_STREAM])
    speakers = rng.permutation(sorted(candidates))[: int(size)]
    return sorted(candidates[s][int(rng.integers(len(candidates[s])))] for s in speakers)


def select_validation_ids(index: pd.DataFrame, subset_size: int, logging_ids: list[str], seed: int) -> list[str]:
    """Fixed validation subset: the logging utterances first, then a random fill up to ``subset_size`` sorted by id.

    The training module logs audio for the first ``logging_subset_size`` items of the validation loader, so the
    logging utterances must lead.
    """
    ids = index["id"].astype(str).to_numpy()
    rest = np.array(sorted(set(ids) - set(logging_ids)))
    fill = max(int(subset_size) - len(logging_ids), 0)
    chosen = np.random.default_rng([int(seed), VAL_FILL_STREAM]).permutation(len(rest))[:fill]
    return list(logging_ids) + sorted(rest[chosen].tolist())


class VocoderDataModule(L.LightningDataModule):
    """Data for training, validation and prediction of the vocoders (contract section 2).

    Training draws aligned crops through a :class:`ResumableSampler` over the splits in ``cfg.data.train_splits``.
    The loader is an endless stream; ``set_samples_consumed(n)`` moves it to ``n`` samples (summed over ranks) into
    the stream, which the training module restores from its checkpoint. The Trainer must be created with
    ``use_distributed_sampler=False`` because the sampler already strides by rank.

    Validation is one loader (batch size 1, T1, gain ``eval_gain_db``) over a fixed dev subset that starts with the
    logging subset; ``logging_ids`` names the utterances whose audio and mel images are logged. Prediction is a list
    of loaders, one per condition in ``cfg.data.predict_conditions``.

    Optional ``cfg.data`` keys: ``train_subset_size`` and ``fixed_crops`` for sanity runs, ``val_subset_seed``,
    ``logging_ids_file``, ``logging_duration_range_s``, ``predict_limit``, ``pin_memory``, ``prefetch_factor``,
    ``eval_num_workers``.
    """

    def __init__(self, cfg: DictConfig, rank: int | None = None, world_size: int | None = None):
        super().__init__()
        self.cfg = cfg
        self._rank = rank
        self._world_size = world_size
        self._samples_consumed = 0
        self._sampler: ResumableSampler | None = None
        self._stats: dict | None = None
        self._logging_ids: list[str] | None = None
        self._val_ids: list[str] | None = None

    @property
    def samples_consumed(self) -> int:
        """Samples (over all ranks) at the start of the training stream."""
        return self._samples_consumed

    def set_samples_consumed(self, samples_consumed: int) -> None:
        """Moves the start of the training stream; applies to the sampler now and to any sampler created later."""
        self._samples_consumed = int(samples_consumed)
        if self._sampler is not None:
            self._sampler.set_samples_consumed(self._samples_consumed)

    @property
    def stats(self) -> dict | None:
        """Training statistics from ``cfg.stats_path``, or ``None`` before ``stats`` have been computed."""
        if self._stats is None and Path(self.cfg.stats_path).is_file():
            self._stats = json.loads(Path(self.cfg.stats_path).read_text())
        return self._stats

    @property
    def logging_ids(self) -> list[str]:
        """Ids of the fixed audio-logging subset (contained in the validation subset)."""
        self._select_validation()
        return list(self._logging_ids)

    @property
    def val_ids(self) -> list[str]:
        self._select_validation()
        return list(self._val_ids)

    def _packed(self, split: str) -> Path:
        return Path(self.cfg.paths.cache_root) / "packed" / split

    def _rank_and_world(self) -> tuple[int, int]:
        if self._rank is not None and self._world_size is not None:
            return self._rank, self._world_size
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            return trainer.global_rank, trainer.world_size
        return 0, 1

    def _select_validation(self) -> None:
        if self._val_ids is not None:
            return
        data = self.cfg.data
        index = pd.read_parquet(self._packed(data.val_split) / "index.parquet", columns=["id", "speaker", "n24"])
        seed = int(data.get("val_subset_seed", 0))
        ids_file = data.get("logging_ids_file")
        if ids_file:
            logging_ids = [line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()]
            missing = set(logging_ids) - set(index["id"].astype(str))
            if missing:
                raise ValueError(
                    f"{len(missing)} ids of {ids_file} are not in {data.val_split}, e.g. {sorted(missing)[:3]}"
                )
        else:
            duration_range = data.get("logging_duration_range_s", LOGGING_DURATION_RANGE_S)
            logging_ids = select_logging_ids(index, data.logging_subset_size, seed, duration_range)
        self._logging_ids = logging_ids
        self._val_ids = select_validation_ids(index, data.val_subset_size, logging_ids, seed)

    def train_dataset(self) -> CropDataset:
        data = self.cfg.data
        return CropDataset(
            [self._packed(split) for split in data.train_splits],
            stats=self.stats,
            crop_frames=data.crop_frames,
            p_cross=self.cfg.speaker.p_cross,
            ref_min_dur=data.ref_min_dur,
            speaker_layer=self.cfg.speaker.layer,
            seed=self.cfg.seed,
            gain_db_range=data.gain_db_range,
            subset_size=data.get("train_subset_size"),
            subset_seed=self.cfg.seed,
            fixed_crops=bool(data.get("fixed_crops", False)),
        )

    def train_dataloader(self) -> DataLoader:
        data = self.cfg.data
        dataset = self.train_dataset()
        rank, world_size = self._rank_and_world()
        connector = getattr(getattr(self, "trainer", None), "_accelerator_connector", None)
        if world_size > 1 and getattr(connector, "use_distributed_sampler", False):
            raise RuntimeError("the sampler strides by rank: create the Trainer with use_distributed_sampler=False")
        self._sampler = ResumableSampler(len(dataset), self.cfg.seed, rank, world_size)
        self._sampler.set_samples_consumed(self._samples_consumed)
        workers = int(data.num_workers)
        return DataLoader(
            dataset,
            batch_size=int(data.batch_size),
            sampler=self._sampler,
            generator=torch.Generator().manual_seed(int(self.cfg.seed)),
            num_workers=workers,
            drop_last=True,
            pin_memory=bool(data.get("pin_memory", True)) and torch.cuda.is_available(),
            worker_init_fn=ignore_stop_signals,
            prefetch_factor=int(data.get("prefetch_factor", 2)) if workers > 0 else None,
            persistent_workers=workers > 0,
        )

    def _eval_loader(self, dataset: FullUtteranceDataset) -> DataLoader:
        workers = int(self.cfg.data.get("eval_num_workers", 0))
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            worker_init_fn=ignore_stop_signals,
            persistent_workers=False,
        )

    def val_dataloader(self) -> DataLoader:
        data = self.cfg.data
        dataset = FullUtteranceDataset(
            self._packed(data.val_split),
            ids=self.val_ids,
            gain_db=data.eval_gain_db,
            condition="T1",
            speaker_layer=self.cfg.speaker.layer,
            ref_min_dur=data.ref_min_dur,
        )
        return self._eval_loader(dataset)

    def predict_ids(self) -> list[str] | None:
        """Target utterances of the prediction split: all of them, or a fixed random ``predict_limit`` of them."""
        limit = self.cfg.data.get("predict_limit")
        if not limit:
            return None
        index = pd.read_parquet(self._packed(self.cfg.data.predict_split) / "index.parquet", columns=["id"])
        ids = np.array(sorted(index["id"].astype(str)))
        chosen = np.random.default_rng([self.cfg.seed, PREDICT_STREAM]).permutation(len(ids))[: int(limit)]
        return sorted(ids[chosen].tolist())

    def predict_dataloader(self) -> list[DataLoader]:
        data = self.cfg.data
        ids = self.predict_ids()
        return [
            self._eval_loader(
                FullUtteranceDataset(
                    self._packed(data.predict_split),
                    ids=ids,
                    gain_db=data.eval_gain_db,
                    condition=condition,
                    speaker_layer=self.cfg.speaker.layer,
                    ref_min_dur=data.ref_min_dur,
                )
            )
            for condition in data.predict_conditions
        ]
