"""Extraction driver: ``trainer.predict`` over a shard of the manifest, writing one cache file per utterance.

``ManifestShardDataModule`` serves one utterance per step (workers only read audio), ``FeatureCacheModule`` runs the
extractor in ``predict_step`` and ``CacheWriter`` writes the files, logs per-utterance errors and counts. A stop
request (SIGUSR1 or SIGTERM) ends the run cleanly after the current utterance.
"""

import json
import signal
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.callbacks import BasePredictionWriter
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from sparc.vocoders.constants import SAMPLE_RATE
from sparc.vocoders.data.datamodule import ignore_stop_signals
from sparc.vocoders.features.cache import build_meta, ensure_meta, is_valid, utt_path, utterance_seed, write_utt
from sparc.vocoders.features.extractor import SparcFeatureExtractor, fork_commit
from sparc.vocoders.features.manifest import load_manifest, select_rows

STOP_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)
EXIT_STOPPED = 75
STOP = threading.Event()


class StopRequested(Exception):
    """Raised from the writer to leave ``trainer.predict`` after a stop signal."""


def request_stop(signum: int, frame) -> None:
    STOP.set()


def identity(item: dict) -> dict:
    """Collate function for ``batch_size=None``: the dataset item is the batch."""
    return item


def shard_name(shard_index: int, num_shards: int) -> str:
    return f"{shard_index:05d}of{num_shards:05d}"


def shard_rows(table: pd.DataFrame, shard_index: int, num_shards: int) -> pd.DataFrame:
    """Every ``num_shards``-th row starting at ``shard_index``, which balances the audio duration between shards."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} is outside [0, {num_shards})")
    return table.iloc[shard_index::num_shards]


class ManifestShardDataset(Dataset):
    """Utterances of one shard. Items are dicts; failures to read audio are reported in ``error``, not raised."""

    def __init__(self, rows: pd.DataFrame, utt_dir: str | Path, skip_valid: bool):
        self.records = rows[["id", "split", "wav_path", "n24", "T"]].to_dict("records")
        self.utt_dir = Path(utt_dir)
        self.skip_valid = skip_valid

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        item = {"id": row["id"], "split": row["split"], "T": int(row["T"]), "n24": int(row["n24"])}
        if self.skip_valid and is_valid(utt_path(self.utt_dir, row["split"], row["id"]), item["T"]):
            return {**item, "skipped": True}
        try:
            wav, rate = sf.read(row["wav_path"], dtype="float64", always_2d=False)
            if rate != SAMPLE_RATE or wav.ndim != 1 or len(wav) != item["n24"]:
                raise ValueError(
                    f"expected {item['n24']} mono samples at {SAMPLE_RATE} Hz, got {wav.shape} at {rate} Hz"
                )
            return {**item, "wav24": wav}
        except Exception:
            return {**item, "error": traceback.format_exc()}


class ManifestShardDataModule(L.LightningDataModule):
    """Predict dataloader over a shard of the manifest; ``encodable == False`` rows are listed, not extracted."""

    def __init__(self, cfg: DictConfig, shard_index: int, num_shards: int, skip_valid: bool = True):
        super().__init__()
        self.cfg = cfg
        self.shard_index = shard_index
        self.num_shards = num_shards
        self.skip_valid = skip_valid
        self.not_encodable: list[str] = []
        self.dataset: ManifestShardDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        table = select_rows(load_manifest(self.cfg), self.cfg)
        rows = shard_rows(table, self.shard_index, self.num_shards)
        self.not_encodable = rows.loc[~rows["encodable"], "id"].tolist()
        self.dataset = ManifestShardDataset(rows[rows["encodable"]], self.cfg.cache.utt_dir, self.skip_valid)

    def predict_dataloader(self) -> DataLoader:
        workers = int(self.cfg.extract.num_workers)
        return DataLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=workers,
            prefetch_factor=int(self.cfg.extract.prefetch_factor) if workers > 0 else None,
            collate_fn=identity,
            worker_init_fn=ignore_stop_signals,
        )

    def transfer_batch_to_device(self, batch: dict, device: torch.device, dataloader_idx: int) -> dict:
        return batch


class FeatureCacheModule(L.LightningModule):
    """Runs the extractor on each utterance; the extractor itself lives outside ``nn.Module`` state."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.extractor: SparcFeatureExtractor | None = None
        self.meta_hash: str | None = None

    def setup(self, stage: str) -> None:
        self.extractor = SparcFeatureExtractor(self.cfg, self.trainer.strategy.root_device)
        meta = build_meta(self.extractor.describe(), fork_commit())
        _, self.meta_hash = ensure_meta(self.cfg.cache.meta_path, meta)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        head = {key: batch[key] for key in ("id", "split", "T", "n24")}
        if batch.get("skipped"):
            return {**head, "status": "skipped"}
        if "error" in batch:
            return {**head, "status": "error", "error": batch["error"]}
        try:
            start = time.perf_counter()
            seed = utterance_seed(batch["id"])
            out = self.extractor.extract(batch["wav24"], seed)
            if out["T"] != batch["T"]:
                raise ValueError(f"extracted {out['T']} frames, manifest says {batch['T']}")
            if self.extractor.device.type == "cuda":
                torch.cuda.synchronize(self.extractor.device)
            seconds = time.perf_counter() - start
        except Exception:
            return {**head, "status": "error", "error": traceback.format_exc()}
        arrays = {
            **out,
            "spk_wsum": np.float64(out["spk_wsum"]),
            "spk_fallback": np.bool_(out["spk_fallback"]),
            "meta_hash": self.meta_hash,
            "gpu": self.extractor.device_name,
        }
        return {**head, "status": "ok", "arrays": arrays, "seconds": seconds}


class CacheWriter(BasePredictionWriter):
    """Writes the cache files, logs errors to ``errors/<shard>.jsonl`` and keeps counts and timings."""

    def __init__(self, cfg: DictConfig, shard: str):
        super().__init__(write_interval="batch")
        self.cfg = cfg
        self.shard = shard
        self.utt_dir = Path(cfg.cache.utt_dir)
        self.error_path = Path(cfg.cache.errors_dir) / f"{shard}.jsonl"
        self.max_consecutive = int(cfg.extract.max_consecutive_errors)
        self.log_every = int(cfg.extract.log_every)
        self.counts = {"written": 0, "skipped": 0, "errors": 0}
        self.consecutive_errors = 0
        self.last_error = ""
        self.extract_seconds: list[float] = []
        self.audio_seconds = 0.0
        self.first_audio_seconds = 0.0
        self.first_written: float | None = None
        self.last_written: float | None = None
        self.started = time.perf_counter()

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ) -> None:
        status = prediction["status"]
        if status == "skipped":
            self.counts["skipped"] += 1
        elif status == "ok":
            path = utt_path(self.utt_dir, prediction["split"], prediction["id"])
            try:
                write_utt(path, prediction["arrays"])
            except OSError:
                self._log_error(prediction, traceback.format_exc())
            else:
                self.counts["written"] += 1
                self.consecutive_errors = 0
                self.extract_seconds.append(prediction["seconds"])
                self.audio_seconds += prediction["n24"] / SAMPLE_RATE
                self.last_written = time.perf_counter()
                if self.first_written is None:
                    self.first_written, self.first_audio_seconds = self.last_written, self.audio_seconds
        else:
            self._log_error(prediction, prediction["error"])
        done = sum(self.counts.values())
        if self.log_every > 0 and done % self.log_every == 0:
            print(f"[{self.shard}] {done} utterances: {self.counts}", flush=True)
        if self.consecutive_errors >= self.max_consecutive:
            raise RuntimeError(f"{self.consecutive_errors} consecutive errors; last: {self.last_error}")
        if STOP.is_set():
            raise StopRequested()

    def _log_error(self, prediction: dict, text: str) -> None:
        self.counts["errors"] += 1
        self.consecutive_errors += 1
        self.last_error = text.strip().splitlines()[-1] if text.strip() else text
        record = {"id": prediction["id"], "split": prediction["split"], "shard": self.shard, "error": text}
        self.error_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.error_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{self.shard}] error on {prediction['id']}: {self.last_error}", flush=True)

    def summary(self) -> dict:
        """Counts and timings of this run."""
        times = np.asarray(self.extract_seconds)
        steady = times[3:] if len(times) > 3 else times
        loop_audio = self.audio_seconds - self.first_audio_seconds
        loop_seconds = (self.last_written - self.first_written) if self.first_written is not None else 0.0
        return {
            "shard": self.shard,
            **self.counts,
            "audio_hours": self.audio_seconds / 3600.0,
            "extract_seconds_total": float(times.sum()),
            "extract_seconds_per_audio_hour": float(times.sum() / (self.audio_seconds / 3600.0))
            if len(times)
            else None,
            "steady_seconds_per_utterance": float(steady.mean()) if len(steady) else None,
            "loop_seconds_per_audio_hour": loop_seconds / (loop_audio / 3600.0) if loop_audio > 0 else None,
            "wall_seconds": time.perf_counter() - self.started,
        }


def write_summary(cfg: DictConfig, shard: str, summary: dict, not_encodable: list[str]) -> None:
    """Appends the run summary to ``summaries/extract_<shard>.jsonl`` and lists the not encodable utterances."""
    summaries = Path(cfg.cache.summaries_dir)
    summaries.mkdir(parents=True, exist_ok=True)
    with open(summaries / f"extract_{shard}.jsonl", "a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **summary}) + "\n")
    errors = Path(cfg.cache.errors_dir)
    errors.mkdir(parents=True, exist_ok=True)
    (errors / f"not_encodable_{shard}.txt").write_text("".join(f"{utt_id}\n" for utt_id in not_encodable))


def run_extract(cfg: DictConfig) -> int:
    """Stage ``extract``: returns 0 on success, 1 if utterances failed, ``EXIT_STOPPED`` after a stop signal."""
    shard_index, num_shards = int(cfg.shard_index), int(cfg.num_shards)
    shard = shard_name(shard_index, num_shards)
    for sig in STOP_SIGNALS:
        signal.signal(sig, request_stop)
    module = FeatureCacheModule(cfg)
    data = ManifestShardDataModule(cfg, shard_index, num_shards, skip_valid=bool(cfg.extract.skip_valid))
    writer = CacheWriter(cfg, shard)
    trainer = L.Trainer(
        accelerator=cfg.extract.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[writer],
        plugins=[LightningEnvironment()],
    )
    stopped = False
    try:
        trainer.predict(module, datamodule=data, return_predictions=False)
    except StopRequested:
        stopped = True
    summary = writer.summary()
    summary["stopped"] = stopped
    write_summary(cfg, shard, summary, data.not_encodable)
    print(f"[{shard}] done: {json.dumps(summary)}", flush=True)
    if stopped:
        return EXIT_STOPPED
    return 1 if writer.counts["errors"] else 0
