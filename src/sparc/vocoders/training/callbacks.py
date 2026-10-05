"""Preemption-safe checkpointing, RNG state and WAV writing for the vocoder training module.

``PreemptionCheckpoint`` follows the design verified bit-exact in ``probes/compute/resume_demo.py`` (Phase 1 notes,
``compute.md``): SIGUSR1/SIGTERM only set a flag, the checkpoint is written at the next batch boundary after the full
discriminator and generator update, and the run resumes from the newest loadable ``step*.ckpt``. Lightning's own
SLURM and SIGTERM paths are not used.
"""

import os
import random
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn

from sparc.vocoders.constants import SAMPLE_RATE

CHECKPOINT_NAME = re.compile(r"step(\d+)\.ckpt")
PREEMPTION_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)
REQUEUE_VAR = "REQUEUE_CMD"
TEMPORARY_SUFFIX = ".tmp"


def checkpoint_path(ckpt_dir: str | Path, step: int) -> Path:
    """Path of the checkpoint written at generator step ``step``."""
    return Path(ckpt_dir) / f"step{step:09d}.ckpt"


def list_step_checkpoints(ckpt_dir: str | Path) -> list[tuple[int, Path]]:
    """``(step, path)`` of every ``step*.ckpt`` in ``ckpt_dir``, oldest step first."""
    found = []
    for path in Path(ckpt_dir).glob("step*.ckpt"):
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def find_resume_checkpoint(ckpt_dir: str | Path) -> tuple[int, Path] | None:
    """Newest loadable ``step*.ckpt`` by step number (never by modification time), or ``None``.

    A checkpoint that fails to load (for example one cut short by a kill during the write) is skipped with a warning.
    """
    for step, path in reversed(list_step_checkpoints(ckpt_dir)):
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            rank_zero_warn(f"skipping unloadable checkpoint {path}: {error!r}")
            continue
        return step, path
    return None


class PreemptionCheckpoint(Callback):
    """Step-boundary checkpoints for preemptible training.

    SIGUSR1 and SIGTERM set a flag. At the end of the next training batch (after the full D and G update) all ranks
    agree on the flag, rank 0 writes ``<ckpt_dir>/step<g_step>.ckpt``, runs the command in ``$REQUEUE_CMD`` if that
    variable is set (only the sbatch script sets it: an ssh session adopted into a job also carries ``SLURM_JOB_ID``,
    so requeueing is never implicit) and the run stops. Time-based checkpoints are written every ``every_minutes``,
    milestones every ``milestone_every`` generator steps; milestones are kept, other files are pruned to the newest
    ``keep_last``. A run that reaches ``max_g_steps`` writes a final checkpoint.
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        every_minutes: float = 20.0,
        milestone_every: int = 0,
        keep_last: int = 3,
        max_g_steps: int | None = None,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.every_seconds = 60.0 * float(every_minutes or 0.0)
        self.milestone_every = int(milestone_every or 0)
        self.keep_last = int(keep_last)
        self.max_g_steps = max_g_steps
        self.signalled = False
        self.signal_number: int | None = None
        self._last_save = time.monotonic()
        self._last_saved_step: int | None = None
        self._previous_handlers: dict[int, object] = {}

    def _on_signal(self, signum, frame) -> None:
        self.signalled = True
        self.signal_number = signum

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._last_save = time.monotonic()
        self.signalled = False
        if trainer.is_global_zero:
            for stale in self.ckpt_dir.glob(f"step*.ckpt{TEMPORARY_SUFFIX}"):
                stale.unlink(missing_ok=True)
        try:
            for sig in PREEMPTION_SIGNALS:
                self._previous_handlers[sig] = signal.signal(sig, self._on_signal)
        except ValueError:
            rank_zero_warn("not in the main thread: preemption signals are not handled")

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        for sig, previous in self._previous_handlers.items():
            if signal.getsignal(sig) == self._on_signal:
                signal.signal(sig, previous if previous is not None else signal.SIG_DFL)
        self._previous_handlers.clear()

    def _is_milestone(self, step: int) -> bool:
        return self.milestone_every > 0 and step % self.milestone_every == 0

    def _save(self, trainer: Trainer, step: int) -> None:
        """Writes ``step<step>.ckpt`` through a temporary file, so a kill during the write leaves no partial file.

        Whether to write is decided from a per-process counter, never from the file system: ranks must all call
        ``save_checkpoint`` (it gathers RNG states) or none of them may.
        """
        if self._last_saved_step != step:
            path = checkpoint_path(self.ckpt_dir, step)
            tmp = path.with_name(path.name + TEMPORARY_SUFFIX)
            trainer.save_checkpoint(tmp)
            if trainer.is_global_zero:
                os.replace(tmp, path)
            self._last_saved_step = step
        self._last_save = time.monotonic()
        if trainer.is_global_zero:
            regular = [p for s, p in list_step_checkpoints(self.ckpt_dir) if not self._is_milestone(s)]
            for old in regular[: max(len(regular) - self.keep_last, 0)]:
                old.unlink(missing_ok=True)

    def _periodic_due(self, step: int) -> bool:
        if self._is_milestone(step):
            return True
        return self.every_seconds > 0 and time.monotonic() - self._last_save >= self.every_seconds

    def _requeue(self) -> None:
        command = os.environ.get(REQUEUE_VAR)
        if command:
            rank_zero_info(f"checkpoint saved, requeueing with: {command}")
            subprocess.run(command, shell=True, check=False)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        step = pl_module.g_step
        due = bool(trainer.strategy.broadcast(self._periodic_due(step), src=0))
        if due:
            self._save(trainer, step)
        if trainer.strategy.reduce_boolean_decision(self.signalled, all=False):
            rank_zero_info(f"preemption signal received: checkpointing at g_step {step}")
            self._save(trainer, step)
            finished = self.max_g_steps is not None and step >= self.max_g_steps
            if trainer.is_global_zero and not finished:
                self._requeue()
            trainer.should_stop = True

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.max_g_steps is not None and pl_module.g_step >= self.max_g_steps:
            self._save(trainer, pl_module.g_step)


def _numpy_state_to_tensors(state: tuple) -> dict:
    name, keys, pos, has_gauss, cached = state
    return {"name": name, "keys": torch.from_numpy(np.asarray(keys).astype(np.int64)), "pos": int(pos),
            "has_gauss": int(has_gauss), "cached": float(cached)}


def _numpy_state_from_tensors(state: dict) -> tuple:
    keys = state["keys"].numpy().astype(np.uint32)
    return state["name"], keys, state["pos"], state["has_gauss"], state["cached"]


def capture_rng_state(device: torch.device | None = None) -> dict:
    """Python, NumPy, torch and (for a CUDA ``device``) CUDA generator states in checkpoint-friendly types."""
    state = {
        "python": random.getstate(),
        "numpy": _numpy_state_to_tensors(np.random.get_state()),
        "torch": torch.get_rng_state(),
    }
    if device is not None and device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict, device: torch.device | None = None) -> None:
    """Inverse of :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(_numpy_state_from_tensors(state["numpy"]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and device is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


class RNGStateCallback(Callback):
    """Saves the RNG state of every rank in the checkpoint and restores it just before the first resumed batch.

    The restore is deferred to ``on_train_batch_start`` because creating the DataLoader iterator draws from the
    global torch generator after the checkpoint has been loaded. A fresh run seeds each rank with ``seed + rank``.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._loaded: list[dict] | None = None
        self._device: torch.device | None = None

    def state_dict(self) -> dict:
        own = capture_rng_state(self._device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, own)
            return {"ranks": gathered}
        return {"ranks": [own]}

    def load_state_dict(self, state_dict: dict) -> None:
        self._loaded = state_dict["ranks"]

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self._device = pl_module.device

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._device = pl_module.device
        if self._loaded is None:
            torch.manual_seed(self.seed + trainer.global_rank)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        if self._loaded is None:
            return
        if len(self._loaded) == trainer.world_size:
            restore_rng_state(self._loaded[trainer.global_rank], pl_module.device)
        else:
            rank_zero_warn(f"checkpoint has {len(self._loaded)} RNG states for {trainer.world_size} ranks: skipped")
        self._loaded = None


def write_wav(path: str | Path, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Writes a mono float32 WAV atomically (temporary file, then ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    sf.write(tmp, np.asarray(wav, dtype=np.float32), sample_rate, format="WAV", subtype="FLOAT")
    os.replace(tmp, path)


class WavWriter(BasePredictionWriter):
    """Saves each predicted utterance to ``<output_dir>/<split>/<condition>/<id>.wav`` (24 kHz, float32)."""

    def __init__(self, output_dir: str | Path, split: str, sample_rate: int = SAMPLE_RATE):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.split = split
        self.sample_rate = sample_rate

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        for i, (utt_id, condition) in enumerate(zip(prediction["id"], prediction["condition"])):
            path = self.output_dir / self.split / condition / f"{utt_id}.wav"
            write_wav(path, prediction["wav"][i, 0].numpy(), self.sample_rate)
