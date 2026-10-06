"""I/O helpers of the Phase 3 evaluation (contract: docs/vocoders/EVALUATION.md sections 0-2).

Utterance table of a split (ids, speaker, chapter, T2 reference), fixed chunking of the sorted id list, atomic writers
(temporary file in the same directory, then ``os.replace``), audio readers and path layout under ``eval_root``, the stop
flag that turns SIGUSR1/SIGTERM into a clean exit with code 75, and the provenance written next to every stage output.
"""

import hashlib
import json
import os
import platform
import signal
import socket
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import soundfile as sf

from sparc.vocoders.constants import HOP, SAMPLE_RATE
from sparc.vocoders.data.dataset import FullUtteranceDataset, PackedStore
from sparc.vocoders.data.gain import apply_gain, gain_factor

if TYPE_CHECKING:
    from sparc.vocoders.eval.systems import SystemSpec

EXIT_STOPPED = 75
STOP_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)
PREDICT_STREAM = 2  # stream of VocoderDataModule.predict_ids; the eval limit uses the same draw
T2_CONDITION = "T2"
SPEAKER_MEAN_REF = "speaker_mean"
EXTRA_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "hydra-core",
    "lightning",
    "jiwer",
    "pesq",
    "pysptk",
    "speechbrain",
    "vocos",
    "huggingface-hub",
)


class StopRequested(Exception):
    """Raised when a stop signal was received; the stage has finished its current chunk and the CLI exits with 75."""


class StopFlag:
    """Flag set by SIGUSR1 and SIGTERM. Stages poll it between chunks; nothing is interrupted mid-chunk."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.signal_number: int | None = None
        self._previous: dict[int, Any] = {}

    def set(self, signum: int | None = None, frame: Any = None) -> None:
        self.signal_number = signum
        self._event.set()

    def clear(self) -> None:
        self._event.clear()
        self.signal_number = None

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_set(self) -> None:
        if self.is_set():
            raise StopRequested(f"stop requested (signal {self.signal_number})")

    def install(self) -> "StopFlag":
        """Installs the handlers (main thread only; elsewhere the flag can still be set by hand)."""
        try:
            for sig in STOP_SIGNALS:
                self._previous[sig] = signal.signal(sig, self.set)
        except ValueError:
            self._previous.clear()
        return self

    def restore(self) -> None:
        for sig, previous in self._previous.items():
            signal.signal(sig, previous if previous is not None else signal.SIG_DFL)
        self._previous.clear()

    @contextmanager
    def installed(self) -> Iterator["StopFlag"]:
        self.install()
        try:
            yield self
        finally:
            self.restore()


# ----------------------------------------------------------------------------------------------- atomic writers


def atomic_write(path: str | Path, writer: Callable[[Path], None], fsync: bool = True) -> Path:
    """Runs ``writer(tmp)`` on a temporary file next to ``path`` and moves it into place; a failure leaves no file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        writer(tmp)
        if fsync:
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_parquet(path: str | Path, table: pd.DataFrame) -> Path:
    return atomic_write(path, lambda tmp: table.to_parquet(tmp, index=False))


def write_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    def write(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)

    return atomic_write(path, write)


def write_json(path: str | Path, obj: Any) -> Path:
    return atomic_write(path, lambda tmp: tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default)))


def write_text(path: str | Path, text: str) -> Path:
    return atomic_write(path, lambda tmp: tmp.write_text(text))


def write_csv(path: str | Path, table: pd.DataFrame) -> Path:
    return atomic_write(path, lambda tmp: table.to_csv(tmp, index=False))


def write_wav(path: str | Path, wav: np.ndarray, sample_rate: int, subtype: str = "FLOAT") -> Path:
    """Mono WAV through a temporary file. ``subtype`` is ``FLOAT`` (float32) or ``PCM_16`` (clipped to [-1, 1])."""
    wav = np.asarray(wav, dtype=np.float32)
    if subtype == "PCM_16":
        wav = np.clip(wav, -1.0, 1.0)
    return atomic_write(
        path, lambda tmp: sf.write(tmp, wav, int(sample_rate), format="WAV", subtype=subtype), fsync=False
    )


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float32 samples and the sample rate of a WAV file."""
    wav, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim != 1:
        raise ValueError(f"{path}: expected a mono file, got shape {wav.shape}")
    return wav, int(rate)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def config_digest(obj: Any) -> str:
    """Short digest of a JSON-compatible object (used to notice that a stage config changed between runs)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- audio helpers


def to_16k(wav: np.ndarray, sample_rate: int, res_type: str = "soxr_hq") -> np.ndarray:
    """16 kHz float32 version of ``wav`` (unchanged for a 16 kHz input)."""
    if sample_rate == 16000:
        return np.asarray(wav, dtype=np.float32)
    import librosa

    return librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sample_rate, target_sr=16000, res_type=res_type)


def expected_samples(T: int, sample_rate: int) -> int:
    """Length of a system's output for an utterance of ``T`` feature frames: ``480 T`` at 24 kHz, ``320 T`` at 16 kHz."""
    return int(T) * (HOP * sample_rate // SAMPLE_RATE)


def read_transcript(wav_path: str | Path) -> str:
    """LibriTTS-R ``<id>.normalized.txt`` next to the original ``<id>.wav``."""
    wav_path = Path(wav_path)
    return wav_path.with_name(wav_path.stem + ".normalized.txt").read_text().strip()


def read_speaker_sex(path: str | Path) -> dict[str, str]:
    """Speaker id -> ``F``/``M`` from LibriTTS ``SPEAKERS.txt`` (pipe separated) or ``speakers.tsv`` (tab separated)."""
    sex: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = [p.strip() for p in (line.split("|") if "|" in line else line.split("\t"))]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("F", "M"):
            sex[parts[0]] = parts[1]
    return sex


# ----------------------------------------------------------------------------------------------- utterance table


def packed_dir(cache_root: str | Path, split: str) -> Path:
    return Path(cache_root) / "packed" / split


def sorted_ids(cache_root: str | Path, split: str) -> list[str]:
    """Sorted utterance ids of a packed split."""
    index = pd.read_parquet(packed_dir(cache_root, split) / "index.parquet", columns=["id"])
    return sorted(index["id"].astype(str))


def select_limit_ids(ids: Sequence[str], limit: int | None, seed: int = 0) -> list[str]:
    """Fixed random subset of ``limit`` ids (all of them if ``limit`` is falsy), sorted.

    Same draw as ``VocoderDataModule.predict_ids`` (``default_rng([seed, 2])`` over the sorted ids), so ``eval.limit = N``
    and ``data.predict_limit = N`` select the same utterances when the seeds agree.
    """
    ids = sorted(ids)
    if not limit or int(limit) >= len(ids):
        return ids
    chosen = np.random.default_rng([int(seed), PREDICT_STREAM]).permutation(len(ids))[: int(limit)]
    return sorted(np.array(ids)[chosen].tolist())


def build_utterance_table(
    cache_root: str | Path, split: str, ref_min_dur: float = 3.0, sex: dict[str, str] | None = None
) -> pd.DataFrame:
    """One row per utterance of ``split``, sorted by id.

    Columns: ``id, speaker, chapter, dur_s, T, n24, peak24, wav_path`` from the packed index; ``ref_id`` (the T2
    reference, chosen by :class:`FullUtteranceDataset` itself, so it is the same utterance the vocoders were
    conditioned on), ``ref_same_chapter`` and ``ref_relaxed`` (its pool flags), ``has_ref`` (False for a speaker with a
    single utterance, which is absent from T2 and T3) and ``sex`` (``F``/``M``/``""``).
    """
    directory = packed_dir(cache_root, split)
    index = pd.read_parquet(directory / "index.parquet")
    table = pd.DataFrame(
        {
            "id": index["id"].astype(str),
            "speaker": index["speaker"].astype(str),
            "chapter": index["chapter"].astype(str),
            "dur_s": index["n24"].to_numpy(dtype=np.float64) / SAMPLE_RATE,
            "T": index["T"].to_numpy(dtype=np.int64),
            "n24": index["n24"].to_numpy(dtype=np.int64),
            "peak24": index["peak24"].to_numpy(dtype=np.float64),
            "wav_path": index["wav_path"].astype(str),
        }
    )
    dataset = FullUtteranceDataset(directory, ids=None, condition=T2_CONDITION, ref_min_dur=ref_min_dur)
    store = dataset.store
    ref_id = {str(store.ids[u]): str(store.ids[r]) for u, r in zip(dataset.items, dataset.references)}
    flags = {str(store.ids[u]): f for u, f in zip(dataset.items, dataset.flags)}
    table["has_ref"] = table["id"].isin(ref_id)
    table["ref_id"] = table["id"].map(ref_id).fillna("")
    table["ref_same_chapter"] = table["id"].map(lambda i: bool(flags[i][0]) if i in flags else False)
    table["ref_relaxed"] = table["id"].map(lambda i: bool(flags[i][1]) if i in flags else False)
    table["sex"] = table["speaker"].map(sex or {}).fillna("")
    return table.sort_values("id").reset_index(drop=True)


def condition_ids(table: pd.DataFrame, condition: str, ids: Sequence[str] | None = None) -> list[str]:
    """Ids evaluated under ``condition``: all for T1, those with a speaker reference for T2 and T3."""
    mask = np.ones(len(table), dtype=bool) if condition == "T1" else table["has_ref"].to_numpy()
    selected = table["id"].to_numpy()[mask]
    if ids is not None:
        selected = np.intersect1d(selected, np.asarray(list(ids), dtype=str))
    return sorted(selected.tolist())


def base_columns(table: pd.DataFrame, ids: Sequence[str], condition: str) -> pd.DataFrame:
    """``id, speaker, chapter, dur_s, ref_id`` rows of ``ids`` (in that order) for a result part.

    ``ref_id`` is what ``FullUtteranceDataset`` reports per condition: the utterance itself for T1, the T2 reference for
    T2 and ``speaker_mean`` for T3.
    """
    ids = list(ids)
    rows = table.set_index("id").loc[ids]
    if condition == "T1":
        refs = np.array(ids, dtype=object)
    elif condition == "T2":
        refs = rows["ref_id"].to_numpy()
    else:
        refs = np.full(len(ids), SPEAKER_MEAN_REF, dtype=object)
    return pd.DataFrame(
        {
            "id": ids,
            "speaker": rows["speaker"].to_numpy(),
            "chapter": rows["chapter"].to_numpy(),
            "dur_s": rows["dur_s"].to_numpy(),
            "ref_id": refs,
        }
    )


def chapter_of(utterance_id: str) -> str:
    """Chapter of a LibriTTS id (second ``_`` field)."""
    return utterance_id.split("_")[1]


# ----------------------------------------------------------------------------------------------- chunking


def chunk_plan(ids: Sequence[str], chunk_size: int) -> list[list[str]]:
    """Fixed chunks of ``chunk_size`` consecutive ids of the sorted list; chunk ``k`` is part ``part-<k:04d>``."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    ids = list(ids)
    return [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]


def chunk_for_condition(chunks: Sequence[Sequence[str]], allowed: Sequence[str]) -> list[list[str]]:
    """Chunks restricted to ``allowed`` ids, keeping the chunk numbering (a chunk may become empty)."""
    keep = set(allowed)
    return [[i for i in chunk if i in keep] for chunk in chunks]


def part_name(k: int, suffix: str) -> str:
    return f"part-{k:04d}.{suffix}"


@dataclass
class ChunkReport:
    """Which chunks a run computed and which it found finished."""

    n_chunks: int = 0
    computed: list[int] | None = None
    skipped: list[int] | None = None

    def __post_init__(self) -> None:
        self.computed = [] if self.computed is None else self.computed
        self.skipped = [] if self.skipped is None else self.skipped


def run_chunks(
    chunks: Sequence[Sequence[str]],
    is_done: Callable[[int, Sequence[str]], bool],
    work: Callable[[int, Sequence[str]], None],
    stop: StopFlag,
    report: ChunkReport | None = None,
) -> ChunkReport:
    """Calls ``work(k, ids)`` for every non-empty chunk that is not done.

    The stop flag is checked before each chunk that still has to be computed, so the chunk in progress always
    finishes; a stop with work remaining raises :class:`StopRequested` (``report`` holds what was done up to then). A
    stop that arrives during the last chunk lets the stage complete normally.
    """
    report = ChunkReport(n_chunks=len(chunks)) if report is None else report
    report.n_chunks = len(chunks)
    for k, ids in enumerate(chunks):
        if not ids:
            continue
        if is_done(k, ids):
            report.skipped.append(k)
            continue
        stop.raise_if_set()
        work(k, ids)
        report.computed.append(k)
    return report


def list_parts(directory: str | Path, suffix: str) -> list[tuple[int, Path]]:
    """``(k, path)`` of every finished ``part-<k>.<suffix>`` in ``directory``, ordered by ``k``."""
    found = []
    for path in Path(directory).glob(f"part-*.{suffix}"):
        stem = path.name[len("part-") : -len(suffix) - 1]
        if stem.isdigit():
            found.append((int(stem), path))
    return sorted(found)


def read_parquet_parts(directory: str | Path) -> pd.DataFrame:
    """All finished parquet parts of a result group concatenated (empty frame if there are none)."""
    parts = [pd.read_parquet(path) for _, path in list_parts(directory, "parquet")]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def read_npz_part(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


# ----------------------------------------------------------------------------------------------- gt audio


def gt_audio(store: PackedStore, utt: int, gain_db: float) -> np.ndarray:
    """Reference audio of utterance ``utt``: the first ``480 T`` samples times the peak gain, as ``FullUtteranceDataset``."""
    frames = int(store.T[utt])
    g = gain_factor(store.peak24[utt], gain_db)
    return apply_gain(store.read_audio(utt, 0, HOP * frames), g)


# ----------------------------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class EvalPaths:
    """Locations under ``eval_root`` (contract section 2). A smoke run (``limit``) lives under ``eval_root/smoke_<N>``."""

    eval_root: Path
    runs_root: Path
    split: str
    limit: int | None = None

    @property
    def root(self) -> Path:
        return self.eval_root / f"smoke_{self.limit}" if self.limit else self.eval_root

    def audio_name(self, spec: "SystemSpec") -> str:
        return spec.audio_from or spec.name

    def predictions_root(self, spec: "SystemSpec") -> Path:
        """Directory whose ``<split>/<cond>/<id>.wav`` files are the system's audio (the run's ``predictions`` dir)."""
        if spec.kind == "vocoder" and not self.limit:
            return self.runs_root / str(spec.experiment) / str(spec.vocoder) / "predictions"
        return self.root / "audio" / self.audio_name(spec)

    def audio_dir(self, spec: "SystemSpec", condition: str) -> Path:
        return self.predictions_root(spec) / self.split / condition

    def audio_path(self, spec: "SystemSpec", condition: str, utterance_id: str) -> Path:
        return self.audio_dir(spec, condition) / f"{utterance_id}.wav"

    def audio_failures_path(self, spec: "SystemSpec", condition: str) -> Path:
        """``failures.json`` of a reference system's audio directory: ids whose audio could not be made, with the error."""
        return self.audio_dir(spec, condition) / "failures.json"

    def synth_meta_path(self, spec: "SystemSpec") -> Path:
        return self.predictions_root(spec) / self.split / "synth_meta.json"

    def results_dir(self, system: str, condition: str, group: str) -> Path:
        return self.root / "results" / self.split / system / condition / group

    def arrays_dir(self, system: str, condition: str, name: str = "reextract") -> Path:
        return self.root / "arrays" / self.split / system / condition / name

    def embeddings_dir(self, system: str, condition: str, model: str) -> Path:
        return self.root / "embeddings" / self.split / system / condition / model

    def probes_dir(self, system: str) -> Path:
        return self.root / "probes" / system

    def efficiency_path(self, system: str, suffix: str = "") -> Path:
        return self.root / "efficiency" / f"{system}{suffix}.json"

    def tables_dir(self) -> Path:
        return self.root / "tables" / self.split

    def samples_dir(self) -> Path:
        return self.root / "samples"


def read_system_audio(paths: EvalPaths, spec: "SystemSpec", condition: str, utterance_id: str) -> np.ndarray:
    """Audio of a system for one utterance; checks the sample rate against the registry."""
    wav, rate = read_wav(paths.audio_path(spec, condition, utterance_id))
    if rate != spec.sr:
        raise ValueError(f"{spec.name}/{condition}/{utterance_id}: sample rate {rate}, expected {spec.sr}")
    return wav


def read_audio_failures(paths: EvalPaths, spec: "SystemSpec", condition: str) -> dict[str, str]:
    """``id -> error`` of the utterances the ``refs`` stage could not synthesize for ``spec`` (empty if none)."""
    path = paths.audio_failures_path(spec, condition)
    return {str(k): str(v) for k, v in read_json(path).items()} if path.is_file() else {}


def audio_problems(
    paths: EvalPaths, spec: "SystemSpec", condition: str, table: pd.DataFrame, ids: Sequence[str]
) -> dict[str, str]:
    """Id -> problem for system audio that is missing or has the wrong rate or length (header reads only)."""
    frames = table.set_index("id")["T"]
    problems: dict[str, str] = {}
    for uid in ids:
        path = paths.audio_path(spec, condition, uid)
        try:
            info = sf.info(str(path))
        except (OSError, RuntimeError) as error:
            problems[uid] = f"unreadable: {error}"
            continue
        want = expected_samples(int(frames[uid]), spec.sr)
        if info.samplerate != spec.sr or info.frames != want or info.channels != 1:
            problems[uid] = f"{info.samplerate} Hz x {info.frames} samples, expected {spec.sr} Hz x {want}"
    return problems


# ----------------------------------------------------------------------------------------------- provenance


def package_versions(extra: Sequence[str] = EXTRA_PACKAGES) -> dict[str, str | None]:
    """Versions of the packages that influence the features (the extractor's list) plus the evaluation packages."""
    import importlib.metadata

    from sparc.vocoders.features.extractor import package_versions as extractor_versions

    versions = dict(extractor_versions())
    for name in extra:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def provenance() -> dict[str, Any]:
    """Who ran what where: fork commit, package versions, GPU, host and job."""
    from sparc.vocoders.features.extractor import fork_commit

    return {
        "fork_commit": fork_commit(),
        "versions": package_versions(),
        "python": platform.python_version(),
        "gpu": gpu_name(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT"),
    }


class StageMeta:
    """``meta.json`` of one stage output directory: configuration, fingerprint and the list of runs that wrote to it.

    ``chunk_size``, ``fingerprint`` (the checkpoint of a vocoder system, for example) and ``ids_digest`` (the ids the
    chunks are cut from, so another ``eval.limit``/``eval.seed`` subset cannot reuse numbered parts) must not change
    between runs that share a directory, otherwise parts from different settings would be mixed; a mismatch raises. A change of the
    configuration digest alone only sets ``config_changed`` on the run record.
    """

    def __init__(self, path: str | Path, header: dict[str, Any], strict: bool = True):
        self.path = Path(path)
        self.header = header
        self.strict = strict
        self.data: dict[str, Any] = {}
        self.run: dict[str, Any] | None = None

    def verify(self) -> None:
        """Raises if an existing ``meta.json`` was written with another ``chunk_size``, ``fingerprint`` or ``ids_digest``.

        Called before any chunk is judged finished, so a directory made with other settings is never reused silently,
        also when every part exists and nothing would run.
        """
        existing = read_json(self.path) if self.path.exists() else {}
        if not (existing and self.strict):
            return
        for key in ("chunk_size", "fingerprint", "ids_digest"):
            if key not in self.header:
                continue
            if key == "ids_digest" and key not in existing:
                continue  # written by older runs without the id digest
            if existing.get(key) != self.header[key]:
                raise RuntimeError(
                    f"{self.path.parent}: existing results were made with {key}={existing.get(key)!r}, now "
                    f"{self.header[key]!r}; delete the directory (or set eval.allow_fingerprint_change=true) to recompute"
                )

    def begin(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        self.verify()
        existing = read_json(self.path) if self.path.exists() else {}
        self.data = {**existing, **{k: v for k, v in self.header.items() if k != "config_digest"}}
        previous = existing.get("config_digest")
        self.data["config_digest"] = self.header.get("config_digest")
        self.data.setdefault("runs", [])
        self.run = {
            "started": now_iso(),
            "finished": None,
            "status": "running",
            "config_changed": bool(previous and previous != self.header.get("config_digest")),
            **provenance(),
            **(extra or {}),
        }
        self.data["runs"].append(self.run)
        write_json(self.path, self.data)
        return self.run

    def update(self, **extra: Any) -> None:
        """Adds fields to the current run record and writes the file (for facts known only after the first chunk)."""
        if self.run is None:
            return
        self.run.update(extra)
        write_json(self.path, self.data)

    def end(self, status: str, report: ChunkReport | None = None, **extra: Any) -> None:
        if self.run is None:
            return
        self.run.update(finished=now_iso(), status=status, **extra)
        if report is not None:
            self.run.update(computed_chunks=report.computed, skipped_chunks=report.skipped, n_chunks=report.n_chunks)
        write_json(self.path, self.data)
