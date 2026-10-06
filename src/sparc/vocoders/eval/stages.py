"""Stage drivers of the Phase 3 evaluation (contract: docs/vocoders/EVALUATION.md sections 1-8 and 10).

One function per stage. Every chunked stage splits the sorted id list of the split into fixed chunks
(``eval.chunk_size``), skips chunks whose output exists, writes each chunk atomically and checks the stop flag between
chunks (a stop raises :class:`~sparc.vocoders.eval.io.StopRequested`, the CLI exits with 75). A stage that has nothing
left to do loads no model and writes no ``meta.json`` entry, so repeating it is free. The metric and feature code lives
in ``metrics/`` and the other modules of this package; this module only reads the audio, calls it and stores the rows.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from sparc.vocoders.constants import SAMPLE_RATE
from sparc.vocoders.data.dataset import PackedStore
from sparc.vocoders.data.gain import gain_factor
from sparc.vocoders.eval.io import (
    ChunkReport,
    EvalPaths,
    StageMeta,
    StopFlag,
    StopRequested,
    audio_problems,
    base_columns,
    build_utterance_table,
    chunk_for_condition,
    chunk_plan,
    condition_ids,
    config_digest,
    expected_samples,
    gt_audio,
    packed_dir,
    part_name,
    read_audio_failures,
    read_json,
    read_npz_part,
    read_speaker_sex,
    read_system_audio,
    read_wav,
    run_chunks,
    select_limit_ids,
    to_16k,
    write_npz,
    write_parquet,
    write_json,
    write_text,
    write_wav,
)
from sparc.vocoders.eval.systems import (
    CheckpointInfo,
    SystemSpec,
    compose_vocoder_config,
    hydra_cleared,
    load_systems,
    resolve_checkpoint,
    resolve_items,
    synth_fingerprint,
)

logger = logging.getLogger(__name__)

GT_SYSTEM = "gt"
EXTRACTOR_SYSTEM = "extractor"  # pseudo system of the efficiency stage: the feature extraction cost, reported once
REFERENCE_SYSTEMS = ("vocos_mel", "enplus16")
SPEAKER_PART_SUFFIX = "npz"


class MissingPrerequisite(FileNotFoundError):
    """An earlier stage has not produced what this one reads (``system=all`` goes on with the other systems)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


# ----------------------------------------------------------------------------------------------- context


class EvalContext:
    """Everything a stage needs: config, layout, utterance table, packed store, stop flag, device and models.

    Models are built on first use (``model(key)``) and kept for the life of the context, so ``system=all`` loads each
    evaluation model once. ``key`` is ``utmos``, ``asr``, ``spk:<name>``, ``reextract:<head>``, ``vocos_mel`` or
    ``enplus16``; tests replace :func:`make_model`.
    """

    def __init__(self, cfg: DictConfig, stop: StopFlag | None = None):
        self.cfg = cfg
        self.split = str(cfg.eval.split)
        self.limit = int(cfg.eval.limit) if cfg.eval.limit else None
        self.paths = EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), self.split, self.limit)
        self.systems = load_systems(cfg)
        self.stop = stop if stop is not None else StopFlag()
        self.chunk_size = int(cfg.eval.chunk_size)
        self.gain_db = float(cfg.eval.gain_db)
        self._table: pd.DataFrame | None = None
        self._ids: list[str] | None = None
        self._store: PackedStore | None = None
        self._device: Any = None
        self._models: dict[str, Any] = {}
        self._stats: dict | None = None

    @property
    def cache_root(self) -> Path:
        return Path(self.cfg.paths.cache_root)

    @property
    def table(self) -> pd.DataFrame:
        """Utterance table of the whole split (``io.build_utterance_table``), sex filled in when the speaker list exists."""
        if self._table is None:
            sex_file = self.cfg.eval.get("sex_file")
            sex = read_speaker_sex(sex_file) if sex_file and Path(str(sex_file)).is_file() else {}
            if not sex:
                logger.warning("no speaker sex list at %s: the samples stage cannot balance sexes", sex_file)
            self._table = build_utterance_table(self.cache_root, self.split, float(self.cfg.data.ref_min_dur), sex)
        return self._table

    @property
    def ids(self) -> list[str]:
        """Sorted ids evaluated: the whole split, or the fixed random subset of ``eval.limit`` ids."""
        if self._ids is None:
            self._ids = select_limit_ids(self.table["id"].tolist(), self.limit, int(self.cfg.eval.seed))
        return self._ids

    @property
    def store(self) -> PackedStore:
        if self._store is None:
            self._store = PackedStore([packed_dir(self.cache_root, self.split)], "l6")
        return self._store

    @property
    def stats(self) -> dict:
        """Training statistics (``ema_std`` is the unit of the EMA RMSE)."""
        if self._stats is None:
            self._stats = read_json(self.cfg.stats_path)
        return self._stats

    @property
    def device(self):
        if self._device is None:
            import torch

            want = str(self.cfg.eval.device)
            self._device = torch.device("cuda" if want == "auto" and torch.cuda.is_available() else ("cpu" if want == "auto" else want))
        return self._device

    @property
    def gt_spec(self) -> SystemSpec:
        return self.systems[GT_SYSTEM]

    def model(self, key: str) -> Any:
        if key not in self._models:
            logger.info("loading model %s on %s", key, self.device)
            self._models[key] = make_model(self, key)
        return self._models[key]

    def condition_ids(self, condition: str) -> list[str]:
        return condition_ids(self.table, condition, self.ids)

    def chunks(self, condition: str) -> list[list[str]]:
        """Chunks of the sorted id list restricted to the ids of ``condition`` (chunk numbers stay those of the full list)."""
        return chunk_for_condition(chunk_plan(self.ids, self.chunk_size), self.condition_ids(condition))

    def read_gt(self, utterance_id: str) -> np.ndarray:
        wav, rate = read_wav(self.paths.audio_path(self.gt_spec, "T1", utterance_id))
        if rate != SAMPLE_RATE:
            raise ValueError(f"gt {utterance_id}: sample rate {rate}")
        return wav

    def gt_from_store(self, utterance_id: str) -> np.ndarray:
        """The gt audio computed from the packed store (what the ``gt`` stage writes), without needing its files."""
        return gt_audio(self.store, self.store.index_of(utterance_id), self.gain_db)


def make_model(ctx: EvalContext, key: str) -> Any:
    """Builds an evaluation model; the imports are local so that a stage only loads what it needs."""
    cfg, device = ctx.cfg, ctx.device
    if key == "utmos":
        from sparc.vocoders.eval.metrics.utmos import UTMOS

        return UTMOS(cfg.eval_metrics.utmos, device)
    if key == "asr":
        from sparc.vocoders.eval.metrics.asr import WhisperASR

        return WhisperASR(cfg.eval_metrics.asr, device)
    if key.startswith("spk:"):
        from sparc.vocoders.eval.metrics.speaker import SpeakerEmbedder

        return SpeakerEmbedder(key.split(":", 1)[1], cfg.eval_metrics.speaker, device)
    if key.startswith("reextract:"):
        from sparc.vocoders.eval.reextract import ReExtractor

        return ReExtractor(cfg, key.split(":", 1)[1], device)
    if key == "vocos_mel":
        from sparc.vocoders.eval.references import VocosMelReference

        return VocosMelReference(cfg, device)
    if key == "enplus16":
        from sparc.vocoders.eval.references import EnPlusReference

        return EnPlusReference(cfg, device)
    raise KeyError(f"unknown model {key!r}")


def describe_model(model: Any) -> Any:
    """Provenance record of a model object (``provenance()`` or ``describe()``), or its class name."""
    for name in ("provenance", "describe"):
        if hasattr(model, name):
            try:
                return getattr(model, name)()
            except Exception as error:  # provenance must never fail a stage
                return {"error": error_text(error)}
    return type(model).__name__


# ----------------------------------------------------------------------------------------------- chunk driver


def _cfg_digest(ctx: EvalContext, *groups: str) -> str:
    return config_digest({g: OmegaConf.to_container(ctx.cfg[g], resolve=True) for g in groups})


def fingerprint_of(ctx: EvalContext, spec: SystemSpec) -> dict | None:
    """The checkpoint a vocoder system's audio was synthesized from (from ``synth_meta.json``); ``None`` otherwise.

    Result directories of a vocoder system refuse to mix parts computed on audio from different checkpoints.
    """
    if spec.kind != "vocoder":
        return None
    path = ctx.paths.synth_meta_path(spec)
    if not path.is_file():
        raise MissingPrerequisite(f"{path} is missing: run stage=synth for {spec.name} first")
    return read_json(path)["fingerprint"]


def drive_chunks(
    ctx: EvalContext,
    *,
    stage: str,
    spec_name: str,
    condition: str,
    out_dir: Path,
    is_done: Callable[[int, Sequence[str]], bool],
    work: Callable[[int, Sequence[str]], None],
    fingerprint: dict | None = None,
    config_groups: Sequence[str] = (),
    models: Callable[[], dict] | None = None,
    precheck: Callable[[list[str]], None] | None = None,
    ids_guard: bool = True,
) -> ChunkReport:
    """Runs ``work(k, ids)`` for every unfinished chunk of ``condition`` and keeps ``out_dir/meta.json`` up to date.

    ``precheck(ids)`` gets the ids of the unfinished chunks before anything is written and raises
    :class:`MissingPrerequisite` if the input audio is not there, so a missing earlier stage never produces finished
    parts full of error rows.
    """
    chunks = ctx.chunks(condition)
    report = ChunkReport(n_chunks=len(chunks))
    header: dict[str, Any] = {
        "stage": stage,
        "system": spec_name,
        "condition": condition,
        "split": ctx.split,
        "limit": ctx.limit,
        "chunk_size": ctx.chunk_size,
        "n_ids": len(ctx.condition_ids(condition)),
        "config_digest": _cfg_digest(ctx, *config_groups),
    }
    if ids_guard:  # numbered parts are only valid for the id list they were cut from (audio files are keyed by id)
        header["ids_digest"] = config_digest(ctx.condition_ids(condition))
    if fingerprint is not None:
        header["fingerprint"] = fingerprint
    meta = StageMeta(out_dir / "meta.json", header, strict=not bool(ctx.cfg.eval.allow_fingerprint_change))
    meta.verify()  # before any part is judged finished: parts of other settings must not be reused silently
    todo = [k for k, ids in enumerate(chunks) if ids and not is_done(k, ids)]
    if not todo:
        report.skipped = [k for k, ids in enumerate(chunks) if ids]
        logger.info("%s %s %s: all %d chunks done", stage, spec_name, condition, len(report.skipped))
        return report
    if precheck is not None:
        precheck([uid for k in todo for uid in chunks[k]])
    meta.begin()
    started = time.time()

    def timed_work(k: int, ids: Sequence[str]) -> None:
        t0 = time.time()
        work(k, ids)
        logger.info("%s %s %s: chunk %d (%d utterances) in %.1f s", stage, spec_name, condition, k, len(ids), time.time() - t0)
        if models is not None:
            meta.update(models=models())

    try:
        run_chunks(chunks, is_done, timed_work, ctx.stop, report)
    except StopRequested:
        meta.end("stopped", report, seconds=time.time() - started)
        raise
    except BaseException as error:
        meta.end("failed", report, error=error_text(error), seconds=time.time() - started)
        raise
    meta.end("done", report, seconds=time.time() - started)
    return report


def require_audio(ctx: EvalContext, spec: SystemSpec, condition: str, ids: Sequence[str]) -> None:
    """Raises :class:`MissingPrerequisite` unless every id has a readable WAV of the right rate and length for ``spec``."""
    known = read_audio_failures(ctx.paths, spec, condition)  # utterances the refs stage could not synthesize: scored as NaN
    problems = audio_problems(ctx.paths, spec, condition, ctx.table, [uid for uid in ids if uid not in known])
    if problems:
        first = next(iter(problems.items()))
        producer = {"gt": "gt", "reference": "refs", "vocoder": "synth"}[spec.kind if not spec.audio_from else "gt"]
        raise MissingPrerequisite(
            f"{spec.name}/{condition}: {len(problems)} of {len(ids)} audio files missing or wrong (run stage={producer} first), e.g. {first}"
        )


def part_frame(ctx: EvalContext, ids: Sequence[str], condition: str, columns: dict[str, Sequence], errors: Sequence[str]) -> pd.DataFrame:
    """Rows of one result part: ``id, speaker, chapter, dur_s, ref_id``, the metric columns and ``err``."""
    frame = base_columns(ctx.table, ids, condition)
    for name, values in columns.items():
        frame[name] = list(values)
    frame["err"] = list(errors)
    return frame


def read_system_wavs(ctx: EvalContext, spec: SystemSpec, condition: str, ids: Sequence[str]) -> tuple[list[np.ndarray | None], list[str]]:
    """Audio of each id at the system's own rate; ``None`` and the error text for a file that cannot be read."""
    wavs: list[np.ndarray | None] = []
    errors: list[str] = []
    for uid in ids:
        try:
            wavs.append(read_system_audio(ctx.paths, spec, condition, uid))
            errors.append("")
        except Exception as error:
            wavs.append(None)
            errors.append(error_text(error))
    return wavs, errors


def _scatter(valid: list[int], values: Sequence, n: int, fill: Any) -> list:
    out = [fill] * n
    for i, v in zip(valid, values):
        out[i] = v
    return out


# ----------------------------------------------------------------------------------------------- audio stages


def _all_exist(ctx: EvalContext, spec: SystemSpec, condition: str) -> Callable[[int, Sequence[str]], bool]:
    return lambda k, ids: all(ctx.paths.audio_path(spec, condition, uid).is_file() for uid in ids)


def _audio_stage(
    ctx: EvalContext,
    stage: str,
    spec: SystemSpec,
    condition: str,
    make: Callable[[str], np.ndarray],
    models: Callable[[], dict] | None = None,
    tolerate_failures: bool = False,
) -> ChunkReport:
    """Writes ``make(id)`` as ``audio/<system>/<split>/<cond>/<id>.wav`` for the ids that have no file yet.

    With ``tolerate_failures`` (the ``refs`` stage) an utterance that the reference model cannot process (for example the
    shortest test utterance, whose ``480 T`` gt samples are below what SPARC's filters accept) does not abort the chunk:
    its error goes to ``failures.json`` next to the audio and the metric stages score it as NaN with that error. At most
    ``eval.max_audio_failures`` utterances may fail per system and condition, so a systematic error still stops the stage,
    and failed ids are tried again on every run.
    """
    frames = ctx.table.set_index("id")["T"]
    budget = int(ctx.cfg.eval.max_audio_failures)
    failures_path = ctx.paths.audio_failures_path(spec, condition)
    failures = read_audio_failures(ctx.paths, spec, condition)

    def work(k: int, ids: Sequence[str]) -> None:
        for uid in ids:
            path = ctx.paths.audio_path(spec, condition, uid)
            if path.is_file():
                continue
            try:
                wav = np.asarray(make(uid), dtype=np.float32)
            except Exception as error:
                if not tolerate_failures:
                    raise
                logger.warning("%s %s/%s/%s: no audio, %s", stage, spec.name, condition, uid, error_text(error))
                failures[uid] = error_text(error)
                write_json(failures_path, failures)
                if len(failures) > budget:
                    raise RuntimeError(
                        f"{spec.name}/{condition}: more than eval.max_audio_failures={budget} utterances failed: {sorted(failures)}"
                    ) from error
                continue
            want = expected_samples(int(frames[uid]), spec.sr)
            if wav.shape != (want,):
                raise ValueError(f"{spec.name}/{condition}/{uid}: produced shape {wav.shape}, expected ({want},)")
            write_wav(path, wav, spec.sr)
            if failures.pop(uid, None) is not None:  # a failure of an earlier run that is gone now
                write_json(failures_path, failures)

    return drive_chunks(
        ctx,
        stage=stage,
        spec_name=spec.name,
        condition=condition,
        out_dir=ctx.paths.audio_dir(spec, condition),
        is_done=_all_exist(ctx, spec, condition),
        work=work,
        config_groups=("eval_features",) if stage == "refs" else (),
        models=models,
        ids_guard=False,
    )


def stage_gt(ctx: EvalContext, spec: SystemSpec, condition: str) -> ChunkReport:
    """Materializes the gt audio: the first ``480 T`` samples times the eval gain, float32, 24 kHz."""
    return _audio_stage(ctx, "gt", spec, condition, ctx.gt_from_store)


def stage_refs(ctx: EvalContext, spec: SystemSpec, condition: str) -> ChunkReport:
    """Synthesizes a reference system (``vocos_mel``: mel copy synthesis; ``enplus16``: SPARC en+) from the gt audio."""
    frames = ctx.table.set_index("id")["T"]
    if spec.name == "vocos_mel":

        def make(uid: str) -> np.ndarray:
            return ctx.model("vocos_mel").synth(ctx.gt_from_store(uid))

        return _audio_stage(
            ctx, "refs", spec, condition, make, models=lambda: {"vocos_mel": describe_model(ctx.model("vocos_mel"))}, tolerate_failures=True
        )
    if spec.name == "enplus16":
        references = ctx.table.set_index("id")["ref_id"]
        speaker_embeddings: dict[str, np.ndarray] = {}  # en+ speaker embedding by utterance id (own encoding)

        def make(uid: str) -> np.ndarray:
            enplus = ctx.model("enplus16")
            code = enplus.encode(ctx.gt_from_store(uid))
            speaker_embeddings[uid] = enplus.spk_emb(code)
            if condition == "T1":
                speaker = speaker_embeddings[uid]
            else:  # T2: the en+ embedding of the reference utterance, obtained the same way (encode its gt audio)
                ref = str(references[uid])
                if ref not in speaker_embeddings:
                    speaker_embeddings[ref] = enplus.spk_emb(enplus.encode(ctx.gt_from_store(ref)))
                speaker = speaker_embeddings[ref]
            return enplus.decode(code, speaker, expected_samples(int(frames[uid]), spec.sr))

        return _audio_stage(
            ctx, "refs", spec, condition, make, models=lambda: {"enplus16": describe_model(ctx.model("enplus16"))}, tolerate_failures=True
        )
    raise ValueError(f"no synthesis is defined for reference system {spec.name!r}")


class StopAfterBatch(Callback):
    """Raises :class:`StopRequested` after the batch in which the stop flag was set (the WAV of that batch is written)."""

    def __init__(self, stop: StopFlag):
        self.stop = stop

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.stop.raise_if_set()


def stage_synth(ctx: EvalContext, spec: SystemSpec, conditions: Sequence[str]) -> None:
    """Synthesizes the split with a trained vocoder through ``sparc-predict`` (``predict.skip_existing=true``).

    The checkpoint is resolved once and pinned (``predict.ckpt``), recorded with its ``g_step`` in
    ``<predictions>/<split>/synth_meta.json`` and must be the final one (``eval.require_final``). A later run that finds
    a different checkpoint refuses to continue in the same directory.
    """
    from sparc.cli.predict_vocoder import run as predict_run
    from sparc.vocoders.data.datamodule import VocoderDataModule

    accelerator = "cuda" if ctx.device.type == "cuda" else "cpu"
    out_dir = ctx.paths.predictions_root(spec)
    vcfg = compose_vocoder_config(ctx.cfg, spec, conditions, out_dir, accelerator, ctx.limit)
    info = resolve_checkpoint(vcfg)
    if bool(ctx.cfg.eval.require_final) and not info.final:
        raise RuntimeError(
            f"{spec.name}: newest checkpoint {info.path} is at g_step {info.g_step} of {info.max_g_steps}; the evaluation "
            "uses final checkpoints only (set eval.require_final=false for a smoke run)"
        )
    vcfg = compose_vocoder_config(ctx.cfg, replace(spec, ckpt=info.path), conditions, out_dir, accelerator, ctx.limit)
    datamodule = VocoderDataModule(vcfg)
    if datamodule.has_pending_prediction():
        header = {
            "stage": "synth",
            "system": spec.name,
            "split": ctx.split,
            "limit": ctx.limit,
            "conditions": list(conditions),
            "checkpoint": info.as_dict(),
            "fingerprint": synth_fingerprint(info),
            "config_digest": config_digest(OmegaConf.to_container(vcfg, resolve=True)),
        }
        meta = StageMeta(ctx.paths.synth_meta_path(spec), header, strict=not bool(ctx.cfg.eval.allow_fingerprint_change))
        meta.begin()
        started = time.time()
        try:
            predict_run(vcfg, datamodule=datamodule, callbacks=[StopAfterBatch(ctx.stop)])
        except StopRequested:
            meta.end("stopped", seconds=time.time() - started)
            raise
        except BaseException as error:
            meta.end("failed", error=error_text(error), seconds=time.time() - started)
            raise
        meta.end("done", seconds=time.time() - started)
    for condition in conditions:
        ids = ctx.condition_ids(condition)
        problems = audio_problems(ctx.paths, spec, condition, ctx.table, ids)
        if problems:
            first = next(iter(problems.items()))
            raise RuntimeError(f"{spec.name}/{condition}: {len(problems)} of {len(ids)} predictions missing or wrong, e.g. {first}")


# ----------------------------------------------------------------------------------------------- metric stages


def results_dir(ctx: EvalContext, spec: SystemSpec, condition: str, group: str) -> Path:
    return ctx.paths.results_dir(spec.name, condition, group)


def _part_done(directory: Path, suffix: str) -> Callable[[int, Sequence[str]], bool]:
    return lambda k, ids: (directory / part_name(k, suffix)).is_file()


def _metric_stage(
    ctx: EvalContext,
    stage: str,
    spec: SystemSpec,
    condition: str,
    compute: Callable[[Sequence[str]], tuple[dict[str, Sequence], list[str]]],
    models: Callable[[], dict] | None,
    config_groups: Sequence[str],
    needs_gt: bool = False,
) -> ChunkReport:
    """Result parquet per chunk: ``compute(ids)`` returns the metric columns and one error string per id."""
    directory = results_dir(ctx, spec, condition, stage)

    def precheck(ids: list[str]) -> None:
        require_audio(ctx, spec, condition, ids)
        if needs_gt:
            require_audio(ctx, ctx.gt_spec, "T1", ids)

    def work(k: int, ids: Sequence[str]) -> None:
        columns, errors = compute(ids)
        write_parquet(directory / part_name(k, "parquet"), part_frame(ctx, ids, condition, columns, errors))

    return drive_chunks(
        ctx,
        stage=stage,
        spec_name=spec.name,
        condition=condition,
        out_dir=directory,
        is_done=_part_done(directory, "parquet"),
        work=work,
        fingerprint=fingerprint_of(ctx, spec),
        config_groups=config_groups,
        models=models,
        precheck=precheck,
    )


def stage_signal(ctx: EvalContext, spec: SystemSpec, condition: str) -> ChunkReport:
    """PESQ, MCD, MR-STFT and mel L1 of the system audio against the gt audio (EVALUATION 4.1)."""
    from sparc.vocoders.eval.metrics.signal import signal_metrics

    signal_cfg = ctx.cfg.eval_metrics.signal

    def compute(ids: Sequence[str]) -> tuple[dict[str, Sequence], list[str]]:
        rows, errors = [], []
        for uid in ids:
            try:
                rows.append(signal_metrics(ctx.read_gt(uid), read_system_audio(ctx.paths, spec, condition, uid), spec.sr, signal_cfg))
                errors.append("")
            except Exception as error:
                logger.warning("signal %s/%s/%s: %s", spec.name, condition, uid, error_text(error))
                rows.append({})
                errors.append(error_text(error))
        frame = pd.DataFrame(rows, index=range(len(ids)))
        return {c: frame[c].to_numpy(dtype=np.float64) for c in frame.columns}, errors

    return _metric_stage(ctx, "signal", spec, condition, compute, None, ("eval_metrics",), needs_gt=True)


def stage_utmos(ctx: EvalContext, spec: SystemSpec, condition: str) -> ChunkReport:
    """UTMOS22 strong of the system audio at 16 kHz (EVALUATION 4.2)."""

    def compute(ids: Sequence[str]) -> tuple[dict[str, Sequence], list[str]]:
        wavs, errors = read_system_wavs(ctx, spec, condition, ids)
        valid = [i for i, w in enumerate(wavs) if w is not None]
        inputs = [to_16k(wavs[i], spec.sr) for i in valid]
        model_errors: list[str] = []
        scores = ctx.model("utmos").score(inputs, model_errors) if inputs else []
        for i, message in zip(valid, model_errors):
            errors[i] = message
        return {"utmos": _scatter(valid, scores, len(ids), np.nan)}, errors

    return _metric_stage(ctx, "utmos", spec, condition, compute, lambda: {"utmos": describe_model(ctx.model("utmos"))}, ("eval_metrics",))


ASR_COUNT_COLUMNS = ("word_errors", "word_ref_len", "char_errors", "char_ref_len")


def stage_asr(ctx: EvalContext, spec: SystemSpec, condition: str) -> ChunkReport:
    """Whisper large-v3 transcripts and edit counts against the LibriTTS-R text (EVALUATION 4.3)."""
    from sparc.vocoders.eval.metrics.asr import edit_counts, normalize_text, read_reference

    wav_paths = ctx.table.set_index("id")["wav_path"]

    def compute(ids: Sequence[str]) -> tuple[dict[str, Sequence], list[str]]:
        wavs, errors = read_system_wavs(ctx, spec, condition, ids)
        valid = [i for i, w in enumerate(wavs) if w is not None]
        model_errors: list[str] = []
        hyps = ctx.model("asr").transcribe([to_16k(wavs[i], spec.sr) for i in valid], model_errors) if valid else []
        for i, message in zip(valid, model_errors):
            errors[i] = message
        hyp = _scatter(valid, hyps, len(ids), "")
        columns: dict[str, list] = {"hyp": hyp, "ref_norm": [], "hyp_norm": [], **{c: [] for c in ASR_COUNT_COLUMNS}}
        for i, (uid, text) in enumerate(zip(ids, hyp)):
            ref_norm = hyp_norm = ""
            counts = dict.fromkeys(ASR_COUNT_COLUMNS, np.nan)
            try:
                ref_norm = normalize_text(read_reference(wav_paths[uid]))
                hyp_norm = normalize_text(text)
                if not errors[i]:
                    counts = edit_counts(ref_norm, hyp_norm)
            except Exception as error:
                errors[i] = errors[i] or error_text(error)
            columns["ref_norm"].append(ref_norm)
            columns["hyp_norm"].append(hyp_norm)
            for c in ASR_COUNT_COLUMNS:
                columns[c].append(counts[c])
        return columns, errors

    return _metric_stage(ctx, "asr", spec, condition, compute, lambda: {"asr": describe_model(ctx.model("asr"))}, ("eval_metrics",))


def speaker_models(ctx: EvalContext) -> list[str]:
    return [str(m) for m in ctx.cfg.eval.spk_models]


def embeddings_dir(ctx: EvalContext, spec: SystemSpec, condition: str, model: str) -> Path:
    return ctx.paths.embeddings_dir(spec.name, condition, model)


def stage_spk(ctx: EvalContext, spec: SystemSpec, condition: str) -> list[ChunkReport]:
    """Speaker embeddings of the system audio at 16 kHz, one npz part per chunk and model (EVALUATION 4.4)."""
    reports = []
    for name in speaker_models(ctx):
        directory = embeddings_dir(ctx, spec, condition, name)

        def work(k: int, ids: Sequence[str], name: str = name, directory: Path = directory) -> None:
            wavs, errors = read_system_wavs(ctx, spec, condition, ids)
            valid = [i for i, w in enumerate(wavs) if w is not None]
            model = ctx.model(f"spk:{name}")
            model_errors: list[str] = []
            rows = model.embed([to_16k(wavs[i], spec.sr) for i in valid], model_errors) if valid else np.zeros((0, model.dim), np.float32)
            for i, message in zip(valid, model_errors):
                errors[i] = message
            emb = np.full((len(ids), model.dim), np.nan, dtype=np.float32)
            emb[valid] = rows
            write_npz(directory / part_name(k, SPEAKER_PART_SUFFIX), {"ids": np.array(list(ids)), "emb": emb, "err": np.array(errors)})

        reports.append(
            drive_chunks(
                ctx,
                stage="spk",
                spec_name=spec.name,
                condition=condition,
                out_dir=directory,
                is_done=_part_done(directory, SPEAKER_PART_SUFFIX),
                work=work,
                fingerprint=fingerprint_of(ctx, spec),
                config_groups=("eval_metrics",),
                models=lambda name=name: {name: describe_model(ctx.model(f"spk:{name}"))},
                precheck=lambda ids: require_audio(ctx, spec, condition, ids),
            )
        )
    return reports


# ----------------------------------------------------------------------------------------------- re-extraction


def stage_reextract(ctx: EvalContext, spec: SystemSpec, condition: str) -> ChunkReport:
    """Re-extracts ``(T, 15)`` features and ``loud_raw`` from the system audio, no CREPE dither (EVALUATION 4.5).

    Part layout: ``ids`` (n,), ``lengths`` (n,), ``feats`` (sum T, 15) and ``loud_raw`` (sum T,) concatenated in id order,
    ``err`` (n,). A failing utterance has length 0 and its error text.
    """
    directory = ctx.paths.arrays_dir(spec.name, condition)
    audio_spec = ctx.systems[spec.audio_from] if spec.audio_from else spec

    def work(k: int, ids: Sequence[str]) -> None:
        extractor = ctx.model(f"reextract:{spec.head}")
        feats, loud, lengths, errors = [], [], [], []
        for uid in ids:
            try:
                out = extractor.extract(read_system_audio(ctx.paths, audio_spec, condition, uid), audio_spec.sr)
                feats.append(np.asarray(out["feats"], dtype=np.float32))
                loud.append(np.asarray(out["loud_raw"], dtype=np.float32))
                errors.append("")
            except Exception as error:
                logger.warning("reextract %s/%s/%s: %s", spec.name, condition, uid, error_text(error))
                feats.append(np.zeros((0, 15), dtype=np.float32))
                loud.append(np.zeros(0, dtype=np.float32))
                errors.append(error_text(error))
            lengths.append(len(feats[-1]))
        write_npz(
            directory / part_name(k, "npz"),
            {
                "ids": np.array(list(ids)),
                "lengths": np.array(lengths, dtype=np.int64),
                "feats": np.concatenate(feats, axis=0),
                "loud_raw": np.concatenate(loud, axis=0),
                "err": np.array(errors),
            },
        )

    return drive_chunks(
        ctx,
        stage="reextract",
        spec_name=spec.name,
        condition=condition,
        out_dir=directory,
        is_done=_part_done(directory, "npz"),
        work=work,
        fingerprint=fingerprint_of(ctx, audio_spec),
        config_groups=("eval_features",),
        models=lambda: {f"reextract:{spec.head}": describe_model(ctx.model(f"reextract:{spec.head}"))},
        precheck=lambda ids: require_audio(ctx, audio_spec, condition, ids),
    )


def unpack_reextraction(part: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """``id -> (feats (T, 15), loud_raw (T,), err)`` from a reextract part."""
    offsets = np.concatenate([[0], np.cumsum(part["lengths"])])
    out = {}
    for i, uid in enumerate(part["ids"].tolist()):
        a, b = int(offsets[i]), int(offsets[i + 1])
        out[str(uid)] = (part["feats"][a:b], part["loud_raw"][a:b], str(part["err"][i]))
    return out


def _load_reextraction(ctx: EvalContext, system: str, condition: str, k: int) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    path = ctx.paths.arrays_dir(system, condition) / part_name(k, "npz")
    if not path.is_file():
        raise MissingPrerequisite(f"{path} is missing: run stage=reextract for {system} {condition} first")
    return unpack_reextraction(read_npz_part(path))


def stage_prosody(ctx: EvalContext, spec: SystemSpec, condition: str) -> ChunkReport:
    """Prosody and articulation metrics from re-extractions (EVALUATION 4.5); needs ``reextract`` of the system and gt.

    Pitch, voicing and periodicity are compared with the gt re-extraction; EMA and loudness with the cached input
    (``enplus16``: the EMA of the shipped-head gt re-extraction, ``ema_ref``). For ``gt`` itself the reference is the
    cached input for every stream, which gives the extraction floor.
    """
    from sparc.vocoders.eval.prosody import prosody_metrics

    options = ctx.cfg.eval_features.prosody
    ema_std = np.asarray(ctx.stats["ema_std"], dtype=np.float64)
    store = ctx.store
    kwargs = {
        "within_cents": float(options.within_cents),
        "floor_percentile": float(options.loud_floor_percentile),
        "loud_eps": float(options.loud_eps),
    }

    def compute_part(k: int, ids: Sequence[str]) -> tuple[dict[str, Sequence], list[str]]:
        own = _load_reextraction(ctx, spec.name, condition, k)
        gt = own if spec.kind == "gt" else _load_reextraction(ctx, GT_SYSTEM, "T1", k)
        ema_ref = _load_reextraction(ctx, spec.ema_ref, "T1", k) if spec.ema_ref else None
        rows, errors = [], []
        for uid in ids:
            try:
                sys_feats, sys_loud, err = own[uid]
                if err or not len(sys_feats):
                    raise ValueError(f"re-extraction failed: {err}")
                utt = store.index_of(uid)
                cached, loud_raw = store.frames(utt, 0, int(store.T[utt]))
                input_loud = loud_raw.astype(np.float64) * gain_factor(store.peak24[utt], ctx.gain_db)
                if spec.kind == "gt":
                    ref_feats, input_ema = cached, cached[:, :12]
                else:
                    ref_feats = gt[uid][0]
                    if not len(ref_feats):
                        raise ValueError(f"gt re-extraction failed: {gt[uid][2]}")
                    input_ema = cached[:, :12]
                    if ema_ref is not None:
                        input_ema = ema_ref[uid][0][:, :12]
                        if not len(input_ema):
                            raise ValueError(f"{spec.ema_ref} re-extraction failed: {ema_ref[uid][2]}")
                rows.append(prosody_metrics(sys_feats, sys_loud, ref_feats, input_ema, input_loud, ema_std, **kwargs))
                errors.append("")
            except Exception as error:
                logger.warning("prosody %s/%s/%s: %s", spec.name, condition, uid, error_text(error))
                rows.append({})
                errors.append(error_text(error))
        frame = pd.DataFrame(rows, index=range(len(ids)))
        return {c: frame[c].to_numpy(dtype=np.float64) for c in frame.columns}, errors

    directory = results_dir(ctx, spec, condition, "prosody")

    def work(k: int, ids: Sequence[str]) -> None:
        columns, errors = compute_part(k, ids)
        write_parquet(directory / part_name(k, "parquet"), part_frame(ctx, ids, condition, columns, errors))

    return drive_chunks(
        ctx,
        stage="prosody",
        spec_name=spec.name,
        condition=condition,
        out_dir=directory,
        is_done=_part_done(directory, "parquet"),
        work=work,
        fingerprint=fingerprint_of(ctx, spec),
        config_groups=("eval_features",),
    )


# ----------------------------------------------------------------------------------------------- B's whole-system stages


def pinned_checkpoint(ctx: EvalContext, spec: SystemSpec) -> tuple[DictConfig, CheckpointInfo]:
    """Resolves the checkpoint of vocoder ``spec`` once, refuses a non-final one (``eval.require_final``) and returns the
    config with ``eval.systems.<name>.ckpt`` pinned to it, so the probes and the efficiency stage evaluate the same
    checkpoint as ``synth`` (and cannot silently pick up a newer ``step*.ckpt`` halfway)."""
    accelerator = "cuda" if ctx.device.type == "cuda" else "cpu"
    vcfg = compose_vocoder_config(ctx.cfg, spec, spec.conditions, ctx.paths.predictions_root(spec), accelerator, ctx.limit)
    info = resolve_checkpoint(vcfg)
    if bool(ctx.cfg.eval.require_final) and not info.final:
        raise RuntimeError(
            f"{spec.name}: newest checkpoint {info.path} is at g_step {info.g_step} of {info.max_g_steps}; the evaluation "
            "uses final checkpoints only (set eval.require_final=false for a smoke run)"
        )
    cfg = OmegaConf.create(OmegaConf.to_container(ctx.cfg, resolve=False))
    cfg.eval.systems[spec.name].ckpt = info.path
    return cfg, info


def check_recorded_checkpoint(ctx: EvalContext, where: Path, recorded: dict | None, info: CheckpointInfo) -> None:
    """Raises if results at ``where`` were made with another checkpoint than ``info`` (unless allowed by config)."""
    if recorded is None or bool(ctx.cfg.eval.allow_fingerprint_change):
        return
    if recorded.get("ckpt") != info.path or recorded.get("g_step") != info.g_step:
        raise RuntimeError(
            f"{where}: results were made with checkpoint {recorded.get('ckpt')} (g_step {recorded.get('g_step')}), now "
            f"{info.path} (g_step {info.g_step}); delete them (or set eval.allow_fingerprint_change=true) to recompute"
        )


def stage_probes(ctx: EvalContext, spec: SystemSpec) -> None:
    """Controllability probes of a vocoder (EVALUATION 6), delegated to ``eval.probes.run_probes``."""
    from sparc.vocoders.eval.probes import run_probes

    out_dir = ctx.paths.probes_dir(spec.name)
    results = out_dir / str(ctx.cfg.eval_features.probes.results_name)
    cfg, info = pinned_checkpoint(ctx, spec)
    meta_path = out_dir / "probes_meta.json"
    recorded = read_json(meta_path).get("checkpoint") if meta_path.is_file() else None
    check_recorded_checkpoint(ctx, out_dir, recorded, info)
    if results.is_file():
        return
    with hydra_cleared():  # the vocoder loader composes its own config
        run_probes(cfg, spec.name, ctx.device, out_dir, ctx.stop.is_set)
    if not results.is_file():  # run_probes returns early on a stop request
        ctx.stop.raise_if_set()
        raise RuntimeError(f"run_probes({spec.name}) finished without writing {results}")


def stage_efficiency(ctx: EvalContext, system: str) -> None:
    """Parameters, RTF and receptive field (EVALUATION 7), delegated to ``eval.efficiency.run_efficiency``.

    ``system`` is a vocoder, ``vocos_mel``, ``enplus16`` or ``extractor``. The GPU and CPU timings are separate runs
    (``eval.device``) that merge into the same ``efficiency/<system>.json``.
    """
    from sparc.vocoders.eval.efficiency import run_efficiency

    cfg, out_path = ctx.cfg, ctx.paths.efficiency_path(system)
    spec = ctx.systems.get(system)
    if spec is not None and spec.kind == "vocoder":
        cfg, info = pinned_checkpoint(ctx, spec)
        recorded = read_json(out_path).get("meta", {}).get("checkpoint") if out_path.is_file() else None
        check_recorded_checkpoint(ctx, out_path, recorded, info)
    with hydra_cleared():
        run_efficiency(cfg, system, str(ctx.device.type), out_path)


# ----------------------------------------------------------------------------------------------- samples and aggregate


def stage_samples(ctx: EvalContext) -> dict:
    from sparc.vocoders.eval.samples import write_samples

    return write_samples(ctx.cfg, ctx.paths, ctx.systems, ctx.table, ctx.ids, allow_fewer=ctx.limit is not None)


def stage_aggregate(ctx: EvalContext) -> dict:
    from sparc.vocoders.eval.aggregate import run_aggregate

    return run_aggregate(ctx.cfg, ctx.paths, ctx.systems, ctx.table, ctx.ids)


# ----------------------------------------------------------------------------------------------- dispatch

CHUNK_STAGES: dict[str, Callable[..., Any]] = {
    "gt": stage_gt,
    "refs": stage_refs,
    "signal": stage_signal,
    "utmos": stage_utmos,
    "asr": stage_asr,
    "spk": stage_spk,
    "reextract": stage_reextract,
    "prosody": stage_prosody,
}
SYSTEM_STAGES = ("probes", "efficiency")
ALWAYS_RERUN = ("aggregate", "samples")  # cheap and dependent on other stages: never short-circuited by a DONE marker


def done_path(root: Path, split: str, stage: str, system: str, condition: str) -> Path:
    """DONE marker of a CLI item (``<eval root>/done/<split>/<stage>__<system>__<condition>``); ``scripts/slurm/eval.sh`` uses the same."""
    return Path(root) / "done" / split / f"{stage}__{system}__{condition}"


def run_stage(ctx: EvalContext, stage: str, system: str = "all", condition: str = "all") -> None:
    """Runs one stage for the selected systems and conditions; raises :class:`StopRequested` after a stop signal.

    With ``system=all`` a system whose prerequisites are missing (for example a vocoder that was not synthesized yet) is
    reported and skipped so the others still run; the stage then fails at the end and writes no DONE marker.
    """
    failures: list[str] = []

    def attempt(label: str, call: Callable[[], Any]) -> None:
        ctx.stop.raise_if_set()
        try:
            call()
        except MissingPrerequisite as error:
            if system != "all":
                raise
            logger.error("%s: skipped, %s", label, error)
            failures.append(f"{label}: {error}")

    if stage == "aggregate":
        stage_aggregate(ctx)
    elif stage == "samples":
        stage_samples(ctx)
    elif stage == "synth":
        grouped: dict[str, list[str]] = {}
        for spec, cond in resolve_items(ctx.systems, stage, system, condition):
            grouped.setdefault(spec.name, []).append(cond)
        for name, conditions in grouped.items():
            attempt(f"synth {name}", lambda name=name, conditions=conditions: stage_synth(ctx, ctx.systems[name], conditions))
    elif stage in SYSTEM_STAGES:
        if stage == "efficiency" and system == EXTRACTOR_SYSTEM:
            targets = [EXTRACTOR_SYSTEM]
        else:
            targets = list(dict.fromkeys(spec.name for spec, _ in resolve_items(ctx.systems, stage, system, condition)))
            if stage == "efficiency" and system == "all":
                targets = [EXTRACTOR_SYSTEM, *targets]
        for name in targets:
            if stage == "probes":
                attempt(f"probes {name}", lambda name=name: stage_probes(ctx, ctx.systems[name]))
            else:
                attempt(f"efficiency {name}", lambda name=name: stage_efficiency(ctx, name))
    elif stage in CHUNK_STAGES:
        for spec, cond in resolve_items(ctx.systems, stage, system, condition):
            logger.info("stage %s: system %s condition %s", stage, spec.name, cond)
            attempt(f"{stage} {spec.name} {cond}", lambda spec=spec, cond=cond: CHUNK_STAGES[stage](ctx, spec, cond))
    else:
        raise ValueError(f"unknown stage {stage!r}")
    if failures:
        raise MissingPrerequisite(f"{len(failures)} item(s) skipped:\n" + "\n".join(failures))
    if stage not in ALWAYS_RERUN:
        write_text(done_path(ctx.paths.root, ctx.split, stage, system, condition), utc_now() + "\n")
