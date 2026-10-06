"""Listening samples of the evaluation (EVALUATION.md section 9, stage ``samples``).

Twenty utterances of distinct speakers (ten female, ten male, 3-12 s) chosen by ``default_rng(seed)`` over the sorted
candidate ids; for each one the gt audio and every system and condition as 16-bit PCM WAV at the system's own sample
rate under ``samples/<id>/<system>_<cond>.wav``, plus ``samples/index.csv``.
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sparc.vocoders.eval.io import (
    EvalPaths,
    read_transcript,
    read_wav,
    write_csv,
    write_json,
    write_wav,
)
from sparc.vocoders.eval.systems import SystemSpec

logger = logging.getLogger(__name__)

SEXES = ("F", "M")


def select_samples(
    table: pd.DataFrame,
    ids: Sequence[str] | None = None,
    n_female: int = 10,
    n_male: int = 10,
    duration_range_s: Sequence[float] = (3.0, 12.0),
    seed: int = 0,
    allow_fewer: bool = False,
) -> list[str]:
    """Ids of the sample utterances: ``n_female`` female then ``n_male`` male speakers, one utterance each.

    Candidates are the utterances of ``ids`` (default: the table) with a duration inside ``duration_range_s``, a known
    speaker sex and a T2 reference (so that every condition exists). For each sex, in the order ``F``, ``M``, the sorted
    candidate ids are permuted with one ``default_rng(seed)`` and the first utterance of each new speaker is taken until
    the quota is met. The result depends only on the table, the id set and the arguments. Raises ``ValueError`` if a
    sex has too few speakers
    (smoke runs on a subset pass ``allow_fewer=True`` and get a warning instead).
    """
    low, high = (float(d) for d in duration_range_s)
    candidates = table
    if ids is not None:
        candidates = candidates[candidates["id"].isin(set(ids))]
    candidates = candidates[
        candidates["dur_s"].between(low, high) & candidates["has_ref"].astype(bool) & candidates["sex"].isin(SEXES)
    ].sort_values("id")
    rng = np.random.default_rng(int(seed))
    chosen: list[str] = []
    for sex, quota in zip(SEXES, (int(n_female), int(n_male))):
        pool = candidates[candidates["sex"] == sex]
        order = rng.permutation(len(pool))
        speakers_seen: set[str] = set()
        picked = 0
        for row in pool.iloc[order].itertuples():
            if picked == quota:
                break
            if row.speaker in speakers_seen:
                continue
            speakers_seen.add(row.speaker)
            chosen.append(str(row.id))
            picked += 1
        if picked < quota and allow_fewer:
            logger.warning("samples: only %d %s speakers have a candidate (wanted %d)", picked, sex, quota)
        elif picked < quota:
            raise ValueError(f"only {picked} {sex} speakers have a candidate utterance of {low}-{high} s, need {quota}")
    return chosen


def sample_systems(systems: dict[str, SystemSpec]) -> list[SystemSpec]:
    """Systems with audio of their own (``audio_from`` systems reuse another system's files)."""
    return [spec for spec in systems.values() if spec.audio_from is None]


def write_samples(
    cfg: DictConfig,
    paths: EvalPaths,
    systems: dict[str, SystemSpec],
    table: pd.DataFrame,
    ids: Sequence[str],
    allow_fewer: bool = False,
) -> dict:
    """Writes the sample WAVs and ``index.csv``; a system or condition without audio is listed, not skipped silently.

    Returns ``{"ids": [...], "written": n, "missing": {id: [name, ...]}}``. The index has one row per utterance with
    ``id, speaker, sex, duration, transcript, ref_id`` (the T2 reference) and ``missing`` (system_condition names whose
    WAV did not exist).
    """
    options = cfg.eval.samples
    selected = select_samples(
        table, ids, int(options.n_female), int(options.n_male), tuple(options.duration_range_s), int(options.seed), allow_fewer
    )
    subtype = str(options.pcm_subtype)
    rows = table.set_index("id")
    out_root = paths.samples_dir()
    written = 0
    missing: dict[str, list[str]] = {}
    index_rows = []
    for uid in selected:
        row = rows.loc[uid]
        lost = []
        for spec in sample_systems(systems):
            for condition in spec.conditions:
                name = f"{spec.name}_{condition}"
                source = paths.audio_path(spec, condition, uid)
                if not source.is_file():
                    lost.append(name)
                    continue
                wav, rate = read_wav(source)
                write_wav(out_root / uid / f"{name}.wav", wav, rate, subtype=subtype)
                written += 1
        if lost:
            missing[uid] = lost
            logger.warning("samples: %s has no audio for %s", uid, ", ".join(lost))
        index_rows.append(
            {
                "id": uid,
                "speaker": row["speaker"],
                "sex": row["sex"],
                "duration": round(float(row["dur_s"]), 3),
                "transcript": read_transcript(row["wav_path"]),
                "ref_id": row["ref_id"],
                "missing": ";".join(lost),
            }
        )
    write_csv(out_root / "index.csv", pd.DataFrame(index_rows))
    summary = {"ids": selected, "written": written, "missing": missing}
    write_json(out_root / "meta.json", {**summary, "seed": int(options.seed), "pcm_subtype": subtype, "split": paths.split})
    return summary
