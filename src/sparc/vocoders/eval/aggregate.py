"""Aggregation of the evaluation results (EVALUATION.md section 8, stage ``aggregate``).

Reads the per-utterance result parts and speaker embeddings under ``eval_root`` and writes ``tables/<split>/``:
``table.md``, ``table.csv``, ``paired.csv``, ``per_speaker.csv`` and ``meta.json``.

Statistics. Utterances of one speaker are not independent, so every interval is a speaker-cluster bootstrap: speakers are
resampled with replacement (all their utterances stay together), the statistic is recomputed on each resample, and the
CI95 is the 2.5th and 97.5th percentile of the resampled statistics (``n_boot`` resamples, fixed seed). A mean over
utterances is a ratio of two per-speaker sums (sum of values over count), and so is a corpus WER or CER (sum of word
errors over sum of reference words), so one vectorized routine serves every metric. A paired difference resamples the
speakers shared by both systems and takes the difference of the two ratios on the ids both systems have. Missing
groups, systems and utterances are reported (``meta.json``, ``table.md``), never dropped silently.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sparc.vocoders.eval.io import (
    EvalPaths,
    base_columns,
    chapter_of,
    condition_ids,
    list_parts,
    now_iso,
    package_versions,
    read_audio_failures,
    read_json,
    read_npz_part,
    read_parquet_parts,
    write_csv,
    write_json,
    write_text,
)
from sparc.vocoders.eval.systems import SystemSpec

logger = logging.getLogger(__name__)

GROUPS_BY_KIND = {"gt": ("utmos", "asr", "prosody"), "reference": ("signal", "utmos", "asr", "prosody"), "vocoder": ("signal", "utmos", "asr", "prosody")}
BASE_COLUMNS = ("id", "speaker", "chapter", "dur_s", "ref_id")
STRATA = ("all", "same_chapter", "diff_chapter")


@dataclass(frozen=True)
class Metric:
    """A reported quantity: the mean of ``key`` over utterances, or the ratio ``sum(num) / sum(den)`` (corpus WER/CER)."""

    key: str
    label: str
    section: str
    digits: int = 3
    num: str | None = None
    den: str | None = None
    note: str = ""

    @property
    def is_ratio(self) -> bool:
        return self.num is not None


METRICS: tuple[Metric, ...] = (
    Metric("utmos", "UTMOS", "quality", 3),
    Metric("pesq_wb", "PESQ", "quality", 3),
    Metric("wer", "WER", "quality", 4, "word_errors", "word_ref_len", "corpus level"),
    Metric("cer", "CER", "quality", 4, "char_errors", "char_ref_len", "corpus level"),
    Metric("sim_same", "Sim same", "quality", 3),
    Metric("sim_spk", "Sim spk", "quality", 3),
    Metric("mcd", "MCD (dB)", "spectral", 3),
    Metric("mrstft", "MR-STFT", "spectral", 3),
    Metric("mel_l1", "Mel L1", "spectral", 3),
    Metric("mel_l1_0_4k", "Mel L1 0-4k", "spectral", 3),
    Metric("mel_l1_4_8k", "Mel L1 4-8k", "spectral", 3),
    Metric("mel_l1_8_12k", "Mel L1 8-12k", "spectral", 3),
    Metric("f0_rmse_cents", "F0 RMSE (cents)", "prosody", 1),
    Metric("f0_med_abs_cents", "F0 median (cents)", "prosody", 1),
    Metric("f0_within50", "F0 within 50c", "prosody", 3),
    Metric("vde", "VDE", "prosody", 3),
    Metric("per_mae", "Periodicity MAE", "prosody", 3),
    Metric("loud_db_rmse", "Loudness RMSE (dB)", "prosody", 2),
    Metric("ema_r_mean", "EMA r", "prosody", 3),
    Metric("ema_rmse_mean", "EMA RMSE (std)", "prosody", 3),
)
SECTION_TITLES = {
    "quality": "Quality, intelligibility and speaker similarity",
    "spectral": "Spectral fidelity (24 kHz)",
    "prosody": "Prosody and articulation (re-extracted features)",
}


METRIC_GROUP = {
    **dict.fromkeys(("pesq_wb", "mcd", "mrstft", "mel_l1", "mel_l1_0_4k", "mel_l1_4_8k", "mel_l1_8_12k"), "signal"),
    "utmos": "utmos",
    **dict.fromkeys(("wer", "cer"), "asr"),
    **dict.fromkeys(("f0_rmse_cents", "f0_med_abs_cents", "f0_within50", "vde", "per_mae", "loud_db_rmse", "ema_r_mean", "ema_rmse_mean"), "prosody"),
    **dict.fromkeys(("sim_same", "sim_spk"), "spk"),
}


def metric_gaps(wide: pd.DataFrame, systems: dict[str, SystemSpec], status: Sequence[dict]) -> list[dict]:
    """Cells where a metric has fewer values than utterances although its result group is complete.

    A mean over utterances ignores NaN (a failed utterance, a pair without a voiced frame in both), so every such
    shortfall is listed instead of staying invisible in the mean: ``{system, condition, stratum, metric, n, n_utt}``.
    Not listed: groups that are missing or partial (they are in ``status``), groups that do not apply to the system
    (the gt has no signal metrics, nor a ``sim_same``) and the 8-12 kHz band of a 16 kHz system.
    """
    open_groups = {(s["system"], s["condition"], s["group"].split(":")[0]) for s in status if s["status"] != "complete"}
    gaps = []
    for row in wide.to_dict("records"):
        spec = systems[row["system"]]
        for m in METRICS:
            group = METRIC_GROUP[m.key]
            if group != "spk" and group not in GROUPS_BY_KIND[spec.kind]:
                continue
            if (spec.name, row["condition"], group) in open_groups:
                continue
            if (m.key == "mel_l1_8_12k" and spec.sr < 24000) or (m.key == "sim_same" and spec.kind == "gt"):
                continue
            n = int(row[f"{m.key}_n"])
            if n < int(row["n_utt"]):
                gaps.append(
                    {"system": spec.name, "condition": row["condition"], "stratum": row["stratum"], "metric": m.key, "n": n, "n_utt": int(row["n_utt"])}
                )
    return gaps


# ----------------------------------------------------------------------------------------------- bootstrap


def draw_clusters(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """Resampled cluster indices ``(n_boot, n_clusters)``; a function of the arguments only, so reruns agree."""
    return np.random.default_rng(int(seed)).integers(0, int(n_clusters), size=(int(n_boot), int(n_clusters)))


def cluster_sums(codes: np.ndarray, n_clusters: int, num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster sums of ``num`` and ``den`` (``codes`` are cluster numbers ``0 .. n_clusters - 1``)."""
    return (
        np.bincount(codes, weights=num, minlength=n_clusters).astype(np.float64),
        np.bincount(codes, weights=den, minlength=n_clusters).astype(np.float64),
    )


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def interval(samples: np.ndarray, ci: float) -> tuple[float, float]:
    """Percentile interval of the finite bootstrap statistics (NaN if there are none)."""
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return float("nan"), float("nan")
    lo, hi = np.percentile(samples, [50.0 * (1.0 - ci), 50.0 * (1.0 + ci)])
    return float(lo), float(hi)


def bootstrap_ratio(num_s: np.ndarray, den_s: np.ndarray, idx: np.ndarray, ci: float = 0.95) -> dict[str, float]:
    """Point estimate ``sum(num) / sum(den)`` and its cluster-bootstrap interval from per-cluster sums."""
    point = float(_ratio(np.array(num_s.sum()), np.array(den_s.sum())))
    if len(num_s) < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan")}
    boots = _ratio(num_s[idx].sum(axis=1), den_s[idx].sum(axis=1))
    lo, hi = interval(boots, ci)
    return {"point": point, "lo": lo, "hi": hi}


def bootstrap_paired(
    num_a: np.ndarray, den_a: np.ndarray, num_b: np.ndarray, den_b: np.ndarray, idx: np.ndarray, ci: float = 0.95
) -> dict[str, float]:
    """Difference of two ratios over the same clusters: point, interval and the share of clusters moving the same way.

    ``frac_same_direction`` is the fraction of clusters (with data) whose own difference has the sign of the overall
    difference; NaN if the overall difference is zero.
    """
    point = float(_ratio(np.array(num_a.sum()), np.array(den_a.sum())) - _ratio(np.array(num_b.sum()), np.array(den_b.sum())))
    own = _ratio(num_a, den_a) - _ratio(num_b, den_b)
    own = own[np.isfinite(own)]
    frac = float(np.mean(np.sign(own) == np.sign(point))) if len(own) and point != 0 and np.isfinite(point) else float("nan")
    if len(num_a) < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "frac_same_direction": frac}
    boots = _ratio(num_a[idx].sum(axis=1), den_a[idx].sum(axis=1)) - _ratio(num_b[idx].sum(axis=1), den_b[idx].sum(axis=1))
    lo, hi = interval(boots, ci)
    return {"point": point, "lo": lo, "hi": hi, "frac_same_direction": frac}


def metric_terms(frame: pd.DataFrame, metric: Metric) -> tuple[np.ndarray, np.ndarray]:
    """``(num, den)`` per utterance: ``(value, 1)`` for a mean, ``(errors, reference length)`` for a ratio; ``(0, 0)`` where
    the value is missing or not finite, so such utterances do not count."""
    n = len(frame)
    if metric.is_ratio:
        a = frame[metric.num].to_numpy(np.float64) if metric.num in frame else np.full(n, np.nan)
        b = frame[metric.den].to_numpy(np.float64) if metric.den in frame else np.full(n, np.nan)
        ok = np.isfinite(a) & np.isfinite(b)
        return np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    v = frame[metric.key].to_numpy(np.float64) if metric.key in frame else np.full(n, np.nan)
    ok = np.isfinite(v)
    return np.where(ok, v, 0.0), ok.astype(np.float64)


def _clusters(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    codes, speakers = pd.factorize(frame["speaker"].astype(str))
    return codes, np.asarray(speakers)


def summarize_cell(frame: pd.DataFrame, n_boot: int, seed: int, ci: float, metrics: Sequence[Metric] = METRICS) -> dict[str, Any]:
    """Mean (or corpus ratio) and CI95 of every metric over the utterances of ``frame``, with counts."""
    out: dict[str, Any] = {"n_utt": int(len(frame)), "n_speakers": int(frame["speaker"].nunique())}
    if not len(frame):
        for m in metrics:
            out[m.key] = {"point": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
        return out
    codes, speakers = _clusters(frame)
    idx = draw_clusters(len(speakers), n_boot, seed)
    for m in metrics:
        num, den = metric_terms(frame, m)
        num_s, den_s = cluster_sums(codes, len(speakers), num, den)
        stats = bootstrap_ratio(num_s, den_s, idx, ci)
        stats["n"] = int((den > 0).sum())
        out[m.key] = stats
    return out


def per_speaker_means(frame: pd.DataFrame, metrics: Sequence[Metric] = METRICS) -> pd.DataFrame:
    """One row per speaker: ``n_utt`` and the mean (or corpus ratio) of every metric over the speaker's utterances."""
    codes, speakers = _clusters(frame)
    out = {"speaker": speakers, "n_utt": np.bincount(codes, minlength=len(speakers))}
    for m in metrics:
        num, den = metric_terms(frame, m)
        num_s, den_s = cluster_sums(codes, len(speakers), num, den)
        out[m.key] = _ratio(num_s, den_s)
    return pd.DataFrame(out)


def paired_cell(
    a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int, ci: float, metrics: Sequence[Metric] = METRICS
) -> dict[str, Any]:
    """Paired differences ``a - b`` of every metric over the ids both frames have (inner join on ``id``)."""
    shared = sorted(set(a["id"]) & set(b["id"]))
    a = a.set_index("id").loc[shared].reset_index()
    b = b.set_index("id").loc[shared].reset_index()
    out: dict[str, Any] = {"n_utt": len(shared), "n_speakers": int(a["speaker"].nunique()) if len(shared) else 0}
    if not shared:
        return out | {m.key: None for m in metrics}
    codes, speakers = _clusters(a)
    idx = draw_clusters(len(speakers), n_boot, seed)
    for m in metrics:
        na, da = metric_terms(a, m)
        nb, db = metric_terms(b, m)
        both = (da > 0) & (db > 0)  # an utterance counts only if both systems have it
        na, da, nb, db = (np.where(both, x, 0.0) for x in (na, da, nb, db))
        sums = [cluster_sums(codes, len(speakers), n, d) for n, d in ((na, da), (nb, db))]
        stats = bootstrap_paired(sums[0][0], sums[0][1], sums[1][0], sums[1][1], idx, ci)
        stats["n"] = int(both.sum())
        out[m.key] = stats
    return out


# ----------------------------------------------------------------------------------------------- speaker similarity


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm > 0, norm, np.nan)


def load_embeddings(directory: Path) -> dict[str, np.ndarray]:
    """``id -> embedding`` (L2-normalized rows as stored; NaN for a failed utterance) from the parts of ``directory``."""
    out: dict[str, np.ndarray] = {}
    for _, path in list_parts(directory, "npz"):
        part = read_npz_part(path)
        for uid, row in zip(part["ids"].tolist(), part["emb"]):
            out[str(uid)] = np.asarray(row, dtype=np.float64)
    return out


def speaker_references(
    ids: Sequence[str], speakers: Sequence[str], t2_refs: Sequence[str], gt_emb: dict[str, np.ndarray]
) -> np.ndarray:
    """Mean L2-normalized gt embedding of each id's speaker over the speaker's other utterances.

    The target and its T2 reference are excluded (EVALUATION 4.4); utterances without a finite gt embedding do not count.
    Row ``i`` is NaN if nothing is left. The same exclusion applies to every condition, so the numbers are comparable.
    """
    d = len(next(iter(gt_emb.values()))) if gt_emb else 1
    out = np.full((len(ids), d), np.nan)
    ids, speakers, t2_refs = list(ids), np.asarray(list(speakers)), list(t2_refs)
    for speaker in np.unique(speakers):
        members = np.flatnonzero(speakers == speaker)
        member_ids = [ids[i] for i in members]
        vectors = np.full((len(members), d), np.nan)
        for j, uid in enumerate(member_ids):
            if uid in gt_emb:
                vectors[j] = gt_emb[uid]
        ok = np.isfinite(vectors).all(axis=1)
        total = np.where(ok[:, None], vectors, 0.0).sum(axis=0)
        position = {uid: j for j, uid in enumerate(member_ids)}
        for j, i in enumerate(members):
            drop = {j}
            ref = t2_refs[i]
            if ref in position:
                drop.add(position[ref])
            drop_ok = [k for k in drop if ok[k]]
            count = int(ok.sum()) - len(drop_ok)
            if count > 0:
                out[i] = (total - vectors[drop_ok].sum(axis=0)) / count
    return out


def similarity_columns(
    frame: pd.DataFrame, sys_emb: dict[str, np.ndarray], gt_emb: dict[str, np.ndarray], is_gt: bool
) -> tuple[np.ndarray, np.ndarray]:
    """``sim_same`` (cosine with the gt of the same utterance) and ``sim_spk`` (cosine with the speaker's other gt
    utterances, EVALUATION 4.4) for the rows of ``frame``. For the gt system ``sim_same`` is not defined (NaN) and
    ``sim_spk`` is the ceiling."""
    ids = frame["id"].tolist()
    n = len(ids)
    d = len(next(iter(gt_emb.values()))) if gt_emb else 1
    sys_rows = np.full((n, d), np.nan)
    gt_rows = np.full((n, d), np.nan)
    for i, uid in enumerate(ids):
        if uid in sys_emb:
            sys_rows[i] = sys_emb[uid]
        if uid in gt_emb:
            gt_rows[i] = gt_emb[uid]
    sys_rows, gt_rows = l2_normalize_rows(sys_rows), l2_normalize_rows(gt_rows)
    same = np.full(n, np.nan) if is_gt else np.sum(sys_rows * gt_rows, axis=1)
    mean = l2_normalize_rows(speaker_references(ids, frame["speaker"].tolist(), frame["t2_ref"].tolist(), gt_emb))
    return same, np.sum(sys_rows * mean, axis=1)


# ----------------------------------------------------------------------------------------------- loading


def stage_models(directory: Path) -> dict:
    """Models recorded by the last run that wrote ``meta.json`` in ``directory``."""
    path = directory / "meta.json"
    if not path.is_file():
        return {}
    for run in reversed(read_json(path).get("runs", [])):
        if run.get("models"):
            return run["models"]
    return {}


def reported_systems(systems: dict[str, SystemSpec]) -> list[SystemSpec]:
    """Systems with a row in the tables (those a ``utmos`` stage applies to: gt, references, vocoders)."""
    return [spec for spec in systems.values() if spec.applies_to("utmos")]


def load_cell(
    paths: EvalPaths, spec: SystemSpec, condition: str, table: pd.DataFrame, ids: Sequence[str]
) -> tuple[pd.DataFrame, list[dict]]:
    """Joined result groups of one system and condition, one row per id of the condition (NaN where a group has none).

    Also returns one status record per group: ``complete``, ``partial`` or ``missing`` with the id counts.
    """
    expected = condition_ids(table, condition, ids)
    frame = base_columns(table, expected, condition)
    frame["t2_ref"] = table.set_index("id").loc[expected, "ref_id"].to_numpy()
    status = []
    for group in GROUPS_BY_KIND[spec.kind]:
        part = read_parquet_parts(paths.results_dir(spec.name, condition, group))
        have = 0
        if len(part):
            part = part[part["id"].isin(set(expected))].drop_duplicates("id")
            have = len(part)
            keep = [c for c in part.columns if c not in BASE_COLUMNS]
            part = part[["id", *keep]].rename(columns={"err": f"err_{group}"})
            frame = frame.merge(part, on="id", how="left")
        status.append(
            {
                "system": spec.name,
                "condition": condition,
                "group": group,
                "n_expected": len(expected),
                "n_found": have,
                "status": "complete" if have == len(expected) and have else ("missing" if not have else "partial"),
            }
        )
    return frame, status


def add_similarity(
    cells: dict[tuple[str, str], pd.DataFrame], paths: EvalPaths, systems: dict[str, SystemSpec], models: Sequence[str]
) -> tuple[list[dict], str | None]:
    """Adds ``sim_same``/``sim_spk`` (primary speaker model) and ``sim_same_<m>``/``sim_spk_<m>`` (other models).

    The primary model is the first of ``models`` that has gt embeddings. Returns status records and the primary model.
    """
    status: list[dict] = []
    primary = None
    gt_name = next((s.name for s in systems.values() if s.kind == "gt" and s.audio_from is None), None)
    for model in models:
        gt_emb = load_embeddings(paths.embeddings_dir(gt_name, "T1", model)) if gt_name else {}
        if gt_emb and primary is None:
            primary = model
        suffix = "" if model == primary else f"_{model}"
        for (name, condition), frame in cells.items():
            spec = systems[name]
            sys_emb = load_embeddings(paths.embeddings_dir(name, condition, model))
            expected = len(frame)
            found = len(set(frame["id"]) & set(sys_emb))
            status.append(
                {
                    "system": name,
                    "condition": condition,
                    "group": f"spk:{model}",
                    "n_expected": expected,
                    "n_found": found,
                    "status": "complete" if found == expected and found else ("missing" if not found else "partial"),
                }
            )
            if gt_emb and sys_emb:
                same, spk = similarity_columns(frame, sys_emb, gt_emb, spec.kind == "gt")
            else:
                same = spk = np.full(len(frame), np.nan)
            frame[f"sim_same{suffix}"] = same
            frame[f"sim_spk{suffix}"] = spk
    return status, primary


def with_strata(frame: pd.DataFrame, condition: str, split_chapters: bool) -> dict[str, pd.DataFrame]:
    """``{"all": frame}`` plus, for T2, the utterances whose reference is in the same or in a different chapter."""
    out = {"all": frame}
    if condition == "T2" and split_chapters and len(frame):
        same = np.array([chapter_of(r) == c for r, c in zip(frame["ref_id"], frame["chapter"])]) if "ref_id" in frame else np.zeros(len(frame), bool)
        out["same_chapter"] = frame[same]
        out["diff_chapter"] = frame[~same]
    return out


# ----------------------------------------------------------------------------------------------- tables


def table_rows(cells: dict[tuple[str, str], pd.DataFrame], systems: dict[str, SystemSpec], options: DictConfig) -> pd.DataFrame:
    """Wide table: one row per system, condition and stratum with ``<metric>``, ``_lo``, ``_hi`` and ``_n`` columns."""
    rows = []
    for (name, condition), frame in cells.items():
        for stratum, sub in with_strata(frame, condition, bool(options.split_chapters)).items():
            summary = summarize_cell(sub, int(options.n_boot), int(options.seed), float(options.ci))
            row = {
                "system": name,
                "label": systems[name].label,
                "condition": condition,
                "stratum": stratum,
                "n_utt": summary["n_utt"],
                "n_speakers": summary["n_speakers"],
            }
            for m in METRICS:
                s = summary[m.key]
                row |= {m.key: s["point"], f"{m.key}_lo": s["lo"], f"{m.key}_hi": s["hi"], f"{m.key}_n": s["n"]}
            rows.append(row)
    return pd.DataFrame(rows)


def per_speaker_table(cells: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for (name, condition), frame in cells.items():
        if not len(frame):
            continue
        part = per_speaker_means(frame)
        part.insert(0, "condition", condition)
        part.insert(0, "system", name)
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def paired_pairs(systems: dict[str, SystemSpec], references: Sequence[str]) -> list[tuple[SystemSpec, str, SystemSpec, str]]:
    """``(a, condition a, b, condition b)``: every pair of vocoders under each shared condition, and each vocoder against
    each reference system (same condition if the reference has it, else its ``T1``)."""
    vocoders = [s for s in systems.values() if s.kind == "vocoder"]
    pairs = []
    for a, b in combinations(vocoders, 2):
        pairs.extend((a, c, b, c) for c in a.conditions if c in b.conditions)
    for v in vocoders:
        for ref in references:
            if ref not in systems:
                continue
            r = systems[ref]
            pairs.extend((v, c, r, c if c in r.conditions else "T1") for c in v.conditions)
    return pairs


def paired_table(cells: dict[tuple[str, str], pd.DataFrame], systems: dict[str, SystemSpec], options: DictConfig) -> pd.DataFrame:
    rows = []
    for a, cond_a, b, cond_b in paired_pairs(systems, list(options.reference_systems)):
        if (a.name, cond_a) not in cells or (b.name, cond_b) not in cells:
            continue
        strata_a = with_strata(cells[(a.name, cond_a)], cond_a, bool(options.split_chapters))
        for stratum, frame_a in strata_a.items():
            result = paired_cell(frame_a, cells[(b.name, cond_b)], int(options.n_boot), int(options.seed), float(options.ci))
            for m in METRICS:
                s = result.get(m.key)
                if s is None:
                    continue
                rows.append(
                    {
                        "system_a": a.name,
                        "condition_a": cond_a,
                        "system_b": b.name,
                        "condition_b": cond_b,
                        "stratum": stratum,
                        "metric": m.key,
                        "n_utt": s["n"],
                        "n_speakers": result["n_speakers"],
                        "diff": s["point"],
                        "lo": s["lo"],
                        "hi": s["hi"],
                        "ci_excludes_zero": bool(np.isfinite(s["lo"]) and (s["lo"] > 0 or s["hi"] < 0)),
                        "frac_speakers_same_direction": s["frac_same_direction"],
                    }
                )
    return pd.DataFrame(rows)


def fmt(point: float, lo: float, hi: float, digits: int) -> str:
    if not np.isfinite(point):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} ({lo:.{digits}f}, {hi:.{digits}f})"


def render_markdown(
    table: pd.DataFrame, status: list[dict], meta: dict, options: DictConfig, split: str, limit: int | None
) -> str:
    """``table.md``: header with the single-run statement, one table per metric section, and the missing-data list."""
    lines = [f"# Evaluation tables: {split}" + (f" (smoke run, {limit} utterances)" if limit else ""), ""]
    lines += [str(options.note).strip(), ""]
    primary = meta.get("speaker_primary")
    lines += [
        f"Cells are mean (CI95); the interval is a speaker-cluster bootstrap ({int(options.n_boot)} resamples, seed {int(options.seed)}). "
        "WER and CER are corpus-level (sum of errors over sum of reference words). "
        "`n` is the number of utterances and `spk` the number of speakers in the cell. "
        + (
            f"Sim same is the cosine with the gt of the same utterance and Sim spk the cosine with the speaker's other gt utterances "
            f"({primary} speaker model; the gt row's Sim spk is the ceiling)."
            if primary
            else "No speaker embeddings were found, so Sim same and Sim spk are n/a."
        ),
        "",
    ]
    narrow = [s for s in meta["systems"].values() if int(s["sr"]) < 24000]
    if narrow:
        names = ", ".join(s["name"] for s in narrow)
        lines += [
            f"Note: the 16 kHz systems ({names}) are upsampled to 24 kHz for the 24 kHz metrics. They have no content "
            "above 8 kHz, so MCD, MR-STFT and mel L1 (all bands) are not comparable with the 24 kHz systems, and the 8-12 kHz "
            "band is n/a; PESQ (computed at 16 kHz) is comparable.",
            "",
        ]
    for section, title in SECTION_TITLES.items():
        metrics = [m for m in METRICS if m.section == section]
        header = ["system", "cond", "n", "spk", *[m.label for m in metrics]]
        lines += [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for row in table.itertuples(index=False):
            cond = row.condition if row.stratum == "all" else f"{row.condition} {row.stratum.replace('_', ' ')}"
            cells = [fmt(getattr(row, m.key), getattr(row, f"{m.key}_lo"), getattr(row, f"{m.key}_hi"), m.digits) for m in metrics]
            lines.append("| " + " | ".join([row.label, cond, str(row.n_utt), str(row.n_speakers), *cells]) + " |")
        lines.append("")
    gaps = [s for s in status if s["status"] != "complete"]
    lines += ["## Data completeness", ""]
    if not gaps:
        lines += ["Every group of every system and condition is complete.", ""]
    else:
        lines += ["Missing or partial groups (their metrics are n/a or computed on fewer utterances):", ""]
        lines += [f"- {s['system']} {s['condition']} {s['group']}: {s['status']} ({s['n_found']} of {s['n_expected']})" for s in gaps]
        lines.append("")
    gaps = meta.get("metric_gaps", [])
    lines += ["## Utterances without a value", ""]
    if not gaps:
        lines += ["Every metric of every complete group has a value for every utterance of its cell.", ""]
    else:
        lines += [
            "Means and CIs above use only utterances with a finite value (a failed utterance, or no frame voiced in both streams "
            "for the F0 metrics); `err` columns of the result parts give the reason.",
            "",
        ]
        by_cell: dict[tuple, list[str]] = {}
        for g in gaps:
            by_cell.setdefault((g["system"], g["condition"], g["stratum"]), []).append(f"{g['metric']} {g['n']}/{g['n_utt']}")
        lines += [f"- {system} {cond} {stratum}: {', '.join(items)}" for (system, cond, stratum), items in by_cell.items()]
        lines.append("")
    failed = meta.get("audio_failures", {})
    if failed:
        lines += ["## Audio that could not be made", "", "The reference model failed on these utterances; they count as missing values above (errors in `failures.json` of the audio directory).", ""]
        for cell, errors in failed.items():
            lines += [f"- {cell}: " + "; ".join(f"{uid} ({err[:80]})" for uid, err in errors.items())]
        lines.append("")
    lines += ["## Checkpoints", ""]
    for name, info in meta["systems"].items():
        if info.get("checkpoint"):
            c = info["checkpoint"]
            lines.append(f"- {name}: {c['path']} (g_step {c['g_step']} of {c['max_g_steps']}{'' if c['final'] else ', NOT final'})")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- driver


def system_meta(paths: EvalPaths, spec: SystemSpec) -> dict:
    info = {"name": spec.name, "kind": spec.kind, "label": spec.label, "sr": spec.sr, "conditions": list(spec.conditions), "head": spec.head}
    if spec.kind == "vocoder":
        info |= {"experiment": spec.experiment, "vocoder": spec.vocoder}
        path = paths.synth_meta_path(spec)
        info["checkpoint"] = read_json(path).get("checkpoint") if path.is_file() else None
    return info


def run_aggregate(cfg: DictConfig, paths: EvalPaths, systems: dict[str, SystemSpec], table: pd.DataFrame, ids: Sequence[str]) -> dict:
    """Builds all tables for the systems of ``systems`` and writes ``paths.tables_dir()``; returns a summary dict."""
    options = cfg.eval.aggregate
    cells: dict[tuple[str, str], pd.DataFrame] = {}
    status: list[dict] = []
    for spec in reported_systems(systems):
        for condition in spec.conditions:
            frame, st = load_cell(paths, spec, condition, table, ids)
            cells[(spec.name, condition)] = frame
            status += st
    sim_status, primary = add_similarity(cells, paths, systems, [str(m) for m in cfg.eval.spk_models])
    status += sim_status
    for key, frame in cells.items():  # a metric column that no group produced is still reported (as n/a)
        for m in METRICS:
            if not m.is_ratio and m.key not in frame:
                frame[m.key] = np.nan

    wide = table_rows(cells, systems, options)
    paired = paired_table(cells, systems, options)
    per_speaker = per_speaker_table(cells)

    models: dict[str, Any] = {}
    for spec in reported_systems(systems):
        for condition in spec.conditions:
            for group in GROUPS_BY_KIND[spec.kind]:
                models.setdefault(group, {}).update(stage_models(paths.results_dir(spec.name, condition, group)))
            for model in cfg.eval.spk_models:
                models.setdefault("spk", {}).update(stage_models(paths.embeddings_dir(spec.name, condition, str(model))))
    meta = {
        "created": now_iso(),
        "split": paths.split,
        "limit": paths.limit,
        "n_ids": len(ids),
        "n_boot": int(options.n_boot),
        "seed": int(options.seed),
        "ci": float(options.ci),
        "speaker_primary": primary,
        "speaker_models": [str(m) for m in cfg.eval.spk_models],
        "systems": {s.name: system_meta(paths, s) for s in reported_systems(systems)},
        "cells": [
            {k: r[k] for k in ("system", "condition", "stratum", "n_utt", "n_speakers")} for r in wide.to_dict("records")
        ],
        "status": status,
        "missing": [s for s in status if s["status"] != "complete"],
        "metric_gaps": metric_gaps(wide, systems, status),
        "audio_failures": {
            f"{spec.name} {condition}": failures
            for spec in reported_systems(systems)
            for condition in spec.conditions
            if (failures := read_audio_failures(paths, spec, condition))
        },
        "models": models,
        "versions": package_versions(),
        "note": str(options.note).strip(),
    }
    try:
        from sparc.vocoders.features.extractor import fork_commit

        meta["fork_commit"] = fork_commit()
    except Exception:  # provenance must not fail the tables
        meta["fork_commit"] = "unknown"

    out = paths.tables_dir()
    write_text(out / "table.md", render_markdown(wide, status, meta, options, paths.split, paths.limit))
    write_csv(out / "table.csv", wide)
    write_csv(out / "paired.csv", paired)
    write_csv(out / "per_speaker.csv", per_speaker)
    write_json(out / "meta.json", meta)
    logger.info("aggregate: %d cells, %d paired rows, %d groups missing or partial", len(wide), len(paired), len(meta["missing"]))
    return {"table": wide, "paired": paired, "per_speaker": per_speaker, "meta": meta}
