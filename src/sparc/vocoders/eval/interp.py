"""Speaker-embedding Phase 3 evaluation harness: interpolation paths, speaker swap and prior samples.

Definitions are pre-registered (``docs/speaker_embedding_plan.md``, "Phase 3 design"); every number below is a pure
function of the packed test split, the seed and the trained run.

``synth`` loads a trained run (generator + speaker encoder) and writes wavs plus ``plan.json``, ``manifest.parquet``,
``meta.json`` and ``codes.npz`` under ``--out``:

* Speaker codes. An evaluation speaker's code is the encoder applied to its references: the first ``n_refs`` (10)
  utterances of at least 3 s in id order, combined as the encoder does (product of Gaussians incl. the prior for the
  VAE, mean of the per-reference outputs for the FFN / deterministic arms). Works for ``spk_l6`` and all-layer arms.
* Interpolation: 40 unseen test.clean speaker pairs (20 same-sex, 20 cross-sex), one 3-8 s content utterance of a third
  speaker per pair, ``a`` in {0, 1/8, ..., 1}, ``z(a) = (1 - a) z_A + a z_B``. F0 of voiced frames (periodicity > 0)
  is scaled by ``F_A^(1-a) F_B^a / F_S`` (medians of the speakers' reference utterances); the ``unscaled`` variant keeps
  the content F0.
* Swap: N (200) target utterances of speaker Y, the code of another speaker X (half same sex), with and without F0
  scaled by ``F_X / F_Y``.
* Prior (VAE arms only): 200 draws ``z ~ N(0, I)`` on 200 content utterances.

``score`` embeds every output with ECAPA and WavLM-SV, scores UTMOS and Whisper WER (the existing metric classes, pinned
and offline) and writes ``summary.json`` and ``summary.md``. ``compare`` puts several summaries side by side.
Synthesis and scoring skip finished work (existing wavs, existing score chunks).
"""

import argparse
import json
import logging
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sparc.vocoders.constants import F0_CHANNEL, N_FEATURES, PERIODICITY_CHANNEL, SAMPLE_RATE, SPEAKER_DIM

logger = logging.getLogger(__name__)

N_STEPS = 8
ALPHAS = tuple(i / N_STEPS for i in range(N_STEPS + 1))
SV_MODELS = ("ecapa", "wavlm_sv")
VARIANTS = ("scaled", "unscaled")
MIN_REF_SAMPLES = 3 * SAMPLE_RATE
CONTENT_RANGE_S = (3.0, 8.0)
ASR_ALPHA_INDICES = (0, N_STEPS // 2, N_STEPS)
PHASE3_ROOT = "/data/user_data/xoy/spkemb/phase3"
SV_ROOT_DEFAULT = "/data/user_data/xoy/sparc-vocoders"
STREAM_PAIRS, STREAM_CONTENT, STREAM_SWAP, STREAM_PRIOR_Z, STREAM_PRIOR_CONTENT, STREAM_REAL = 1, 2, 3, 4, 5, 6


# ------------------------------------------------------------------------------------------------ pure: F0 and refs


def f0_scale_factor(a: float, f_a: float, f_b: float, f_s: float) -> float:
    """``F_A^(1-a) F_B^a / F_S``: the pitch of the interpolated speaker over the content speaker's pitch."""
    return float(math.exp((1.0 - a) * math.log(f_a) + a * math.log(f_b) - math.log(f_s)))


def scale_f0(features: np.ndarray, factor: float) -> np.ndarray:
    """Copy of ``features [15, T]`` with the F0 of voiced frames (periodicity > 0) multiplied by ``factor``."""
    out = np.array(features, dtype=np.float32, copy=True)
    voiced = out[PERIODICITY_CHANNEL] > 0.0
    out[F0_CHANNEL, voiced] = out[F0_CHANNEL, voiced] * np.float32(factor)
    return out


def median_voiced_f0(frames: Sequence[np.ndarray]) -> float:
    """Median F0 (Hz) over the voiced frames (periodicity > 0, F0 > 0) of utterances ``[T, 15]``; NaN if none."""
    values = [f[(f[:, PERIODICITY_CHANNEL] > 0.0) & (f[:, F0_CHANNEL] > 0.0), F0_CHANNEL] for f in frames]
    values = np.concatenate(values) if values else np.zeros(0)
    return float(np.median(values)) if len(values) else math.nan


def select_references(ids: Sequence[str], n24: Sequence[int], n_refs: int = 10, min_samples: int = MIN_REF_SAMPLES) -> list[str]:
    """The first ``n_refs`` utterances of at least ``min_samples`` samples in id order (a speaker's reference set)."""
    order = sorted(range(len(ids)), key=lambda i: str(ids[i]))
    return [str(ids[i]) for i in order if int(n24[i]) >= min_samples][:n_refs]


# ------------------------------------------------------------------------------------------------ pure: the plan


def build_plan(
    table: pd.DataFrame,
    sex: dict[str, str],
    seed: int = 0,
    n_same: int = 20,
    n_cross: int = 20,
    n_swap: int = 200,
    n_prior: int = 200,
    n_refs: int = 10,
    min_refs: int = 3,
) -> dict[str, Any]:
    """Deterministic evaluation plan from a table with columns ``id, speaker, n24`` and the speaker -> ``F``/``M`` map.

    Eligible speakers have a known sex and at least ``min_refs`` utterances of >= 3 s. Returns ``refs`` (speaker ->
    reference ids), ``pairs`` (``pair_id, a, b, same_gender, content_id, content_speaker``), ``swaps`` (``swap_id,
    target_id, target_speaker (Y), donor (X), same_gender``), ``prior`` (``prior_id, content_id``) and ``real_ids``
    (utterances of as many distinct speakers as possible, round robin, for the diversity baseline).
    """
    table = table.assign(id=table["id"].astype(str), speaker=table["speaker"].astype(str)).sort_values("id")
    ids, spk, n24 = table["id"].to_numpy(), table["speaker"].to_numpy(), table["n24"].to_numpy(dtype=np.int64)
    refs: dict[str, list[str]] = {}
    for s in sorted(set(spk)):
        m = spk == s
        chosen = select_references(ids[m], n24[m], n_refs)
        if len(chosen) >= min_refs and s in sex:
            refs[s] = chosen
    speakers = sorted(refs)
    lo, hi = (int(round(x * SAMPLE_RATE)) for x in CONTENT_RANGE_S)
    in_range = (n24 >= lo) & (n24 <= hi)
    ref_ids = {u for r in refs.values() for u in r}
    utts_of = {s: ids[(spk == s) & in_range] for s in speakers}

    # pairs
    same = [(a, b) for i, a in enumerate(speakers) for b in speakers[i + 1 :] if sex[a] == sex[b]]
    cross = [(a, b) for i, a in enumerate(speakers) for b in speakers[i + 1 :] if sex[a] != sex[b]]
    rng = np.random.default_rng([seed, STREAM_PAIRS])
    chosen_pairs = [(p, True) for p in _pick(same, n_same, rng)] + [(p, False) for p in _pick(cross, n_cross, rng)]
    pairs = []
    for k, ((a, b), is_same) in enumerate(chosen_pairs):
        prng = np.random.default_rng([seed, STREAM_CONTENT, k])
        if prng.random() < 0.5:
            a, b = b, a
        others = [s for s in speakers if s not in (a, b) and len(utts_of[s])]
        if not others:
            raise ValueError("no third speaker with a 3-8 s utterance for the content of pair %d" % k)
        content_spk = others[int(prng.integers(len(others)))]
        content_id = str(utts_of[content_spk][int(prng.integers(len(utts_of[content_spk])))])
        pairs.append(
            {"pair_id": f"p{k:02d}", "a": a, "b": b, "same_gender": is_same, "content_id": content_id, "content_speaker": content_spk}
        )

    # swaps: targets outside their own speaker's reference set
    rng = np.random.default_rng([seed, STREAM_SWAP])
    cand = [(str(u), s) for s in speakers for u in utts_of[s] if u not in ref_ids]
    picked = _pick(cand, n_swap, rng)
    swaps = []
    for k, (target, y) in enumerate(picked):
        want_same = k % 2 == 0
        donors = [s for s in speakers if s != y and (sex[s] == sex[y]) == want_same]
        swaps.append(
            {
                "swap_id": f"s{k:03d}",
                "target_id": target,
                "target_speaker": y,
                "donor": donors[int(rng.integers(len(donors)))],
                "same_gender": want_same,
            }
        )

    # prior: content utterances and the real baseline
    all_in_range = ids[in_range]
    prior_content = _pick([str(u) for u in all_in_range], n_prior, np.random.default_rng([seed, STREAM_PRIOR_CONTENT]))
    prior = [{"prior_id": f"n{k:03d}", "content_id": u} for k, u in enumerate(prior_content)]
    rrng = np.random.default_rng([seed, STREAM_REAL])
    queues = {s: list(utts_of[s][rrng.permutation(len(utts_of[s]))]) for s in speakers}
    real: list[str] = []
    while len(real) < n_prior and any(queues.values()):
        for s in speakers:
            if queues[s] and len(real) < n_prior:
                real.append(str(queues[s].pop(0)))
    return {
        "seed": seed,
        "speakers": speakers,
        "sex": {s: sex[s] for s in speakers},
        "refs": refs,
        "pairs": pairs,
        "swaps": swaps,
        "prior": prior,
        "real_ids": real,
        "n_prior": n_prior,
    }


def _pick(items: Sequence, n: int, rng: np.random.Generator) -> list:
    """``n`` distinct elements of ``items`` in a seeded random order (all of them if fewer)."""
    idx = rng.permutation(len(items))[: int(n)]
    return [items[int(i)] for i in idx]


def prior_codes(n: int, seed: int = 0, dim: int = SPEAKER_DIM) -> np.ndarray:
    """``[n, dim]`` standard-normal speaker codes of the prior samples (seeded)."""
    return np.random.default_rng([seed, STREAM_PRIOR_Z]).standard_normal((n, dim)).astype(np.float32)


@dataclass
class Item:
    """One output wav: what to synthesize (``spec``) and the manifest fields that describe it."""

    kind: str
    key: str
    content_id: str
    f0_factor: float
    spec: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


def build_items(
    plan: dict[str, Any], medians: dict[str, float], parts: Sequence[str], variants: Sequence[str] = VARIANTS, vae: bool = True
) -> list[Item]:
    """Outputs of the requested parts. ``medians`` maps speaker -> median voiced F0 over its references."""
    items: list[Item] = []
    sp_of = {p["content_id"]: p["content_speaker"] for p in plan["pairs"]}
    if "interp" in parts:
        for p in plan["pairs"]:
            for v in variants:
                for i, a in enumerate(ALPHAS):
                    factor = 1.0
                    if v == "scaled":
                        factor = f0_scale_factor(a, medians[p["a"]], medians[p["b"]], medians[p["content_speaker"]])
                    items.append(
                        Item(
                            "interp",
                            f"interp/{p['pair_id']}_{v}_a{i}",
                            p["content_id"],
                            factor,
                            {"a": p["a"], "b": p["b"], "alpha": a},
                            {
                                "pair_id": p["pair_id"], "alpha_idx": i, "variant": v, "spk_a": p["a"], "spk_b": p["b"],
                                "content_spk": p["content_speaker"], "same_gender": p["same_gender"],
                                "asr": i in ASR_ALPHA_INDICES,
                            },
                        )
                    )
    if "swap" in parts:
        for s in plan["swaps"]:
            for v in variants:
                factor = medians[s["donor"]] / medians[s["target_speaker"]] if v == "scaled" else 1.0
                items.append(
                    Item(
                        "swap",
                        f"swap/{s['swap_id']}_{v}",
                        s["target_id"],
                        float(factor),
                        {"donor": s["donor"]},
                        {
                            "pair_id": s["swap_id"], "alpha_idx": -1, "variant": v, "spk_a": s["donor"],
                            "spk_b": s["target_speaker"], "content_spk": s["target_speaker"],
                            "same_gender": s["same_gender"], "asr": True,
                        },
                    )
                )
    if "prior" in parts and vae:
        for k, p in enumerate(plan["prior"]):
            items.append(
                Item(
                    "prior", f"prior/{p['prior_id']}", p["content_id"], 1.0, {"index": k},
                    {"pair_id": p["prior_id"], "alpha_idx": -1, "variant": "prior", "spk_a": "", "spk_b": "",
                     "content_spk": sp_of.get(p["content_id"], ""), "same_gender": False, "asr": True},
                )
            )
    return items


# ------------------------------------------------------------------------------------------------ pure: metrics


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks); NaN for a constant input or non-finite values."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return math.nan
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def crossing_point(d: Sequence[float], alphas: Sequence[float] = ALPHAS) -> float:
    """``a*`` where ``d`` first crosses 0 (linear interpolation between grid points); NaN if it never does."""
    d = np.asarray(d, dtype=np.float64)
    if not np.isfinite(d).all():
        return math.nan
    for i in range(len(d) - 1):
        if d[i] == 0.0:
            return float(alphas[i])
        if d[i] * d[i + 1] < 0.0:
            return float(alphas[i] + (alphas[i + 1] - alphas[i]) * (-d[i]) / (d[i + 1] - d[i]))
    return float(alphas[-1]) if d[-1] == 0.0 else math.nan


def interp_metrics(d: Sequence[float], alphas: Sequence[float] = ALPHAS) -> dict[str, float]:
    """Path metrics of ``d(a) = SIM_B - SIM_A``: spearman, frac_increasing, evenness, range, crossing.

    ``evenness = max_step |d(a+1/8) - d(a)| / (d(1) - d(0))`` (ideal 1/8; NaN when the range is not positive),
    ``crossing = |a* - 0.5|`` (NaN when ``d`` never crosses 0).
    """
    d = np.asarray(d, dtype=np.float64)
    nan = math.nan
    if not np.isfinite(d).all():
        return {"spearman": nan, "frac_increasing": nan, "evenness": nan, "range": nan, "crossing": nan}
    rng = float(d[-1] - d[0])
    steps = np.abs(np.diff(d))
    a_star = crossing_point(d, alphas)
    return {
        "spearman": spearman(d, alphas),
        "frac_increasing": float(np.mean(np.diff(d) > 0)),
        "evenness": float(steps.max() / rng) if rng > 0 else nan,
        "range": rng,
        "crossing": abs(a_star - 0.5) if np.isfinite(a_star) else nan,
    }


def quality_dip(utmos: Sequence[float]) -> float:
    """Minimum UTMOS along the path minus the mean UTMOS of the two endpoints."""
    u = np.asarray(utmos, dtype=np.float64)
    if not np.isfinite(u).all():
        return math.nan
    return float(u.min() - 0.5 * (u[0] + u[-1]))


def interp_score(spearman_value: float, evenness: float, dip: float, steps: int = N_STEPS) -> float:
    """Mean of (Spearman, ``1 - (evenness - 1/steps) / (1 - 1/steps)`` clipped to [0, 1], ``1 + dip`` clipped to [0, 1]).

    A failed path (non-finite Spearman or evenness) contributes 0 for that term; a non-finite dip gives NaN.
    """
    if not np.isfinite(dip):
        return math.nan
    s = spearman_value if np.isfinite(spearman_value) else 0.0
    e = 0.0 if not np.isfinite(evenness) else float(np.clip(1.0 - (evenness - 1.0 / steps) / (1.0 - 1.0 / steps), 0.0, 1.0))
    q = float(np.clip(1.0 + dip, 0.0, 1.0))
    return float((s + e + q) / 3.0)


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """Mean over the finite values and a percentile bootstrap interval over resampled units (e.g. speaker pairs).

    Non-finite values are ignored in each resample; ``n`` counts the finite ones.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    s = np.where(ok[idx], v[idx], 0.0).sum(1)
    c = ok[idx].sum(1)
    means = (s / np.maximum(c, 1))[c > 0]
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {"mean": float(v[ok].mean()), "lo": float(lo), "hi": float(hi), "n": int(ok.sum())}


def rate_ci(errors: Sequence[float], lengths: Sequence[float], n_boot: int = 10000, seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """Corpus-level rate ``sum(errors) / sum(lengths)`` with a bootstrap interval over units (pairs / utterances)."""
    e, n = np.asarray(errors, dtype=np.float64), np.asarray(lengths, dtype=np.float64)
    if len(e) == 0 or n.sum() <= 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": int(len(e))}
    idx = np.random.default_rng(seed).integers(0, len(e), size=(n_boot, len(e)))
    den = n[idx].sum(1)
    rates = e[idx].sum(1)[den > 0] / den[den > 0]
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(rates, [alpha, 1.0 - alpha])
    return {"mean": float(e.sum() / n.sum()), "lo": float(lo), "hi": float(hi), "n": int(len(e))}


def pairwise_diversity(emb: np.ndarray, index: np.ndarray | None = None) -> float:
    """Mean pairwise ``1 - cosine`` over distinct rows ``index`` of ``emb`` (all rows by default)."""
    e = np.asarray(emb, dtype=np.float64)
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)
    cos = e @ e.T
    idx = np.arange(len(e)) if index is None else np.asarray(index)
    sub = cos[np.ix_(idx, idx)]
    off = idx[:, None] != idx[None, :]
    return float((1.0 - sub[off]).mean()) if off.any() else math.nan


def diversity_ci(emb: np.ndarray, n_boot: int = 1000, seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """:func:`pairwise_diversity` with a bootstrap interval over the utterances (self pairs of a resample are dropped)."""
    rng = np.random.default_rng(seed)
    vals = np.array([pairwise_diversity(emb, rng.integers(0, len(emb), len(emb))) for _ in range(n_boot)])
    vals = vals[np.isfinite(vals)]
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha]) if len(vals) else (math.nan, math.nan)
    return {"mean": pairwise_diversity(emb), "lo": float(lo), "hi": float(hi), "n": int(len(emb))}


def novelty_scores(emb: np.ndarray, means: np.ndarray, exclude: Sequence[int] | None = None) -> np.ndarray:
    """Max cosine of each row of ``emb`` to the rows of ``means``; ``exclude[i]`` (a row of ``means``) is skipped for row ``i``."""
    e = np.asarray(emb, dtype=np.float64)
    m = np.asarray(means, dtype=np.float64)
    cos = (e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)) @ (
        m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)
    ).T
    if exclude is not None:
        for i, j in enumerate(exclude):
            if j is not None and j >= 0:
                cos[i, j] = -np.inf
    return cos.max(1)


def cosine_rows(emb: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Cosine of each row of ``emb`` with ``vec`` (NaN rows give NaN)."""
    e = np.asarray(emb, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    return (e @ v) / (np.maximum(np.linalg.norm(e, axis=-1), 1e-12) * max(np.linalg.norm(v), 1e-12))


# ------------------------------------------------------------------------------------------------ summaries (pure)


INTERP_METRICS = ("spearman", "frac_increasing", "evenness", "range", "crossing")


def summarize(
    man: pd.DataFrame,
    emb: dict[str, np.ndarray],
    utmos: np.ndarray,
    text: Sequence[str],
    ref_means: dict[str, dict[str, np.ndarray]],
    real_emb: np.ndarray | None,
    real_speakers: Sequence[str] | None,
    wer_fn: Callable[[str, str], tuple[int, int]],
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Scores of every part present in the manifest ``man`` (columns of ``synth``, plus ``ref_text``).

    ``emb[model]`` is ``[len(man), dim]`` (NaN rows for failures), ``utmos`` ``[len(man)]``, ``text`` the Whisper
    transcripts (``""`` where none was made), ``ref_means[model][speaker]`` the speaker's reference-mean embedding,
    ``wer_fn(ref, hyp) -> (word_errors, ref_words)`` and ``real_emb`` / ``real_speakers`` the ECAPA embeddings of the
    real-utterance baseline of the prior part.
    """
    man = man.reset_index(drop=True)
    out: dict[str, Any] = {}
    if (man["kind"] == "interp").any():
        out["interp"] = _summarize_interp(man, emb, utmos, text, ref_means, wer_fn, n_boot, seed)
    if (man["kind"] == "swap").any():
        out["swap"] = _summarize_swap(man, emb, utmos, text, ref_means, wer_fn, n_boot, seed)
    if (man["kind"] == "prior").any():
        out["prior"] = _summarize_prior(man, emb, utmos, text, ref_means, real_emb, real_speakers, wer_fn, n_boot, seed)
    return out


def _wer(man: pd.DataFrame, rows: np.ndarray, text: Sequence[str], wer_fn) -> tuple[np.ndarray, np.ndarray]:
    counts = [wer_fn(str(man["ref_text"].iloc[r]), str(text[r])) for r in rows]
    return np.array([c[0] for c in counts], dtype=float), np.array([c[1] for c in counts], dtype=float)


def _summarize_interp(man, emb, utmos, text, ref_means, wer_fn, n_boot, seed) -> dict[str, Any]:
    result: dict[str, Any] = {}
    sub_all = man[man["kind"] == "interp"]
    for variant in [v for v in VARIANTS if (sub_all["variant"] == v).any()]:
        sub = sub_all[sub_all["variant"] == variant]
        records, skipped = [], 0
        for pair_id, g in sub.groupby("pair_id", sort=True):
            if len(g) != len(ALPHAS) or sorted(g["alpha_idx"]) != list(range(len(ALPHAS))):
                skipped += 1
                continue
            g = g.sort_values("alpha_idx")
            rows = g.index.to_numpy()
            a, b = str(g["spk_a"].iloc[0]), str(g["spk_b"].iloc[0])
            rec: dict[str, Any] = {"pair_id": pair_id, "same_gender": bool(g["same_gender"].iloc[0]), "spk_a": a, "spk_b": b}
            dip = quality_dip(utmos[rows])
            rec["quality_dip"] = dip
            rec["utmos_endpoints"] = float(0.5 * (utmos[rows[0]] + utmos[rows[-1]]))
            for m in emb:
                d = cosine_rows(emb[m][rows], ref_means[m][b]) - cosine_rows(emb[m][rows], ref_means[m][a])
                rec[f"{m}/d"] = d.tolist()
                for k, v in interp_metrics(d).items():
                    rec[f"{m}/{k}"] = v
                rec[f"{m}/score"] = interp_score(rec[f"{m}/spearman"], rec[f"{m}/evenness"], dip)
            for i in ASR_ALPHA_INDICES:
                e, n = wer_fn(str(g["ref_text"].iloc[i]), str(text[rows[i]]))
                rec[f"wer_errors_a{i}"], rec[f"wer_len_a{i}"] = e, n
            records.append(rec)
        if not records:
            continue
        df = pd.DataFrame(records)
        groups = {"all": np.ones(len(df), bool), "same_gender": df["same_gender"].to_numpy(), "cross_gender": ~df["same_gender"].to_numpy()}
        res: dict[str, Any] = {"n_pairs": len(df), "n_incomplete_pairs": skipped, "metrics": {}, "wer": {}}
        for gname, mask in groups.items():
            if not mask.any():
                continue
            met: dict[str, Any] = {"quality_dip": bootstrap_ci(df["quality_dip"][mask], n_boot, seed)}
            for m in emb:
                for k in (*INTERP_METRICS, "score"):
                    met[f"{m}/{k}"] = bootstrap_ci(df[f"{m}/{k}"][mask], n_boot, seed)
            res["metrics"][gname] = met
        for i in ASR_ALPHA_INDICES:
            res["wer"][f"a{i}"] = rate_ci(df[f"wer_errors_a{i}"], df[f"wer_len_a{i}"], n_boot, seed)
        res["pairs"] = df.to_dict("records")
        result[variant] = res
    return result


def _summarize_swap(man, emb, utmos, text, ref_means, wer_fn, n_boot, seed) -> dict[str, Any]:
    result: dict[str, Any] = {}
    sub_all = man[man["kind"] == "swap"]
    for variant in [v for v in VARIANTS if (sub_all["variant"] == v).any()]:
        sub = sub_all[sub_all["variant"] == variant]
        rows = sub.index.to_numpy()
        x = sub["spk_a"].astype(str).to_numpy()
        y = sub["spk_b"].astype(str).to_numpy()
        same = sub["same_gender"].to_numpy(dtype=bool)
        e, n = _wer(man, rows, text, wer_fn)
        groups = {"all": np.ones(len(rows), bool), "same_gender": same, "cross_gender": ~same}
        res: dict[str, Any] = {"n": len(rows), "metrics": {}, "wer": {}, "utmos": {}}
        for m in emb:
            sim_x = np.array([cosine_rows(emb[m][r][None], ref_means[m][s])[0] for r, s in zip(rows, x)])
            sim_y = np.array([cosine_rows(emb[m][r][None], ref_means[m][s])[0] for r, s in zip(rows, y)])
            for gname, mask in groups.items():
                if mask.any():
                    met = res["metrics"].setdefault(gname, {})
                    met[f"{m}/sim_target"] = bootstrap_ci(sim_x[mask], n_boot, seed)
                    met[f"{m}/sim_source"] = bootstrap_ci(sim_y[mask], n_boot, seed)
                    met[f"{m}/sim_diff"] = bootstrap_ci((sim_x - sim_y)[mask], n_boot, seed)
        for gname, mask in groups.items():
            if mask.any():
                res["wer"][gname] = rate_ci(e[mask], n[mask], n_boot, seed)
                res["utmos"][gname] = bootstrap_ci(utmos[rows][mask], n_boot, seed)
        result[variant] = res
    return result


def _summarize_prior(man, emb, utmos, text, ref_means, real_emb, real_speakers, wer_fn, n_boot, seed) -> dict[str, Any]:
    sub = man[man["kind"] == "prior"]
    rows = sub.index.to_numpy()
    e, n = _wer(man, rows, text, wer_fn)
    gen = emb["ecapa"][rows]
    ok = np.isfinite(gen).all(1)
    speakers = sorted(ref_means["ecapa"])
    means = np.stack([ref_means["ecapa"][s] for s in speakers])
    res: dict[str, Any] = {
        "n": len(rows),
        "utmos": bootstrap_ci(utmos[rows], n_boot, seed),
        "wer": rate_ci(e, n, n_boot, seed),
        "diversity_generated": diversity_ci(gen[ok], min(n_boot, 1000), seed),
        "novelty_generated": bootstrap_ci(novelty_scores(gen[ok], means), n_boot, seed),
    }
    if real_emb is not None and len(real_emb):
        rok = np.isfinite(real_emb).all(1)
        real = real_emb[rok]
        own = [speakers.index(s) if s in speakers else -1 for s in np.asarray(real_speakers)[rok]]
        res["diversity_real"] = diversity_ci(real, min(n_boot, 1000), seed)
        res["novelty_real_other_speakers"] = bootstrap_ci(novelty_scores(real, means, own), n_boot, seed)
    return res


# ------------------------------------------------------------------------------------------------ rendering


def flatten_summary(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    """``{row name: {mean, lo, hi}}`` of the headline numbers of a summary (rows are shared across runs)."""
    rows: dict[str, dict[str, float]] = {}

    def add(name: str, ci: dict[str, float]) -> None:
        rows[name] = {k: ci.get(k, math.nan) for k in ("mean", "lo", "hi")}

    for variant, res in summary.get("interp", {}).items():
        for gname, met in res["metrics"].items():
            if gname != "all":
                continue
            for k, ci in met.items():
                add(f"interp/{variant}/{k}", ci)
        for k, ci in res["wer"].items():
            add(f"interp/{variant}/wer_{k}", ci)
    for variant, res in summary.get("swap", {}).items():
        for gname, met in res["metrics"].items():
            for k, ci in met.items():
                add(f"swap/{variant}/{gname}/{k}", ci)
        for gname, ci in res["wer"].items():
            add(f"swap/{variant}/{gname}/wer", ci)
        for gname, ci in res["utmos"].items():
            add(f"swap/{variant}/{gname}/utmos", ci)
    if "prior" in summary:
        for k, v in summary["prior"].items():
            if isinstance(v, dict):
                add(f"prior/{k}", v)
    return rows


def _fmt(ci: dict[str, float] | None) -> str:
    if ci is None or not np.isfinite(ci.get("mean", math.nan)):
        return "n/a"
    return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def render_markdown(summary: dict[str, Any], title: str = "Phase 3 interpolation / swap / prior evaluation") -> str:
    """Markdown report of one summary (mean and bootstrap 95% CI)."""
    lines = [f"# {title}", ""]
    meta = summary.get("meta", {})
    if meta:
        lines += [f"Run `{meta.get('run', '?')}`, checkpoint step {meta.get('g_step', '?')}, seed {meta.get('seed', '?')}.", ""]
    rows = flatten_summary(summary)
    for part in ("interp", "swap", "prior"):
        names = [k for k in rows if k.startswith(part + "/")]
        if not names:
            continue
        lines += [f"## {part}", "", "| metric | mean [95% CI] |", "|---|---|"]
        lines += [f"| {k[len(part) + 1:]} | {_fmt(rows[k])} |" for k in names]
        lines.append("")
    lines += [
        "Interval: bootstrap over speaker pairs (interp), utterances (swap, prior); WER is corpus level.",
        "Evenness ideal 1/8; interp score = mean(Spearman, evenness term, 1 + quality dip).",
    ]
    return "\n".join(lines) + "\n"


def render_compare(summaries: Sequence[dict[str, Any]], labels: Sequence[str]) -> str:
    """Side-by-side Markdown table of the headline numbers of several summaries."""
    flat = [flatten_summary(s) for s in summaries]
    names: list[str] = []
    for f in flat:
        names += [k for k in f if k not in names]
    lines = ["| metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for k in names:
        lines.append(f"| {k} | " + " | ".join(_fmt(f.get(k)) for f in flat) + " |")
    return "\n".join(lines) + "\n"


def sanitize(obj: Any) -> Any:
    """JSON-safe copy: NaN / inf become ``None``, numpy scalars and arrays become Python values."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ------------------------------------------------------------------------------------------------ environment / loading


def setup_offline_env() -> None:
    """Pinned, offline model loading: private hub cache, ``HF_HUB_OFFLINE=1``, torch hub under the user's cache."""
    sv = os.environ.get("SV_ROOT", SV_ROOT_DEFAULT)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_CACHE", f"{sv}/hf_hub")
    os.environ.setdefault("TORCH_HOME", f"/data/user_data/{os.environ.get('USER', 'xoy')}/.cache/torch")


def metrics_config():
    """``cfg.eval_metrics`` of the packaged default (``conf/eval_metrics/default.yaml``)."""
    import importlib.resources

    from omegaconf import OmegaConf

    default = importlib.resources.files("sparc.conf").joinpath("eval_metrics", "default.yaml")
    with importlib.resources.as_file(default) as path:
        return OmegaConf.load(path)


def pick_device(name: str = "auto"):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_run(run_dir: str | Path, ckpt: str | Path | None = None, device: str = "auto"):
    """``(module, cfg, info)``: the run's module restored from ``ckpt`` (default: newest loadable step checkpoint).

    The config is the one stored in the checkpoint (``hyper_parameters.cfg``), so no Hydra composition or environment
    variable is needed. The module is in eval mode on the device.
    """
    import torch
    from omegaconf import OmegaConf

    from sparc.vocoders.training.callbacks import list_step_checkpoints
    from sparc.vocoders.training.module import VocoderGANModule

    run_dir = Path(run_dir)
    if ckpt is not None and Path(ckpt).is_file():
        candidates = [(None, Path(ckpt))]
    else:
        candidates = list(reversed(list_step_checkpoints(Path(ckpt) if ckpt else run_dir / "ckpt")))
    for step, path in candidates:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # a checkpoint cut short by a kill
            logger.warning("skipping unloadable checkpoint %s: %r", path, error)
            continue
        cfg = OmegaConf.create(state["hyper_parameters"]["cfg"])
        module = VocoderGANModule(cfg)
        module.load_state_dict(state["state_dict"], strict=True)
        g_step = int(state["vocoder_counters"]["g_step"])
        info = {"ckpt": str(path), "g_step": g_step, "run": str(run_dir), "experiment": str(cfg.experiment_name)}
        return module.eval().to(pick_device(device)), cfg, info
    raise FileNotFoundError(f"no loadable checkpoint for {run_dir} (ckpt={ckpt})")


# ------------------------------------------------------------------------------------------------ synthesis


def speaker_code(module, store, utt_ids: Sequence[str], device) -> np.ndarray:
    """Code ``[64]`` of one speaker from its reference utterances, through the module's own multi-reference ``embed``."""
    import torch

    vecs = np.stack([store.speaker_vector(store.index_of(u)) for u in utt_ids])
    raw = torch.from_numpy(vecs)[None].to(device)  # [1, K, ...]
    with torch.inference_mode():
        z, _ = module.speaker.embed(raw, torch.ones(1, len(utt_ids), dtype=torch.bool, device=device))
    return z[0].float().cpu().numpy()


def utterance_features(store, utt_id: str, gain_db: float) -> np.ndarray:
    """Features ``[15, T]`` of an utterance with the evaluation gain on the loudness channel (as ``FullUtteranceDataset``)."""
    from sparc.vocoders.constants import LOUDNESS_CHANNEL
    from sparc.vocoders.data.gain import apply_gain, gain_factor

    u = store.index_of(utt_id)
    feats, loud = store.frames(u, 0, int(store.T[u]))
    out = np.ascontiguousarray(feats.T)
    out[LOUDNESS_CHANNEL] = apply_gain(loud, gain_factor(store.peak24[u], gain_db))
    assert out.shape[0] == N_FEATURES
    return out


def reference_medians(store, plan: dict[str, Any]) -> dict[str, float]:
    """Median voiced F0 per eligible speaker over its reference utterances (from the cached feature channel 12)."""
    medians = {}
    for s, ids in plan["refs"].items():
        medians[s] = median_voiced_f0([store.frames(store.index_of(u), 0, int(store.T[store.index_of(u)]))[0] for u in ids])
    return medians


def _read_sex(path: str | Path | None) -> dict[str, str]:
    from sparc.vocoders.eval.io import read_speaker_sex

    candidates = [path] if path else []
    sv = os.environ.get("SV_ROOT", SV_ROOT_DEFAULT)
    candidates += [f"{sv}/eval/ref_data/LibriTTS_R/SPEAKERS.txt"]
    for c in candidates:
        if c and Path(c).is_file():
            return read_speaker_sex(c)
    raise FileNotFoundError(f"no LibriTTS speaker list among {candidates}; pass --sex-file")


def run_synth(args: argparse.Namespace) -> Path:
    """Loads the run, builds the plan, writes plan / codes / manifest and every missing wav."""
    import torch

    from sparc.vocoders.eval.io import atomic_write, packed_dir, write_json, write_parquet, write_wav
    from sparc.vocoders.data.dataset import PackedStore
    from sparc.vocoders.training.callbacks import EVAL_SEED, fixed_torch_rng

    setup_offline_env()
    module, cfg, info = load_run(args.run, args.ckpt, args.device)
    device = next(module.parameters()).device
    out = Path(args.out) if args.out else Path(PHASE3_ROOT) / "eval" / info["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    parts = [p for p in args.parts.split(",") if p]
    vae = bool(getattr(module, "vae", False))
    if "prior" in parts and not vae:
        logger.warning("%s has no VAE latent: the prior part is skipped", info["experiment"])
    variants = VARIANTS if args.f0 == "both" else (args.f0,)
    cache_root = Path(args.cache_root or cfg.paths.cache_root)
    gain_db = float(cfg.data.eval_gain_db) if args.gain_db is None else args.gain_db
    spk_cfg = cfg.speaker
    layer = str(spk_cfg.layer)
    root = spk_cfg.get("layer_stats_root") if layer == "layers" else None
    store = PackedStore([packed_dir(cache_root, args.split)], layer, root)
    table = pd.DataFrame({"id": store.ids, "speaker": store.speakers[store.speaker_codes], "n24": store.n24})
    plan = build_plan(table, _read_sex(args.sex_file), args.seed, args.n_same, args.n_cross, args.n_swap, args.n_prior, args.n_refs)
    plan["wav_paths"] = {u: str(store.wav_paths[store.index_of(u)]) for u in _plan_utterances(plan)}
    medians = reference_medians(store, plan)
    bad = [s for s, m in medians.items() if not np.isfinite(m)]
    if bad:
        raise RuntimeError(f"speakers without voiced reference frames: {bad}")
    plan["f0_median"] = medians
    write_json(out / "plan.json", plan)
    items = build_items(plan, medians, parts, variants, vae)

    needed = set()
    for it in items:
        if it.kind == "interp":
            needed |= {it.spec["a"], it.spec["b"]}
        elif it.kind == "swap":
            needed.add(it.spec["donor"])
    zs = {s: speaker_code(module, store, plan["refs"][s], device) for s in sorted(needed)}
    zp = prior_codes(args.n_prior, args.seed) if vae else None
    np.savez(out / "codes.npz", **{f"spk/{s}": z for s, z in zs.items()}, **({"prior": zp} if zp is not None else {}))
    rows = []
    for it in items:
        rows.append(
            {"key": it.key, "kind": it.kind, "wav": it.key + ".wav", "content_id": it.content_id,
             "ref_wav": plan["wav_paths"][it.content_id], "f0_factor": it.f0_factor, **it.meta}
        )
    manifest = pd.DataFrame(rows)
    write_parquet(out / "manifest.parquet", manifest)
    write_json(out / "meta.json", {**info, "parts": parts, "variants": list(variants), "seed": args.seed, "split": args.split,
                                   "cache_root": str(cache_root), "gain_db": gain_db, "vae": vae,
                                   "speaker_type": str(spk_cfg.get("type", "ffn")), "n_items": len(items)})
    done = skipped = 0
    cache: dict[str, np.ndarray] = {}
    for it in items:
        path = out / "wav" / (it.key + ".wav")
        if path.is_file():
            skipped += 1
            continue
        if it.content_id not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[it.content_id] = utterance_features(store, it.content_id, gain_db)
        feats = cache[it.content_id]
        if it.f0_factor != 1.0:
            feats = scale_f0(feats, it.f0_factor)
        z = _item_code(it, zs, zp)
        batch = {"features": torch.from_numpy(feats)[None].to(device)}
        with torch.inference_mode(), fixed_torch_rng(device, EVAL_SEED):
            wav = module.synthesize(batch, z=torch.from_numpy(z)[None].to(device))
        wav = wav[0, 0].float().cpu().numpy()
        write_wav(path, wav, SAMPLE_RATE)
        done += 1
        if done % 50 == 0:
            logger.info("synthesized %d / %d", done + skipped, len(items))
    logger.info("synth finished: %d new, %d existing, %d total in %s", done, skipped, len(items), out)
    return out


def _plan_utterances(plan: dict[str, Any]) -> list[str]:
    ids = {p["content_id"] for p in plan["pairs"]} | {s["target_id"] for s in plan["swaps"]}
    ids |= {p["content_id"] for p in plan["prior"]} | set(plan["real_ids"])
    ids |= {u for r in plan["refs"].values() for u in r}
    return sorted(ids)


def _item_code(it: Item, zs: dict[str, np.ndarray], zp: np.ndarray | None) -> np.ndarray:
    if it.kind == "interp":
        a = float(it.spec["alpha"])
        return ((1.0 - a) * zs[it.spec["a"]] + a * zs[it.spec["b"]]).astype(np.float32)
    if it.kind == "swap":
        return zs[it.spec["donor"]]
    return zp[int(it.spec["index"])]


# ------------------------------------------------------------------------------------------------ scoring


class Scorers:
    """Lazily loaded SV embedders, UTMOS and Whisper (pinned, offline)."""

    def __init__(self, device):
        self.device = device
        self.cfg = metrics_config()
        self._models: dict[str, Any] = {}

    def get(self, key: str):
        if key not in self._models:
            logger.info("loading %s", key)
            if key == "utmos":
                from sparc.vocoders.eval.metrics.utmos import UTMOS

                self._models[key] = UTMOS(self.cfg.utmos, self.device)
            elif key == "asr":
                from sparc.vocoders.eval.metrics.asr import WhisperASR

                self._models[key] = WhisperASR(self.cfg.asr, self.device)
            else:
                from sparc.vocoders.eval.metrics.speaker import SpeakerEmbedder

                self._models[key] = SpeakerEmbedder(key, self.cfg.speaker, self.device)
        return self._models[key]


def real_wer(ref: str, hyp: str) -> tuple[int, int]:
    """``(word errors, reference words)`` after Whisper English normalization of both strings."""
    from sparc.vocoders.eval.metrics.asr import edit_counts, normalize_text

    c = edit_counts(normalize_text(ref), normalize_text(hyp))
    return c["word_errors"], c["word_ref_len"]


def _gt16(store, utt_id: str, gain_db: float) -> np.ndarray:
    from sparc.vocoders.eval.io import gt_audio, to_16k

    return to_16k(gt_audio(store, store.index_of(utt_id), gain_db), SAMPLE_RATE)


def reference_embeddings(out: Path, scorers: Scorers, store, plan: dict[str, Any], gain_db: float) -> dict[str, dict[str, np.ndarray]]:
    """Reference-mean embedding of every eligible speaker per SV model (cached in ``scores/refs.npz``)."""
    from sparc.vocoders.eval.io import write_npz
    from sparc.vocoders.eval.metrics.speaker import mean_embedding

    path = out / "scores" / "refs.npz"
    if path.is_file():
        with np.load(path) as z:
            return {m: {s: z[f"{m}/{s}"] for s in plan["refs"]} for m in SV_MODELS}
    result: dict[str, dict[str, np.ndarray]] = {m: {} for m in SV_MODELS}
    wavs = {s: [_gt16(store, u, gain_db) for u in ids] for s, ids in plan["refs"].items()}
    for m in SV_MODELS:
        model = scorers.get(m)
        for s, w in wavs.items():
            result[m][s] = mean_embedding(model.embed(w, errors=[]))
    write_npz(path, {f"{m}/{s}": v for m, d in result.items() for s, v in d.items()})
    return result


def real_embeddings(out: Path, scorers: Scorers, store, plan: dict[str, Any], gain_db: float) -> np.ndarray:
    """ECAPA embeddings of the real-utterance baseline (cached in ``scores/real.npz``)."""
    from sparc.vocoders.eval.io import write_npz

    path = out / "scores" / "real.npz"
    if path.is_file():
        with np.load(path) as z:
            return z["ecapa"]
    emb = scorers.get("ecapa").embed([_gt16(store, u, gain_db) for u in plan["real_ids"]], errors=[])
    write_npz(path, {"ecapa": emb})
    return emb


def score_chunks(out: Path, manifest: pd.DataFrame, scorers: Scorers, chunk_size: int) -> dict[str, np.ndarray]:
    """Embeddings, UTMOS and transcripts of every manifest row, in chunks that are skipped when already scored."""
    from sparc.vocoders.eval.io import read_wav, to_16k, write_npz

    n = len(manifest)
    parts = []
    for k, start in enumerate(range(0, n, chunk_size)):
        block = manifest.iloc[start : start + chunk_size]
        keys = np.array(block["key"].astype(str).tolist())
        path = out / "scores" / f"chunk_{k:04d}.npz"
        if path.is_file():
            with np.load(path) as z:
                if np.array_equal(z["keys"], keys):
                    parts.append({name: z[name] for name in z.files})
                    continue
        wavs = []
        for w in block["wav"]:
            wav, rate = read_wav(out / "wav" / w)
            wavs.append(to_16k(wav, rate))
        data: dict[str, np.ndarray] = {"keys": keys}
        for m in SV_MODELS:
            data[f"emb_{m}"] = scorers.get(m).embed(wavs, errors=[])
        errors: list[str] = []
        data["utmos"] = np.array(scorers.get("utmos").score(wavs, errors), dtype=np.float64)
        want = [i for i, a in enumerate(block["asr"].to_numpy(dtype=bool)) if a]
        text = [""] * len(block)
        if want:
            hyp = scorers.get("asr").transcribe([wavs[i] for i in want], errors=[])
            for i, h in zip(want, hyp):
                text[i] = h
        data["text"] = np.array(text, dtype="<U1024")
        write_npz(path, data)
        parts.append(data)
        logger.info("scored %d / %d", min(start + chunk_size, n), n)
    cat = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]} if parts else {}
    return cat


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    """Scores a ``synth`` output directory and writes ``summary.json`` / ``summary.md``."""
    from sparc.vocoders.data.dataset import PackedStore
    from sparc.vocoders.eval.io import packed_dir, read_json, write_json, write_text
    from sparc.vocoders.eval.metrics.asr import read_reference

    setup_offline_env()
    out = Path(args.out)
    meta = read_json(out / "meta.json")
    plan = read_json(out / "plan.json")
    manifest = pd.read_parquet(out / "manifest.parquet")
    missing = [w for w in manifest["wav"] if not (out / "wav" / w).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} wavs missing (e.g. {missing[0]}); run synth first")
    cache_root = Path(args.cache_root or meta["cache_root"])
    store = PackedStore([packed_dir(cache_root, meta["split"])], "l6")
    scorers = Scorers(pick_device(args.device))
    gain_db = float(meta["gain_db"])
    ref_means = reference_embeddings(out, scorers, store, plan, gain_db)
    real = real_embeddings(out, scorers, store, plan, gain_db) if (manifest["kind"] == "prior").any() else None
    cat = score_chunks(out, manifest, scorers, args.chunk_size)
    manifest = manifest.assign(ref_text=[read_reference(w) for w in manifest["ref_wav"]])
    real_speakers = [store.speakers[store.speaker_codes[store.index_of(u)]] for u in plan["real_ids"]]
    summary = summarize(
        manifest, {m: cat[f"emb_{m}"] for m in SV_MODELS}, cat["utmos"], cat["text"], ref_means, real, real_speakers,
        real_wer, args.n_boot, args.seed,
    )
    summary["meta"] = {**meta, "bootstrap": {"n_boot": args.n_boot, "seed": args.seed}}
    pd.DataFrame({"key": cat["keys"], "utmos": cat["utmos"], "text": cat["text"]}).to_csv(out / "scores" / "items.csv", index=False)
    write_json(out / "summary.json", sanitize(summary))
    write_text(out / "summary.md", render_markdown(summary))
    return summary


def run_compare(args: argparse.Namespace) -> str:
    """Side-by-side table of several ``summary.json`` files."""
    outs = [Path(o) for o in args.outs]
    labels = args.labels or [o.name for o in outs]
    if len(labels) != len(outs):
        raise ValueError("--labels must have one entry per --outs directory")
    text = render_compare([json.loads((o / "summary.json").read_text()) for o in outs], labels)
    if args.md:
        Path(args.md).write_text(text)
    print(text)
    return text


# ------------------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="write wavs, plan and manifest of a trained run")
    s.add_argument("--run", required=True, help="run directory (<runs>/<experiment>/<vocoder>)")
    s.add_argument("--out", default=None, help=f"default {PHASE3_ROOT}/eval/<experiment>")
    s.add_argument("--ckpt", default=None, help="checkpoint file or directory (default: newest loadable step in <run>/ckpt)")
    s.add_argument("--parts", default="interp,swap,prior")
    s.add_argument("--split", default="test.clean")
    s.add_argument("--cache-root", default=None)
    s.add_argument("--sex-file", default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-same", type=int, default=20)
    s.add_argument("--n-cross", type=int, default=20)
    s.add_argument("--n-swap", type=int, default=200)
    s.add_argument("--n-prior", type=int, default=200)
    s.add_argument("--n-refs", type=int, default=10)
    s.add_argument("--f0", choices=("both", *VARIANTS), default="both", help="interpolation / swap pitch variants")
    s.add_argument("--gain-db", type=float, default=None)
    s.add_argument("--device", default="auto")
    c = sub.add_parser("score", help="embeddings, UTMOS, WER, summary.json / summary.md")
    c.add_argument("--out", required=True)
    c.add_argument("--cache-root", default=None)
    c.add_argument("--device", default="auto")
    c.add_argument("--chunk-size", type=int, default=48)
    c.add_argument("--n-boot", type=int, default=10000)
    c.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("compare", help="side-by-side table of summaries")
    m.add_argument("--outs", nargs="+", required=True)
    m.add_argument("--labels", nargs="+", default=None)
    m.add_argument("--md", default=None, help="also write the table to this file")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    {"synth": run_synth, "score": run_score, "compare": run_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
