"""Speaker-embedding Phase 4 evaluation: zero-shot cloning of unseen voices (``docs/speaker_embedding_plan.md``, Phase 4).

Everything here is a pure function of the pool tables, the seed and the trained run; definitions follow the
pre-registration.

Pools: ``libritts`` (test.clean), ``vctk``, ``globe`` (and ``libritts_dev`` for selecting the inference baseline); see
:mod:`sparc.vocoders.eval.clone_pools` for how VCTK / GLOBE subsets are chosen and cached.

``plan`` writes ``<root>/plans/<pool>.json`` (independent of any run, so every arm is scored on identical items):

* 400 swap pairs per pool (content utterance of speaker S rendered with the voice of speaker Y), half same-gender and
  half cross-gender, content duration stratified over <3 / 3-6 / 6-10 / >10 s (equal quotas, a bin with too few
  utterances hands its quota to the others), targets used evenly; Y has up to 10 references of >= 3 s (a seeded draw
  of its long utterances), disjoint from every content utterance because content is never a reference of its own
  speaker; a wrong-speaker W (same gender as Y, not S or Y); seeded permutations of each speaker's references (``K``
  references = the first ``K``); 100 of the pairs are flagged for the reference-length / noise curves;
* ``own`` items for the guards: an utterance of speaker S rendered with S's own references (T2: 1 reference, T3: K).

``synth`` renders them with a trained run. Variants (names in ``VARIANTS``; K = ``--k-main``, default 5):

* ``base`` (K = 1), ``k5``, ``k10``: code of K references (mean of the per-reference codes, ``speaker.embed`` with a
  mask); ``k5_f0``: + F0 median and semitone-range matching to the target references; ``k5_f0_loud``: + loudness
  matching; ``k5_f0_loud_ema``: + EMA per-channel mean / std matching (ablation);
* controls: ``ctl_src`` (the source speaker's own code with the target's F0 / loudness matching) and ``ctl_wrong`` (a
  third speaker's code with the same matching);
* curves (flagged pairs only): ``len{1,3,10,30}_{base,best}`` (one code and the matching statistics from the first L
  seconds of the target's references concatenated, re-encoded from audio) and ``noisy_{base,best}`` (references with
  white noise at 20 dB SNR, re-encoded); ``best`` = ``--best-variant``;
* ``own_t2`` / ``own_t3``.

Matching definitions (all on the vocoder's feature frames; ``voiced`` = periodicity > 0 and F0 > 0):

* F0: ``ln F0' = m_t + r (ln F0 - m_c)`` on voiced frames, ``m`` = median of ln F0, ``r`` = ratio of the P10-P90
  ranges (target refs over the utterance) clipped to [0.5, 2], F0' clipped to [50, 550] Hz;
* loudness: channel 13 (loud_raw x evaluation gain) times ``exp(med_t - med_c)``, ``med`` = median ln loudness over
  voiced frames (the references carry the same peak gain as the content);
* EMA: channels 0-11 standardized with the utterance's per-channel mean / std (all frames) and re-scaled to those of the
  target references' frames concatenated.

``score`` embeds every output with ECAPA, WavLM-SV and an optional held-out SV model (``--holdout-sv``), UTMOS, Whisper
WER, then ``summarize`` aggregates. Aggregation is per target speaker first, then the mean over speakers with a
speaker-level bootstrap 95% interval (10k). Per pool the ground truth gives the same-/different-speaker cosine
distributions (utterance vs the mean of the speaker's other references / of another speaker's references) and the EER
threshold. Per output: cosine to the target's reference-mean embedding (primary), to the source's, top-1 speaker ID
among the pool's speakers, acceptance at the EER threshold, and the ground-truth similarity of the content utterance to
its own speaker (``gt_same``, duration matched) and to the target (``gt_diff``).

``compare`` pairs two output directories (arm vs baseline) on the shared plan and evaluates the pre-registered decision
rule fields.

Order of preparation per pool: list, extract, pack, then ``plan`` (``synth`` checks that every planned utterance exists in
the packed store). Conventions: unvoiced frames keep their filled-in F0 contour in every F0 operation (as
``eval/interp.py`` and swap-in-training); the held-out SV model is required by the decision (``decision_rule`` raises
without it unless ``--allow-missing-holdout``) and should be an architecture / training set different from the
primary ECAPA; an arm trained against WavLM-SV (``meta.json`` ``trained_against``) is judged without it automatically;
GLOBE speakers are ECAPA-clustered pseudo-speakers, so ECAPA does not count towards the GLOBE same-sign check and the
GLOBE speaker-level intervals are anti-conservative (clusters may split one person). ``sim_target`` uses all of the
target's references and so includes the ones that produced the code; ``sim_target_heldout`` uses only the references
that neither gave the code nor the matching statistics (NaN if none is left, e.g. K = 10) and is what ``select-baseline``
ranks on. Speaker-relative-F0 checkpoints (``speaker.rel_f0``) get ``utt_f0`` / ``ref_f0`` batches and the pitch code;
their F0 matching keeps the content median (the generator sees ratios, only the range matters).

Usage (GPU for synth / score)::

    python -m sparc.vocoders.eval.clone plan --pools libritts,vctk,globe
    python -m sparc.vocoders.eval.clone synth --run <runs>/<exp>/hifigan --out <out> --shard 0 --nshards 4 --max-minutes 50
    python -m sparc.vocoders.eval.clone score --out <out> --shard 0 --nshards 4      # embeddings, UTMOS, Whisper
    python -m sparc.vocoders.eval.clone score --out <out> --stage summarize          # CPU, needs all chunks
    python -m sparc.vocoders.eval.clone compare --a <out_arm> --b <out_base> --margin 0.02
    python -m sparc.vocoders.eval.clone margin --a <b0_seed0_out> --b <b0_seed1_out> --json margin.json     # noise floor m
    python -m sparc.vocoders.eval.clone select-baseline --out <b0_dev_out> --pool libritts_dev --json best.json
    # then: synth --best-variant best.json
"""

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sparc.vocoders.constants import (
    F0_CHANNEL,
    F0_MAX_HZ,
    F0_MIN_HZ,
    LOUDNESS_CHANNEL,
    LOUDNESS_EPS,
    N_EMA,
    PERIODICITY_CHANNEL,
    SAMPLE_RATE,
)
from sparc.vocoders.eval.interp import bootstrap_ci, cosine_rows, rate_ci, sanitize

logger = logging.getLogger(__name__)

PHASE4_ROOT = "/data/user_data/xoy/spkemb/phase4"
POOLS_DEFAULT = ("libritts", "vctk", "globe")
SV_MODELS = ("ecapa", "wavlm_sv")
PRIMARY_MODEL = "ecapa"
DUR_EDGES = (3.0, 6.0, 10.0)
BIN_NAMES = ("lt3", "3-6", "6-10", "gt10")
MIN_REF_S = 3.0
MIN_CONTENT_S = 1.0
F0_RANGE_CLIP = (0.5, 2.0)
NOISE_SNR_DB = 20.0
CURVE_SECONDS = (1, 3, 10, 30)
STREAM_REFS, STREAM_SWAP, STREAM_OWN, STREAM_ORDER = 21, 22, 23, 24
GUARDS = {"wer_points": 0.5, "utmos": -0.10, "source": 0.02, "t23": -0.01}  # t23 (-0.01) is provisional: confirm with the user
# Per pool, SV models that are optimistic there and so do not count towards the same-sign requirement: GLOBE speakers are
# ECAPA-clustered pseudo-speakers, so ECAPA identity metrics on GLOBE are partly circular.
SIGN_EXEMPT = {"globe": ("ecapa",)}
REQUIRED_MODELS = ("holdout",)


# ------------------------------------------------------------------------------------------------ pure: plan


def duration_bin(dur_s: float) -> int:
    """Index into ``BIN_NAMES`` of a duration in seconds (<3, 3-6, 6-10, >=10)."""
    return int(sum(dur_s >= e for e in DUR_EDGES))


def allocate_quota(total: int, available: Sequence[int]) -> list[int]:
    """Splits ``total`` over bins as evenly as possible without exceeding ``available``; shortfalls go to the others."""
    quota = [0] * len(available)
    left = int(total)
    open_bins = [i for i, a in enumerate(available) if a > 0]
    while left > 0 and open_bins:
        share, extra = divmod(left, len(open_bins))
        given = 0
        for j, i in enumerate(list(open_bins)):
            want = share + (1 if j < extra else 0)
            take = min(want, available[i] - quota[i])
            quota[i] += take
            given += take
        left -= given
        open_bins = [i for i in open_bins if quota[i] < available[i]]
        if given == 0:
            break
    return quota


def _rng(seed: int, stream: int, *extra: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream, *extra])


def select_speaker_refs(
    table: pd.DataFrame, seed: int, n_refs: int, min_refs: int, min_ref_s: float = MIN_REF_S
) -> dict[str, list[str]]:
    """References per eligible speaker: a seeded draw of its utterances of >= ``min_ref_s``, at most ``n_refs``.

    One long utterance is left out of the references when the speaker has more than ``min_refs`` of them, so it can
    serve as content. Speakers with fewer than ``min_refs`` references are not eligible.
    """
    refs: dict[str, list[str]] = {}
    for spk, g in table.groupby("speaker", sort=True):
        long_ids = sorted(g.loc[g["n24"] >= min_ref_s * SAMPLE_RATE, "id"].astype(str))
        if len(long_ids) < min_refs:
            continue
        perm = _rng(seed, STREAM_REFS, zlib.crc32(str(spk).encode())).permutation(len(long_ids))
        spare = 1 if len(long_ids) > min_refs else 0
        refs[str(spk)] = [long_ids[int(i)] for i in perm[: min(n_refs, len(long_ids) - spare)]]
    return refs


def _least_used(candidates: list[str], usage: dict[str, int], rng: np.random.Generator) -> str:
    low = min(usage[c] for c in candidates)
    pool = [c for c in candidates if usage[c] == low]
    return pool[int(rng.integers(len(pool)))]


def build_plan(
    table: pd.DataFrame,
    pool: str,
    seed: int = 0,
    n_swap: int = 400,
    n_own: int = 100,
    n_curve: int = 100,
    n_refs: int = 10,
    min_refs: int = 10,
) -> dict[str, Any]:
    """Deterministic plan of one pool from a table with columns ``id, speaker, gender (F/M), n24``."""
    table = table.assign(id=table["id"].astype(str), speaker=table["speaker"].astype(str)).sort_values("id")
    table = table[table["gender"].isin(["F", "M"])]
    refs = select_speaker_refs(table, seed, n_refs, min_refs)
    speakers = sorted(refs)
    if len(speakers) < 3:
        raise ValueError(f"{pool}: only {len(speakers)} eligible speakers (need >= 3)")
    gender = {s: str(table.loc[table["speaker"] == s, "gender"].iloc[0]) for s in speakers}
    ref_set = {s: set(r) for s, r in refs.items()}
    t = table[table["speaker"].isin(speakers)]
    cand = [
        (str(r.id), str(r.speaker), r.n24 / SAMPLE_RATE)
        for r in t.itertuples()
        if r.id not in ref_set[str(r.speaker)] and r.n24 / SAMPLE_RATE >= MIN_CONTENT_S
    ]
    by_bin: list[list[tuple[str, str, float]]] = [[] for _ in BIN_NAMES]
    for c in cand:
        by_bin[duration_bin(c[2])].append(c)
    avail = [len(b) for b in by_bin]
    quota = allocate_quota(n_swap, avail)
    rng = _rng(seed, STREAM_SWAP)
    usage = {s: 0 for s in speakers}
    swaps: list[dict[str, Any]] = []
    for b, q in enumerate(quota):
        picked = [by_bin[b][int(i)] for i in rng.permutation(len(by_bin[b]))[:q]]
        for k, (uid, src, dur) in enumerate(picked):
            same = k % 2 == 0
            targets = [s for s in speakers if s != src and (gender[s] == gender[src]) == same]
            if not targets:
                same = not same
                targets = [s for s in speakers if s != src and (gender[s] == gender[src]) == same]
            tgt = _least_used(targets, usage, rng)
            usage[tgt] += 1
            wrong_c = [s for s in speakers if s not in (src, tgt) and gender[s] == gender[tgt]]
            wrong = wrong_c[int(rng.integers(len(wrong_c)))] if wrong_c else [s for s in speakers if s not in (src, tgt)][0]
            swaps.append(
                {"content_id": uid, "source": src, "target": tgt, "wrong": wrong, "same_gender": bool(gender[src] == gender[tgt]),
                 "dur_s": float(dur), "dur_bin": b}
            )
    order = _rng(seed, STREAM_ORDER).permutation(len(swaps))
    swaps = [swaps[int(i)] for i in order]
    for k, s in enumerate(swaps):
        s["swap_id"] = f"s{k:03d}"
        for role in ("target", "source", "wrong"):
            prng = _rng(seed, STREAM_ORDER, k, zlib.crc32(role.encode()))
            r = refs[s[role]]
            s[f"{role}_order"] = [r[int(i)] for i in prng.permutation(len(r))]
    # own-speaker items
    orng = _rng(seed, STREAM_OWN)
    own_q = allocate_quota(n_own, avail)
    own: list[dict[str, Any]] = []
    for b, q in enumerate(own_q):
        for i in orng.permutation(len(by_bin[b]))[:q]:
            uid, spk, dur = by_bin[b][int(i)]
            r = refs[spk]
            own.append({"content_id": uid, "speaker": spk, "dur_s": float(dur), "dur_bin": b,
                        "order": [r[int(j)] for j in orng.permutation(len(r))]})
    own = [own[int(i)] for i in orng.permutation(len(own))]
    for k, o in enumerate(own):
        o["own_id"] = f"o{k:03d}"
    return {
        "pool": pool, "seed": seed, "n_refs": n_refs, "min_refs": min_refs, "speakers": speakers, "gender": gender,
        "refs": refs, "swaps": swaps, "own": own, "curve_ids": [s["swap_id"] for s in swaps[:n_curve]],
        "bins": list(BIN_NAMES), "available_per_bin": avail, "quota_per_bin": quota,
    }


# ------------------------------------------------------------------------------------------------ pure: matching


def voiced(feats: np.ndarray) -> np.ndarray:
    """Boolean mask ``[T]`` of voiced frames (periodicity > 0 and F0 > 0) of features ``[15, T]``."""
    return (feats[PERIODICITY_CHANNEL] > 0.0) & (feats[F0_CHANNEL] > 0.0)


def f0_stats(feats_list: Sequence[np.ndarray]) -> tuple[float, float]:
    """``(median ln F0, P90 - P10 of ln F0)`` over the voiced frames of all utterances ``[15, T]``; NaN if none."""
    v = [np.log(f[F0_CHANNEL][voiced(f)].astype(np.float64)) for f in feats_list]
    v = np.concatenate(v) if v else np.zeros(0)
    if len(v) == 0:
        return math.nan, math.nan
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return float(p50), float(p90 - p10)


def semitone_range(ln_range: float) -> float:
    """P10-P90 range in semitones of a range in ln Hz."""
    return float(12.0 * ln_range / math.log(2.0))


def median_f0_hz(feats: np.ndarray) -> float:
    """Median voiced F0 in Hz of features ``[15, T]`` (NaN if none): ``utt_f0`` of the speaker-relative-F0 batch."""
    f = feats[F0_CHANNEL][voiced(feats)]
    return float(np.median(f)) if len(f) else math.nan


def geometric_mean_hz(values: Sequence[float]) -> float:
    """Geometric mean of the finite positive values (NaN if none)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.log(v).mean())) if len(v) else math.nan


def reference_f0_hz(feats_list: Sequence[np.ndarray]) -> float:
    """``ref_f0``: geometric mean of the references' median voiced F0 (as ``f0_conditioning`` in the dataset)."""
    return geometric_mean_hz([median_f0_hz(f) for f in feats_list])


def match_f0(feats: np.ndarray, target_median: float, target_range: float, clip: tuple[float, float] = F0_RANGE_CLIP) -> np.ndarray:
    """Copy of ``feats [15, T]`` with voiced F0 mapped to the target's median and (clipped) range, see the module doc."""
    out = np.array(feats, dtype=np.float32, copy=True)
    med, rng = f0_stats([feats])
    if not (np.isfinite(med) and np.isfinite(target_median)):
        return out
    r = 1.0
    if np.isfinite(rng) and rng > 0 and np.isfinite(target_range):
        r = float(np.clip(target_range / rng, *clip))
    m = voiced(out)
    ln = np.log(out[F0_CHANNEL, m].astype(np.float64))
    new = np.exp(target_median + r * (ln - med))
    out[F0_CHANNEL, m] = np.clip(new, F0_MIN_HZ, F0_MAX_HZ).astype(np.float32)
    return out


def loudness_stat(feats_list: Sequence[np.ndarray]) -> float:
    """Median ln loudness (channel 13) over the voiced frames of all utterances; NaN if none."""
    v = [np.log(np.maximum(f[LOUDNESS_CHANNEL][voiced(f)].astype(np.float64), LOUDNESS_EPS)) for f in feats_list]
    v = np.concatenate(v) if v else np.zeros(0)
    return float(np.median(v)) if len(v) else math.nan


def match_loudness(feats: np.ndarray, target_median: float) -> np.ndarray:
    """Copy of ``feats`` with channel 13 scaled so that its voiced median ln loudness equals ``target_median``."""
    out = np.array(feats, dtype=np.float32, copy=True)
    own = loudness_stat([feats])
    if np.isfinite(own) and np.isfinite(target_median):
        out[LOUDNESS_CHANNEL] = out[LOUDNESS_CHANNEL] * np.float32(math.exp(target_median - own))
    return out


def ema_stats(feats_list: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of the 12 EMA channels over all frames of all utterances ``[15, T]``."""
    x = np.concatenate([f[:N_EMA].astype(np.float64) for f in feats_list], axis=1)
    return x.mean(1), x.std(1)


def match_ema(feats: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    """Copy of ``feats`` with the EMA channels standardized by the utterance's own statistics and re-scaled to the target's."""
    out = np.array(feats, dtype=np.float32, copy=True)
    mean, std = ema_stats([feats])
    out[:N_EMA] = ((out[:N_EMA] - mean[:, None]) / np.maximum(std[:, None], 1e-6) * target_std[:, None] + target_mean[:, None]).astype(np.float32)
    return out


@dataclass(frozen=True)
class Matching:
    """Statistics of the reference frames that drive the matching (any may be unused)."""

    f0_median: float
    f0_range: float
    loud_median: float
    ema_mean: np.ndarray
    ema_std: np.ndarray

    @classmethod
    def from_features(cls, feats_list: Sequence[np.ndarray]) -> "Matching":
        med, rng = f0_stats(feats_list)
        mean, std = ema_stats(feats_list)
        return cls(med, rng, loudness_stat(feats_list), mean, std)


def apply_matching(feats: np.ndarray, stats: Matching, f0: bool, loud: bool, ema: bool, f0_relative: bool = False) -> np.ndarray:
    """Content features with the requested matchings (order: F0, loudness, EMA; each reads the original frames).

    ``f0_relative`` (speaker-relative-F0 checkpoints): the F0 median is kept (the generator divides it out), only the
    semitone range is matched."""
    out = feats
    if f0:
        median = f0_stats([feats])[0] if f0_relative else stats.f0_median
        out = match_f0(out, median, stats.f0_range)
    if loud:
        out = match_loudness(out, stats.loud_median)
    if ema:
        out = match_ema(out, stats.ema_mean, stats.ema_std)
    return np.array(out, dtype=np.float32, copy=True)


# ------------------------------------------------------------------------------------------------ pure: variants / items


@dataclass(frozen=True)
class Variant:
    """One way to render a swap. ``who``: whose references give the code (``target``, ``source``, ``wrong``);
    ``k``: references used (capped by what the speaker has); matching statistics always come from the target's
    references; ``cond``: ``clean``, ``len<L>`` (L seconds of concatenated reference audio, one code) or ``noisy``."""

    name: str
    k: int
    who: str = "target"
    f0: bool = False
    loud: bool = False
    ema: bool = False
    cond: str = "clean"


def main_variants(k_main: int = 5) -> dict[str, Variant]:
    k = f"k{k_main}"
    vs = [
        Variant("base", 1), Variant("k5", 5), Variant("k10", 10),
        Variant(f"{k}_f0", k_main, f0=True),
        Variant(f"{k}_f0_loud", k_main, f0=True, loud=True),
        Variant(f"{k}_f0_loud_ema", k_main, f0=True, loud=True, ema=True),
    ]
    return {v.name: v for v in vs}


def control_variants(k_main: int = 5) -> dict[str, Variant]:
    return {
        "ctl_src": Variant("ctl_src", k_main, who="source", f0=True, loud=True),
        "ctl_wrong": Variant("ctl_wrong", k_main, who="wrong", f0=True, loud=True),
    }


def curve_variants(best: Variant, k_main: int = 5) -> dict[str, Variant]:
    out: dict[str, Variant] = {}
    for sec in CURVE_SECONDS:
        out[f"len{sec}_base"] = Variant(f"len{sec}_base", 1, cond=f"len{sec}")
        out[f"len{sec}_best"] = replace(best, name=f"len{sec}_best", k=1, cond=f"len{sec}")
    out["noisy_base"] = Variant("noisy_base", 1, cond="noisy")
    out["noisy_best"] = replace(best, name="noisy_best", cond="noisy")
    return out


def own_variants(k_main: int = 5) -> dict[str, Variant]:
    return {"own_t2": Variant("own_t2", 1), "own_t3": Variant("own_t3", k_main)}


def all_variants(k_main: int = 5, best_name: str | None = None) -> dict[str, Variant]:
    """Main, control, curve and own variants by name (the best variant defaults to ``k<K>_f0_loud``)."""
    main = main_variants(k_main)
    best = main[best_name or f"k{k_main}_f0_loud"]
    return {**main, **control_variants(k_main), **curve_variants(best, k_main), **own_variants(k_main)}


@dataclass
class Item:
    """One output wav and the manifest fields that describe it."""

    key: str
    kind: str  # swap | own
    pool: str
    pair_id: str
    variant: str
    content_id: str
    source: str
    target: str
    code_speaker: str
    code_ids: list[str]
    match_ids: list[str]
    cond: str
    f0: bool
    loud: bool
    ema: bool
    same_gender: bool
    dur_bin: int
    dur_s: float
    meta: dict[str, Any] = field(default_factory=dict)


def build_items(plan: dict[str, Any], variants: dict[str, Variant], names: Sequence[str]) -> list[Item]:
    """All outputs of a pool for the variants ``names``; curve variants only on the plan's flagged pairs, ``own_*`` on
    own items, the rest on swaps. Wiring: the code comes from ``who``'s references (``<who>_order[:k]``), the matching
    statistics from the target's (``target_order[:k]``)."""
    pool = plan["pool"]
    curve = set(plan["curve_ids"])
    items: list[Item] = []
    for name in names:
        v = variants[name]
        if name.startswith("own_"):
            for o in plan["own"]:
                ids = o["order"][: v.k]
                items.append(Item(f"{pool}/{o['own_id']}/{name}", "own", pool, o["own_id"], name, o["content_id"], o["speaker"], o["speaker"],
                                  o["speaker"], ids, ids, "clean", False, False, False, True, o["dur_bin"], o["dur_s"]))
            continue
        is_curve = v.cond != "clean"
        for s in plan["swaps"]:
            if is_curve and s["swap_id"] not in curve:
                continue
            k = len(s[f"{v.who}_order"]) if v.cond.startswith("len") else v.k  # a length clip concatenates the references
            code_ids = s[f"{v.who}_order"][:k]
            match_ids = s["target_order"][:k]
            items.append(
                Item(f"{pool}/{s['swap_id']}/{name}", "swap", pool, s["swap_id"], name, s["content_id"], s["source"], s["target"],
                     s[v.who], code_ids, match_ids, v.cond, v.f0, v.loud, v.ema, s["same_gender"], s["dur_bin"], s["dur_s"],
                     {"wrong": s["wrong"]})
            )
    return items


# ------------------------------------------------------------------------------------------------ pure: scoring maths


def eer_threshold(same: Sequence[float], diff: Sequence[float]) -> tuple[float, float]:
    """``(threshold, eer)`` where false-accept (diff >= t) and false-reject (same < t) rates are closest to equal."""
    same, diff = np.asarray(same, dtype=np.float64), np.asarray(diff, dtype=np.float64)
    same, diff = same[np.isfinite(same)], diff[np.isfinite(diff)]
    if len(same) == 0 or len(diff) == 0:
        return math.nan, math.nan
    cands = np.unique(np.concatenate([same, diff]))
    far = 1.0 - np.searchsorted(np.sort(diff), cands, side="left") / len(diff)
    frr = np.searchsorted(np.sort(same), cands, side="left") / len(same)
    i = int(np.argmin(np.abs(far - frr)))
    return float(cands[i]), float(0.5 * (far[i] + frr[i]))


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def mean_unit(emb: np.ndarray) -> np.ndarray:
    """Mean of the L2-normalized rows (NaN rows ignored)."""
    e = _unit(emb)
    e = e[np.isfinite(e).all(1)]
    return e.mean(0)


def top1_ids(emb: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Index of the highest-cosine row of ``means [S, d]`` for each row of ``emb [n, d]`` (-1 for non-finite rows)."""
    e = np.asarray(emb, dtype=np.float64)
    cos = _unit(e) @ _unit(means).T
    ok = np.isfinite(e).all(1)
    out = np.full(len(e), -1, dtype=np.int64)
    out[ok] = cos[ok].argmax(1)
    return out


def calibration_stats(ref_emb: dict[str, np.ndarray]) -> dict[str, Any]:
    """Ground-truth calibration of one SV model on a pool from ``{speaker: [n_i, d]}`` reference embeddings.

    Trials: each reference vs the mean embedding of its speaker's other references (same) and vs the full reference mean
    of every other speaker (different). Returns ``eer_threshold``, ``eer``, ``same_mean``, ``diff_mean``, ``top1`` (ID of
    each reference among all speakers, own speaker with leave-one-out mean) and the trial counts.
    """
    spk = [s for s in sorted(ref_emb) if len(ref_emb[s]) >= 2]
    if not spk:  # one reference per speaker: no same-speaker trial, so no threshold and no calibration
        nan = math.nan
        return {"eer_threshold": nan, "eer": nan, "same_mean": nan, "diff_mean": nan, "top1": nan, "n_same": 0, "n_diff": 0}
    full = {s: mean_unit(ref_emb[s]) for s in spk}
    same, diff, hit = [], [], []
    names = list(spk)
    mat = np.stack([full[s] for s in names])
    for i, s in enumerate(names):
        e = _unit(ref_emb[s])
        for j in range(len(e)):
            loo = mean_unit(np.delete(e, j, axis=0))
            c = _unit(e[j]) @ _unit(mat).T
            c[i] = float(_unit(e[j]) @ _unit(loo))
            same.append(c[i])
            diff.extend(np.delete(c, i).tolist())
            hit.append(int(np.argmax(c) == i))
    thr, eer = eer_threshold(same, diff)
    return {"eer_threshold": thr, "eer": eer, "same_mean": float(np.mean(same)), "diff_mean": float(np.mean(diff)),
            "top1": float(np.mean(hit)), "n_same": len(same), "n_diff": len(diff)}


def holdout_refs(order: Sequence[str], used: Sequence[str]) -> list[str]:
    """References of a speaker not used by the item (possibly none: no fallback to the used ones, which would inflate
    the similarity): the target of the own-speaker ``sim_spk`` and of ``sim_target_heldout``."""
    gone = set(used)
    return [u for u in order if u not in gone]


def plan_hash(plan: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a plan: arms are only comparable on identical plans."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def assert_same_plan(a: pd.DataFrame, b: pd.DataFrame, what: str = "") -> None:
    """Raises if two single-variant output tables do not describe the same items (pair ids, content, source, target)."""
    if set(a["pair_id"]) != set(b["pair_id"]):
        raise ValueError(f"{what}: the two outputs cover different pair ids; they were not rendered from the same plan")
    cols = [c for c in ("content_id", "source", "target") if c in a and c in b]
    m = a[["pair_id", *cols]].drop_duplicates().merge(b[["pair_id", *cols]].drop_duplicates(), on="pair_id", suffixes=("_a", "_b"))
    for c in cols:
        if (m[f"{c}_a"] != m[f"{c}_b"]).any():
            raise ValueError(f"{what}: {c} differs between the two outputs for the same pair id (different plans?)")


# ------------------------------------------------------------------------------------------------ pure: aggregation


def speaker_means(df: pd.DataFrame, col: str, by: str = "target") -> pd.Series:
    """Mean of ``col`` per speaker (finite values only); speakers without any are dropped."""
    v = df[[by, col]].copy()
    v[col] = pd.to_numeric(v[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return v.dropna().groupby(by)[col].mean()


def speaker_ci(df: pd.DataFrame, col: str, by: str = "target", n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """Mean over speakers of the per-speaker means of ``col`` with a speaker-level bootstrap 95% interval."""
    return bootstrap_ci(speaker_means(df, col, by).to_numpy(), n_boot, seed)


def speaker_wer(df: pd.DataFrame, by: str = "target", n_boot: int = 10000, seed: int = 0) -> dict[str, dict[str, float]]:
    """Per-speaker WER (errors / words) averaged over speakers, and the corpus-level rate; both with bootstrap intervals."""
    g = df.groupby(by)[["wer_err", "wer_len"]].sum()
    g = g[g["wer_len"] > 0]
    rates = (g["wer_err"] / g["wer_len"]).to_numpy()
    return {"per_speaker": bootstrap_ci(rates, n_boot, seed), "corpus": rate_ci(df["wer_err"], df["wer_len"], n_boot, seed)}


def speaker_diff_ci(a: pd.DataFrame, b: pd.DataFrame, col: str, on: str = "pair_id", by: str = "target", n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """Paired difference ``a - b`` of ``col`` joined on ``on`` (items), averaged per speaker, bootstrapped over speakers."""
    m = a[[on, by, col]].merge(b[[on, col]], on=on, suffixes=("_a", "_b"))
    m["d"] = m[f"{col}_a"] - m[f"{col}_b"]
    return speaker_ci(m, "d", by, n_boot, seed)


def ratio_ci(df: pd.DataFrame, num: str, den: str, by: str = "target", n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """Mean over speakers of ``mean(num) / mean(den)`` per speaker with a speaker-level bootstrap interval."""
    g = df.groupby(by)[[num, den]].mean()
    return bootstrap_ci((g[num] / g[den]).to_numpy(), n_boot, seed)


def strata(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``all``, ``same_gender`` / ``cross_gender`` and ``bin:<name>`` subsets of a swap table (empty ones omitted)."""
    out = {"all": df, "same_gender": df[df["same_gender"]], "cross_gender": df[~df["same_gender"]]}
    for i, name in enumerate(BIN_NAMES):
        out[f"bin:{name}"] = df[df["dur_bin"] == i]
    return {k: v for k, v in out.items() if len(v)}


def metric_columns(models: Sequence[str]) -> list[str]:
    return [f"{m}/{k}" for m in models for k in ("sim_target", "sim_target_heldout", "sim_source", "top1", "accept", "gt_same", "gt_diff")] + ["utmos"]


def summarize_variant(df: pd.DataFrame, models: Sequence[str], n_boot: int, seed: int) -> dict[str, Any]:
    """Per stratum: ``n`` items / speakers, a speaker-aggregated CI for every metric column, WER and the success ratios."""
    res: dict[str, Any] = {}
    for name, sub in strata(df).items():
        entry: dict[str, Any] = {"n_items": int(len(sub)), "n_speakers": int(sub["target"].nunique()), "metrics": {}}
        for col in metric_columns(models):
            if col in sub:
                entry["metrics"][col] = speaker_ci(sub, col, "target", n_boot, seed)
        entry["wer"] = speaker_wer(sub, "target", n_boot, seed)
        entry["success"] = {
            m: {"target_over_gt_same": ratio_ci(sub, f"{m}/sim_target", f"{m}/gt_same", "target", n_boot, seed),
                "source_minus_gt_diff": speaker_ci(sub.assign(d=sub[f"{m}/sim_source"] - sub[f"{m}/gt_diff"]), "d", "target", n_boot, seed)}
            for m in models if f"{m}/gt_same" in sub
        }
        res[name] = entry
    return res


def summarize_margins(items: pd.DataFrame, variant: str, controls: Sequence[str], models: Sequence[str], n_boot: int, seed: int) -> dict[str, Any]:
    """Paired margin of ``variant`` over each control in ``sim_target`` (per model), speaker-level bootstrap."""
    a = items[items["variant"] == variant]
    out: dict[str, Any] = {}
    for c in controls:
        b = items[items["variant"] == c]
        if len(a) and len(b):
            out[c] = {m: speaker_diff_ci(a, b, f"{m}/sim_target", "pair_id", "target", n_boot, seed) for m in models}
    return out


def summarize_own(df: pd.DataFrame, models: Sequence[str], n_boot: int, seed: int) -> dict[str, Any]:
    """Guard table of the own-speaker items: ``sim_same`` and ``sim_spk`` overall and for content > 4 s."""
    out: dict[str, Any] = {}
    for name, sub in {"all": df, "gt4s": df[df["dur_s"] > 4.0]}.items():
        if not len(sub):
            continue
        out[name] = {f"{m}/{k}": speaker_ci(sub.assign(s=sub[f"{m}/{k}"]), "s", "target", n_boot, seed)
                     for m in models for k in ("sim_same", "sim_spk") if f"{m}/{k}" in sub}
        out[name]["utmos"] = speaker_ci(sub, "utmos", "target", n_boot, seed)
        out[name]["wer"] = speaker_wer(sub, "target", n_boot, seed)
    return out


def item_table(
    manifest: pd.DataFrame,
    emb: dict[str, np.ndarray],
    utmos: np.ndarray,
    wer_counts: Sequence[tuple[int, int]],
    ref_means: dict[str, dict[str, dict[str, np.ndarray]]],
    ref_emb_by_utt: dict[str, dict[str, dict[str, np.ndarray]]],
    gt_content: dict[str, dict[str, dict[str, np.ndarray]]],
    thresholds: dict[str, dict[str, float]],
    plans: dict[str, dict[str, Any]],
    rank_speakers: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Per-output metric table (one row per manifest row).

    ``rank_speakers[pool]``, when given, restricts the speakers top-1 identification ranks over (a split's speakers); the
    similarities themselves still use every speaker's reference mean (a source outside the split keeps its ``sim_source``).
    ``top1_source`` is NaN when the source is not among the ranked speakers.

    ``ref_means[pool][model][speaker]`` is the reference-mean embedding, ``ref_emb_by_utt[pool][model][utt]`` the embedding
    of each reference, ``gt_content[pool][model][utt]`` the ground-truth embedding of each content utterance,
    ``thresholds[pool][model]`` the EER threshold.
    """
    manifest = manifest.reset_index(drop=True)
    rows = manifest.copy()
    rows["wer_err"] = [c[0] for c in wer_counts]
    rows["wer_len"] = [c[1] for c in wer_counts]
    rows["utmos"] = np.asarray(utmos, dtype=np.float64)
    pool_col, tgt_col, src_col = rows["pool"].tolist(), rows["target"].tolist(), rows["source"].tolist()
    cid_col, kind_col, ids_col = rows["content_id"].tolist(), rows["kind"].tolist(), rows["code_ids"].tolist()
    mids_col = rows["match_ids"].tolist() if "match_ids" in rows else ids_col
    for m, e in emb.items():
        cols = {k: np.full(len(rows), np.nan) for k in ("sim_target", "sim_target_heldout", "sim_source", "top1", "top1_source", "accept", "gt_same", "gt_diff", "sim_same", "sim_spk")}
        en = _unit(e)
        for pool, idx in rows.groupby("pool").groups.items():
            idx = np.asarray(list(idx))
            spk = plans[pool]["speakers"]
            pos = {s: i for i, s in enumerate(spk)}
            means = np.stack([ref_means[pool][m][s] for s in spk])
            rank = np.arange(len(spk)) if rank_speakers is None else np.array([i for i, s in enumerate(spk) if s in rank_speakers[pool]], dtype=np.int64)
            top = top1_ids(e[idx], means[rank])
            top = np.where(top >= 0, rank[np.maximum(top, 0)], -1)
            ranked = set(rank.tolist())
            cos = en[idx] @ _unit(means).T
            for j, (r, t) in enumerate(zip(idx, top)):
                tgt, src = tgt_col[r], src_col[r]
                cols["top1"][r] = float(t == pos[tgt]) if t >= 0 else np.nan
                cols["top1_source"][r] = float(t == pos[src]) if t >= 0 and pos[src] in ranked else np.nan
                cols["sim_target"][r], cols["sim_source"][r] = cos[j, pos[tgt]], cos[j, pos[src]]
                if kind_col[r] == "swap":  # references that neither gave the code nor the matching statistics
                    rest = [u for u in holdout_refs(plans[pool]["refs"][tgt], [*ids_col[r], *mids_col[r]]) if u in ref_emb_by_utt[pool][m]]
                    if rest:
                        cols["sim_target_heldout"][r] = cosine_rows(e[r][None], mean_unit(np.stack([ref_emb_by_utt[pool][m][u] for u in rest])))[0]
                g = gt_content[pool][m].get(cid_col[r])
                if g is not None:
                    cols["gt_same"][r] = cosine_rows(g[None], ref_means[pool][m][src])[0]
                    cols["gt_diff"][r] = cosine_rows(g[None], ref_means[pool][m][tgt])[0]
                if kind_col[r] == "own":
                    hold = [u for u in holdout_refs(plans[pool]["refs"][tgt], ids_col[r]) if u in ref_emb_by_utt[pool][m]]
                    if hold:  # no held-out reference left (few references per speaker): NaN, never the code's own references
                        cols["sim_spk"][r] = cosine_rows(e[r][None], mean_unit(np.stack([ref_emb_by_utt[pool][m][u] for u in hold])))[0]
                    if g is not None:
                        cols["sim_same"][r] = cosine_rows(e[r][None], g)[0]
        thr = np.array([thresholds[p][m] for p in rows["pool"]])
        # A pool whose speakers all have a single reference has no same-speaker trial, hence no EER threshold (NaN):
        # acceptance is then not available (NaN), not 0 (``x >= nan`` is False).
        cols["accept"] = np.where(np.isfinite(cols["sim_target"]) & np.isfinite(thr), (cols["sim_target"] >= thr).astype(float), np.nan)
        for k, v in cols.items():
            rows[f"{m}/{k}"] = v
    return rows


def summarize_items(
    items: pd.DataFrame, calibration: dict[str, Any], models: Sequence[str], controls: Sequence[str] = ("ctl_src", "ctl_wrong"),
    n_boot: int = 10000, seed: int = 0,
) -> dict[str, Any]:
    """Full per-pool summary from the per-output table (see the module doc for the layout)."""
    out: dict[str, Any] = {}
    for pool, pdf in items.groupby("pool"):
        res: dict[str, Any] = {"calibration": calibration.get(pool, {}), "variants": {}, "margins": {}}
        swaps = pdf[pdf["kind"] == "swap"]
        for v, g in swaps.groupby("variant"):
            res["variants"][v] = summarize_variant(g, models, n_boot, seed)
            res["margins"][v] = summarize_margins(swaps, v, [c for c in controls if c != v], models, n_boot, seed)
        own = pdf[pdf["kind"] == "own"]
        res["own"] = {v: summarize_own(g, models, n_boot, seed) for v, g in own.groupby("variant")}
        out[str(pool)] = res
    return out


# ------------------------------------------------------------------------------------------------ pure: decision rule


def decision_rule(
    a: pd.DataFrame,
    b: pd.DataFrame,
    models: Sequence[str],
    margin: float,
    primary_pool: str = "libritts",
    primary_model: str = PRIMARY_MODEL,
    exclude_models: Sequence[str] = (),
    variant_a: str | None = None,
    variant_b: str | None = None,
    guards: dict[str, float] | None = None,
    n_boot: int = 10000,
    seed: int = 0,
    required_models: Sequence[str] = REQUIRED_MODELS,
    allow_missing_holdout: bool = False,
    sign_exempt: dict[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Pre-registered advancing rule for arm ``a`` against baseline ``b`` (per-output tables of ``item_table``).

    Fields: ``gain[pool][model]`` (speaker-level CI of the paired ``sim_target`` difference of the swap variants),
    ``gain_heldout`` (the same on ``sim_target_heldout``, reported for the K-reference bias check), ``primary_ok``
    (gain >= margin with a CI above 0 on the primary pool / model), ``same_sign`` (gain > 0 for every pool and every
    decision model, except ``sign_exempt`` models in their pool, default ECAPA on GLOBE), per-pool ``guards`` (WER
    points, UTMOS, source similarity of every decision model, own-speaker T2 / T3 ``sim_spk`` overall and for > 4 s per
    decision model) with ``guards_ok``, and ``advance``. ``exclude_models`` drops SV models from the decision (an arm
    trained against one of them is judged on the others). ``required_models`` (the held-out SV model) must be among
    the decision models: otherwise this raises, unless ``allow_missing_holdout`` (then ``holdout_missing`` is set and
    ``advance`` stays False). The two tables must come from the same plan (checked per pool and variant).
    """
    g = {**GUARDS, **(guards or {})}
    exempt = SIGN_EXEMPT if sign_exempt is None else sign_exempt
    decision_models = [m for m in models if m not in exclude_models]
    missing = [m for m in required_models if m not in decision_models]
    if missing and not allow_missing_holdout:
        raise ValueError(f"decision models {decision_models} lack the required {missing}; score with --holdout-sv / --require-holdout, or pass allow_missing_holdout")
    if missing:
        logger.warning("DECISION WITHOUT REQUIRED MODEL(S) %s: the pre-registered held-out sign check is NOT applied; advance is forced False", missing)
    out: dict[str, Any] = {"margin": margin, "decision_models": decision_models, "holdout_missing": missing, "gain": {}, "gain_heldout": {}, "guards": {}}
    pools = sorted(set(a["pool"]) & set(b["pool"]))
    same_sign = True
    guards_ok = True
    for pool in pools:
        sa = a[(a["pool"] == pool) & (a["kind"] == "swap")]
        sb = b[(b["pool"] == pool) & (b["kind"] == "swap")]
        va, vb = variant_a, variant_b or variant_a
        sa, sb = (sa[sa["variant"] == va] if va else sa), (sb[sb["variant"] == vb] if vb else sb)
        if sa["variant"].nunique() > 1 or sb["variant"].nunique() > 1:
            raise ValueError("several variants present: pass variant_a / variant_b")
        assert_same_plan(sa, sb, f"{pool} swaps")
        out["gain"][pool] = {m: speaker_diff_ci(sa, sb, f"{m}/sim_target", "pair_id", "target", n_boot, seed) for m in models}
        out["gain_heldout"][pool] = {m: speaker_diff_ci(sa, sb, f"{m}/sim_target_heldout", "pair_id", "target", n_boot, seed)
                                     for m in models if f"{m}/sim_target_heldout" in sa and f"{m}/sim_target_heldout" in sb}
        for m in decision_models:
            if m not in exempt.get(pool, ()):
                same_sign &= bool(out["gain"][pool][m]["mean"] > 0)
        cols = ["pair_id", "wer_err", "wer_len", "utmos"] + [f"{m}/sim_source" for m in models]
        j = sa[cols + ["target"]].merge(sb[cols], on="pair_id", suffixes=("_a", "_b"))
        wer_a = 100.0 * j["wer_err_a"].sum() / max(j["wer_len_a"].sum(), 1)
        wer_b = 100.0 * j["wer_err_b"].sum() / max(j["wer_len_b"].sum(), 1)
        pg: dict[str, Any] = {"wer_points": float(wer_a - wer_b), "utmos": float((j["utmos_a"] - j["utmos_b"]).mean()),
                              "source": {m: float((j[f"{m}/sim_source_a"] - j[f"{m}/sim_source_b"]).mean()) for m in models}}
        oa, ob = a[(a["pool"] == pool) & (a["kind"] == "own")], b[(b["pool"] == pool) & (b["kind"] == "own")]
        t23: dict[str, float] = {}
        for var in sorted(set(oa["variant"]) & set(ob["variant"])):
            assert_same_plan(oa[oa["variant"] == var], ob[ob["variant"] == var], f"{pool} {var}")
            for nm, flt in (("all", lambda d: d), ("gt4s", lambda d: d[d["dur_s"] > 4.0])):
                x, y = flt(oa[oa["variant"] == var]), flt(ob[ob["variant"] == var])
                for m in models:
                    if f"{m}/sim_spk" in x and f"{m}/sim_spk" in y:
                        t23[f"{var}/{nm}/{m}"] = float(speaker_diff_ci(x, y, f"{m}/sim_spk", "pair_id", "target", n_boot, seed)["mean"])
        pg["t23_sim_spk"] = t23
        pg["pass"] = {
            "wer": pg["wer_points"] <= g["wer_points"], "utmos": pg["utmos"] >= g["utmos"],
            "source": all(v <= g["source"] for m, v in pg["source"].items() if m in decision_models),
            "t23": all((not np.isfinite(v)) or v >= g["t23"] for k, v in t23.items() if k.rsplit("/", 1)[1] in decision_models),
        }
        guards_ok &= all(pg["pass"].values())
        out["guards"][pool] = pg
    prim = out["gain"].get(primary_pool, {}).get(primary_model, {})
    out["primary_ok"] = bool(prim and prim["mean"] >= margin and prim["lo"] > 0)
    out["same_sign"] = bool(same_sign and pools)
    out["guards_ok"] = bool(guards_ok and pools)
    out["advance"] = bool(out["primary_ok"] and out["same_sign"] and out["guards_ok"] and not missing)
    return out


def render_decision(d: dict[str, Any], label_a: str = "arm", label_b: str = "baseline") -> str:
    """Markdown of a :func:`decision_rule` result."""
    f = lambda c: "n/a" if not np.isfinite(c.get("mean", math.nan)) else f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"  # noqa: E731
    models = list(next(iter(d["gain"].values())).keys()) if d["gain"] else []
    lines = [f"# {label_a} vs {label_b}", "", f"Margin {d['margin']:.4f}; decision SV models: {', '.join(d['decision_models'])}.", "",
             "Gain in sim_target (arm - baseline), speaker-level 95% CI:", "",
             "| pool | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for pool, gains in d["gain"].items():
        lines.append(f"| {pool} | " + " | ".join(f(c) for c in gains.values()) + " |")
    lines += ["", "| pool | WER points | UTMOS | source sim | T2/T3 sim_spk | pass |", "|---|---|---|---|---|---|"]
    for pool, g in d["guards"].items():
        src = ", ".join(f"{m} {v:+.3f}" for m, v in g["source"].items())
        t23 = ", ".join(f"{k} {v:+.3f}" for k, v in g["t23_sim_spk"].items())
        lines.append(f"| {pool} | {g['wer_points']:+.2f} | {g['utmos']:+.3f} | {src} | {t23} | {all(g['pass'].values())} |")
    if d.get("holdout_missing"):
        lines += ["", f"**WARNING: required model(s) {', '.join(d['holdout_missing'])} missing, the held-out sign check was not applied; advance is forced False.**"]
    if "globe" in d["gain"]:
        lines += ["", "GLOBE speakers are ECAPA-clustered pseudo-speakers: ECAPA is optimistic there (not in the sign check) and speaker-level intervals are anti-conservative."]
    lines += ["", f"primary_ok={d['primary_ok']}, same_sign={d['same_sign']}, guards_ok={d['guards_ok']}, **advance={d['advance']}**", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ plans


def plan_path(root: str | Path, pool: str) -> Path:
    return Path(root) / "plans" / f"{pool}.json"


def run_plan(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    from sparc.vocoders.eval.clone_pools import POOL_SPECS, Pool
    from sparc.vocoders.eval.io import write_json

    plans = {}
    for name in [p for p in args.pools.split(",") if p]:
        pool = Pool(name, args.root, args.cache_root, sex_file=args.sex_file)
        plan = build_plan(pool.table, name, args.seed, args.n_swap, args.n_own, args.n_curve, args.n_refs, POOL_SPECS[name].min_refs)
        write_json(plan_path(args.root, name), plan)
        counts = {k: int(v) for k, v in zip(BIN_NAMES, plan["quota_per_bin"])}
        logger.info("%s: %d speakers, %d swaps (%s), %d own", name, len(plan["speakers"]), len(plan["swaps"]), counts, len(plan["own"]))
        plans[name] = plan
    return plans


# ------------------------------------------------------------------------------------------------ synthesis


class RefConditions:
    """Reference vectors and matching statistics for clean, truncated-length and noisy references.

    Encodings are cached in memory and, with ``cache_dir``, on disk (``<cache_dir>/<cond>/<hash>.npz``, written
    atomically) so that shards and arms do not repeat the WavLM / CREPE / inverter passes: the result depends only on the
    audio, the condition, the layer and the gain."""

    def __init__(self, pool, store, layer: str, gain_db: float, device, cache_dir: str | Path | None = None,
                 inverter: str | None = None):
        self.pool, self.store, self.layer, self.gain_db, self.device = pool, store, layer, gain_db, device
        self.inverter = inverter  # Phase 9: k12 inverter supplying the articulatory channels of reference encodings
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._enc = None
        self._cache: dict[Any, Any] = {}

    def encoder(self):
        if self._enc is None:
            from sparc.vocoders.eval.clone_pools import FeatureEncoder

            self._enc = FeatureEncoder(self.device, inverter=self.inverter)
        return self._enc

    def clean_features(self, ids: Sequence[str]) -> list[np.ndarray]:
        from sparc.vocoders.eval.interp import utterance_features

        return [utterance_features(self.store, u, self.gain_db) for u in ids]

    def _audio(self, uid: str) -> np.ndarray:
        u = self.store.index_of(uid)
        return self.store.read_audio(u, 0, int(self.store.n24[u]))

    def _gained(self, enc: dict[str, np.ndarray]) -> np.ndarray:
        from sparc.vocoders.data.gain import apply_gain, gain_factor

        feats = np.ascontiguousarray(enc["feats"].T).astype(np.float32)
        feats[LOUDNESS_CHANNEL] = apply_gain(enc["loud_raw"], gain_factor(float(enc["peak24"]), self.gain_db))
        return feats

    def _disk_path(self, cond: str, ids: Sequence[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        tag = f"|inv={self.inverter}" if self.inverter else ""  # unchanged key without an inverter
        h = hashlib.sha256(f"{cond}|{self.layer}|{self.gain_db}|{'|'.join(ids)}{tag}".encode()).hexdigest()[:32]
        return self.cache_dir / cond / f"{h}.npz"

    def _encode(self, cond: str, ids: Sequence[str]) -> list[dict[str, np.ndarray]]:
        from sparc.vocoders.eval.clone_pools import utterance_seed

        enc = self.encoder()
        if cond.startswith("len"):
            seconds = float(cond[3:])
            cat = np.concatenate([self._audio(u) for u in ids])
            clip = cat[: int(seconds * SAMPLE_RATE)]
            if not np.any(clip):  # GLOBE clips can open with > 1 s of digital silence: start at the first non-zero sample instead
                nz = np.flatnonzero(cat)
                clip = cat[int(nz[0]) :][: int(seconds * SAMPLE_RATE)] if len(nz) else clip
            return [enc.encode(clip, utterance_seed(f"{ids[0]}|{cond}"))]
        if cond == "noisy":
            encs = []
            for u in ids:
                x = self._audio(u)
                rng = np.random.default_rng([zlib.crc32(u.encode()), 7])
                noise = rng.standard_normal(len(x)).astype(np.float32) * math.sqrt(float(np.mean(x.astype(np.float64) ** 2)) / 10 ** (NOISE_SNR_DB / 10))
                encs.append(enc.encode(x + noise, utterance_seed(f"{u}|noisy")))
            return encs
        raise ValueError(f"unknown reference condition {cond!r}")

    def encode_conditioned(self, cond: str, ids: Sequence[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """``(speaker vectors, feature arrays [15, T])`` of the references under ``cond`` (``len<L>`` gives one clip)."""
        key = (cond, tuple(ids))
        if key in self._cache:
            return self._cache[key]
        path = self._disk_path(cond, ids)
        out = None
        if path is not None and path.is_file():
            try:
                with np.load(path) as z:
                    n = int(z["n"])
                    out = ([z[f"vec{i}"] for i in range(n)], [z[f"feat{i}"] for i in range(n)])
            except (OSError, ValueError, KeyError):  # cut short by a kill: recompute
                out = None
        if out is None:
            from sparc.vocoders.eval.clone_pools import speaker_vector

            encs = self._encode(cond, ids)
            out = ([speaker_vector(e, self.layer) for e in encs], [self._gained(e) for e in encs])
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
                np.savez(tmp, n=len(out[0]), **{f"vec{i}": v for i, v in enumerate(out[0])}, **{f"feat{i}": f for i, f in enumerate(out[1])})
                os.replace(tmp, path)
        self._cache[key] = out
        return out


def reference_f0_std(feats_list: Sequence[np.ndarray]) -> float:
    """``ref_f0_std`` of the conditioning-statistics checkpoints: mean over the references of the std of ln F0 over the
    voiced frames (as ``CropDataset(cond_stats=True)``); NaN if no reference has 2 voiced frames."""
    stds = []
    for f in feats_list:
        v = f[F0_CHANNEL][voiced(f)]
        if len(v) > 1:
            stds.append(float(np.log(v.astype(np.float64)).std()))
    return float(np.mean(stds)) if stds else math.nan


def utterance_log_loudness(feats: np.ndarray) -> float:
    """``utt_loud``: mean ``ln(loudness + eps)`` over the voiced frames of features ``[15, T]`` (NaN if none)."""
    m = voiced(feats)
    if not m.any():
        return math.nan
    return float(np.log(np.clip(feats[LOUDNESS_CHANNEL][m].astype(np.float64), 0.0, None) + LOUDNESS_EPS).mean())


def code_from_vectors(module, vectors: Sequence[np.ndarray], device, ref_f0: float | None = None) -> np.ndarray:
    """Speaker code ``[64]`` of reference vectors through the module's own multi-reference ``embed``; with a
    speaker-relative-F0 module (``module.pitch_code``) the reference pitch level ``ref_f0`` (Hz) is added, as in
    ``module.speaker_code``."""
    import torch

    raw = torch.from_numpy(np.stack(vectors).astype(np.float32))[None].to(device)
    with torch.inference_mode():
        z, _ = module.speaker.embed(raw, torch.ones(1, len(vectors), dtype=torch.bool, device=device))
        pitch = getattr(module, "pitch_code", None)
        if pitch is not None:
            z = z + pitch(torch.tensor([math.nan if ref_f0 is None else ref_f0], dtype=torch.float32, device=device))
    return z[0].float().cpu().numpy()


class ItemRenderer:
    """Renders manifest items with a loaded module (caches content features, codes and matching statistics).

    ``features_fn(store, utt_id, gain_db)`` gives content features ``[15, T]`` (default
    :func:`sparc.vocoders.eval.interp.utterance_features`)."""

    def __init__(self, module, stores: dict[str, Any], conds: dict[str, Any], gain_db: float, device, features_fn=None):
        self.module, self.stores, self.conds, self.gain_db, self.device = module, stores, conds, gain_db, device
        self.rel_f0 = bool(getattr(module, "rel_f0", False))
        if features_fn is None:
            from sparc.vocoders.eval.interp import utterance_features as features_fn
        self.features_fn = features_fn
        self.content_cache: dict[tuple[str, str], np.ndarray] = {}
        self.code_cache: dict[Any, np.ndarray] = {}
        self.stats_cache: dict[Any, Matching] = {}

    def code(self, it: Item) -> tuple[np.ndarray, list[np.ndarray] | None]:
        """``(z, features of the code's references or None for clean)``; ``ref_f0`` is the code references' geometric-mean median F0."""
        store, rc = self.stores[it.pool], self.conds[it.pool]
        if it.cond == "clean":
            kc = (it.pool, tuple(it.code_ids))
            if kc not in self.code_cache:
                ref_f0 = reference_f0_hz(rc.clean_features(it.code_ids)) if self.rel_f0 else None
                vectors = [store.speaker_vector(store.index_of(u)) for u in it.code_ids]
                self.code_cache[kc] = code_from_vectors(self.module, vectors, self.device, ref_f0)
            return self.code_cache[kc], None
        vecs, feats = rc.encode_conditioned(it.cond, it.code_ids)
        return code_from_vectors(self.module, vecs, self.device, reference_f0_hz(feats) if self.rel_f0 else None), feats

    def render(self, it: Item) -> np.ndarray:
        """Waveform ``[n]`` of one item."""
        import torch

        from sparc.vocoders.training.callbacks import EVAL_SEED, fixed_torch_rng

        store, rc = self.stores[it.pool], self.conds[it.pool]
        ck = (it.pool, it.content_id)
        if ck not in self.content_cache:
            if len(self.content_cache) > 64:
                self.content_cache.clear()
            self.content_cache[ck] = self.features_fn(store, it.content_id, self.gain_db)
        feats = self.content_cache[ck]
        z, match_feats = self.code(it)
        if it.f0 or it.loud or it.ema:
            sk = (it.pool, it.cond, tuple(it.match_ids))
            if sk not in self.stats_cache:
                self.stats_cache[sk] = Matching.from_features(match_feats if match_feats is not None else rc.clean_features(it.match_ids))
            feats = apply_matching(feats, self.stats_cache[sk], it.f0, it.loud, it.ema, f0_relative=self.rel_f0)
        batch = {"features": torch.from_numpy(np.ascontiguousarray(feats))[None].to(self.device)}
        if getattr(self.module, "stats_code", None) is not None:  # Phase 8: utterance-level statistics in the code
            ref_feats = match_feats if match_feats is not None else rc.clean_features(it.code_ids)
            utt_loud = utterance_log_loudness(feats)
            batch["utt_loud"] = torch.tensor([utt_loud], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                z = z + self.module.stats_term(reference_f0_std(ref_feats), utt_loud).float().cpu().numpy()
        if self.rel_f0:  # synthesize divides the F0 channel by the source utterance's median voiced F0
            batch["utt_f0"] = torch.tensor([median_f0_hz(feats)], dtype=torch.float32, device=self.device)
        with torch.inference_mode(), fixed_torch_rng(self.device, EVAL_SEED):
            wav = self.module.synthesize(batch, z=torch.from_numpy(z)[None].to(self.device))
        return wav[0, 0].float().cpu().numpy()


def missing_plan_ids(plan: dict[str, Any], store) -> list[str]:
    """Utterance ids of a plan (references and contents) that the packed store does not hold."""
    ids = {u for r in plan["refs"].values() for u in r} | {s["content_id"] for s in plan["swaps"]} | {o["content_id"] for o in plan["own"]}
    bad = []
    for u in sorted(ids):
        try:
            store.index_of(u)
        except (KeyError, ValueError):
            bad.append(u)
    return bad


def resolve_best_variant(spec: str | None) -> str | None:
    """``--best-variant``: a variant name, or the json written by ``select-baseline``."""
    if spec and spec.endswith(".json"):
        return str(json.loads(Path(spec).read_text())["best_variant"])
    return spec


def run_synth(args: argparse.Namespace) -> Path:
    """Loads the run and writes the wavs of this shard (resumable; ``--max-minutes`` stops cleanly)."""
    from sparc.vocoders.eval.clone_pools import Pool
    from sparc.vocoders.eval.interp import load_run, setup_offline_env
    from sparc.vocoders.eval.io import read_json, write_json, write_parquet, write_wav

    setup_offline_env()
    module, cfg, info = load_run(args.run, args.ckpt, args.device)
    device = next(module.parameters()).device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layer = str(cfg.speaker.layer)
    gain_db = float(cfg.data.eval_gain_db) if args.gain_db is None else args.gain_db
    best_name = resolve_best_variant(args.best_variant)
    variants = all_variants(args.k_main, best_name)
    wanted = [n for n in variants if _wanted(n, args.variants)]
    pools = [p for p in args.pools.split(",") if p]
    plans = {p: read_json(plan_path(args.root, p)) for p in pools}
    items: list[Item] = []
    for p in pools:
        items += build_items(plans[p], variants, wanted)
    manifest = pd.DataFrame(
        [{"key": it.key, "kind": it.kind, "pool": it.pool, "pair_id": it.pair_id, "variant": it.variant, "content_id": it.content_id,
          "source": it.source, "target": it.target, "code_speaker": it.code_speaker, "code_ids": it.code_ids, "match_ids": it.match_ids,
          "cond": it.cond, "same_gender": it.same_gender, "dur_bin": it.dur_bin, "dur_s": it.dur_s, "wav": it.key + ".wav", **it.meta}
         for it in items]
    )
    pool_objs = {p: Pool(p, args.root, args.cache_root) for p in pools}
    stores = {p: pool_objs[p].store(layer) for p in pools}
    for p in pools:
        bad = missing_plan_ids(plans[p], stores[p])
        if bad:
            raise KeyError(f"{p}: {len(bad)} planned utterances are not in the packed store (e.g. {bad[:3]}); re-run `plan` after extract / pack")
        write_json(out / "plans" / f"{p}.json", plans[p])
    ident = cfg.speaker.get("identity", None) or {}
    trained_against = ["wavlm_sv"] if float(ident.get("weight", 0.0) or 0.0) > 0 else []
    write_parquet(out / "manifest.parquet", manifest)
    write_json(out / "meta.json", {**info, "pools": pools, "variants": wanted, "k_main": args.k_main, "best_variant": best_name or f"k{args.k_main}_f0_loud",
                                   "gain_db": gain_db, "speaker_layer": layer, "n_items": len(items), "root": str(args.root),
                                   "plan_hashes": {p: plan_hash(plans[p]) for p in pools}, "trained_against": trained_against,
                                   "rel_f0": bool(getattr(module, "rel_f0", False))})
    cache_dir = Path(args.refcond_dir) if args.refcond_dir else Path(args.root) / "refcond"
    inverter = (cfg.get("inverter") or {}).get("name")  # Phase 9 runs: reference encodings use the run's inverter
    conds = {p: RefConditions(pool_objs[p], stores[p], layer, gain_db, device, cache_dir / p, inverter=inverter) for p in pools}
    renderer = ItemRenderer(module, stores, conds, gain_db, device)
    mine = [it for i, it in enumerate(items) if i % args.nshards == args.shard]
    start, done, skipped = time.time(), 0, 0
    for it in mine:
        path = out / "wav" / (it.key + ".wav")
        if path.is_file():
            skipped += 1
            continue
        if args.max_minutes and time.time() - start > 60 * args.max_minutes:
            logger.info("time budget reached after %d wavs", done)
            break
        write_wav(path, renderer.render(it), SAMPLE_RATE)
        done += 1
        if done % 50 == 0:
            logger.info("synthesized %d (+%d existing) of %d in this shard", done, skipped, len(mine))
    logger.info("synth shard %d/%d: %d new, %d existing, %d total items", args.shard, args.nshards, done, skipped, len(items))
    return out


def _wanted(name: str, spec: str) -> bool:
    """``--variants`` is ``all`` or a comma list of names and groups (``main``, ``controls``, ``curve``, ``own``)."""
    if spec == "all":
        return True
    groups = {"main": lambda n: n in ("base", "k5", "k10") or re.match(r"k\d+_f0", n) is not None, "controls": lambda n: n.startswith("ctl_"),
              "curve": lambda n: n.startswith(("len", "noisy")), "own": lambda n: n.startswith("own_")}
    return any(n == name or (n in groups and groups[n](name)) for n in spec.split(","))


# ------------------------------------------------------------------------------------------------ scoring


class HoldoutEmbedder:
    """Held-out SV model (used in no loss, no selection). ``spec``: ``speechbrain:<local dir>`` (an x-vector / ECAPA /
    CAM++ style ``EncoderClassifier`` snapshot) or ``wespeaker:<local dir>``; anything else registers via ``LOADERS``."""

    LOADERS: dict[str, Any] = {}

    def __init__(self, spec: str, device):
        import torch

        kind, _, location = spec.partition(":")
        self.name, self.device = "holdout", torch.device(device)
        if kind in self.LOADERS:
            self.model = self.LOADERS[kind](location, self.device)
            self.fn = self.model
        elif kind == "campplus":
            from sparc.vocoders.eval.metrics.campplus import CHECKPOINT, CAMPPlusEmbedder

            self.fn = CAMPPlusEmbedder(location or CHECKPOINT, self.device)
        elif kind == "speechbrain":
            from speechbrain.inference.speaker import EncoderClassifier

            clf = EncoderClassifier.from_hparams(source=location, savedir=location, run_opts={"device": str(self.device)})
            self.fn = lambda w: clf.encode_batch(torch.from_numpy(w)[None].to(self.device), normalize=False)[0, 0].float().cpu().numpy()
        elif kind == "wespeaker":
            import wespeaker

            model = wespeaker.load_model_local(location)
            model.set_device(str(self.device))
            self.fn = lambda w: model.extract_embedding_from_pcm(torch.from_numpy(w)[None], 16000).float().cpu().numpy().reshape(-1)
        else:
            raise ValueError(f"unknown held-out SV spec {spec!r}")

    def embed(self, wavs16: list[np.ndarray], errors: list[str] | None = None) -> np.ndarray:
        rows: list[np.ndarray | None] = []
        for w in wavs16:
            try:
                v = np.asarray(self.fn(np.asarray(w, dtype=np.float32)), dtype=np.float64)
                rows.append(v / max(np.linalg.norm(v), 1e-12))
            except (ValueError, RuntimeError):
                if errors is None:
                    raise
                rows.append(None)
        dim = next((len(r) for r in rows if r is not None), 192)  # NaN rows take the width of the model's first success
        return np.stack([np.full(dim, np.nan) if r is None else r for r in rows]).astype(np.float32)


def load_holdout(spec: str | None, device, require: bool = False):
    """The held-out embedder, or ``None`` (with a warning) if it is not requested or not available offline. With
    ``require`` a missing or failing model raises: the pre-registered decision cannot be made without it."""
    if not spec:
        if require:
            raise ValueError("a held-out SV model is required: pass --holdout-sv <speechbrain|wespeaker>:<local dir>")
        return None
    try:
        return HoldoutEmbedder(spec, device)
    except Exception as error:  # missing package / snapshot: the decision then lacks the held-out model, loudly
        if require:
            raise RuntimeError(f"held-out SV model {spec!r} could not be loaded") from error
        logger.warning("held-out SV model %r unavailable (%r); skipped. The decision will refuse to run without it", spec, error)
        return None


class CloneScorers:
    """:class:`~sparc.vocoders.eval.interp.Scorers` plus the optional held-out SV model."""

    def __init__(self, device, holdout: str | None = None, require_holdout: bool = False):
        from sparc.vocoders.eval.interp import Scorers

        self.base = Scorers(device)
        self.holdout_spec = holdout
        self.require_holdout = require_holdout
        self._holdout = None
        self._tried = False

    def holdout(self):
        if not self._tried:
            self._holdout, self._tried = load_holdout(self.holdout_spec, self.base.device, self.require_holdout), True
        return self._holdout

    def models(self) -> list[str]:
        return [*SV_MODELS, *(["holdout"] if self.holdout() is not None else [])]

    def get(self, key: str):
        return self.holdout() if key == "holdout" else self.base.get(key)


def _gt16(store, uid: str, gain_db: float) -> np.ndarray:
    from sparc.vocoders.eval.io import gt_audio, to_16k

    return to_16k(gt_audio(store, store.index_of(uid), gain_db), SAMPLE_RATE)


def ground_truth_embeddings(
    out: Path, pool_obj, plan: dict[str, Any], scorers: CloneScorers | None, gain_db: float, layer: str, models: Sequence[str]
) -> dict[str, Any]:
    """Embeddings of every reference and content utterance of a pool, per model (cached in ``scores/gt_<pool>.npz``;
    computed only when the cache lacks a model and ``scorers`` is given)."""
    from sparc.vocoders.eval.io import write_npz

    path = out / "scores" / f"gt_{plan['pool']}.npz"
    ref_ids = sorted({u for r in plan["refs"].values() for u in r})
    content_ids = sorted({s["content_id"] for s in plan["swaps"]} | {o["content_id"] for o in plan["own"]})
    if path.is_file():
        with np.load(path) as z:
            if all(f"ref/{m}" in z.files for m in models) and list(z["ref_ids"]) == ref_ids and list(z["content_ids"]) == content_ids:
                return {"ref_ids": ref_ids, "content_ids": list(z["content_ids"]),
                        **{f"{k}/{m}": z[f"{k}/{m}"] for k in ("ref", "content") for m in models}}
    if scorers is None:
        raise FileNotFoundError(f"{path} lacks the ground-truth embeddings of {list(models)}; run `score --stage embed` (shard 0)")
    store = pool_obj.store(layer)
    data: dict[str, Any] = {"ref_ids": np.array(ref_ids), "content_ids": np.array(content_ids)}
    for m in models:
        model = scorers.get(m)
        data[f"ref/{m}"] = model.embed([_gt16(store, u, gain_db) for u in ref_ids], errors=[])
        data[f"content/{m}"] = model.embed([_gt16(store, u, gain_db) for u in content_ids], errors=[])
    write_npz(path, data)
    return {**data, "ref_ids": ref_ids, "content_ids": content_ids}


def score_chunks(out: Path, manifest: pd.DataFrame, scorers: CloneScorers, chunk_size: int, shard: int, nshards: int) -> None:
    """Embeddings, UTMOS and Whisper transcripts of the manifest in chunks; this shard takes chunks ``k % nshards == shard``."""
    from sparc.vocoders.eval.io import read_wav, to_16k, write_npz

    models = scorers.models()
    for k, start in enumerate(range(0, len(manifest), chunk_size)):
        if k % nshards != shard:
            continue
        block = manifest.iloc[start : start + chunk_size]
        keys = np.array(block["key"].astype(str).tolist())
        path = out / "scores" / f"chunk_{k:04d}.npz"
        if path.is_file():
            with np.load(path) as z:
                if np.array_equal(z["keys"], keys) and all(f"emb_{m}" in z.files for m in models):
                    continue
        wavs = [to_16k(*read_wav(out / "wav" / w)) for w in block["wav"]]
        data: dict[str, np.ndarray] = {"keys": keys}
        for m in models:
            data[f"emb_{m}"] = scorers.get(m).embed(wavs, errors=[])
        data["utmos"] = np.array(scorers.base.get("utmos").score(wavs, []), dtype=np.float64)
        data["text"] = np.array(scorers.base.get("asr").transcribe(wavs, errors=[]), dtype="<U1024")
        write_npz(path, data)
        logger.info("scored chunk %d (%d items)", k, len(block))


def read_chunks(out: Path, manifest: pd.DataFrame, chunk_size: int) -> dict[str, np.ndarray]:
    """Concatenated chunk arrays in manifest order; raises if a chunk is missing."""
    n = math.ceil(len(manifest) / chunk_size)
    parts = []
    for k in range(n):
        path = out / "scores" / f"chunk_{k:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"chunk {k} not scored ({path}); run score on every shard first")
        with np.load(path) as z:
            parts.append({name: z[name] for name in z.files})
    models = [m for m in ("ecapa", "wavlm_sv", "holdout") if all(f"emb_{m}" in p for p in parts)]
    partial = [m for m in ("ecapa", "wavlm_sv", "holdout") if m not in models and any(f"emb_{m}" in p for p in parts)]
    if partial:
        raise ValueError(f"model(s) {partial} were scored in some chunks only (shards ran with different --holdout-sv?); re-score the others")
    cat = {name: np.concatenate([p[name] for p in parts]) for name in ("keys", "utmos", "text", *[f"emb_{m}" for m in models])}
    if list(cat["keys"]) != list(manifest["key"].astype(str)):
        raise ValueError("chunk keys do not match the manifest")
    return cat


def run_score(args: argparse.Namespace) -> dict[str, Any] | None:
    """``--stage embed`` scores this shard's chunks, ``summarize`` aggregates all chunks, ``all`` does both (1 shard)."""
    from sparc.vocoders.eval.interp import pick_device, real_wer, setup_offline_env
    from sparc.vocoders.eval.clone_pools import Pool
    from sparc.vocoders.eval.io import read_json, write_json, write_parquet, write_text

    setup_offline_env()
    out = Path(args.out)
    meta = read_json(out / "meta.json")
    manifest = pd.read_parquet(out / "manifest.parquet")
    missing = [w for w in manifest["wav"] if not (out / "wav" / w).is_file()]
    if missing and args.stage != "summarize":
        raise FileNotFoundError(f"{len(missing)} wavs missing (e.g. {missing[0]}); run synth on every shard first")
    scorers = CloneScorers(pick_device(args.device), args.holdout_sv or meta.get("holdout_sv"), getattr(args, "require_holdout", False))
    root = Path(meta.get("root", args.root))
    if args.stage in ("embed", "all"):
        if args.holdout_sv:
            write_json(out / "meta.json", {**meta, "holdout_sv": args.holdout_sv})
        score_chunks(out, manifest, scorers, args.chunk_size, args.shard, args.nshards)
        if args.shard == 0:
            for p in meta["pools"]:
                ground_truth_embeddings(out, Pool(p, root, args.cache_root), read_json(out / "plans" / f"{p}.json"), scorers,
                                        float(meta["gain_db"]), meta["speaker_layer"], scorers.models())
        if args.stage == "embed":
            return None
    cat = read_chunks(out, manifest, args.chunk_size)
    models = [m for m in ("ecapa", "wavlm_sv", "holdout") if f"emb_{m}" in cat]
    if getattr(args, "require_holdout", False) and "holdout" not in models:
        raise RuntimeError("--require-holdout: the scored chunks have no held-out model embeddings")
    gain_db, layer = float(meta["gain_db"]), meta["speaker_layer"]
    plans = {p: read_json(out / "plans" / f"{p}.json") for p in meta["pools"]}
    ref_means: dict = {}
    ref_by_utt: dict = {}
    gt_content: dict = {}
    thresholds: dict = {}
    calibration: dict = {}
    for pool, plan in plans.items():
        gt = ground_truth_embeddings(out, Pool(pool, root, args.cache_root), plan, scorers if args.stage == "all" else None, gain_db, layer, models)
        ref_means[pool], ref_by_utt[pool], gt_content[pool], thresholds[pool], calibration[pool] = {}, {}, {}, {}, {}
        for m in models:
            e = dict(zip(gt["ref_ids"], gt[f"ref/{m}"]))
            ref_by_utt[pool][m] = e
            ref_means[pool][m] = {s: mean_unit(np.stack([e[u] for u in r])) for s, r in plan["refs"].items()}
            gt_content[pool][m] = dict(zip(gt["content_ids"], gt[f"content/{m}"]))
            cal = calibration_stats({s: np.stack([e[u] for u in r]) for s, r in plan["refs"].items()})
            calibration[pool][m] = cal
            thresholds[pool][m] = cal["eer_threshold"]
    pools_obj = {p: Pool(p, root, args.cache_root) for p in plans}
    texts = {(p, u): pools_obj[p].text(u) for p in plans for u in set(manifest.loc[manifest["pool"] == p, "content_id"])}
    counts = [real_wer(texts[(p, u)], t) for p, u, t in zip(manifest["pool"], manifest["content_id"], cat["text"])]
    items = item_table(manifest, {m: cat[f"emb_{m}"] for m in models}, cat["utmos"], counts, ref_means, ref_by_utt, gt_content, thresholds, plans)
    items["text"] = cat["text"]
    write_parquet(out / "items.parquet", items.drop(columns=["code_ids", "match_ids"]))
    summary = summarize_items(items, calibration, models, n_boot=args.n_boot, seed=args.seed)
    result = {"meta": {**meta, "models": models, "bootstrap": {"n_boot": args.n_boot, "seed": args.seed}}, "pools": summary}
    write_json(out / "summary.json", sanitize(result))
    write_text(out / "summary.md", render_summary(result))
    return result


def stratum_counts(summary: dict[str, Any]) -> dict[str, Any]:
    """``{pool: {variant: {stratum: {n_items, n_speakers}}}}`` of a summary (swap variants only)."""
    return {pool: {v: {st: {"n_items": e["n_items"], "n_speakers": e["n_speakers"]} for st, e in strata_.items()}
                   for v, strata_ in res["variants"].items()} for pool, res in summary.items()}


def split_members(splits: dict[str, Any], split: str, plans: dict[str, Any]) -> dict[str, set[str]]:
    """Split speakers per pool present in ``plans`` (pools the split does not define are absent)."""
    return {p: set(v) & set(plans[p]["speakers"]) for p, v in splits["cloning"][split].items() if p in plans}


def rescore_split(
    manifest: pd.DataFrame, items: pd.DataFrame, emb: dict[str, np.ndarray], gt: dict[str, dict[str, Any]], plans: dict[str, dict[str, Any]],
    members: dict[str, set[str]], models: Sequence[str], require_source: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Per-output table of a split with everything that depends on the pool recomputed from the split's own speakers.

    Rows kept: target in the split (and the source too with ``require_source``). Sources may come from either split; their
    number is reported. Acceptance threshold and ground-truth calibration come from the split speakers' own references
    (:func:`calibration_stats` on them), top-1 identification ranks only the split speakers. ``manifest`` / ``items`` / ``emb[m]``
    are row-aligned (``items`` supplies UTMOS and WER counts); ``gt[pool]`` holds ``ref_ids``, ``ref/<m>``, ``content_ids``,
    ``content/<m>``. Returns ``(table, calibration[pool][model], info)``.
    """
    keep = np.zeros(len(manifest), dtype=bool)
    for pool, spk in members.items():
        k = (manifest["pool"] == pool).to_numpy() & manifest["target"].isin(spk).to_numpy()
        if require_source:
            k &= manifest["source"].isin(spk).to_numpy()
        keep |= k
    idx = np.flatnonzero(keep)
    pools_kept = set(manifest["pool"].to_numpy()[idx])
    plans = {p: plans[p] for p in members if p in pools_kept}
    ref_means: dict = {}
    ref_by: dict = {}
    gt_c: dict = {}
    thr: dict = {}
    cal: dict = {}
    for pool, plan in plans.items():
        ref_means[pool], ref_by[pool], gt_c[pool], thr[pool], cal[pool] = {}, {}, {}, {}, {}
        for m in models:
            e = dict(zip(gt[pool]["ref_ids"], gt[pool][f"ref/{m}"]))
            ref_by[pool][m] = e
            ref_means[pool][m] = {s: mean_unit(np.stack([e[u] for u in r])) for s, r in plan["refs"].items()}
            gt_c[pool][m] = dict(zip(gt[pool]["content_ids"], gt[pool][f"content/{m}"]))
            c = calibration_stats({s: np.stack([e[u] for u in plan["refs"][s]]) for s in plan["speakers"] if s in members[pool]})
            cal[pool][m], thr[pool][m] = c, c["eer_threshold"]
    sub = manifest.iloc[idx].reset_index(drop=True)
    it = items.iloc[idx]
    table = item_table(sub, {m: emb[m][idx] for m in models}, it["utmos"].to_numpy(), list(zip(it["wer_err"], it["wer_len"])), ref_means, ref_by, gt_c, thr,
                       plans, rank_speakers={p: members[p] for p in plans})
    if "text" in it:
        table["text"] = it["text"].to_numpy()
    sw = table[table["kind"] == "swap"]
    outside = np.array([r not in members[p] for p, r in zip(sw["pool"], sw["source"])], dtype=bool)
    info = {"n_swap_rows": int(len(sw)), "n_cross_split_source_rows": int(outside.sum()),
            "n_cross_split_sources": int(sw["source"][outside].nunique()),
            "n_split_speakers": {p: len(members[p]) for p in plans}, "calibration_scope": "split speakers"}
    return table, cal, info


def load_split_inputs(out: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """``(manifest, items, emb, gt, plans, meta)`` of a scored output directory, from its stored chunks (no scoring models)."""
    from sparc.vocoders.eval.io import read_json

    out = Path(out)
    meta = read_json(out / "meta.json")
    manifest = pd.read_parquet(out / "manifest.parquet")
    items = pd.read_parquet(out / "items.parquet")
    with np.load(out / "scores" / "chunk_0000.npz") as z:  # the chunk size is not recorded: the first chunk is full unless it is the only one
        chunk_size = len(z["keys"])
    cat = read_chunks(out, manifest, chunk_size)
    models = [m for m in ("ecapa", "wavlm_sv", "holdout") if f"emb_{m}" in cat]
    plans = {p: read_json(out / "plans" / f"{p}.json") for p in meta["pools"]}
    gt = {}
    for p in plans:
        with np.load(out / "scores" / f"gt_{p}.npz") as z:
            gt[p] = {k: z[k] for k in z.files}
    return manifest, items, {m: cat[f"emb_{m}"] for m in models}, gt, plans, meta


def split_table(out: str | Path, splits: dict[str, Any], split: str, require_source: bool = False) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    """Rescored per-output table of a scored directory restricted to ``split`` (see :func:`rescore_split`)."""
    manifest, items, emb, gt, plans, meta = load_split_inputs(out)
    models = list(emb)
    table, cal, info = rescore_split(manifest, items, emb, gt, plans, split_members(splits, split, plans), models, require_source)
    return table, cal, info, models


def render_summary(result: dict[str, Any]) -> str:
    """Markdown headline table: per pool and variant, primary model ``sim_target``, top-1, accept, UTMOS, WER."""
    f = lambda c: "n/a" if not c or c.get("mean") is None or not np.isfinite(c["mean"]) else f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"  # noqa: E731
    lines = ["# Phase 4 cloning evaluation", "", f"Run `{result['meta'].get('run', '?')}` step {result['meta'].get('g_step', '?')}; "
             "mean over target speakers, speaker-level bootstrap 95% CI.", ""]
    for pool, res in result["pools"].items():
        cal = res.get("calibration", {})
        lines += [f"## {pool}", ""]
        for m, c in cal.items():
            g = {k: (math.nan if c.get(k) is None else c[k]) for k in ("eer", "eer_threshold", "same_mean", "diff_mean", "top1")}
            lines.append(f"- {m}: EER {g['eer']:.3f} at cos {g['eer_threshold']:.3f}; GT same {g['same_mean']:.3f}, different {g['diff_mean']:.3f}, top-1 ID {g['top1']:.3f}")
        lines += ["", "| variant | sim_target (ECAPA) | sim_target (WavLM-SV) | top-1 | accept | sim_source | UTMOS | WER |", "|---|---|---|---|---|---|---|---|"]
        for v, strata_ in res["variants"].items():
            a = strata_["all"]["metrics"]
            lines.append(f"| {v} | {f(a.get('ecapa/sim_target'))} | {f(a.get('wavlm_sv/sim_target'))} | {f(a.get('ecapa/top1'))} | {f(a.get('ecapa/accept'))} | "
                         f"{f(a.get('ecapa/sim_source'))} | {f(a.get('utmos'))} | {f(strata_['all']['wer']['corpus'])} |")
        lines.append("")
    info = result["meta"].get("split_info") or {}
    if info:
        lines += [f"Split `{result['meta'].get('split')}`: {info['n_swap_rows']} swap rows, {info['n_cross_split_source_rows']} with a source from the other split "
                  f"({info['n_cross_split_sources']} such sources); calibration from the split's own speakers {info['n_split_speakers']}.", ""]
    for pool, vs in (result.get("stratum_counts") or {}).items():
        lines += [f"Strata of `{pool}` (items / speakers), per variant:", ""]
        for v, st in vs.items():
            lines.append(f"- {v}: " + ", ".join(f"{k} {e['n_items']}/{e['n_speakers']}" for k, e in st.items()))
        lines.append("")
    return "\n".join(lines) + "\n"


def _load_scored(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sparc.vocoders.eval.io import read_json

    return pd.read_parquet(Path(path) / "items.parquet"), read_json(Path(path) / "meta.json")


def check_same_plans(meta_a: dict[str, Any], meta_b: dict[str, Any], pools: Sequence[str]) -> None:
    """Raises unless both outputs recorded the same plan hash for every compared pool."""
    ha, hb = meta_a.get("plan_hashes", {}), meta_b.get("plan_hashes", {})
    for p in pools:
        if p not in ha or p not in hb:
            raise ValueError(f"{p}: plan hash missing in meta.json (re-synthesize with the current code)")
        if ha[p] != hb[p]:
            raise ValueError(f"{p}: the two outputs were rendered from different plans ({ha[p][:8]} vs {hb[p][:8]})")


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    """Paired decision-rule comparison of two scored output directories (``--a`` arm, ``--b`` baseline).

    SV models the arm (or the baseline) was trained against (``meta.json`` ``trained_against``) are dropped from the
    decision automatically, in addition to ``--exclude-sv``."""
    from sparc.vocoders.eval import splits as S

    a, meta_a = _load_scored(args.a)
    b, meta_b = _load_scored(args.b)
    pools = sorted(set(a["pool"]) & set(b["pool"]))
    check_same_plans(meta_a, meta_b, pools)
    split = getattr(args, "split", "all")
    if split != "all":  # validation / test speakers only (docs/spkemb_validation_protocol.md); items unchanged
        sp = S.load(args.splits_file)
        a, b = (split_table(d, sp, split, args.require_source)[0] for d in (args.a, args.b))
        if args.primary_pool is None:
            args.primary_pool = sp["primary_pool"][split]
        logger.info("split %s: pools %s, %d / %d outputs", split, sorted(set(a["pool"])), len(a), len(b))
    if args.primary_pool is None:
        args.primary_pool = "libritts"
    models = [m for m in ("ecapa", "wavlm_sv", "holdout") if f"{m}/sim_target" in a and f"{m}/sim_target" in b]
    exclude = sorted({*(args.exclude_sv or ()), *meta_a.get("trained_against", []), *meta_b.get("trained_against", [])})
    if exclude:
        logger.info("excluded from the decision: %s", exclude)
    d = decision_rule(a, b, models, args.margin, args.primary_pool, PRIMARY_MODEL, exclude, args.variant_a, args.variant_b,
                      n_boot=args.n_boot, seed=args.seed, allow_missing_holdout=args.allow_missing_holdout)
    d["excluded"] = exclude
    text = render_decision(d, Path(args.a).name, Path(args.b).name)
    if args.md:
        Path(args.md).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(sanitize(d), indent=2))
    print(text)
    return d


def run_summarize_split(args: argparse.Namespace) -> dict[str, Any]:
    """``summarize-split``: re-score a scored output directory on the validation / test speakers of
    ``docs/spkemb_validation_protocol.md`` without re-synthesis (from the stored chunk embeddings). Writes
    ``summary_<split>.json`` / ``.md`` next to the original (which is not touched). Only swap pairs whose target is in the split
    are used (sources may come from either split; their number is reported); the acceptance threshold and ground-truth
    calibration come from the split's own speakers and top-1 ranks only the split's speakers. ``--split all`` reproduces
    ``summary.json`` from the stored table."""
    from sparc.vocoders.eval import splits as S
    from sparc.vocoders.eval.io import read_json, write_json, write_text

    out = Path(args.out)
    summ = read_json(out / "summary.json")
    meta = summ["meta"]
    models = meta["models"]
    info: dict[str, Any] = {}
    if args.split == "all":
        sub = pd.read_parquet(out / "items.parquet")
        cal = {p: v.get("calibration", {}) for p, v in summ["pools"].items()}
    else:
        sub, cal, info, models = split_table(out, S.load(args.splits_file), args.split, args.require_source)
    pools = summarize_items(sub, cal, models, n_boot=args.n_boot, seed=args.seed)
    result = {"meta": {**meta, "split": args.split, "require_source": bool(args.require_source), "models": models, "split_info": info,
                       "calibration_scope": "split speakers" if args.split != "all" else "full pool",
                       "bootstrap": {"n_boot": args.n_boot, "seed": args.seed}},
              "pools": pools, "stratum_counts": stratum_counts(pools)}
    result = sanitize(result)
    tag = "" if args.split == "all" else f"_{args.split}"
    write_json(out / f"summary_{args.split}.json" if args.split != "all" else Path(args.json or out / "summary_all_rescored.json"), result)
    write_text(out / f"summary{tag or '_all_rescored'}.md", render_summary(result))
    return result


def noise_floor(a: pd.DataFrame, b: pd.DataFrame, pool: str, model: str, variant: str | None, floor: float = 0.02, n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Pre-registered margin ``m = max(floor, 2 x |B0 seed-to-seed difference|)`` of the primary metric (speaker-paired mean)."""
    sa = a[(a["pool"] == pool) & (a["kind"] == "swap")]
    sb = b[(b["pool"] == pool) & (b["kind"] == "swap")]
    if variant:
        sa, sb = sa[sa["variant"] == variant], sb[sb["variant"] == variant]
    if sa["variant"].nunique() > 1 or sb["variant"].nunique() > 1:
        raise ValueError("several variants present: pass --variant")
    assert_same_plan(sa, sb, f"{pool} swaps")
    diff = speaker_diff_ci(sa, sb, f"{model}/sim_target", "pair_id", "target", n_boot, seed)
    return {"pool": pool, "model": model, "diff": diff, "floor": floor, "margin": float(max(floor, 2.0 * abs(diff["mean"])))}


def run_margin(args: argparse.Namespace) -> dict[str, Any]:
    """``margin``: the noise floor from two scored B0 seeds (same plan)."""
    a, meta_a = _load_scored(args.a)
    b, meta_b = _load_scored(args.b)
    check_same_plans(meta_a, meta_b, [args.primary_pool])
    res = noise_floor(a, b, args.primary_pool, PRIMARY_MODEL, args.variant, args.floor, args.n_boot, args.seed)
    if args.json:
        Path(args.json).write_text(json.dumps(sanitize(res), indent=2))
    print(f"seed difference {res['diff']['mean']:+.4f} [{res['diff']['lo']:+.4f}, {res['diff']['hi']:+.4f}]; margin m = {res['margin']:.4f}")
    return res


def select_baseline(summary: dict[str, Any], pool: str, model: str = PRIMARY_MODEL, candidates: Sequence[str] | None = None) -> dict[str, Any]:
    """Best inference variant on a dev pool from a ``summary.json``: ranks the swap variants (``base``, ``k*``, no controls /
    curves) by the held-out-reference ``sim_target`` (references that gave neither the code nor the matching statistics),
    which does not favour large K; variants without it (K = all references) are ranked by the primary ``sim_target`` only
    when no variant has one. The primary-metric ranking is returned alongside, with a flag if its best differs."""
    variants = summary["pools"][pool]["variants"]
    names = [v for v in variants if (v in candidates if candidates else (v in ("base", "k5", "k10") or re.match(r"k\d+_f0", v)))]
    table = {}
    for v in names:
        m = variants[v]["all"]["metrics"]
        # NaN means are stored as JSON null
        get = lambda k: float((m.get(f"{model}/{k}") or {}).get("mean") if (m.get(f"{model}/{k}") or {}).get("mean") is not None else math.nan)  # noqa: E731
        table[v] = {"sim_target": get("sim_target"), "sim_target_heldout": get("sim_target_heldout")}
    held = {v: t["sim_target_heldout"] for v, t in table.items() if np.isfinite(t["sim_target_heldout"])}
    key = "sim_target_heldout" if held else "sim_target"
    pool_ = held or {v: t["sim_target"] for v, t in table.items() if np.isfinite(t["sim_target"])}
    if not pool_:
        raise ValueError(f"no scored swap variants on {pool}")
    best = max(pool_, key=pool_.get)
    primary = {v: t["sim_target"] for v, t in table.items() if np.isfinite(t["sim_target"])}
    best_primary = max(primary, key=primary.get) if primary else None
    return {"pool": pool, "model": model, "ranked_by": key, "best_variant": best, "best_by_primary": best_primary,
            "agrees_with_primary": best == best_primary, "table": table}


def run_select_baseline(args: argparse.Namespace) -> dict[str, Any]:
    from sparc.vocoders.eval.io import read_json

    res = select_baseline(read_json(Path(args.out) / "summary.json"), args.pool)
    if args.json:
        Path(args.json).write_text(json.dumps(sanitize(res), indent=2))
    print(f"best variant on {args.pool}: {res['best_variant']} (ranked by {res['ranked_by']}; primary metric picks {res['best_by_primary']})")
    return res


# ------------------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=PHASE4_ROOT, help="phase 4 root (pools/, plans/)")
    common.add_argument("--cache-root", default=None, help="LibriTTS-R feature cache (default $SPARC_VOC_CACHE)")
    pl = sub.add_parser("plan", parents=[common], help="write plans/<pool>.json")
    pl.add_argument("--pools", default=",".join(POOLS_DEFAULT))
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--n-swap", type=int, default=400)
    pl.add_argument("--n-own", type=int, default=100)
    pl.add_argument("--n-curve", type=int, default=100)
    pl.add_argument("--n-refs", type=int, default=10)
    pl.add_argument("--sex-file", default=None)
    s = sub.add_parser("synth", parents=[common], help="render the plan with a trained run")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--pools", default=",".join(POOLS_DEFAULT))
    s.add_argument("--variants", default="all", help="all, or names / groups main,controls,curve,own")
    s.add_argument("--k-main", type=int, default=5)
    s.add_argument("--best-variant", default=None, help="variant used for the curve 'best' arms (default k<K>_f0_loud); a name or the json of select-baseline")
    s.add_argument("--refcond-dir", default=None, help="disk cache of re-encoded noisy / length-clipped references (default <root>/refcond)")
    s.add_argument("--gain-db", type=float, default=None)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1)
    s.add_argument("--max-minutes", type=float, default=0.0, help="stop cleanly after this many minutes (0 = no limit)")
    s.add_argument("--device", default="auto")
    c = sub.add_parser("score", parents=[common], help="embeddings / UTMOS / Whisper, then summary.json")
    c.add_argument("--out", required=True)
    c.add_argument("--stage", choices=("embed", "summarize", "all"), default="all")
    c.add_argument("--holdout-sv", default=None, help="held-out SV model, e.g. speechbrain:<local dir> or wespeaker:<local dir>")
    c.add_argument("--require-holdout", action="store_true", help="fail unless the held-out SV model loads (needed for any decision)")
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--nshards", type=int, default=1)
    c.add_argument("--chunk-size", type=int, default=48)
    c.add_argument("--n-boot", type=int, default=10000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--device", default="auto")
    m = sub.add_parser("compare", help="paired decision-rule comparison of two scored directories")
    m.add_argument("--a", required=True, help="arm")
    m.add_argument("--b", required=True, help="baseline")
    m.add_argument("--variant-a", default=None)
    m.add_argument("--variant-b", default=None)
    m.add_argument("--margin", type=float, default=0.02)
    m.add_argument("--primary-pool", default=None, help="default: libritts, or the split's primary pool (libritts_dev for val)")
    m.add_argument("--split", choices=("all", "val", "test"), default="all", help="restrict both outputs to the validation / test target speakers")
    m.add_argument("--splits-file", default=None)
    m.add_argument("--require-source", action="store_true", help="with --split: content speaker must also be in the split")
    m.add_argument("--exclude-sv", nargs="*", default=None, help="extra SV models to leave out (arms trained against wavlm_sv are excluded automatically)")
    m.add_argument("--allow-missing-holdout", action="store_true", help="run without the held-out SV model (loud warning, advance forced False)")
    m.add_argument("--n-boot", type=int, default=10000)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--md", default=None)
    m.add_argument("--json", default=None)
    g = sub.add_parser("margin", help="noise floor m from two scored B0 seeds")
    g.add_argument("--a", required=True, help="B0 seed 0 output")
    g.add_argument("--b", required=True, help="B0 seed 1 output")
    g.add_argument("--variant", default=None)
    g.add_argument("--primary-pool", default="libritts")
    g.add_argument("--floor", type=float, default=0.02)
    g.add_argument("--n-boot", type=int, default=10000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--json", default=None)
    ss = sub.add_parser("summarize-split", help="re-score stored outputs on the validation / test speakers (no synthesis)")
    ss.add_argument("--out", required=True, help="scored output directory (items.parquet, summary.json)")
    ss.add_argument("--split", choices=("all", "val", "test"), required=True)
    ss.add_argument("--splits-file", default=None)
    ss.add_argument("--require-source", action="store_true")
    ss.add_argument("--n-boot", type=int, default=10000)
    ss.add_argument("--seed", type=int, default=0)
    ss.add_argument("--json", default=None, help="with --split all: where to write the re-scored summary (default <out>/summary_all_rescored.json)")
    sb = sub.add_parser("select-baseline", help="best inference variant on the dev pool (held-out-reference sim_target)")
    sb.add_argument("--out", required=True, help="scored B0 output on libritts_dev")
    sb.add_argument("--pool", default="libritts_dev")
    sb.add_argument("--json", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    {"plan": run_plan, "synth": run_synth, "score": run_score, "compare": run_compare, "margin": run_margin,
     "select-baseline": run_select_baseline, "summarize-split": run_summarize_split}[args.cmd](args)


if __name__ == "__main__":
    main()
