"""Datasets over the packed feature cache: random training crops and full evaluation utterances.

Features and speaker vectors come from the packed ``.npy`` files (memory-mapped); audio is read from the original
24 kHz files with ``soundfile`` seeks, so the crop of features and audio always covers the same samples.
"""

import argparse
import hashlib
import json
import os
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from sparc.vocoders.constants import (
    F0_CHANNEL,
    HOP,
    LOUDNESS_CHANNEL,
    LOUDNESS_EPS,
    PERIODICITY_CHANNEL,
    SAMPLE_RATE,
)
from sparc.vocoders.data.gain import DEFAULT_GAIN_DB_RANGE, apply_gain, gain_factor, sample_gain_db
from sparc.vocoders.data.sampler import SampleKey
from sparc.vocoders.features.spk_phase6 import NEEDS_SIDE, P6_LAYERS, P6_ROOT, SideIndex, compose_input

CONDITIONS = ("T1", "T2", "T3")
SPEAKER_LAYERS = ("l0", "l6", "enplus64", "layers", *P6_LAYERS)  # "layers": all-layer statistics of a separate cache
FIXED_CROP_STREAM = 2**32 - 1
INDEX_COLUMNS = ("id", "speaker", "chapter", "wav_path", "n24", "T", "offset", "peak24", "spk_wsum")
F0_STATS_DEFAULT = "/data/user_data/xoy/spkemb/phase4/stats/speaker_f0.json"


def voiced_f0(feats: np.ndarray) -> np.ndarray:
    """F0 values (Hz) of the voiced frames of ``feats [T, 15]``: periodicity > 0 and F0 > 0.

    The cached F0 channel is never 0 (unvoiced frames carry filled-in values), so voicing comes from the
    periodicity channel, as in ``eval/interp.py``.
    """
    voiced = (feats[:, PERIODICITY_CHANNEL] > 0.0) & (feats[:, F0_CHANNEL] > 0.0)
    return feats[voiced, F0_CHANNEL]


def median_voiced_f0(feats: np.ndarray) -> float:
    """Median F0 (Hz) over the voiced frames of ``feats [T, 15]``; NaN when there is none."""
    values = voiced_f0(feats)
    return float(np.median(values)) if len(values) else float("nan")


def voiced_logf0_std(feats: np.ndarray) -> float:
    """Standard deviation of ln F0 over the voiced frames of ``feats [T, 15]`` (the utterance's pitch range); NaN if
    fewer than 2 voiced frames."""
    values = voiced_f0(feats)
    return float(np.log(values).std()) if len(values) > 1 else float("nan")


def voiced_log_loudness(feats: np.ndarray, loudness: np.ndarray) -> float:
    """Mean of ``ln(loudness + LOUDNESS_EPS)`` over the voiced frames (the frontend's loudness transform; ``loudness``
    is the gained loudness ``[T]`` of the same frames as ``feats [T, 15]``); NaN when there is no voiced frame."""
    voiced = (feats[:, PERIODICITY_CHANNEL] > 0.0) & (feats[:, F0_CHANNEL] > 0.0)
    if not voiced.any():
        return float("nan")
    return float(np.log(np.clip(np.asarray(loudness, dtype=np.float64)[voiced], 0.0, None) + LOUDNESS_EPS).mean())


def geometric_mean_f0(values: Sequence[float]) -> float:
    """Geometric mean of the finite positive entries of ``values`` (the mean in the log-F0 domain); NaN if none."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.log(v).mean())) if len(v) else float("nan")


def build_speaker_f0_stats(packed_dirs: Sequence[str | Path]) -> dict:
    """Median voiced F0 (Hz) of every speaker over all voiced frames of the given packed splits.

    Returns ``{"splits": [...], "median_f0": {speaker: Hz}, "n_voiced_frames": {speaker: n}}``. Speakers without a
    voiced frame are left out. Frames are pooled over a speaker's utterances (not the median of utterance medians), so
    long utterances weigh more, like the frame-level F0 statistics of the frontend.
    """
    store = PackedStore(packed_dirs)
    pools: dict[int, list[np.ndarray]] = {}
    for utt in range(len(store)):
        feats, _ = store.frames(utt, 0, int(store.T[utt]))
        pools.setdefault(int(store.speaker_codes[utt]), []).append(voiced_f0(feats))
    median, count = {}, {}
    for code, parts in pools.items():
        values = np.concatenate(parts)
        if len(values):
            median[str(store.speakers[code])] = float(np.median(values))
            count[str(store.speakers[code])] = int(len(values))
    return {"splits": [Path(d).name for d in store.packed_dirs], "median_f0": median, "n_voiced_frames": count}


def save_speaker_f0_stats(stats: dict, path: str | Path) -> None:
    """Writes the statistics as JSON through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(stats, indent=1, sort_keys=True))
    os.replace(tmp, path)


def load_speaker_f0(path: str | Path, speakers: Sequence[str]) -> np.ndarray:
    """Median F0 (Hz, float32) of each of ``speakers`` from a statistics file; NaN for speakers it does not hold."""
    median = json.loads(Path(path).read_text())["median_f0"]
    return np.array([median.get(str(s), np.nan) for s in speakers], dtype=np.float32)


class PackedStore:
    """Read-only view of one or more packed split directories (``feats.npy``, ``loud_raw.npy``, ``spk_*.npy``).

    Utterances are numbered ``0 .. len-1`` in the order of the directories and of each ``index.parquet``. The arrays
    are opened on first use and dropped when pickled, so the store can be handed to ``DataLoader`` workers.

    ``speaker_layer="layers"`` reads the all-layer statistics of ``layer_stats_root/<split>/{mean,std}.npy`` (written by
    ``sparc.vocoders.features.spk_layers`` in the row order of each ``index.parquet``) instead of ``spk_*.npy``;
    :meth:`speaker_vector` then returns ``[2, 25, 1024]`` (mean, std).

    ``speaker_layer`` in ``l1l6`` / ``vus3_l6`` / ``l6_rep3`` (speaker-embedding Phase 6, see
    ``features.spk_phase6``) returns the 2048- / 3072- / 3072-d input built from ``spk_l6`` and the side files under
    ``side_root`` (default ``$SPKEMB_P6_CACHE``), which are looked up by utterance id.
    """

    def __init__(
        self,
        packed_dirs: Sequence[str | Path],
        speaker_layer: str = "l6",
        layer_stats_root: str | Path | None = None,
        side_root: str | Path | None = None,
    ):
        if speaker_layer not in SPEAKER_LAYERS:
            raise ValueError(f"speaker_layer must be one of {SPEAKER_LAYERS}, got {speaker_layer!r}")
        self.packed_dirs = [Path(d) for d in packed_dirs]
        if not self.packed_dirs:
            raise ValueError("no packed directories given")
        self.speaker_layer = speaker_layer
        self.layer_stats_dirs = None
        if speaker_layer == "layers":
            if layer_stats_root is None:
                raise ValueError('speaker_layer="layers" needs layer_stats_root')
            self.layer_stats_dirs = [Path(layer_stats_root) / d.name for d in self.packed_dirs]
        frames = []
        for dir_id, directory in enumerate(self.packed_dirs):
            table = pd.read_parquet(directory / "index.parquet", columns=list(INDEX_COLUMNS))
            table = table.reset_index(drop=True)
            table["dir_id"] = dir_id
            table["row"] = np.arange(len(table))
            frames.append(table)
        index = pd.concat(frames, ignore_index=True)
        self.ids = index["id"].astype(str).to_numpy().astype(str)
        self.speaker_codes, self.speakers = pd.factorize(index["speaker"].astype(str))
        self.chapter_codes, _ = pd.factorize(index["speaker"].astype(str) + "/" + index["chapter"].astype(str))
        self.wav_paths = index["wav_path"].astype(str).to_numpy().astype(str)
        self.n24 = index["n24"].to_numpy(dtype=np.int64)
        self.T = index["T"].to_numpy(dtype=np.int64)
        self.offset = index["offset"].to_numpy(dtype=np.int64)
        self.peak24 = index["peak24"].to_numpy(dtype=np.float64)
        self.spk_wsum = index["spk_wsum"].to_numpy(dtype=np.float64)
        self.dir_id = index["dir_id"].to_numpy(dtype=np.int64)
        self.row = index["row"].to_numpy(dtype=np.int64)
        self._id_to_index = {str(uid): i for i, uid in enumerate(self.ids)}
        if len(self._id_to_index) != len(self.ids):
            raise ValueError("duplicate utterance ids across the packed directories")
        if self.layer_stats_dirs is not None:
            for d, directory in enumerate(self.layer_stats_dirs):
                ids = (directory / "ids.txt").read_text().split()
                if not np.array_equal(np.array(ids), self.ids[self.dir_id == d]):
                    raise ValueError(f"{directory}/ids.txt does not match the index of {self.packed_dirs[d]}")
        self.side = None
        if speaker_layer in P6_LAYERS and NEEDS_SIDE[speaker_layer]:
            self.side = SideIndex(side_root or P6_ROOT)
            self.side_dir, self.side_row = self.side.locate(self.ids)
            self.side.where = {}  # only needed to locate
        self._arrays: tuple | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def index_of(self, utterance_id: str) -> int:
        return self._id_to_index[utterance_id]

    def _open(self) -> tuple:
        if self._arrays is None:
            feats, loud, spk = [], [], []
            for d, directory in enumerate(self.packed_dirs):
                feats.append(np.load(directory / "feats.npy", mmap_mode="r"))
                loud.append(np.load(directory / "loud_raw.npy", mmap_mode="r"))
                if self.layer_stats_dirs is None:
                    base = "l6" if self.speaker_layer in P6_LAYERS else self.speaker_layer
                    spk.append(np.load(directory / f"spk_{base}.npy", mmap_mode="r"))
                else:
                    stats_dir = self.layer_stats_dirs[d]
                    spk.append(tuple(np.load(stats_dir / f"{k}.npy", mmap_mode="r") for k in ("mean", "std")))
            self._arrays = (feats, loud, spk)
        return self._arrays

    def frames(self, utt: int, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw features ``[length, 15]`` and ``loud_raw [length]`` of frames ``start .. start + length`` of ``utt``."""
        if start < 0 or start + length > self.T[utt]:
            raise IndexError(f"{self.ids[utt]}: frames {start}..{start + length} outside 0..{self.T[utt]}")
        feats, loud, _ = self._open()
        d = self.dir_id[utt]
        begin = int(self.offset[utt]) + start
        return np.array(feats[d][begin : begin + length]), np.array(loud[d][begin : begin + length])

    def median_f0(self, utt: int) -> float:
        """Median voiced F0 (Hz) of the whole utterance ``utt``; NaN when it has no voiced frame."""
        return median_voiced_f0(self.frames(utt, 0, int(self.T[utt]))[0])

    def f0_range(self, utt: int) -> float:
        """Std of ln F0 over the voiced frames of the whole utterance ``utt`` (NaN if there are fewer than 2)."""
        return voiced_logf0_std(self.frames(utt, 0, int(self.T[utt]))[0])

    def log_loudness(self, utt: int, gain: float) -> float:
        """Mean ``ln(loudness + eps)`` over the voiced frames of the whole utterance after the linear gain ``gain``."""
        feats, loud = self.frames(utt, 0, int(self.T[utt]))
        return voiced_log_loudness(feats, apply_gain(loud, gain))

    def speaker_vector(self, utt: int) -> np.ndarray:
        """Pooled speaker vector of ``utt``, float32 (``[2, 25, 1024]`` for ``speaker_layer="layers"``)."""
        spk = self._open()[2][self.dir_id[utt]]
        if self.speaker_layer in P6_LAYERS:
            return self._phase6_vector(utt, np.array(spk[self.row[utt]], dtype=np.float32))
        if self.layer_stats_dirs is None:
            return np.array(spk[self.row[utt]], dtype=np.float32)
        return np.stack([np.asarray(a[self.row[utt]], dtype=np.float32) for a in spk])

    def _phase6_vector(self, utt: int, l6: np.ndarray) -> np.ndarray:
        kind = self.speaker_layer
        need = NEEDS_SIDE[kind]
        extra = {k: np.array(self.side.array(k, int(self.side_dir[utt]))[int(self.side_row[utt])], np.float32) for k in need}
        return compose_input(kind, l6, extra.get("l1"), extra.get("vus3"))

    def speaker_references(self, refs: Sequence[int], n_slots: int) -> tuple[np.ndarray, np.ndarray]:
        """Speaker vectors of ``refs`` stacked into ``n_slots`` slots (zero padded) and the mask of filled slots."""
        vectors = [self.speaker_vector(int(r)) for r in refs]
        out = np.zeros((n_slots, *vectors[0].shape), np.float32)
        out[: len(vectors)] = np.stack(vectors)
        return out, np.arange(n_slots) < len(vectors)

    def read_audio(self, utt: int, start_sample: int, num_samples: int) -> np.ndarray:
        """Mono float32 audio ``[num_samples]`` of ``utt`` starting at ``start_sample``."""
        audio, rate = sf.read(
            str(self.wav_paths[utt]), start=start_sample, frames=num_samples, dtype="float32", always_2d=False
        )
        if rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) != num_samples:
            raise ValueError(
                f"{self.ids[utt]}: expected {num_samples} mono samples at {SAMPLE_RATE} Hz, "
                f"got shape {audio.shape} at {rate} Hz"
            )
        return audio


class SpeakerGroups:
    """Utterances of each speaker ordered by utterance id, the order the reference rules are defined on.

    ``members[g]`` holds the store indices of group ``g``; ``group_of[u]`` and ``position[u]`` locate utterance ``u``.
    """

    def __init__(self, store: PackedStore):
        order = np.argsort(store.ids, kind="stable")
        order = order[np.argsort(store.speaker_codes[order], kind="stable")]
        boundaries = np.flatnonzero(np.diff(store.speaker_codes[order])) + 1
        self.members = np.split(order, boundaries)
        self.group_of = np.empty(len(store), dtype=np.int64)
        self.position = np.empty(len(store), dtype=np.int64)
        for g, members in enumerate(self.members):
            self.group_of[members] = g
            self.position[members] = np.arange(len(members))


@dataclass(frozen=True)
class ReferencePool:
    """Candidate references of one target (positions into the speaker group) and how they were found."""

    members: np.ndarray
    same_chapter: bool
    relaxed: bool


def reference_rng(utterance_id: str) -> np.random.Generator:
    """Generator of the deterministic evaluation reference draw for ``utterance_id``."""
    key = int(hashlib.sha1(utterance_id.encode()).hexdigest()[:16], 16)
    return np.random.default_rng([0, key])


def evaluation_pool(chapters: np.ndarray, n24: np.ndarray, target: int, min_samples: int) -> ReferencePool | None:
    """Reference pool for the T2 and T3 conditions.

    ``chapters`` and ``n24`` describe one speaker's utterances of one split, ordered by id; ``target`` is a position
    in them. The pool holds the other utterances of at least ``min_samples`` samples, restricted to other chapters
    when any exist. When no other utterance is long enough, it holds the longest other utterance (other chapter
    first, ties to the smaller id) and ``relaxed`` is set. Returns ``None`` for a speaker with a single utterance.
    """
    positions = np.arange(len(chapters))
    others = positions != target
    if not others.any():
        return None
    long_enough = positions[others & (n24 >= min_samples)]
    if len(long_enough) == 0:
        other_chapter = positions[others & (chapters != chapters[target])]
        source = other_chapter if len(other_chapter) else positions[others]
        longest = source[np.argmax(n24[source])]
        return ReferencePool(np.array([longest]), bool(chapters[longest] == chapters[target]), True)
    other_chapter = long_enough[chapters[long_enough] != chapters[target]]
    if len(other_chapter):
        return ReferencePool(other_chapter, False, False)
    return ReferencePool(long_enough, True, False)


def evaluation_reference(pool: ReferencePool, target_id: str) -> int:
    """Position (in the speaker group) of the single T2 reference for ``target_id``."""
    if pool.relaxed:
        return int(pool.members[0])
    return int(pool.members[reference_rng(target_id).integers(len(pool.members))])


def training_reference(
    chapters: np.ndarray, n24: np.ndarray, target: int, min_samples: int, rng: np.random.Generator
) -> int:
    """Position of a reference for ``target`` in training, drawn uniformly from the first non-empty tier.

    Tiers: other chapter and at least ``min_samples`` samples; same chapter and at least ``min_samples``; any other
    utterance of the speaker; the target itself.
    """
    positions = np.arange(len(chapters))
    others = positions != target
    long_enough = n24 >= min_samples
    other_chapter = chapters != chapters[target]
    for mask in (others & long_enough & other_chapter, others & long_enough, others):
        candidates = np.flatnonzero(mask)
        if len(candidates):
            return int(candidates[rng.integers(len(candidates))])
    return target


def reference_manifest(table: pd.DataFrame, min_dur: float = 3.0) -> list[dict]:
    """T2 reference manifest of one split from a table with columns ``id, speaker, chapter, n24``.

    Row format: ``id, speaker, chapter, skipped`` and, for targets with a reference, ``ref_id, ref_chapter,
    ref_same_chapter, ref_relaxed, n_pool``. Used to check the rule against externally built manifests.
    """
    table = table.assign(
        id=table["id"].astype(str), speaker=table["speaker"].astype(str), chapter=table["chapter"].astype(str)
    ).sort_values("id")
    min_samples = int(round(min_dur * SAMPLE_RATE))
    rows = []
    for _, group in table.groupby("speaker", sort=True):
        ids = group["id"].to_numpy()
        chapters = group["chapter"].to_numpy()
        n24 = group["n24"].to_numpy(dtype=np.int64)
        for i, uid in enumerate(ids):
            row = {"id": uid, "speaker": group["speaker"].iloc[0], "chapter": chapters[i]}
            pool = evaluation_pool(chapters, n24, i, min_samples)
            if pool is None:
                rows.append({**row, "skipped": True})
                continue
            ref = evaluation_reference(pool, uid)
            rows.append(
                {
                    **row,
                    "skipped": False,
                    "ref_id": ids[ref],
                    "ref_chapter": chapters[ref],
                    "ref_same_chapter": pool.same_chapter,
                    "ref_relaxed": pool.relaxed,
                    "n_pool": int(len(pool.members)),
                }
            )
    rows.sort(key=lambda r: r["id"])
    return rows


class CropDataset(Dataset):
    """Random aligned crops of ``crop_frames`` feature frames and the matching audio, for training.

    Items are the utterances with ``T >= crop_frames``. ``__getitem__`` takes a sampler key ``(epoch, position,
    item_index)`` and draws crop start, gain and speaker reference from ``default_rng([seed, epoch, position])``,
    so the sample is a pure function of the key. With probability ``p_cross`` the speaker vector comes from another
    utterance of the same speaker (see :func:`training_reference`), otherwise from the utterance itself.

    ``stats`` is accepted for interface symmetry with the models and is not used: speaker normalization lives in
    the model. ``subset_size`` keeps only that many items, chosen once from ``subset_seed``; ``fixed_crops`` makes
    crop start, gain and speaker reference a function of the utterance alone (overfitting and sanity runs).

    ``n_refs = (lo, hi)`` (speaker-embedding Phase 3) draws ``k ~ U{lo..hi}`` references per sample, each one by the
    ``p_cross`` rule above (references may repeat); ``spk_raw`` is then ``[hi, ...]`` zero padded with ``ref_mask
    [hi]``, ``ref_index`` is the first reference and ``is_cross`` is true when any reference is another utterance.
    ``n_refs=None`` keeps the single-reference sample and its random stream unchanged.

    Speaker-embedding Phase 4 (both off by default; neither draws from the random stream): ``f0_stats`` (a file of
    :func:`build_speaker_f0_stats`) adds ``speaker_id`` (int, the store's speaker code) and ``speaker_f0`` (median voiced
    F0 of the sample's speaker in Hz, NaN if unknown) for swap-in-training; ``f0_conditioning`` adds ``utt_f0`` (median
    voiced F0 of the whole source utterance) and ``ref_f0`` (geometric mean of the references' medians) for the
    speaker-relative F0 arm. Full-utterance medians (not crop medians) are used so training matches inference.
    """

    def __init__(
        self,
        packed_dirs: Sequence[str | Path],
        stats: dict | None = None,
        crop_frames: int = 64,
        p_cross: float = 0.5,
        ref_min_dur: float = 3.0,
        speaker_layer: str = "l6",
        seed: int = 0,
        gain_db_range: Sequence[float] = DEFAULT_GAIN_DB_RANGE,
        subset_size: int | None = None,
        subset_seed: int = 0,
        fixed_crops: bool = False,
        n_refs: Sequence[int] | None = None,
        layer_stats_root: str | Path | None = None,
        f0_stats: str | Path | None = None,
        f0_conditioning: bool = False,
        cond_stats: bool = False,
    ):
        self.cond_stats = bool(cond_stats)
        self.store = PackedStore(packed_dirs, speaker_layer, layer_stats_root)
        self.speaker_f0 = None if f0_stats is None else load_speaker_f0(f0_stats, self.store.speakers)
        if self.speaker_f0 is not None:
            missing = float(np.isnan(self.speaker_f0).mean())
            if missing > 0.01:  # a stale or partial file would silently turn the swap's F0 scaling off
                raise ValueError(
                    f"{f0_stats}: no median F0 for {100 * missing:.1f} % of the {len(self.speaker_f0)} speakers; "
                    "rebuild it with `python -m sparc.vocoders.data.dataset f0-stats` for the training splits"
                )
        self.f0_conditioning = bool(f0_conditioning)
        self.n_refs = None if n_refs is None else (int(n_refs[0]), int(n_refs[1]))
        if self.n_refs is not None and not 1 <= self.n_refs[0] <= self.n_refs[1]:
            raise ValueError(f"n_refs must be (lo, hi) with 1 <= lo <= hi, got {n_refs}")
        self.crop_frames = int(crop_frames)
        self.p_cross = float(p_cross)
        self.min_ref_samples = int(round(ref_min_dur * SAMPLE_RATE))
        self.seed = int(seed)
        self.gain_db_range = (float(gain_db_range[0]), float(gain_db_range[1]))
        self.fixed_crops = bool(fixed_crops)
        items = np.flatnonzero(self.store.T >= self.crop_frames)
        if subset_size is not None:
            keep = np.random.default_rng([int(subset_seed), 1]).permutation(len(items))[: int(subset_size)]
            items = np.sort(items[keep])
        if len(items) == 0:
            raise ValueError(f"no utterance has at least {self.crop_frames} frames")
        self.items = items
        self.groups = SpeakerGroups(self.store)

    def __len__(self) -> int:
        return len(self.items)

    def _reference(self, utt: int, rng: np.random.Generator) -> int:
        members = self.groups.members[self.groups.group_of[utt]]
        pos = training_reference(
            self.store.chapter_codes[members],
            self.store.n24[members],
            int(self.groups.position[utt]),
            self.min_ref_samples,
            rng,
        )
        return int(members[pos])

    def __getitem__(self, key: SampleKey) -> dict[str, torch.Tensor]:
        epoch, position, item = key
        store = self.store
        utt = int(self.items[item])
        n = self.crop_frames
        if self.fixed_crops:
            rng = np.random.default_rng([self.seed, zlib.crc32(str(store.ids[utt]).encode()), FIXED_CROP_STREAM])
        else:
            rng = np.random.default_rng([self.seed, int(epoch), int(position)])
        start = int(rng.integers(0, int(store.T[utt]) - n + 1))
        gain_db = sample_gain_db(rng, self.gain_db_range)
        if self.n_refs is None:
            refs = [self._reference(utt, rng) if rng.random() < self.p_cross else utt]
        else:
            k = int(rng.integers(self.n_refs[0], self.n_refs[1] + 1))
            refs = [self._reference(utt, rng) if rng.random() < self.p_cross else utt for _ in range(k)]
        g = gain_factor(store.peak24[utt], gain_db)
        feats, loud_raw = store.frames(utt, start, n)
        features = np.ascontiguousarray(feats.T)
        features[LOUDNESS_CHANNEL] = apply_gain(loud_raw, g)
        audio = apply_gain(store.read_audio(utt, HOP * start, HOP * n), g)
        sample = {
            "features": torch.from_numpy(features),
            "audio": torch.from_numpy(audio)[None],
            "gain_db": torch.tensor(gain_db, dtype=torch.float32),
            "utt_index": torch.tensor(utt, dtype=torch.int64),
            "ref_index": torch.tensor(refs[0], dtype=torch.int64),
            "is_cross": torch.tensor(any(r != utt for r in refs)),
            "position": torch.tensor(int(position), dtype=torch.int64),
        }
        if self.n_refs is None:
            sample["spk_raw"] = torch.from_numpy(store.speaker_vector(refs[0]))
        else:
            vectors, mask = store.speaker_references(refs, self.n_refs[1])
            sample["spk_raw"], sample["ref_mask"] = torch.from_numpy(vectors), torch.from_numpy(mask)
        if self.speaker_f0 is not None:
            code = int(store.speaker_codes[utt])
            sample["speaker_id"] = torch.tensor(code, dtype=torch.int64)
            sample["speaker_f0"] = torch.tensor(float(self.speaker_f0[code]), dtype=torch.float32)
        if self.f0_conditioning:
            medians = {utt: store.median_f0(utt)}
            medians.update({r: store.median_f0(r) for r in refs if r not in medians})
            sample["utt_f0"] = torch.tensor(medians[utt], dtype=torch.float32)
            sample["ref_f0"] = torch.tensor(geometric_mean_f0([medians[r] for r in refs]), dtype=torch.float32)
        if self.cond_stats:  # Phase 8: utterance-level statistics for the conditioning path (full utterances)
            ranges = [store.f0_range(r) for r in refs]
            ranges = [x for x in ranges if np.isfinite(x)]
            sample["ref_f0_std"] = torch.tensor(float(np.mean(ranges)) if ranges else float("nan"), dtype=torch.float32)
            sample["utt_loud"] = torch.tensor(store.log_loudness(utt, g), dtype=torch.float32)
        return sample


class FullUtteranceDataset(Dataset):
    """Full utterances of one packed split for validation and prediction; use with batch size 1.

    Features cover all ``T`` frames and audio the first ``480 T`` samples, both with the peak gain for ``gain_db``.
    The speaker vector depends on ``condition``: T1 the utterance's own; T2 one reference chosen deterministically
    from the pool of :func:`evaluation_pool`; T3 the ``spk_wsum``-weighted mean of that pool. Under T2 and T3 the
    utterances without a pool (speaker with a single utterance) are left out and listed in ``skipped_ids``.
    ``flags[i]`` is ``(ref_same_chapter, ref_relaxed, n_pool)`` of item ``i`` for stratified reporting (T1 reports
    ``(True, False, 0)``).

    ``multi_ref`` (encoders that take several references, speaker-embedding Phase 3): ``spk_raw`` gets a reference
    axis with ``ref_mask``; T1 and T2 hold their one reference, and T3 holds every vector of the pool, so the
    encoder combines them (instead of the mean of the raw vectors).

    ``f0_conditioning`` (Phase 4, speaker-relative F0) adds ``utt_f0``, the median voiced F0 of the utterance, and
    ``ref_f0``, the geometric mean of the reference utterances' medians (the own utterance for T1, the T2 reference, the
    whole pool for T3).
    """

    def __init__(
        self,
        packed_dir: str | Path,
        ids: Sequence[str] | None = None,
        gain_db: float = -3.0,
        condition: str = "T1",
        speaker_layer: str = "l6",
        ref_min_dur: float = 3.0,
        multi_ref: bool = False,
        layer_stats_root: str | Path | None = None,
        f0_conditioning: bool = False,
        cond_stats: bool = False,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        self.store = PackedStore([packed_dir], speaker_layer, layer_stats_root)
        self.multi_ref = bool(multi_ref)
        self.f0_conditioning = bool(f0_conditioning)
        self.cond_stats = bool(cond_stats)
        self.gain_db = float(gain_db)
        self.condition = condition
        chosen = range(len(self.store)) if ids is None else [self.store.index_of(str(i)) for i in ids]
        self.items: list[int] = []
        self.pools: list[np.ndarray | None] = []
        self.references: list[int] = []
        self.skipped_ids: list[str] = []
        self.flags: list[tuple[bool, bool, int]] = []
        if condition == "T1":
            self.items = list(chosen)
            self.pools = [None] * len(self.items)
            self.references = list(self.items)
            self.flags = [(True, False, 0)] * len(self.items)
            return
        min_samples = int(round(ref_min_dur * SAMPLE_RATE))
        groups = SpeakerGroups(self.store)
        for utt in chosen:
            members = groups.members[groups.group_of[utt]]
            p = int(groups.position[utt])
            pool = evaluation_pool(self.store.chapter_codes[members], self.store.n24[members], p, min_samples)
            if pool is None:
                self.skipped_ids.append(str(self.store.ids[utt]))
                continue
            self.items.append(int(utt))
            self.flags.append((pool.same_chapter, pool.relaxed, int(len(pool.members))))
            if condition == "T2":
                self.references.append(int(members[evaluation_reference(pool, str(self.store.ids[utt]))]))
                self.pools.append(None)
            else:
                self.references.append(-1)
                self.pools.append(members[pool.members])

    def __len__(self) -> int:
        return len(self.items)

    def _speaker_vector(self, i: int) -> np.ndarray:
        if self.multi_ref:
            refs = self.pools[i] if self.condition == "T3" else [self.references[i]]
            return np.stack([self.store.speaker_vector(int(u)) for u in refs])
        if self.condition != "T3":
            return self.store.speaker_vector(self.references[i])
        pool = self.pools[i]
        vectors = np.stack([self.store.speaker_vector(int(u)) for u in pool]).astype(np.float64)
        weights = self.store.spk_wsum[pool]
        if not weights.sum() > 0.0:
            weights = np.ones(len(pool))
        return (weights @ vectors / weights.sum()).astype(np.float32)

    def __getitem__(self, i: int) -> dict:
        store = self.store
        utt = self.items[i]
        frames = int(store.T[utt])
        g = gain_factor(store.peak24[utt], self.gain_db)
        feats, loud_raw = store.frames(utt, 0, frames)
        features = np.ascontiguousarray(feats.T)
        features[LOUDNESS_CHANNEL] = apply_gain(loud_raw, g)
        audio = apply_gain(store.read_audio(utt, 0, HOP * frames), g)
        ref = self.references[i]
        spk = self._speaker_vector(i)
        extra = {"ref_mask": torch.ones(len(spk), dtype=torch.bool)} if self.multi_ref else {}
        if self.f0_conditioning:
            refs = self.pools[i] if self.condition == "T3" else [ref]
            own = median_voiced_f0(feats)
            ref_medians = [own if int(r) == utt else store.median_f0(int(r)) for r in refs]
            extra["utt_f0"] = torch.tensor(own, dtype=torch.float32)
            extra["ref_f0"] = torch.tensor(geometric_mean_f0(ref_medians), dtype=torch.float32)
        if self.cond_stats:
            refs = self.pools[i] if self.condition == "T3" else [ref]
            ranges = [voiced_logf0_std(feats) if int(r) == utt else store.f0_range(int(r)) for r in refs]
            ranges = [x for x in ranges if np.isfinite(x)]
            extra["ref_f0_std"] = torch.tensor(float(np.mean(ranges)) if ranges else float("nan"), dtype=torch.float32)
            extra["utt_loud"] = torch.tensor(voiced_log_loudness(feats, features[LOUDNESS_CHANNEL]), dtype=torch.float32)
        return {
            **extra,
            "features": torch.from_numpy(features),
            "audio": torch.from_numpy(audio)[None],
            "spk_raw": torch.from_numpy(spk),
            "gain_db": torch.tensor(self.gain_db, dtype=torch.float32),
            "utt_index": torch.tensor(utt, dtype=torch.int64),
            "ref_index": torch.tensor(ref, dtype=torch.int64),
            "is_cross": torch.tensor(ref != utt),
            "position": torch.tensor(i, dtype=torch.int64),
            "id": str(store.ids[utt]),
            "condition": self.condition,
            "ref_id": str(store.ids[ref]) if ref >= 0 else "speaker_mean",
        }


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m sparc.vocoders.data.dataset f0-stats``: writes the per-speaker median voiced F0 file."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f0 = sub.add_parser("f0-stats", help="median voiced F0 per speaker of the packed training splits (swap-in-training)")
    f0.add_argument("--cache-root", default=os.environ.get("SPARC_VOC_CACHE"), help="default: $SPARC_VOC_CACHE")
    f0.add_argument("--splits", nargs="+", default=["train.clean.100", "train.clean.360", "train.other.500"])
    f0.add_argument("--out", default=F0_STATS_DEFAULT)
    args = parser.parse_args(argv)
    if not args.cache_root:
        parser.error("--cache-root or $SPARC_VOC_CACHE is required")
    stats = build_speaker_f0_stats([Path(args.cache_root) / "packed" / s for s in args.splits])
    save_speaker_f0_stats(stats, args.out)
    print(f"{len(stats['median_f0'])} speakers -> {args.out}")


if __name__ == "__main__":
    main()
