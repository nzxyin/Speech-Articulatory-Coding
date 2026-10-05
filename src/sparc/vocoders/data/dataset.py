"""Datasets over the packed feature cache: random training crops and full evaluation utterances.

Features and speaker vectors come from the packed ``.npy`` files (memory-mapped); audio is read from the original
24 kHz files with ``soundfile`` seeks, so the crop of features and audio always covers the same samples.
"""

import hashlib
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from sparc.vocoders.constants import HOP, LOUDNESS_CHANNEL, SAMPLE_RATE
from sparc.vocoders.data.gain import DEFAULT_GAIN_DB_RANGE, apply_gain, gain_factor, sample_gain_db
from sparc.vocoders.data.sampler import SampleKey

CONDITIONS = ("T1", "T2", "T3")
SPEAKER_LAYERS = ("l0", "l6", "enplus64")
FIXED_CROP_STREAM = 2**32 - 1
INDEX_COLUMNS = ("id", "speaker", "chapter", "wav_path", "n24", "T", "offset", "peak24", "spk_wsum")


class PackedStore:
    """Read-only view of one or more packed split directories (``feats.npy``, ``loud_raw.npy``, ``spk_*.npy``).

    Utterances are numbered ``0 .. len-1`` in the order of the directories and of each ``index.parquet``. The arrays
    are opened on first use and dropped when pickled, so the store can be handed to ``DataLoader`` workers.
    """

    def __init__(self, packed_dirs: Sequence[str | Path], speaker_layer: str = "l6"):
        if speaker_layer not in SPEAKER_LAYERS:
            raise ValueError(f"speaker_layer must be one of {SPEAKER_LAYERS}, got {speaker_layer!r}")
        self.packed_dirs = [Path(d) for d in packed_dirs]
        if not self.packed_dirs:
            raise ValueError("no packed directories given")
        self.speaker_layer = speaker_layer
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
            for directory in self.packed_dirs:
                feats.append(np.load(directory / "feats.npy", mmap_mode="r"))
                loud.append(np.load(directory / "loud_raw.npy", mmap_mode="r"))
                spk.append(np.load(directory / f"spk_{self.speaker_layer}.npy", mmap_mode="r"))
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

    def speaker_vector(self, utt: int) -> np.ndarray:
        """Pooled speaker vector of ``utt``, float32."""
        spk = self._open()[2]
        return np.array(spk[self.dir_id[utt]][self.row[utt]], dtype=np.float32)

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
    ):
        self.store = PackedStore(packed_dirs, speaker_layer)
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
        ref = self._reference(utt, rng) if rng.random() < self.p_cross else utt
        g = gain_factor(store.peak24[utt], gain_db)
        feats, loud_raw = store.frames(utt, start, n)
        features = np.ascontiguousarray(feats.T)
        features[LOUDNESS_CHANNEL] = apply_gain(loud_raw, g)
        audio = apply_gain(store.read_audio(utt, HOP * start, HOP * n), g)
        return {
            "features": torch.from_numpy(features),
            "audio": torch.from_numpy(audio)[None],
            "spk_raw": torch.from_numpy(store.speaker_vector(ref)),
            "gain_db": torch.tensor(gain_db, dtype=torch.float32),
            "utt_index": torch.tensor(utt, dtype=torch.int64),
            "ref_index": torch.tensor(ref, dtype=torch.int64),
            "is_cross": torch.tensor(ref != utt),
            "position": torch.tensor(int(position), dtype=torch.int64),
        }


class FullUtteranceDataset(Dataset):
    """Full utterances of one packed split for validation and prediction; use with batch size 1.

    Features cover all ``T`` frames and audio the first ``480 T`` samples, both with the peak gain for ``gain_db``.
    The speaker vector depends on ``condition``: T1 the utterance's own; T2 one reference chosen deterministically
    from the pool of :func:`evaluation_pool`; T3 the ``spk_wsum``-weighted mean of that pool. Under T2 and T3 the
    utterances without a pool (speaker with a single utterance) are left out and listed in ``skipped_ids``.
    ``flags[i]`` is ``(ref_same_chapter, ref_relaxed, n_pool)`` of item ``i`` for stratified reporting (T1 reports
    ``(True, False, 0)``).
    """

    def __init__(
        self,
        packed_dir: str | Path,
        ids: Sequence[str] | None = None,
        gain_db: float = -3.0,
        condition: str = "T1",
        speaker_layer: str = "l6",
        ref_min_dur: float = 3.0,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        self.store = PackedStore([packed_dir], speaker_layer)
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
        return {
            "features": torch.from_numpy(features),
            "audio": torch.from_numpy(audio)[None],
            "spk_raw": torch.from_numpy(self._speaker_vector(i)),
            "gain_db": torch.tensor(self.gain_db, dtype=torch.float32),
            "utt_index": torch.tensor(utt, dtype=torch.int64),
            "ref_index": torch.tensor(ref, dtype=torch.int64),
            "is_cross": torch.tensor(ref != utt),
            "position": torch.tensor(i, dtype=torch.int64),
            "id": str(store.ids[utt]),
            "condition": self.condition,
            "ref_id": str(store.ids[ref]) if ref >= 0 else "speaker_mean",
        }
