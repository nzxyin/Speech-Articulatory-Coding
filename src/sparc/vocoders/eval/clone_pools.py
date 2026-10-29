"""Unseen-speaker pools of the Phase 4 cloning evaluation: listing, subset choice, feature cache and store access.

Pools (``docs/speaker_embedding_plan.md``, Phase 4):

* ``libritts`` -- LibriTTS-R test.clean (39 speakers, packed cache of the vocoder comparison, all-layer speaker
  statistics of Phase 3); ``libritts_dev`` -- dev.clean, used only to select the inference baseline.
* ``vctk`` -- VCTK corpus (``wav48``, 48 kHz, 109 speakers; gender and accent from ``speaker-info.txt``). Subset: 40
  speakers, half of each gender as far as available, round robin over accent groups, 80 random utterances each
  (seed 0). None of them is in LibriTTS-R, so none was seen by a vocoder arm.
* ``globe`` -- GLOBE V3 dev + test (Common Voice, 24 kHz flac inside the parquet shards). The corpus has **no speaker
  labels**; ``pseudo_speakers.tsv`` groups utterances by ECAPA clustering and almost all groups hold one utterance
  (max 9), so a GLOBE target has only 2-9 references of >= 3 s (``min_refs`` = 2, a documented deviation from the 10
  references of the other pools; K-reference variants use what exists). Subset: pseudo-speakers with at least
  ``min_refs`` utterances of >= 3 s plus one more utterance (562 of 14k), 400 of them balanced by gender / accent,
  all their utterances (<= 10); gender = majority of the rows' ``predicted_gender`` (ties dropped), accent from the manifest. Because the groups were
  formed by ECAPA similarity, ECAPA scores on GLOBE are optimistic for same-speaker comparisons; the WavLM-SV and the
  held-out model are the cleaner read there.

Every non-packed pool is stored under ``<root>/pools/<pool>/``: ``table.parquet`` (``id, speaker, gender, accent, n24,
text, src, wav_path``), ``wav24/<id>.wav`` (mono 24 kHz float), ``utt/<id>.npz`` (per-utterance features and all-layer
pooled statistics, written atomically so extraction is resumable and shardable), and after ``pack`` a packed split
``packed/<pool>/`` + ``layers/<pool>/`` in exactly the layout :class:`~sparc.vocoders.data.dataset.PackedStore` reads.
Features come from :class:`~sparc.vocoders.features.extractor.SparcFeatureExtractor` with the cache configuration
(per-utterance CREPE seed as in the LibriTTS-R cache) and the speaker statistics from
:mod:`sparc.vocoders.features.spk_layers` (periodicity weights from the extracted feature channel), i.e. the same
procedure as the packed LibriTTS-R splits.

Usage (GPU for ``extract``)::

    python -m sparc.vocoders.eval.clone_pools list --pool vctk       # choose the subset, write wav24 + table.parquet
    python -m sparc.vocoders.eval.clone_pools extract --pool vctk --shard 0 --nshards 4
    python -m sparc.vocoders.eval.clone_pools pack --pool vctk
"""

import argparse
import io
import logging
import os
import shutil
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.vocoders.constants import LOUDNESS_CHANNEL, N_EMA, PERIODICITY_CHANNEL, SAMPLE_RATE
from sparc.vocoders.features.spk_phase6 import P6_LAYERS, compose_input, pool_inputs

logger = logging.getLogger(__name__)

PHASE4_ROOT = "/data/user_data/xoy/spkemb/phase4"
P3_LAYER_ROOT = "/data/user_data/xoy/spkemb/phase3/cache"
VCTK_ROOT = "/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus"
GLOBE_USER = "/data/user_data/xoy/GLOBE_V3"
GLOBE_RAW = "/data/group_data/UTD-NAS/Databases/GLOBE_V3/data"
MIN_UTT_S = 0.5
INDEX_COLUMNS = ("id", "speaker", "chapter", "wav_path", "n24", "T", "offset", "peak24", "spk_wsum")


@dataclass(frozen=True)
class PoolSpec:
    """Static description of a pool. ``min_refs``: minimum number of >= 3 s utterances for an eligible speaker."""

    name: str
    kind: str  # "packed" | "vctk" | "globe" | "globe_spk"
    split: str | None = None
    min_refs: int = 10
    n_speakers: int = 0
    per_speaker: int = 0
    splits: tuple[str, ...] = ("dev", "test")  # GLOBE splits the contributors come from (``globe_spk`` kind)
    min_long: int | None = None  # utterances of >= 3 s a speaker needs (default ``min_refs``)


POOL_SPECS = {
    "libritts": PoolSpec("libritts", "packed", "test.clean", 10),
    "libritts_dev": PoolSpec("libritts_dev", "packed", "dev.clean", 10),
    "vctk": PoolSpec("vctk", "vctk", None, 10, n_speakers=40, per_speaker=80),
    "globe": PoolSpec("globe", "globe", None, 2, n_speakers=400, per_speaker=10),
    # Phase 7: the same GLOBE dev + test audio grouped by the real Common Voice contributor (same size rules as ``globe``)
    "globe_spk": PoolSpec("globe_spk", "globe_spk", None, 2, n_speakers=400, per_speaker=10),
    # Validation protocol: real GLOBE test contributors (no test contributor has 3 utterances). Relaxed rule: exactly 2
    # utterances of >= 3 s, one reference and one content utterance per speaker; swap plan only (no own-speaker items).
    "globe_spk_test": PoolSpec("globe_spk_test", "globe_spk", None, 1, n_speakers=400, per_speaker=2, splits=("test",), min_long=2),
}


# ------------------------------------------------------------------------------------------------ pure helpers


def stratified_speakers(speakers: pd.DataFrame, n: int, seed: int = 0) -> list[str]:
    """``n`` speakers (all if fewer) from a table with columns ``speaker, gender, accent``: half per gender, round robin
    over accent groups inside a gender (random order inside each group), seeded. A short gender hands its quota over."""
    rng = np.random.default_rng([seed, 11])
    per_gender: dict[str, list[str]] = {}
    for gender in sorted(speakers["gender"].unique()):
        sub = speakers[speakers["gender"] == gender]
        groups = {a: list(rng.permutation(sorted(g["speaker"]))) for a, g in sub.groupby("accent")}
        order = [groups[a] for a in sorted(groups)]
        picked: list[str] = []
        while any(order):
            for g in order:
                if g:
                    picked.append(str(g.pop()))
        per_gender[gender] = picked
    genders = sorted(per_gender)
    quota = {g: n // len(genders) for g in genders}
    for g in genders[: n - sum(quota.values())]:
        quota[g] += 1
    chosen: list[str] = []
    spare: list[str] = []
    for g in genders:
        chosen += per_gender[g][: quota[g]]
        spare += per_gender[g][quota[g] :]
    short = n - len(chosen)
    if short > 0:
        chosen += spare[:short]
    return sorted(chosen[:n])


def majority_gender(labels: Sequence[str]) -> str | None:
    """``"F"`` / ``"M"`` by strict majority of ``labels`` (``female`` / ``male`` / ``F`` / ``M``); ``None`` on a tie."""
    f = sum(str(x).lower().startswith("f") for x in labels)
    m = sum(str(x).lower().startswith("m") for x in labels)
    if f == m:
        return None
    return "F" if f > m else "M"


def parse_vctk_speaker_info(text: str) -> dict[str, tuple[str, str]]:
    """``speaker-info.txt`` -> ``{"p225": ("F", "English")}`` (header skipped; accent = first word after the gender)."""
    info = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4 and parts[2] in ("F", "M"):
            info[f"p{parts[0]}"] = (parts[2], parts[3])
    return info


def utterance_seed(utt_id: str) -> int:
    """Per-utterance CREPE dither seed, the rule of the LibriTTS-R cache."""
    return zlib.crc32(utt_id.encode()) & 0x7FFFFFFF


# ------------------------------------------------------------------------------------------------ paths and access


def pool_dir(root: str | Path, name: str) -> Path:
    return Path(root) / "pools" / name


def cache_root_default() -> Path:
    sv = os.environ.get("SV_ROOT", "/data/user_data/xoy/sparc-vocoders")
    return Path(os.environ.get("SPARC_VOC_CACHE", f"{sv}/cache"))


class Pool:
    """One evaluation pool: its utterance ``table`` and a :class:`PackedStore` over its features.

    ``table`` columns: ``id, speaker, gender (F/M), accent, n24, wav_path``; ``text`` is read lazily by :meth:`text`.
    Only speakers with a known gender are kept.
    """

    def __init__(
        self,
        spec: PoolSpec | str,
        root: str | Path = PHASE4_ROOT,
        cache_root: str | Path | None = None,
        layer_root: str | Path | None = None,
        sex_file: str | Path | None = None,
    ):
        self.spec = POOL_SPECS[spec] if isinstance(spec, str) else spec
        self.root = Path(root)
        self.cache_root = Path(cache_root) if cache_root else cache_root_default()
        self.layer_root = Path(layer_root or os.environ.get("SPKEMB_P3_CACHE", P3_LAYER_ROOT))
        self.sex_file = sex_file
        self._table: pd.DataFrame | None = None
        self._texts: pd.Series | None = None
        self._paths: pd.Series | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def directory(self) -> Path:
        return pool_dir(self.root, self.name)

    @property
    def table(self) -> pd.DataFrame:
        if self._table is None:
            self._table = self._load_table()
        return self._table

    def _load_table(self) -> pd.DataFrame:
        if self.spec.kind == "packed":
            from sparc.vocoders.eval.io import packed_dir

            index = pd.read_parquet(packed_dir(self.cache_root, self.spec.split) / "index.parquet")
            table = index[["id", "speaker", "n24", "wav_path"]].copy()
            table["id"], table["speaker"] = table["id"].astype(str), table["speaker"].astype(str)
            sex = _read_libri_sex(self.sex_file)
            table["gender"] = table["speaker"].map(sex)
            table["accent"] = ""
            table = table[table["gender"].notna()]
        else:
            table = pd.read_parquet(self.directory / "table.parquet")
        return table.sort_values("id").reset_index(drop=True)

    def store(self, layer: str = "l6", layer_root: str | Path | None = None):
        """:class:`PackedStore` of the pool for the given speaker-vector layout (``l6`` or ``layers``)."""
        from sparc.vocoders.data.dataset import PackedStore
        from sparc.vocoders.eval.io import packed_dir

        if self.spec.kind == "packed":
            d = packed_dir(self.cache_root, self.spec.split)
            root = layer_root or self.layer_root
        else:
            d = self.directory / "packed" / self.name
            root = layer_root or self.directory / "layers"
        return PackedStore([d], layer, root if layer == "layers" else None)

    def text(self, utt_id: str) -> str:
        """Reference transcript of an utterance (LibriTTS-R ``normalized.txt``, or the ``text`` column)."""
        if self.spec.kind == "packed":
            from sparc.vocoders.eval.io import read_transcript

            if self._paths is None:
                self._paths = self.table.set_index("id")["wav_path"]
            return read_transcript(self._paths.loc[utt_id])
        if self._texts is None:
            self._texts = pd.read_parquet(self.directory / "table.parquet", columns=["id", "text"]).set_index("id")["text"]
        return str(self._texts.loc[utt_id])


def _read_libri_sex(path: str | Path | None) -> dict[str, str]:
    from sparc.vocoders.eval.io import read_speaker_sex

    sv = os.environ.get("SV_ROOT", "/data/user_data/xoy/sparc-vocoders")
    for c in [path, f"{sv}/eval/ref_data/LibriTTS_R/SPEAKERS.txt"]:
        if c and Path(c).is_file():
            return read_speaker_sex(c)
    raise FileNotFoundError("no LibriTTS speaker list; pass --sex-file")


# ------------------------------------------------------------------------------------------------ listing (VCTK, GLOBE)


def list_vctk(root: str | Path = VCTK_ROOT, spec: PoolSpec = POOL_SPECS["vctk"], seed: int = 0) -> pd.DataFrame:
    """Subset table of VCTK (see the module docstring); ``n24`` filled from the file headers of the chosen files only."""
    import soundfile as sf

    root = Path(root)
    info = parse_vctk_speaker_info((root / "speaker-info.txt").read_text())
    wavs = sorted(p for p in (root / "wav48").glob("p*/*.wav") if "mic2" not in p.name)
    by_spk: dict[str, list[Path]] = {}
    for p in wavs:
        if (root / "txt" / p.parent.name / (p.stem + ".txt")).is_file() and p.parent.name in info:
            by_spk.setdefault(p.parent.name, []).append(p)
    spk_table = pd.DataFrame([{"speaker": s, "gender": info[s][0], "accent": info[s][1]} for s in sorted(by_spk)])
    chosen = stratified_speakers(spk_table, spec.n_speakers, seed)
    rng = np.random.default_rng([seed, 12])
    rows = []
    for s in chosen:
        files = by_spk[s]
        for i in sorted(rng.permutation(len(files))[: spec.per_speaker]):
            p = files[int(i)]
            meta = sf.info(str(p))
            n24 = int(round(meta.frames * SAMPLE_RATE / meta.samplerate))
            if n24 < MIN_UTT_S * SAMPLE_RATE:
                continue
            rows.append(
                {"id": p.stem, "speaker": s, "gender": info[s][0], "accent": info[s][1], "n24": n24,
                 "text": (root / "txt" / s / (p.stem + ".txt")).read_text().strip(), "src": str(p)}
            )
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def list_globe(
    user_root: str | Path = GLOBE_USER, spec: PoolSpec = POOL_SPECS["globe"], seed: int = 0, raw_root: str | Path = GLOBE_RAW
) -> pd.DataFrame:
    """Subset table of GLOBE dev + test pseudo-speakers (see the module docstring); ``src`` is ``<parquet>#<row>``."""
    import pyarrow.parquet as pq
    import soundfile as sf

    user_root, raw_root = Path(user_root), Path(raw_root)
    ps = pd.read_csv(user_root / "pseudo_speakers.tsv", sep="\t")
    ps = ps[ps["split"].isin(["dev", "test"])]
    man = pd.read_csv(user_root / "manifest.tsv", sep="\t").set_index("stem")
    counts = ps.groupby("pseudo_speaker")["stem"].transform("size")
    ps = ps[counts >= spec.min_refs + 1].copy()  # at least one utterance besides the references
    ps["dur"] = [sf.info(str(user_root / "wavs" / f"{s}.wav")).duration for s in ps["stem"]]
    ps["shard"] = ps["stem"].str.rsplit("-", n=1).str[0]
    ps["row"] = ps["stem"].str.rsplit("-", n=1).str[1].astype(int)
    gender_of = {}
    for shard, g in ps.groupby("shard"):
        col = pq.read_table(str(raw_root / f"{shard}.parquet"), columns=["predicted_gender"]).column(0).to_pylist()
        for stem, row in zip(g["stem"], g["row"]):
            gender_of[stem] = col[int(row)]
    ps["g"] = ps["stem"].map(gender_of)
    rows = []
    for spk, g in ps.groupby("pseudo_speaker"):
        gender = majority_gender(g["g"])
        if gender is None or (g["dur"] >= 3.0).sum() < spec.min_refs:
            continue
        for _, r in g.sort_values("stem").head(spec.per_speaker).iterrows():
            rows.append(
                {"id": r["stem"], "speaker": spk, "gender": gender, "accent": str(man.loc[r["stem"], "accent"]),
                 "n24": int(round(r["dur"] * SAMPLE_RATE)), "text": str(man.loc[r["stem"], "transcript"]),
                 "src": f"{raw_root / (r['shard'] + '.parquet')}#{int(r['row'])}"}
            )
    table = pd.DataFrame(rows)
    keep = stratified_speakers(table.drop_duplicates("speaker")[["speaker", "gender", "accent"]], spec.n_speakers, seed)
    return table[table["speaker"].isin(keep)].sort_values("id").reset_index(drop=True)


def list_globe_spk(
    user_root: str | Path = GLOBE_USER, spec: PoolSpec = POOL_SPECS["globe_spk"], seed: int = 0, raw_root: str | Path = GLOBE_RAW
) -> pd.DataFrame:
    """Like :func:`list_globe`, but speakers are the real Common Voice contributors of GLOBE dev + test
    (:mod:`sparc.vocoders.features.globe_speakers`, label ``gcv<index>``). Same rules: speakers with at least
    ``min_refs + 1`` utterances of which ``min_refs`` last >= 3 s, gender by majority of ``predicted_gender``, 400 speakers
    stratified by gender and accent, the first ``per_speaker`` utterances by stem. A speaker's accent is the most common
    accent label of its utterances (ties: alphabetical)."""
    import pyarrow.parquet as pq
    import soundfile as sf

    from sparc.vocoders.features.globe_speakers import globe_manifest, join_speakers, read_cv_clients

    user_root, raw_root = Path(user_root), Path(raw_root)
    joined = join_speakers(globe_manifest(user_root), read_cv_clients())
    ps = joined[joined["split"].isin(list(spec.splits))].copy()
    counts = ps.groupby("speaker")["stem"].transform("size")
    ps = ps[counts >= spec.min_refs + 1].copy()
    ps["dur"] = [sf.info(str(user_root / "wavs" / f"{s}.wav")).duration for s in ps["stem"]]
    ps["shard"] = ps["stem"].str.rsplit("-", n=1).str[0]
    ps["row"] = ps["stem"].str.rsplit("-", n=1).str[1].astype(int)
    gender_of = {}
    for shard, g in ps.groupby("shard"):
        col = pq.read_table(str(raw_root / f"{shard}.parquet"), columns=["predicted_gender"]).column(0).to_pylist()
        for stem, row in zip(g["stem"], g["row"]):
            gender_of[stem] = col[int(row)]
    ps["g"] = ps["stem"].map(gender_of)
    man = pd.read_csv(user_root / "manifest.tsv", sep="\t").set_index("stem")
    rows = []
    for spk, g in ps.groupby("speaker"):
        gender = majority_gender(g["g"])
        if gender is None or (g["dur"] >= 3.0).sum() < (spec.min_long or spec.min_refs):
            continue
        accents = g["accent"].value_counts()
        accent = sorted(accents[accents == accents.max()].index)[0]
        for _, r in g.sort_values("stem").head(spec.per_speaker).iterrows():
            rows.append(
                {"id": r["stem"], "speaker": spk, "gender": gender, "accent": str(accent),
                 "n24": int(round(r["dur"] * SAMPLE_RATE)), "text": str(man.loc[r["stem"], "transcript"]),
                 "src": f"{raw_root / (r['shard'] + '.parquet')}#{int(r['row'])}"}
            )
    table = pd.DataFrame(rows)
    keep = stratified_speakers(table.drop_duplicates("speaker")[["speaker", "gender", "accent"]], spec.n_speakers, seed)
    return table[table["speaker"].isin(keep)].sort_values("id").reset_index(drop=True)


def _globe_audio(src: str) -> np.ndarray:
    """24 kHz mono float32 of one GLOBE parquet row (``<shard.parquet>#<row>``)."""
    import pyarrow.parquet as pq
    import soundfile as sf

    path, row = src.rsplit("#", 1)
    f = pq.ParquetFile(path)
    row, start = int(row), 0
    for g in range(f.metadata.num_row_groups):
        n = f.metadata.row_group(g).num_rows
        if row < start + n:
            audio = f.read_row_group(g, columns=["audio"]).column(0)[row - start].as_py()
            wav, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
            if rate != SAMPLE_RATE or wav.ndim != 1:
                raise ValueError(f"{src}: expected mono {SAMPLE_RATE} Hz, got {wav.shape} at {rate}")
            return wav
        start += n
    raise IndexError(f"{src}: row outside the shard")


def write_wav24(table: pd.DataFrame, kind: str, out_dir: Path) -> pd.DataFrame:
    """Writes ``<out_dir>/<id>.wav`` (mono 24 kHz float; resumable) for every row and returns the table with ``wav_path``
    and the exact ``n24`` of the written file."""
    import librosa
    import soundfile as sf

    from sparc.vocoders.eval.io import write_wav

    out_dir.mkdir(parents=True, exist_ok=True)
    paths, lengths = [], []
    for _, r in table.iterrows():
        path = out_dir / f"{r['id']}.wav"
        if not path.is_file():
            if kind == "vctk":
                wav, rate = sf.read(r["src"], dtype="float32", always_2d=False)
                wav = librosa.resample(wav, orig_sr=rate, target_sr=SAMPLE_RATE, res_type="soxr_hq") if rate != SAMPLE_RATE else wav
            else:
                wav = _globe_audio(r["src"])
            write_wav(path, wav, SAMPLE_RATE)
        paths.append(str(path))
        lengths.append(int(sf.info(str(path)).frames))
    return table.assign(wav_path=paths, n24=lengths)


# ------------------------------------------------------------------------------------------------ feature extraction


class FeatureEncoder:
    """Cache-identical SPARC features plus all-layer pooled WavLM statistics of one 24 kHz waveform."""

    def __init__(self, device="auto", dither: bool = True, inverter: str | None = None):
        import torch
        from omegaconf import OmegaConf

        from sparc.vocoders.eval.reextract import extractor_config
        from sparc.vocoders.features.extractor import SparcFeatureExtractor
        from sparc.vocoders.features.spk_layers import load_wavlm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        cfg = extractor_config(OmegaConf.create({}))
        cfg.extractor.dither = bool(dither)
        self.extractor = SparcFeatureExtractor(cfg, self.device)
        self.wavlm = load_wavlm(self.device)
        self.inverter = None  # Phase 9: a WavLM k12 inverter ('mngu0' | 'all7') supplies the 12 articulatory channels
        if inverter:
            from sparc.vocoders.features.k12_inverter import K12Inverter

            self.inverter = K12Inverter(inverter, self.device)

    def encode(self, wav24: np.ndarray, seed: int) -> dict[str, np.ndarray]:
        """``feats [T, 15]``, ``loud_raw``, ``spk_l0``, ``spk_l6``, ``spk_wsum``, ``peak24``, ``n24`` and the pooled
        ``mean`` / ``std`` ``[25, 1024]`` (weights: the periodicity channel of ``feats``)."""
        from sparc.vocoders.features.spk_layers import hidden_states, pooled_stats

        out = self.extractor.extract(np.ascontiguousarray(wav24, dtype=np.float32), seed)
        if self.inverter is not None:
            feats = np.array(out["feats"], dtype=np.float32, copy=True)
            feats[:, :N_EMA] = self.inverter.channels24(np.ascontiguousarray(wav24, dtype=np.float32))
            out = {**out, "feats": feats}
        hidden = hidden_states(self.wavlm, wav24, self.device)
        mean, std = pooled_stats(hidden, out["feats"][:, PERIODICITY_CHANNEL])
        keep = ("feats", "loud_raw", "spk_l0", "spk_l6", "spk_wsum", "peak24", "n24", "T")
        feats = np.asarray(out["feats"])
        p6 = pool_inputs(hidden[1], hidden[6], feats[:, PERIODICITY_CHANNEL], feats[:, LOUDNESS_CHANNEL])
        return {
            **{k: np.asarray(out[k]) for k in keep},
            "mean": mean.astype(np.float16),
            "std": std.astype(np.float16),
            "spk_l1": p6["l1"],
            "spk_vus3": p6["vus3"],
        }


def speaker_vector(enc: dict[str, np.ndarray], layer: str) -> np.ndarray:
    """Speaker-encoder input of an encoded waveform for the layout ``layer`` (``l6``, ``l0``, ``layers`` or a Phase 6 one)."""
    if layer == "layers":
        return np.stack([enc["mean"], enc["std"]]).astype(np.float32)
    if layer in ("l0", "l6"):
        return np.asarray(enc[f"spk_{layer}"], dtype=np.float32)
    if layer in P6_LAYERS:  # Phase 6 layouts, pooled by the same code as the training side files
        return compose_input(layer, np.asarray(enc["spk_l6"], np.float32), enc.get("spk_l1"), enc.get("spk_vus3"))
    raise ValueError(f"unsupported speaker layer {layer!r}")


def extract_pool(
    spec: PoolSpec, root: str | Path, shard: int = 0, nshards: int = 1, device="auto", limit: int | None = None,
    inverter: str | None = None,
) -> int:
    """Encodes the utterances of this shard that have no ``utt/<id>.npz`` yet; returns how many were written."""
    from sparc.vocoders.eval.io import write_npz

    pool = Pool(spec, root)
    ids = list(pool.table["id"])[shard::nshards][:limit]
    todo = [u for u in ids if not (pool.directory / "utt" / f"{u}.npz").is_file()]
    if not todo:
        return 0
    enc = FeatureEncoder(device, inverter=inverter)
    paths = pool.table.set_index("id")["wav_path"]
    import soundfile as sf

    done = 0
    for u in todo:
        wav, rate = sf.read(paths[u], dtype="float32", always_2d=False)
        if rate != SAMPLE_RATE or wav.ndim != 1:
            raise ValueError(f"{u}: expected mono {SAMPLE_RATE} Hz")
        try:
            out = enc.encode(wav, utterance_seed(u))
        except ValueError as error:  # too short to encode
            logger.warning("skipping %s: %s", u, error)
            write_npz(pool.directory / "utt" / f"{u}.npz", {"skipped": np.array(1)})
            continue
        write_npz(pool.directory / "utt" / f"{u}.npz", {k: np.asarray(v) for k, v in out.items()})
        done += 1
        if done % 100 == 0:
            logger.info("%s: %d / %d", spec.name, done, len(todo))
    return done


def pack_pool(spec: PoolSpec, root: str | Path) -> Path:
    """Packs the per-utterance files of a pool into ``packed/<pool>`` and ``layers/<pool>`` (all utterances must exist)."""
    from sparc.vocoders.constants import N_FEATURES

    pool = Pool(spec, root)
    table = pool.table
    files = {u: pool.directory / "utt" / f"{u}.npz" for u in table["id"]}
    missing = [u for u, p in files.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} utterances not extracted, first {missing[:3]}")
    loaded = {}
    for u, p in files.items():
        with np.load(p) as z:
            if "skipped" not in z.files:
                loaded[u] = {k: z[k] for k in z.files}
    table = table[table["id"].isin(loaded)].reset_index(drop=True)
    frames = np.array([int(loaded[u]["T"]) for u in table["id"]], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(frames)[:-1]])
    feats = np.zeros((int(frames.sum()), N_FEATURES), np.float32)
    loud = np.zeros(int(frames.sum()), np.float32)
    n = len(table)
    spk = {k: np.zeros((n, 1024), np.float32) for k in ("spk_l0", "spk_l6")}
    mean, std = np.zeros((n, 25, 1024), np.float16), np.zeros((n, 25, 1024), np.float16)
    peak, wsum = np.zeros(n), np.zeros(n)
    for i, u in enumerate(table["id"]):
        d, lo = loaded[u], int(offsets[i])
        feats[lo : lo + frames[i]], loud[lo : lo + frames[i]] = d["feats"], d["loud_raw"]
        for k in spk:
            spk[k][i] = d[k]
        mean[i], std[i], peak[i], wsum[i] = d["mean"], d["std"], float(d["peak24"]), float(d["spk_wsum"])
    index = pd.DataFrame(
        {"id": table["id"], "speaker": table["speaker"], "chapter": table["speaker"], "wav_path": table["wav_path"],
         "n24": table["n24"].astype("int64"), "T": frames, "offset": offsets, "peak24": peak, "spk_wsum": wsum}
    )[list(INDEX_COLUMNS)]
    final_p, final_l = pool.directory / "packed" / spec.name, pool.directory / "layers" / spec.name
    for final in (final_p, final_l):
        shutil.rmtree(final.with_name(final.name + ".tmp"), ignore_errors=True)
        final.with_name(final.name + ".tmp").mkdir(parents=True)
    tp, tl = (f.with_name(f.name + ".tmp") for f in (final_p, final_l))
    np.save(tp / "feats.npy", feats)
    np.save(tp / "loud_raw.npy", loud)
    for k, v in spk.items():
        np.save(tp / f"{k}.npy", v)
    index.to_parquet(tp / "index.parquet", index=False)
    np.save(tl / "mean.npy", mean)
    np.save(tl / "std.npy", std)
    (tl / "ids.txt").write_text("\n".join(index["id"].astype(str)) + "\n")
    for final, tmp in ((final_p, tp), (final_l, tl)):
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
    pd.DataFrame({"id": index["id"]}).merge(table, on="id").to_parquet(pool.directory / "table.parquet", index=False)
    logger.info("packed %s: %d utterances, %d frames", spec.name, n, int(frames.sum()))
    return final_p


def prepare_table(spec: PoolSpec, root: str | Path, seed: int = 0) -> pd.DataFrame:
    """Lists the pool, writes the 24 kHz wavs and ``table.parquet``; returns the table."""
    if spec.kind == "vctk":
        table = list_vctk(spec=spec, seed=seed)
    elif spec.kind == "globe":
        table = list_globe(spec=spec, seed=seed)
    elif spec.kind == "globe_spk":
        table = list_globe_spk(spec=spec, seed=seed)
    else:
        raise ValueError(f"{spec.name} is a packed pool; nothing to prepare")
    d = pool_dir(root, spec.name)
    table = write_wav24(table, spec.kind, d / "wav24")
    d.mkdir(parents=True, exist_ok=True)
    table.to_parquet(d / "table.parquet", index=False)
    logger.info("%s: %d speakers, %d utterances", spec.name, table["speaker"].nunique(), len(table))
    return table


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "extract", "pack"):
        s = sub.add_parser(name)
        s.add_argument("--pool", required=True, choices=[k for k, v in POOL_SPECS.items() if v.kind != "packed"])
        s.add_argument("--root", default=PHASE4_ROOT)
        if name == "list":
            s.add_argument("--seed", type=int, default=0)
        if name == "extract":
            s.add_argument("--shard", type=int, default=0)
            s.add_argument("--nshards", type=int, default=1)
            s.add_argument("--limit", type=int, default=None)
            s.add_argument("--device", default="auto")
            s.add_argument("--inverter", default=None, choices=("mngu0", "all7"),
                           help="Phase 9: articulatory channels from this k12 inverter (use a separate --root)")
    a = p.parse_args(argv)
    spec = POOL_SPECS[a.pool]
    if a.cmd == "list":
        prepare_table(spec, a.root, a.seed)
    elif a.cmd == "extract":
        extract_pool(spec, a.root, a.shard, a.nshards, a.device, a.limit, a.inverter)
    else:
        pack_pool(spec, a.root)


if __name__ == "__main__":
    main()
