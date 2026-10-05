"""Per-utterance cache files and the run metadata (``meta.json``) of the feature cache.

Every utterance is one ``<utt_dir>/<split>/<id>.npz``. Files are written to ``<id>.npz.tmp`` and moved into place with
``os.replace``, so a reader never sees a partial file, and :func:`is_valid` decides whether an existing file can be
skipped by a re-run.
"""

import hashlib
import json
import os
import socket
import uuid
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sparc.vocoders.constants import N_FEATURES, SPEAKER_RAW_DIM

ENPLUS_DIM = 64
ARRAY_SPECS = {
    "loud_raw": ("float32", lambda T: (T,)),
    "spk_l0": ("float32", lambda T: (SPEAKER_RAW_DIM,)),
    "spk_l6": ("float32", lambda T: (SPEAKER_RAW_DIM,)),
    "spk_enplus64": ("float32", lambda T: (ENPLUS_DIM,)),
}
FEATS_SPEC = ("float32", lambda T: (T, N_FEATURES))
SCALAR_KEYS = ("spk_wsum", "spk_fallback", "n24", "T", "peak24", "seed", "meta_hash", "gpu")
UTT_KEYS = ("feats", *ARRAY_SPECS, *SCALAR_KEYS)
VOLATILE_META_KEYS = ("created", "fork_commit")
READ_ERRORS = (OSError, ValueError, EOFError, KeyError, zlib.error, zipfile.BadZipFile)


def utterance_seed(utt_id: str) -> int:
    """Seed of the per-utterance numpy RNG that fixes torchcrepe's pitch dither."""
    return zlib.crc32(utt_id.encode()) & 0x7FFFFFFF


def utt_path(utt_dir: str | Path, split: str, utt_id: str) -> Path:
    """Location of the cache file of one utterance."""
    return Path(utt_dir) / split / f"{utt_id}.npz"


def write_utt(path: str | Path, arrays: dict) -> None:
    """Writes ``arrays`` as an ``.npz`` file atomically (temporary file in the same directory, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_utt(path: str | Path) -> dict:
    """Loads every array of a cache file into memory."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def check_arrays(data: dict, frames: int) -> str | None:
    """Returns why ``data`` is not a valid cache entry of ``frames`` frames, or ``None`` if it is valid."""
    missing = [key for key in UTT_KEYS if key not in data]
    if missing:
        return f"missing keys {missing}"
    specs = {"feats": FEATS_SPEC, **ARRAY_SPECS}
    for key, (dtype, shape_of) in specs.items():
        array = data[key]
        if array.dtype != np.dtype(dtype) or array.shape != shape_of(frames):
            return f"{key} has dtype {array.dtype} and shape {array.shape}, expected {dtype} {shape_of(frames)}"
        if not np.isfinite(array).all():
            return f"{key} is not finite"
    if int(data["T"]) != frames:
        return f"stored T {int(data['T'])} differs from expected {frames}"
    if not float(data["peak24"]) > 0.0 or not np.isfinite(float(data["spk_wsum"])):
        return "peak24 or spk_wsum is not usable"
    return None


def invalid_reason(path: str | Path, frames: int) -> str | None:
    """Returns why the file at ``path`` is not valid (including a missing or unreadable file), else ``None``."""
    path = Path(path)
    if not path.is_file():
        return "missing file"
    try:
        data = read_utt(path)
    except READ_ERRORS as error:
        return f"unreadable: {error!r}"
    return check_arrays(data, frames)


def is_valid(path: str | Path, T_expected: int) -> bool:
    """True if ``path`` loads, has all keys with the right shapes and dtypes, and every float array is finite."""
    return invalid_reason(path, T_expected) is None


def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def build_meta(description: dict, fork_commit: str) -> dict:
    """Run metadata: ``description`` (versions, checkpoints, extractor settings) plus provenance fields."""
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fork_commit": fork_commit,
        **description,
    }


def meta_compat(meta: dict) -> dict:
    """The part of ``meta`` that must agree between all tasks writing into one cache."""
    return {key: value for key, value in meta.items() if key not in VOLATILE_META_KEYS}


def ensure_meta(path: str | Path, meta: dict) -> tuple[dict, str]:
    """Creates ``meta.json`` if absent, else checks that it agrees with ``meta``; returns the stored meta and hash.

    Creation is exclusive (hard link), so concurrent array tasks end up with one file. A disagreement means the cache
    was built with different software or settings and raises ``RuntimeError``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        except OSError:
            if not path.exists():
                os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    stored = json.loads(path.read_text())
    if meta_compat(stored) != meta_compat(meta):
        raise RuntimeError(f"{path} was written with different settings; remove the cache or restore them")
    return stored, file_sha256(path)


def read_meta_hash(path: str | Path) -> str:
    """SHA-256 of ``meta.json``."""
    return file_sha256(path)


def validate_rows(rows, utt_dir: str | Path, num_threads: int) -> dict:
    """Checks the cache files of manifest ``rows`` (needs ``id``, ``split``, ``T``, ``encodable``).

    Returns counts of valid, missing and invalid files among the encodable rows, the number of rows that are not
    encodable, and up to ten examples of each problem.
    """
    encodable = rows[rows["encodable"]]
    jobs = [(utt_path(utt_dir, s, i), int(t)) for i, s, t in zip(encodable["id"], encodable["split"], encodable["T"])]
    with ThreadPoolExecutor(num_threads) as pool:
        reasons = list(pool.map(lambda job: invalid_reason(*job), jobs))
    missing = [str(p) for (p, _), r in zip(jobs, reasons) if r == "missing file"]
    invalid = [f"{p}: {r}" for (p, _), r in zip(jobs, reasons) if r not in (None, "missing file")]
    return {
        "expected": len(jobs),
        "valid": sum(r is None for r in reasons),
        "missing": len(missing),
        "invalid": len(invalid),
        "not_encodable": int((~rows["encodable"]).sum()),
        "missing_examples": missing[:10],
        "invalid_examples": invalid[:10],
    }
