"""Toy packed split, environment and config for the runner tests (tests/eval/test_io.py, test_stages.py, ...).

Not a test module. ``build_toy_cache`` writes a packed ``test.clean`` split (contract INTERFACES 1.5) with real WAV files and
LibriTTS-style ``.normalized.txt`` transcripts, a ``train_stats.json`` and a speaker list, so that the evaluation stages run on
CPU in seconds without the real cache.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from sparc.vocoders.constants import HOP, N_EMA, N_FEATURES, SAMPLE_RATE, feature_length
from sparc.vocoders.eval.systems import compose_config

SPLIT = "test.clean"
# (speaker, chapter -> durations in seconds); speaker 305 has a single utterance (no T2 reference)
TOY_SPEAKERS = {
    "300": {"1": [4.0, 5.0, 3.5, 1.5], "2": [3.2, 6.0]},
    "301": {"1": [4.0, 5.0, 3.5], "3": [3.5, 2.0]},
    "302": {"1": [3.6, 3.1, 4.4], "2": [5.0]},
    "303": {"7": [3.3, 4.2, 3.9, 1.0]},
    "305": {"1": [5.0]},
}
SEX = {"300": "F", "301": "M", "302": "F", "303": "M", "305": "F"}
WORDS = "the quick brown fox jumps over a lazy dog".split()


def utterance_rows() -> list[tuple[str, str, str, float]]:
    rows = []
    for speaker, chapters in TOY_SPEAKERS.items():
        for chapter, durations in chapters.items():
            for k, seconds in enumerate(durations):
                rows.append((f"{speaker}_{chapter}_{k:06d}_000000", speaker, chapter, seconds))
    return rows


def build_toy_cache(root: Path, seed: int = 0) -> Path:
    """Writes ``root/cache/packed/test.clean``, ``root/cache/stats/train_stats.json`` and ``root/eval/ref_data/.../SPEAKERS.txt``."""
    rng = np.random.default_rng(seed)
    cache = root / "cache"
    out = cache / "packed" / SPLIT
    wav_dir = root / "wav" / SPLIT
    out.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    feats, loud, records = [], [], []
    offset = 0
    rows = utterance_rows()
    for uid, speaker, chapter, seconds in rows:
        n24 = int(round(seconds * SAMPLE_RATE))
        T = feature_length(n24)
        audio = rng.uniform(-0.3, 0.3, n24)
        path = wav_dir / f"{uid}.wav"
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        path.with_name(f"{uid}.normalized.txt").write_text(" ".join(rng.choice(WORDS, size=5)) + "\n")
        audio = sf.read(path, dtype="float32")[0]
        f = rng.standard_normal((T, N_FEATURES)).astype(np.float32)
        f[:, 12] = 100.0 + 50.0 * rng.random(T)  # F0 Hz
        f[:, 13] = 0.1 * rng.random(T)  # SPARC's z-scored loudness (unused by the evaluation)
        f[:, 14] = np.where(rng.random(T) > 0.4, 0.5 + 0.4 * rng.random(T), 0.0)  # periodicity: 0 or in [0.5, 0.9)
        feats.append(f)
        loud.append(np.abs(audio[: HOP * T]).reshape(T, HOP).mean(axis=1).astype(np.float32))
        records.append(
            dict(
                id=uid, speaker=speaker, chapter=chapter, wav_path=str(path), n24=n24, T=T, offset=offset,
                peak24=float(np.abs(audio).max()), seed=0, duration=n24 / SAMPLE_RATE,
                spk_wsum=float(rng.uniform(5.0, 50.0)), spk_fallback=False,
            )
        )  # fmt: skip
        offset += T
    n = len(rows)
    np.save(out / "feats.npy", np.concatenate(feats))
    np.save(out / "loud_raw.npy", np.concatenate(loud))
    np.save(out / "spk_l0.npy", rng.standard_normal((n, 1024)).astype(np.float32))
    np.save(out / "spk_l6.npy", rng.standard_normal((n, 1024)).astype(np.float32))
    np.save(out / "spk_enplus64.npy", rng.standard_normal((n, 64)).astype(np.float32))
    pd.DataFrame(records).to_parquet(out / "index.parquet")
    stats = cache / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    (stats / "train_stats.json").write_text(json.dumps({"ema_std": [1.0 + 0.1 * c for c in range(N_EMA)]}))
    ref = root / "eval" / "ref_data" / "LibriTTS_R"
    ref.mkdir(parents=True, exist_ok=True)
    lines = ["; READER | SEX | SUBSET | MINUTES | NAME"] + [f"{s} | {SEX[s]} | test-clean | 8.0 | name {s}" for s in SEX]
    (ref / "SPEAKERS.txt").write_text("\n".join(lines) + "\n")
    return out


def set_env(monkeypatch, root: Path) -> None:
    """The environment variables the Hydra path interpolations read."""
    monkeypatch.setenv("SV_ROOT", str(root))
    monkeypatch.setenv("SPARC_VOC_CACHE", str(root / "cache"))
    monkeypatch.setenv("SPARC_VOC_RUNS", str(root / "runs"))
    monkeypatch.setenv("LIBRITTSR_RAW", str(root / "raw"))
    monkeypatch.setenv("SPARC_REFIT_NPZ", str(root / "refit.npz"))
    monkeypatch.setenv("HF_HUB_CACHE", str(root / "hf_hub"))


def toy_config(root: Path, *overrides: str):
    """``eval_config`` composed for the toy cache (call after ``set_env``)."""
    return compose_config("eval_config", [f"paths.eval_root={root / 'eval'}", f"paths.runs_root={root / 'runs'}", *overrides])
