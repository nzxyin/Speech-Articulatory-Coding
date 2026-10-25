"""Shared audio-quality scorer for the Phase 5 data filters (``docs/speaker_embedding_plan.md``, Phase 5 Amendment 1).

One implementation, run unchanged on every corpus (LibriTTS-R train and test.clean, the GLOBE v3 training list, the VCTK
evaluation pool and, later, the VCTK training list). One row per utterance:

* ``duration_s``, ``native_sr``, ``native_unavailable`` (the native-rate audio is not on disk, bandwidth and clipping are
  then computed on the file that was scored), ``upsampled`` (native rate below 24 kHz);
* noise: ``wada_snr`` (WADA-SNR, Kim and Stern 2008, on the 16 kHz audio) and ``squim_si_sdr`` (torchaudio
  ``SQUIM_OBJECTIVE``, reference-free);
* reverberation / degradation: ``squim_stoi``. The Brouhaha C50 estimator is NOT used: it needs ``pyannote.audio`` 3
  with ``speechbrain`` 0.5.16, which would downgrade the ``speechbrain`` 1.1.1 of the shared venv (the ECAPA models of
  the cloning evaluation). The Squim STOI estimate is the pre-registered fallback; it measures intelligibility loss from
  any degradation, so it is a proxy for reverberation, not a C50 estimate;
* naturalness: ``utmos`` (UTMOS22 strong, the Phase 4 predictor, one utterance at a time) and ``squim_pesq``
  (reference-free PESQ estimate);
* bandwidth: ``bw99_hz`` (lowest frequency below which 99 % of the spectral energy lies, on the native-rate audio before
  resampling, 2048-point Hann frames, energy summed over frames) and ``bw_floor_hz`` (highest frequency whose long-term
  average level is within 60 dB of the spectral peak; detects a hard low-pass cut that ``bw99_hz`` cannot see because
  speech energy is concentrated below 4 kHz);
* clipping: ``clip_frac`` (fraction of native samples with ``|x| >= 0.999``) and ``peak``;
* text: ``has_wer`` rows carry the Whisper large-v3 transcript ``hyp`` and the per-utterance ``word_errors``,
  ``word_ref_len``, ``char_errors``, ``char_ref_len``, ``wer``, ``cer``, with the normalization of the Phase 4
  evaluation (:mod:`sparc.vocoders.eval.metrics.asr`).

Subcommands::

    python -m sparc.vocoders.features.quality make-list --corpus libritts_train|libritts_test_clean|vctk_eval|globe_train
    python -m sparc.vocoders.features.quality run --corpus globe_train --shard 0 --nshards 8 [--max-minutes 50] [--limit N]
    python -m sparc.vocoders.features.quality report

Lists live in ``<ROOT>/<corpus>/list.parquet`` with columns ``id, audio, corpus, accent, speaker`` and optional
``native_path, start_s, end_s, text, do_wer`` (any list given with ``--list`` works, for example the VCTK training
list). Scores are written as ``<ROOT>/<corpus>/scores/chunk-<k>.parquet`` (``--chunk`` rows each, atomic, a finished
chunk is never recomputed), so a shard that is killed or requeued resumes at the first missing chunk; the job exits with
code 75 while work is left.
"""

import argparse
import hashlib
import importlib.resources
import json
import logging
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("quality")

ROOT = Path(os.environ.get("SPKEMB_P5_QUALITY", "/data/user_data/xoy/spkemb/phase5/quality"))
EXIT_STOPPED = 75
MODEL_SR = 16000
TARGET_SR = 24000
CLIP_LEVEL = 0.999
N_FFT = 2048
CHUNK = 500
SEED_WER_SAMPLE = 0
WER_SAMPLE_SIZE = 5000

PACKED = Path("/data/user_data/xoy/sparc-vocoders/cache/packed")
TRAIN_SPLITS = ("train.clean.100", "train.clean.360", "train.other.500")
GLOBE_LIST = Path("/data/user_data/xoy/spkemb/phase4/globe_train/list.parquet")
GLOBE_WAV24 = Path("/data/user_data/xoy/spkemb/phase4/globe_train/wav24")
GLOBE_MANIFEST = Path("/data/user_data/xoy/GLOBE_V3/manifest.tsv")
VCTK_POOL = Path("/data/user_data/xoy/spkemb/phase4/pools/vctk/table.parquet")

# measure -> True if higher is better; the single-measure and joint thresholds of the report use these
MEASURES = {
    "wada_snr": True,
    "squim_si_sdr": True,
    "squim_stoi": True,
    "utmos": True,
    "squim_pesq": True,
    "bw99_hz": True,
    "clip_frac": False,
}
TEXT_MEASURES = {"wer": False, "cer": False}
ANCHORS = ("libritts_test_clean", "vctk_eval")
ANCHOR_PERCENTILES = (5, 10, 25)

SCORE_COLUMNS = [
    "id", "corpus", "accent", "speaker", "duration_s", "native_sr", "native_unavailable", "upsampled",
    "bw99_hz", "bw_floor_hz", "clip_frac", "peak", "rms_dbfs", "wada_snr",
    "squim_si_sdr", "squim_stoi", "squim_pesq", "utmos",
    "has_wer", "hyp", "word_errors", "word_ref_len", "char_errors", "char_ref_len", "wer", "cer", "error",
]  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# signal measures (numpy only)
# --------------------------------------------------------------------------------------------------------------------
def _frames(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    count = 1 + (len(x) - n) // hop
    return np.lib.stride_tricks.sliding_window_view(x, n)[::hop][:count]


def spectrum_measures(x: np.ndarray, sr: int, n_fft: int = N_FFT) -> tuple[float, float]:
    """``(bw99_hz, bw_floor_hz)`` of a mono waveform at its own rate (see the module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    window = np.hanning(n_fft)
    power = np.zeros(n_fft // 2 + 1)
    frames = _frames(x, n_fft, n_fft // 2)
    for start in range(0, len(frames), 256):
        spec = np.fft.rfft(frames[start : start + 256] * window, axis=1)
        power += (spec.real**2 + spec.imag**2).sum(axis=0)
    total = power.sum()
    if not np.isfinite(total) or total <= 0:
        return math.nan, math.nan
    freqs = np.arange(len(power)) * sr / n_fft
    bw99 = float(freqs[min(int(np.searchsorted(np.cumsum(power) / total, 0.99)), len(power) - 1)])
    smooth = np.convolve(power, np.ones(9) / 9, mode="same")
    level = 10 * np.log10(smooth + 1e-30)
    above = np.nonzero(level >= level.max() - 60.0)[0]
    return bw99, float(freqs[above[-1]])


def clipping(x: np.ndarray) -> tuple[float, float]:
    """``(fraction of samples with |x| >= 0.999, peak |x|)`` of a waveform in full-scale units."""
    mag = np.abs(np.asarray(x, dtype=np.float64))
    return float((mag >= CLIP_LEVEL).mean()), float(mag.max())


_WADA_DB = np.arange(-20, 101)
_WADA_G: np.ndarray | None = None


def wada_table() -> tuple[np.ndarray, np.ndarray]:
    """The WADA lookup ``G(SNR)`` for SNR = -20..100 dB, built from the model of the paper.

    Clean speech samples have a symmetric Gamma(0.4) amplitude distribution, the noise is Gaussian, and
    ``G = log E|z| - E log|z|`` of the sum ``z``. The table is computed once per process by Monte Carlo with a fixed
    seed (4e6 samples shared across all SNRs), then made monotone. It reproduces the published table (0.4097 at -20 dB,
    1.645 for noise-free speech) to about 1e-3.
    """
    global _WADA_G
    if _WADA_G is None:
        rng = np.random.default_rng(12345)
        n = 4_000_000
        s = rng.gamma(0.4, 1.0, n) * rng.choice([-1.0, 1.0], n)
        noise = rng.standard_normal(n)
        sig_power = 0.4 * 1.4
        g = np.empty(len(_WADA_DB))
        for i, db in enumerate(_WADA_DB):
            z = s + noise * math.sqrt(sig_power / 10 ** (db / 10))
            a = np.abs(z)
            g[i] = math.log(a.mean()) - np.log(a + 1e-10).mean()
        _WADA_G = np.maximum.accumulate(g)
    return _WADA_DB, _WADA_G


def wada_snr(x: np.ndarray) -> float:
    """WADA-SNR in dB (blind, waveform amplitude distribution analysis), clamped to [-20, 100]."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    peak = a.max() if len(a) else 0.0
    if peak <= 0:
        return math.nan
    a = a / peak
    g = math.log(max(a.mean(), 1e-10)) - np.log(a + 1e-10).mean()
    db, table = wada_table()
    return float(np.interp(g, table, db))


def read_audio(path: str, start_s: float | None = None, end_s: float | None = None) -> tuple[np.ndarray, int]:
    """Mono float32 waveform of a file (optionally the span ``[start_s, end_s)``) and its sample rate."""
    import soundfile as sf

    info = sf.info(path)
    sr = info.samplerate
    start = 0 if start_s is None or not np.isfinite(start_s) else max(0, int(round(start_s * sr)))
    stop = None if end_s is None or not np.isfinite(end_s) else int(round(end_s * sr))
    audio, _ = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def resample16(x: np.ndarray, sr: int) -> np.ndarray:
    import soxr

    if sr == MODEL_SR:
        return np.ascontiguousarray(x, dtype=np.float32)
    return soxr.resample(np.asarray(x, dtype=np.float32), sr, MODEL_SR, quality="HQ")


def cpu_measures(model_audio: np.ndarray, model_sr: int, native: np.ndarray, native_sr: int) -> dict:
    """Every measure that needs no network: returns the row fields plus the 16 kHz waveform under ``"_a16"``."""
    a16 = resample16(model_audio, model_sr)
    bw99, bw_floor = spectrum_measures(native, native_sr)
    clip, peak = clipping(native)
    rms = float(np.sqrt(np.mean(np.square(native.astype(np.float64))))) if len(native) else 0.0
    return {
        "duration_s": len(model_audio) / model_sr,
        "native_sr": int(native_sr),
        "upsampled": bool(native_sr < TARGET_SR),
        "bw99_hz": bw99,
        "bw_floor_hz": bw_floor,
        "clip_frac": clip,
        "peak": peak,
        "rms_dbfs": 20 * math.log10(rms) if rms > 0 else math.nan,
        "wada_snr": wada_snr(a16),
        "_a16": a16,
    }


# --------------------------------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------------------------------
def metrics_cfg():
    """``cfg.eval_metrics`` of the packaged default, as the Phase 4 evaluation uses it."""
    from omegaconf import OmegaConf

    default = importlib.resources.files("sparc.conf").joinpath("eval_metrics", "default.yaml")
    with importlib.resources.as_file(default) as path:
        return OmegaConf.load(path)


class Models:
    """UTMOS22 and Squim (one utterance per call) and Whisper large-v3 (length-sorted batches)."""

    def __init__(self, device: str, utmos: bool = True, squim: bool = True, whisper: bool = True):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        cfg = metrics_cfg()
        self.utmos = self.squim = self.asr = None
        if utmos:
            from sparc.vocoders.eval.metrics.utmos import UTMOS

            self.utmos = UTMOS(cfg.utmos, self.device)
        if squim:
            import torchaudio

            self.squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(self.device).eval()
        if whisper:
            from sparc.vocoders.eval.metrics.asr import WhisperASR

            self.asr = WhisperASR(cfg.asr, self.device)

    def provenance(self) -> dict:
        info: dict = {"torch": self.torch.__version__}
        if self.utmos is not None:
            info["utmos"] = self.utmos.provenance()
        if self.asr is not None:
            info["whisper"] = self.asr.provenance()
        if self.squim is not None:
            info["squim"] = "torchaudio.pipelines.SQUIM_OBJECTIVE"
        return info

    def score_squim(self, a16: np.ndarray) -> tuple[float, float, float]:
        """``(stoi, pesq, si_sdr)`` estimates of one 16 kHz waveform."""
        with self.torch.inference_mode():
            x = self.torch.from_numpy(np.ascontiguousarray(a16, dtype=np.float32))[None].to(self.device)
            stoi, pesq, si_sdr = self.squim(x)
        return float(stoi.item()), float(pesq.item()), float(si_sdr.item())


def apply_models(rows: list[dict], models: Models | None) -> None:
    """Fills ``utmos``, ``squim_*`` and the WER fields of the loaded rows in place (failures become NaN + ``error``)."""
    for r in rows:
        r.setdefault("error", "")
        for k in ("utmos", "squim_si_sdr", "squim_stoi", "squim_pesq"):
            r.setdefault(k, math.nan)
        if "_a16" not in r:
            continue
        if models is not None and models.utmos is not None:
            try:
                r["utmos"] = models.utmos.score_one(r["_a16"])
            except (ValueError, RuntimeError) as err:
                r["error"] += f"utmos: {err}; "
        if models is not None and models.squim is not None:
            try:
                r["squim_stoi"], r["squim_pesq"], r["squim_si_sdr"] = models.score_squim(r["_a16"])
            except (ValueError, RuntimeError) as err:
                r["error"] += f"squim: {err}; "
    todo = [i for i, r in enumerate(rows) if r.get("has_wer") and "_a16" in r]
    if models is not None and models.asr is not None and todo:
        from sparc.vocoders.eval.metrics.asr import edit_counts, normalize_text

        errors: list[str] = []
        hyps = models.asr.transcribe([rows[i]["_a16"] for i in todo], errors)
        for i, hyp, err in zip(todo, hyps, errors):
            r = rows[i]
            if err:
                r["error"] += f"whisper: {err}; "
                continue
            counts = edit_counts(normalize_text(r["_text"]), normalize_text(hyp))
            r.update(counts)
            r["hyp"] = hyp
            r["wer"] = counts["word_errors"] / counts["word_ref_len"] if counts["word_ref_len"] else math.nan
            r["cer"] = counts["char_errors"] / counts["char_ref_len"] if counts["char_ref_len"] else math.nan


# --------------------------------------------------------------------------------------------------------------------
# lists
# --------------------------------------------------------------------------------------------------------------------
def _read_text(path: str) -> str:
    from sparc.vocoders.eval.metrics.asr import read_reference

    try:
        return read_reference(path)
    except OSError:
        return ""


def make_list(corpus: str, workers: int = 16) -> pd.DataFrame:
    """Scoring list of one corpus (see the module docstring for the columns)."""
    if corpus in ("libritts_train", "libritts_test_clean"):
        splits = TRAIN_SPLITS if corpus == "libritts_train" else ("test.clean",)
        d = pd.concat([pd.read_parquet(PACKED / s / "index.parquet", columns=["id", "split", "speaker", "wav_path"]) for s in splits])
        d = d.drop_duplicates("id").reset_index(drop=True)
        with ThreadPoolExecutor(workers) as pool:
            text = list(pool.map(_read_text, d["wav_path"], chunksize=256))
        out = pd.DataFrame({
            "id": d["id"], "audio": d["wav_path"], "native_path": d["wav_path"], "corpus": corpus, "accent": "",
            "speaker": d["speaker"].astype(str), "text": text, "do_wer": True,
        })  # fmt: skip
    elif corpus == "vctk_eval":
        d = pd.read_parquet(VCTK_POOL)
        out = pd.DataFrame({
            "id": d["id"], "audio": d["wav_path"], "native_path": d["src"], "corpus": corpus, "accent": d["accent"],
            "speaker": d["speaker"], "text": d["text"], "do_wer": True,
        })  # fmt: skip
    elif corpus == "globe_train":
        d = pd.read_parquet(GLOBE_LIST)
        man = pd.read_csv(GLOBE_MANIFEST, sep="\t", usecols=["stem", "transcript"]).drop_duplicates("stem").set_index("stem")
        rng = np.random.default_rng(SEED_WER_SAMPLE)
        sample = np.zeros(len(d), dtype=bool)
        sample[rng.choice(len(d), WER_SAMPLE_SIZE, replace=False)] = True
        out = pd.DataFrame({
            "id": d["stem"], "audio": [str(GLOBE_WAV24 / s / f"{t}.wav") for s, t in zip(d["shard"], d["stem"])],
            "native_path": "", "corpus": corpus, "accent": d["accent"], "speaker": "", "do_wer": sample,
            "text": [man["transcript"].get(t, "") if smp else "" for t, smp in zip(d["stem"], sample)],
        })  # fmt: skip
        missing = [s for s in d["shard"].unique() if not (GLOBE_WAV24 / s).is_dir()]
        logger.info("GLOBE wav24 shard dirs missing: %d", len(missing))
    else:
        raise SystemExit(f"unknown corpus {corpus}")
    return out


def load_list(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, sep="\t")
    for col, default in (("native_path", ""), ("corpus", path.parent.name), ("accent", ""), ("speaker", ""), ("text", ""), ("do_wer", False)):
        if col not in d:
            d[col] = default
    for col in ("start_s", "end_s"):
        if col not in d:
            d[col] = np.nan
    d["native_path"] = d["native_path"].fillna("").astype(str)
    d["text"] = d["text"].fillna("").astype(str)
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------------------------------
def load_row(row: dict) -> dict:
    """CPU part of one list row: audio, native-rate measures, WADA. Failures keep the row with ``error``."""
    out = {
        "id": row["id"], "corpus": row["corpus"], "accent": row["accent"], "speaker": row["speaker"],
        "has_wer": False, "hyp": "", "error": "", "native_unavailable": not row["native_path"],
    }  # fmt: skip
    try:
        audio, sr = read_audio(row["audio"])
        if row["native_path"]:
            native, nsr = read_audio(row["native_path"], row.get("start_s"), row.get("end_s"))
        else:
            native, nsr = audio, sr
        if len(audio) < 400 or len(native) < N_FFT // 4:
            raise ValueError(f"audio too short ({len(audio)} samples)")
        out.update(cpu_measures(audio, sr, native, nsr))
        if bool(row.get("do_wer")) and row["text"].strip():
            out["has_wer"] = True
            out["_text"] = row["text"]
    except Exception as err:  # noqa: BLE001 - a bad file must not stop a 200k-utterance shard
        out["error"] = f"load: {type(err).__name__}: {err}; "
    return out


def finish_rows(rows: list[dict]) -> pd.DataFrame:
    for r in rows:
        r.pop("_a16", None)
        r.pop("_text", None)
    df = pd.DataFrame(rows)
    for col in SCORE_COLUMNS:
        if col not in df:
            df[col] = math.nan if col not in ("hyp", "error", "id", "corpus", "accent", "speaker") else ""
    df["has_wer"] = df["has_wer"].fillna(False).astype(bool)
    df["native_unavailable"] = df["native_unavailable"].fillna(True).astype(bool)
    df["upsampled"] = df["upsampled"].fillna(False).astype(bool)
    for col in ("native_sr", "word_errors", "word_ref_len", "char_errors", "char_ref_len"):
        df[col] = df[col].astype("float64")
    return df[SCORE_COLUMNS]


def list_digest(d: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(d["id"].astype(str)).encode()).hexdigest()[:16]


def chunk_path(scores: Path, k: int) -> Path:
    return scores / f"chunk-{k:05d}.parquet"


def chunk_done(path: Path, ids: list[str]) -> bool:
    """A chunk file counts only if it holds exactly the list's ids of that chunk (a stale list invalidates it)."""
    if not path.is_file():
        return False
    try:
        got = pd.read_parquet(path, columns=["id"])["id"].tolist()
    except Exception:  # noqa: BLE001 - truncated or corrupt file
        return False
    return got == ids


def write_chunk(df: pd.DataFrame, path: Path, meta: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    table = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata({"quality": json.dumps(meta)})
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out) if args.out else ROOT / args.corpus
    list_path = Path(args.list) if args.list else ROOT / args.corpus / "list.parquet"
    d = load_list(list_path)
    if args.limit:
        d = d.sample(n=min(args.limit, len(d)), random_state=0).sort_index().reset_index(drop=True)
    scores = out_dir / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(len(d) / args.chunk)
    mine = [k for k in range(n_chunks) if k % args.nshards == args.shard]
    ids = {k: d["id"].iloc[k * args.chunk : (k + 1) * args.chunk].astype(str).tolist() for k in mine}
    todo = [k for k in mine if not chunk_done(chunk_path(scores, k), ids[k])]
    logger.info("%s: %d rows, %d chunks, shard %d/%d owns %d, %d to do", args.corpus, len(d), n_chunks, args.shard, args.nshards, len(mine), len(todo))
    if not todo:
        return 0
    stop = {"flag": False}
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("flag", True))
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    wer_needed = bool(d["do_wer"].astype(bool).any())
    models = None if args.no_models else Models("cuda" if args.device == "auto" else args.device, whisper=wer_needed)
    meta = {"list": str(list_path), "list_digest": list_digest(d), "models": models.provenance() if models else {},
            "wada_table_G0": float(wada_table()[1][20]), "chunk": args.chunk}  # fmt: skip

    def load_chunk(k: int) -> list[dict]:
        rows = d.iloc[k * args.chunk : (k + 1) * args.chunk].to_dict("records")
        return list(pool.map(load_row, rows))

    t_start = time.time()
    done_rows = 0
    with ThreadPoolExecutor(args.workers) as pool, ThreadPoolExecutor(1) as prefetch:
        future = prefetch.submit(load_chunk, todo[0])
        for n, k in enumerate(todo):
            rows = future.result()
            if n + 1 < len(todo):
                future = prefetch.submit(load_chunk, todo[n + 1])
            t0 = time.time()
            apply_models(rows, models)
            df = finish_rows(rows)
            write_chunk(df, chunk_path(scores, k), {**meta, "gpu_seconds": time.time() - t0})
            done_rows += len(df)
            rate = done_rows / (time.time() - t_start)
            logger.info("chunk %d done (%d rows, %d errors); %.1f utt/s, %d chunks left", k, len(df), int((df["error"] != "").sum()), rate, len(todo) - n - 1)
            if n + 1 < len(todo) and (stop["flag"] or (deadline and time.time() > deadline)):
                future.cancel()
                logger.info("stopping with %d chunks left", len(todo) - n - 1)
                return EXIT_STOPPED
    return 0


def load_scores(corpus_dir: Path) -> pd.DataFrame:
    files = sorted((corpus_dir / "scores").glob("chunk-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=SCORE_COLUMNS)


# --------------------------------------------------------------------------------------------------------------------
# calibration report
# --------------------------------------------------------------------------------------------------------------------
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def anchor_thresholds(anchor: pd.DataFrame, measures: dict[str, bool]) -> dict[str, dict[int, float]]:
    """Per measure and percentile p: the p-th percentile (higher-better) or the (100-p)-th (lower-better) of an anchor."""
    out: dict[str, dict[int, float]] = {}
    for m, higher in measures.items():
        v = anchor[m].dropna()
        if len(v) == 0:
            continue
        out[m] = {p: float(np.quantile(v, p / 100 if higher else 1 - p / 100)) for p in ANCHOR_PERCENTILES}
    return out


def passes(df: pd.DataFrame, m: str, thr: float, higher: bool) -> pd.Series:
    """NaN measures fail (a row with a failed measure is not retained)."""
    return (df[m] >= thr) if higher else (df[m] <= thr)


def retention(df: pd.DataFrame, thresholds: dict[str, dict[int, float]], measures: dict[str, bool], p: int) -> dict:
    """Retained utterances and hours at percentile ``p``: each measure alone and the joint over all of them."""
    out: dict = {"n": int(len(df)), "hours": float(df["duration_s"].sum() / 3600)}
    joint = pd.Series(True, index=df.index)
    for m, higher in measures.items():
        if m not in thresholds or m not in df or df[m].notna().sum() == 0:
            continue
        ok = passes(df, m, thresholds[m][p], higher)
        joint &= ok
        out[m] = {"kept": int(ok.sum()), "frac": float(ok.mean()), "hours": float(df.loc[ok, "duration_s"].sum() / 3600)}
    out["joint"] = {"kept": int(joint.sum()), "frac": float(joint.mean()), "hours": float(df.loc[joint, "duration_s"].sum() / 3600)}
    return out


def group_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The whole corpus plus one frame per accent label if there are at least two."""
    groups = {"all": df}
    accents = [a for a in df["accent"].unique() if a]
    if len(accents) >= 2:
        for a in sorted(accents):
            groups[f"accent={a}"] = df[df["accent"] == a]
    return groups


def build_report(root: Path) -> dict:
    corpora = {p.name: load_scores(p) for p in sorted(root.iterdir()) if (p / "scores").is_dir()}
    corpora = {k: v for k, v in corpora.items() if len(v)}
    clean = {k: v[v["error"] == ""] if len(v) else v for k, v in corpora.items()}
    measures = dict(MEASURES)
    report: dict = {"corpora": {}, "anchors": {}, "errors": {}, "wada_table_G_at_0dB": float(wada_table()[1][20])}
    for name, df in corpora.items():
        report["errors"][name] = {"rows": int(len(df)), "with_error": int((df["error"] != "").sum()),
                                  "native_unavailable": int(df["native_unavailable"].sum()), "upsampled": int(df["upsampled"].sum())}  # fmt: skip
    thresholds = {a: anchor_thresholds(clean[a], {**measures, **TEXT_MEASURES}) for a in ANCHORS if a in clean}
    report["anchors"] = {a: {m: {str(p): v for p, v in t.items()} for m, t in th.items()} for a, th in thresholds.items()}
    for name, df in clean.items():
        entry: dict = {"quantiles": {}, "retention": {}, "accent_loss_flag": {}}
        for gname, g in group_frames(df).items():
            entry["quantiles"][gname] = {"n": int(len(g)), "hours": float(g["duration_s"].sum() / 3600)}
            for m in [*measures, "bw_floor_hz", "duration_s", *TEXT_MEASURES]:
                v = g.loc[g["has_wer"], m].dropna() if m in TEXT_MEASURES else g[m].dropna()
                if len(v):
                    entry["quantiles"][gname][m] = {f"q{int(q * 100):02d}": float(np.quantile(v, q)) for q in QUANTILES} | {"mean": float(v.mean())}
        for a, th in thresholds.items():
            for p in ANCHOR_PERCENTILES:
                res = {gname: retention(g, th, measures, p) for gname, g in group_frames(df).items()}
                entry["retention"][f"{a}@p{p}"] = res
                entry["accent_loss_flag"][f"{a}@p{p}"] = sorted(
                    gname for gname, r in res.items() if gname != "all" and r["joint"]["frac"] < 0.5
                )
        report["corpora"][name] = entry
    return report


def _fmt(x: float) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.4g}" if abs(x) < 1000 else f"{x:.0f}")


def render_markdown(rep: dict) -> str:
    lines = ["# Phase 5 quality calibration report", "",
             "Generated by `python -m sparc.vocoders.features.quality report`. Thresholds are NOT fixed here: candidates only.",
             "Reverberation measure: Squim STOI estimate (proxy; Brouhaha C50 not installed, see the module docstring).",
             f"WADA table check: G at 0 dB = {rep['wada_table_G_at_0dB']:.4f} (published 0.4622 (and 0.6675 at 10 dB, 0.9566 at 20 dB)).", "",
             "## Rows and errors", "", "| corpus | rows | with error | native unavailable | upsampled (native < 24 kHz) |", "|---|---|---|---|---|"]  # fmt: skip
    for name, e in rep["errors"].items():
        lines.append(f"| {name} | {e['rows']} | {e['with_error']} | {e['native_unavailable']} | {e['upsampled']} |")
    lines += ["", "## Anchor thresholds (p = percentile of the anchor; lower-better measures use the mirrored percentile)", ""]
    for a, th in rep["anchors"].items():
        lines += [f"### {a}", "", "| measure | p5 | p10 | p25 |", "|---|---|---|---|"]
        for m, t in th.items():
            lines.append(f"| {m} | {_fmt(t['5'])} | {_fmt(t['10'])} | {_fmt(t['25'])} |")
        lines.append("")
    show = ["wada_snr", "squim_si_sdr", "squim_stoi", "utmos", "squim_pesq", "bw99_hz", "bw_floor_hz", "clip_frac", "wer", "cer"]
    lines += ["## Quantiles per corpus and accent", ""]
    for name, entry in rep["corpora"].items():
        lines += [f"### {name}", "", "| group | n | hours | measure | q05 | q25 | median | q75 | q95 | mean |", "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for gname, q in entry["quantiles"].items():
            for m in show:
                if m in q:
                    s = q[m]
                    lines.append(f"| {gname} | {q['n']} | {q['hours']:.1f} | {m} | {_fmt(s['q05'])} | {_fmt(s['q25'])} | {_fmt(s['q50'])} | {_fmt(s['q75'])} | {_fmt(s['q95'])} | {_fmt(s['mean'])} |")
        lines.append("")
    lines += ["## Retained utterances and hours at candidate thresholds (joint = every measure above at once)", ""]
    for name, entry in rep["corpora"].items():
        lines += [f"### {name}", "", "| anchor@p | group | n | hours | joint kept | joint frac | joint hours |", "|---|---|---|---|---|---|---|"]
        for key, res in entry["retention"].items():
            for gname, r in res.items():
                lines.append(f"| {key} | {gname} | {r['n']} | {r['hours']:.1f} | {r['joint']['kept']} | {r['joint']['frac']:.3f} | {r['joint']['hours']:.1f} |")
        flags = {k: v for k, v in entry["accent_loss_flag"].items() if v}
        lines += ["", "Accent-loss flag (joint retention below 50 %): " + (json.dumps(flags) if flags else "none"), ""]
        lines += ["Single-measure retention (fraction kept), group all:", "", "| anchor@p | " + " | ".join(MEASURES) + " |", "|---|" + "---|" * len(MEASURES)]
        for key, res in entry["retention"].items():
            r = res["all"]
            lines.append(f"| {key} | " + " | ".join(_fmt(r[m]["frac"]) if m in r else "-" for m in MEASURES) + " |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-list")
    p.add_argument("--corpus", required=True)
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("run")
    p.add_argument("--corpus", required=True)
    p.add_argument("--list", default=None, help="list parquet/tsv (default <ROOT>/<corpus>/list.parquet)")
    p.add_argument("--out", default=None, help="output directory (default <ROOT>/<corpus>)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-minutes", type=float, default=0)
    p.add_argument("--limit", type=int, default=0, help="score a random subset of this many rows (timing runs)")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-models", action="store_true", help="signal measures only")
    p = sub.add_parser("report")
    p.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    if args.cmd == "make-list":
        out = make_list(args.corpus, args.workers)
        path = ROOT / args.corpus / "list.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        out.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        logger.info("wrote %s (%d rows, %d with WER text)", path, len(out), int(out["do_wer"].sum()))
        return 0
    if args.cmd == "run":
        return run(args)
    root = Path(args.root)
    rep = build_report(root)
    (root / "report.json").write_text(json.dumps(rep, indent=1))
    (root / "report.md").write_text(render_markdown(rep))
    logger.info("wrote %s/report.{md,json}", root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
