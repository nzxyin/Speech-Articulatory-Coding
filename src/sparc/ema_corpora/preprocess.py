"""Preprocess USC-TIMIT EMA and USC EMA_5EMO into audio-aligned utterances at 50 Hz.

Per utterance (see corpora.py for segmentation and the per-speaker frame facts):
  1. Audio: the embedded AUDIO track (identical to / a gain-scaled copy of the .wav) cut to the segment and
     resampled 22050 -> 16000 Hz.
  2. EMA clock: sample i of a file is at i / SRATE from the start of the audio (start-aligned; SRATE is the
     file's own rate, 97.5-100.08 Hz), plus an optional per-speaker latency correction (LAG_S).
  3. Frame: the 5emo_jn frame is rotated about the lateral axis by corpora.rotation_deg so that x points
     anterior and y up, like the other speakers (all others are used as recorded). No other cross-speaker
     transform is applied.
  4. Dropouts (NaN): gaps up to MAX_GAP_S are linearly interpolated; longer gaps stay NaN and are flagged in
     `valid`.
  5. 20 Hz zero-phase Butterworth low-pass (order 5, per contiguous valid run), as in EMA_5EMO's own
     post-processing; also the anti-aliasing filter for the 50 Hz grid.
  6. Sampling at the centres of 50 Hz SSL frames: frame j is at segment time 0.02 j + 0.0125 s, matching
     a 20 ms-hop / 25 ms-window encoder (WavLM, XLS-R) run on the 16 kHz segment.

Outputs under <out>/<corpus>/<speaker>/: <utt>.wav (16 kHz PCM_16) and <utt>.npz with
    ema       (T, 12) float32 mm, channels mngu0.CHANNELS order: TD, TB, TT, LI(jaw), UL, LL x (anterior +), y (up +)
    lateral   (T, 6)  float32 mm, lateral coordinate of each sensor (same sensor order)
    valid     (T, 6)  bool, False where a sensor was in a dropout longer than MAX_GAP_S
    native    (N, 6, 3) float32 mm, the segment at the native EMA rate after rotation, before filtering (NaN kept)
    native_sr, native_t0 (time of native[0] relative to the segment start)
plus <out>/manifest.csv (one row per utterance) and <out>/stats.json (per normalization group, per channel
mean / std over valid frames: standard per-articulator normalization, no cross-speaker transform).

Usage: python -m sparc.ema_corpora.preprocess [--out DIR] [--corpora usc_timit ema_5emo] [--workers N]
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from . import corpora
from .mview import read_mview

OUT_ROOT = Path("/data/user_data/xoy/ema_corpora")
SENSOR_ORDER = ("TD", "TB", "TT", "JAW", "UL", "LL")  # = mngu0.ARTICULATORS (JAW is LI)
CHANNELS = ("TD_x", "TD_y", "TB_x", "TB_y", "TT_x", "TT_y", "LI_x", "LI_y", "UL_x", "UL_y", "LL_x", "LL_y")
FT_SR = 50
FRAME_CENTER = 0.0125
MAX_GAP_S = 0.1
LOWPASS_HZ = 20.0
OUT_SR = 16000
# Per-speaker latency correction added to the EMA clock (seconds): minus the model-based lag measured by
# sync.py (EMA events are recorded that much earlier than the audio implies). Zero is SPARC's MNGU0-derived
# timing convention, the one the rest of this repository uses; the between-speaker differences (~30 ms
# between 5emo_jn and 5emo_jr / 5emo_kf) are what a single global shift could not fix. --no-lag disables it.
LAG_S = {"usc_M1": 0.0221, "usc_F1": 0.0217, "usc_F5": 0.0230, "usc_M3": 0.0230,
         "5emo_jn": 0.0258, "5emo_jr": -0.0019, "5emo_kf": -0.0084}


@lru_cache(maxsize=8)
def _record(mat):
    return read_mview(mat)


def read_audio(mat):
    r = _record(Path(mat))
    return r.audio, r.audio_sr


def rotate_xy(xyz, deg):
    if not deg:
        return xyz
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    out = xyz.copy()
    out[..., 0] = c * xyz[..., 0] - s * xyz[..., 1]
    out[..., 1] = s * xyz[..., 0] + c * xyz[..., 1]
    return out


def fill_short_gaps(x, max_gap):
    """Linearly interpolate NaN runs of at most max_gap samples (per column); returns (filled, still_nan)."""
    x = x.copy()
    bad = np.isnan(x).any(axis=1)
    n = len(x)
    i = 0
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                for c in range(x.shape[1]):
                    x[i:j, c] = np.interp(np.arange(i, j), [i - 1, j], [x[i - 1, c], x[j, c]])
                bad[i:j] = False
            i = j
        else:
            i += 1
    return x, bad


def lowpass_runs(x, bad, sr, cut=LOWPASS_HZ, order=5):
    """Zero-phase low-pass over each contiguous run of valid samples (runs too short to filter are kept)."""
    sos = butter(order, cut, fs=sr, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    y = x.copy()
    n = len(x)
    i = 0
    while i < n:
        if not bad[i]:
            j = i
            while j < n and not bad[j]:
                j += 1
            if j - i > padlen:
                y[i:j] = sosfiltfilt(sos, x[i:j], axis=0)
            i = j
        else:
            i += 1
    return y


def process_segment(seg, out_dir):
    rec = _record(Path(seg.mat))
    sr = rec.ema_sr
    lag = LAG_S.get(seg.speaker, 0.0)
    deg = corpora.rotation_deg(seg.speaker)
    start, end = seg.start, seg.end if seg.end is not None else rec.audio_dur
    end = min(end, rec.audio_dur)

    # audio
    a0, a1 = int(round(start * rec.audio_sr)), int(round(end * rec.audio_sr))
    audio = resample_poly(rec.audio[a0:a1], 320, 441) if rec.audio_sr == 22050 else \
        resample_poly(rec.audio[a0:a1], OUT_SR, int(rec.audio_sr))
    peak = np.abs(audio).max()
    if peak > 1:
        audio = audio / peak * 0.999

    # EMA: whole file, rotated, gap-filled, low-passed, then sampled on the segment's frame grid
    raw = np.stack([rec.ema[s] for s in SENSOR_ORDER], 1)  # (N, 6, 3)
    raw = rotate_xy(raw, deg)
    t_native = np.arange(len(raw)) / sr + lag
    nan_frac = np.isnan(raw).any(axis=2).mean(0)
    filled, bad = [], []
    for k in range(6):
        f, b = fill_short_gaps(raw[:, k], int(round(MAX_GAP_S * sr)))
        filled.append(lowpass_runs(f, b, sr))
        bad.append(b)
    filled, bad = np.stack(filled, 1), np.stack(bad, 1)  # (N, 6, 3), (N, 6)

    n_frames = int(np.floor((end - start) * FT_SR))
    tf = start + np.arange(n_frames) / FT_SR + FRAME_CENTER
    ema = np.full((n_frames, 6, 3), np.nan)
    valid = np.zeros((n_frames, 6), bool)
    for k in range(6):
        for c in range(3):
            ema[:, k, c] = np.interp(tf, t_native, filled[:, k, c], left=np.nan, right=np.nan)
        # a frame is valid if its two neighbouring native samples are valid
        idx = np.searchsorted(t_native, tf)
        lo, hi = np.clip(idx - 1, 0, len(t_native) - 1), np.clip(idx, 0, len(t_native) - 1)
        valid[:, k] = ~bad[lo, k] & ~bad[hi, k] & (tf >= t_native[0]) & (tf <= t_native[-1])
    ema[~valid] = np.nan

    sel = (t_native >= start) & (t_native < end)
    native = raw[sel]
    native_t0 = float(t_native[sel][0] - start) if sel.any() else 0.0

    d = out_dir / seg.corpus / seg.speaker
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / f"{seg.utt_id}.wav", audio.astype(np.float32), OUT_SR, subtype="PCM_16")
    np.savez_compressed(d / f"{seg.utt_id}.npz", ema=ema[:, :, :2].reshape(n_frames, 12).astype(np.float32),
                        lateral=ema[:, :, 2].astype(np.float32), valid=valid, native=native.astype(np.float32),
                        native_sr=np.float64(sr), native_t0=np.float64(native_t0))
    row = {"utt_id": seg.utt_id, "corpus": seg.corpus, "speaker": seg.speaker, "norm_group": seg.norm_group,
           "text": seg.text, "source": str(seg.mat), "start": round(start, 4), "end": round(end, 4),
           "duration": round(end - start, 4), "n_frames": n_frames, "ema_sr": round(sr, 5),
           "lag_s": lag, "rotation_deg": deg,
           "audio": str((d / f"{seg.utt_id}.wav").relative_to(out_dir)),
           "ema": str((d / f"{seg.utt_id}.npz").relative_to(out_dir)),
           "valid_frac": round(float(valid.all(axis=1).mean()), 4) if n_frames else 0.0,
           **{f"nan_frac_{s}": round(float(v), 4) for s, v in zip(SENSOR_ORDER, nan_frac)}}
    row.update({k: v for k, v in seg.meta.items()})
    medians = np.nanmedian(ema, axis=0) if valid.any() else np.full((6, 3), np.nan)
    return row, medians


def _job(args):
    seg, out_dir = args
    try:
        return process_segment(seg, out_dir)
    except Exception as e:  # report and continue: one broken file should not stop the corpus
        print(f"FAILED {seg.utt_id}: {e!r}", flush=True)
        return None


def collect_segments(which):
    segs = []
    if "usc_timit" in which:
        for spk in corpora.USC_SPEAKERS:
            segs += list(corpora.usc_segments(spk, read_audio))
    if "ema_5emo" in which:
        segs += list(corpora.emo_segments(read_audio))
    return segs


def frame_offsets(rows, medians):
    """EMA_5EMO: common-mode (all-sensor) offset of each utterance from the median of the same speaker's
    utterances of the same prompt, in mm (median over sensors of the per-sensor median difference)."""
    out = {}
    groups = {}
    for i, r in enumerate(rows):
        if r["corpus"] == "ema_5emo":
            groups.setdefault((r["speaker"], r.get("prompt")), []).append(i)
    for idx in groups.values():
        ref = np.nanmedian(np.stack([medians[i] for i in idx]), 0)
        for i in idx:
            common = np.nanmedian(medians[i] - ref, axis=0)
            out[i] = common
    return out


def norm_stats(rows, out_dir):
    acc = {}
    for r in rows:
        z = np.load(out_dir / r["ema"])
        e = z["ema"].astype(np.float64)
        ok = ~np.isnan(e)
        a = acc.setdefault(r["norm_group"], [np.zeros(12), np.zeros(12), np.zeros(12)])
        a[0] += np.where(ok, e, 0).sum(0)
        a[1] += np.where(ok, e * e, 0).sum(0)
        a[2] += ok.sum(0)
    stats = {}
    for g, (s, q, n) in acc.items():
        mean = s / n
        stats[g] = {"channels": list(CHANNELS), "mean": mean.round(5).tolist(),
                    "std": np.sqrt(q / n - mean**2).round(5).tolist(), "n_frames": n.astype(int).tolist()}
    return stats


def usc_split(sentence_id):
    """Text-disjoint USC-TIMIT split: the same sentences are held out for every speaker."""
    if sentence_id % 10 == 0:
        return "test"
    if sentence_id % 10 == 5:
        return "valid"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--corpora", nargs="+", default=["usc_timit", "ema_5emo"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-lag", action="store_true", help="do not apply the per-speaker latency correction")
    args = ap.parse_args()
    if args.no_lag:
        LAG_S.clear()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    segs = collect_segments(args.corpora)
    print(f"{len(segs)} utterances", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(_job, [(s, out_dir) for s in segs], chunksize=8))
    ok = [r for r in results if r is not None]
    rows, medians = [r[0] for r in ok], [r[1] for r in ok]
    offs = frame_offsets(rows, medians)
    for i, r in enumerate(rows):
        if i in offs:
            c = offs[i]
            r.update({"frame_offset_mm": round(float(np.linalg.norm(c)), 3), "frame_offset_x": round(float(c[0]), 3),
                      "frame_offset_y": round(float(c[1]), 3), "frame_offset_lat": round(float(c[2]), 3)})
        r["split"] = usc_split(r["sentence_id"]) if r["corpus"] == "usc_timit" else ""
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    stats = norm_stats(rows, out_dir)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {len(rows)} utterances ({len(segs) - len(rows)} failed) to {out_dir}")


if __name__ == "__main__":
    main()
