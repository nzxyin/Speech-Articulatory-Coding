"""Audio/EMA synchronization checks for the mview corpora.

Two independent estimates of the lag between the EMA and audio clocks of each speaker:

  model:     SPARC 'en' inversion (WavLM + linear, 50 Hz, trained on MNGU0) predicts EMA from the audio; for each
             signed vertical channel (jaw, tongue dorsum/body/tip heights, vertical lip aperture) the lag that
             maximizes the correlation between measured and predicted trajectories is found (parabolic peak
             refinement, pairs with peak correlation >= 0.3 kept); the per-speaker median is reported.
  bilabial:  USC-TIMIT only: lag that maximizes the correlation between minus the 3-D lip aperture and an
             indicator of /p b m/ intervals in the .trans phone alignment.

Lag convention: measured EMA at EMA-clock time t + L corresponds to audio time t; L < 0 means EMA events are
recorded earlier than the audio implies, and preprocess.LAG_S = -L delays the EMA to align it. The model-based
lags define zero by SPARC's (MNGU0-derived) timing convention, so between-speaker differences are robust while
the absolute zero is that convention. Results (2026-10-07, 15-25 files per speaker):

    model:    usc_M1 -22.1, usc_F1 -21.7, usc_F5 -23.0, usc_M3 -23.0, 5emo_jn -25.8, 5emo_jr +1.9, 5emo_kf +8.4 ms
    bilabial: usc_M1 -35, usc_F1 -35, usc_F5 -30 ms (M3's transcripts belong to its MRI session: no peak)

Usage: python -m sparc.ema_corpora.sync [--per-speaker N] [--out results.json]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import corpora
from .mview import read_mview
from .preprocess import rotate_xy

LAGS = np.arange(-0.12, 0.1201, 0.005)
CACHE = Path("/data/user_data/xoy/ema_corpora/_sync_cache")


def _sparc_prediction(rec, cache_path, coder):
    if cache_path.exists():
        return np.load(cache_path)
    import soundfile as sf

    a16 = resample_poly(rec.audio, 320, 441).astype(np.float32) if rec.audio_sr == 22050 else \
        resample_poly(rec.audio, 16000, int(rec.audio_sr)).astype(np.float32)
    tmp = cache_path.with_suffix(".tmp.wav")
    sf.write(tmp, a16 / (np.abs(a16).max() + 1e-9) * 0.9, 16000)
    pred = coder.inverter(str(tmp))["ema"][0]
    tmp.unlink()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pred)
    return pred


def peak_lag(t_meas, y_meas, t_pred, y_pred, lags=LAGS):
    """(lag, peak corr) maximizing corr(y_meas(t + lag), y_pred(t)), parabolic refinement of the peak."""
    ok = ~np.isnan(y_meas)
    if ok.sum() < 50:
        return None
    cs = []
    for L in lags:
        yi = np.interp(t_pred + L, t_meas[ok], y_meas[ok], left=np.nan, right=np.nan)
        v = ~np.isnan(yi)
        cs.append(np.corrcoef(yi[v], y_pred[v])[0, 1] if v.sum() > 50 else np.nan)
    cs = np.array(cs)
    if np.isnan(cs).all():
        return None
    i = int(np.nanargmax(cs))
    lag = lags[i]
    if 0 < i < len(cs) - 1 and not np.isnan(cs[i - 1:i + 2]).any():
        a, b, c = cs[i - 1], cs[i], cs[i + 1]
        den = a - 2 * b + c
        if den < 0:
            lag += 0.5 * (a - c) / den * (lags[1] - lags[0])
    return float(lag), float(cs[i])


def model_lags(mat, speaker, coder):
    rec = read_mview(mat)
    pred = _sparc_prediction(rec, CACHE / (Path(mat).stem + ".npy"), coder)
    P = pred.reshape(len(pred), 6, 2)  # SPARC order TD TB TT LI UL LL, y = up
    pv = {"TD": P[:, 0, 1], "TB": P[:, 1, 1], "TT": P[:, 2, 1], "JAW": P[:, 3, 1], "LA": P[:, 4, 1] - P[:, 5, 1]}
    E = {k: rotate_xy(v, corpora.rotation_deg(speaker)) for k, v in rec.ema.items()}
    mv = {"TD": E["TD"][:, 1], "TB": E["TB"][:, 1], "TT": E["TT"][:, 1], "JAW": E["JAW"][:, 1],
          "LA": E["UL"][:, 1] - E["LL"][:, 1]}
    te = np.arange(len(mv["TD"])) / rec.ema_sr
    tp = np.arange(len(pred)) * 0.02
    return {ch: r for ch in pv if (r := peak_lag(te, mv[ch], tp, pv[ch])) is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=20)
    ap.add_argument("--min-corr", type=float, default=0.3)
    ap.add_argument("--out", default=str(CACHE.parent / "sync_results.json"))
    args = ap.parse_args()
    from sparc import load_model

    coder = load_model("en", device="cpu")
    rng = random.Random(0)
    files = {}
    for spk in corpora.USC_SPEAKERS:
        files[f"usc_{spk}"] = [m for m, *_ in corpora.usc_files(spk)]
    for spk in corpora.EMO_SPEAKERS:
        files[f"5emo_{spk}"] = sorted((corpora.EMO_ROOT / spk / "wav_mat").glob("ema_5emo_*sent*.mat"))
    out = {}
    for speaker, mats in files.items():
        lags = []
        for m in rng.sample(mats, min(args.per_speaker, len(mats))):
            lags += [lag for lag, c in model_lags(m, speaker, coder).values() if c >= args.min_corr]
        out[speaker] = {"median_lag_ms": round(1000 * float(np.median(lags)), 1),
                        "iqr_ms": [round(1000 * float(np.percentile(lags, q)), 1) for q in (25, 75)], "n": len(lags)}
        print(speaker, out[speaker], flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
