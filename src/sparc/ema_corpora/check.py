"""End-to-end checks on preprocessed outputs.

  orientation -- per speaker, angle of the lip line (LL->UL; 0 = UL straight above LL) and tongue line
                 (TD->TT; above horizontal) from the output EMA: all speakers should share x = anterior, y = up
  sync        -- model-based lag (sync.peak_lag with SPARC predictions on the output 16 kHz audio) against the
                 output 50 Hz EMA; after the per-speaker latency correction it should be ~0 for every speaker
  stats       -- per normalization group: utterances, hours, valid-frame fraction, channel std (mm)

Usage: python -m sparc.ema_corpora.check [--per-speaker N]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .dataset import load_manifest, load_stats, load_utterance
from .preprocess import FRAME_CENTER, OUT_ROOT
from .sync import peak_lag


def orientation(rows):
    out = {}
    by = {}
    for r in rows:
        by.setdefault(r["speaker"], []).append(r)
    for spk, rs in by.items():
        meds = []
        for r in rs[:: max(1, len(rs) // 60)]:
            e = load_utterance(r, normalize=False)["ema_mm"].reshape(-1, 6, 2)
            meds.append(np.nanmedian(e, 0))
        m = np.nanmedian(np.stack(meds), 0)  # (6, 2): TD TB TT LI UL LL
        lip, tng = m[4] - m[5], m[2] - m[0]
        out[spk] = {"lip_deg": round(float(np.degrees(np.arctan2(lip[0], lip[1]))), 1),
                    "tongue_deg": round(float(np.degrees(np.arctan2(tng[1], tng[0]))), 1)}
    return out


def sync(rows, per_speaker, min_corr=0.3):
    from sparc import load_model

    coder = load_model("en", device="cpu")
    rng = random.Random(0)
    by = {}
    for r in rows:
        if r["duration"] >= 2.0:
            by.setdefault(r["speaker"], []).append(r)
    out = {}
    for spk, rs in by.items():
        lags = []
        for r in rng.sample(rs, min(per_speaker, len(rs))):
            pred = coder.inverter(str(Path(OUT_ROOT) / r["audio"]))["ema"][0].reshape(-1, 6, 2)
            u = load_utterance(r, normalize=False)
            E = u["ema_mm"].reshape(-1, 6, 2)
            meas = {"TD": E[:, 0, 1], "TB": E[:, 1, 1], "TT": E[:, 2, 1], "JAW": E[:, 3, 1], "LA": E[:, 4, 1] - E[:, 5, 1]}
            pv = {"TD": pred[:, 0, 1], "TB": pred[:, 1, 1], "TT": pred[:, 2, 1], "JAW": pred[:, 3, 1],
                  "LA": pred[:, 4, 1] - pred[:, 5, 1]}
            tm = np.arange(len(E)) / 50 + FRAME_CENTER
            tp = np.arange(len(pred)) * 0.02
            for ch in pv:
                res = peak_lag(tm, meas[ch], tp, pv[ch])
                if res and res[1] >= min_corr:
                    lags.append(res[0])
        out[spk] = {"median_lag_ms": round(1000 * float(np.median(lags)), 1) if lags else None, "n": len(lags)}
        print(spk, out[spk], flush=True)
    return out


def stats_summary(rows):
    st = load_stats()
    out = {}
    for g, s in st.items():
        rs = [r for r in rows if r["norm_group"] == g]
        out[g] = {"utterances": len(rs), "hours": round(sum(r["duration"] for r in rs) / 3600, 3),
                  "valid_frac": round(float(np.mean([r["valid_frac"] for r in rs])), 3),
                  "std_mm": [round(x, 2) for x in s["std"]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=20)
    ap.add_argument("--skip-sync", action="store_true")
    args = ap.parse_args()
    rows = load_manifest()
    res = {"orientation": orientation(rows), "stats": stats_summary(rows)}
    print(json.dumps(res, indent=1))
    if not args.skip_sync:
        res["sync_after_correction"] = sync(rows, args.per_speaker)
    (Path(OUT_ROOT) / "checks.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
