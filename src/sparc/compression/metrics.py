"""EMA prediction metrics. All inputs are lists of per-utterance (T, 12) arrays in mm at 50 Hz.

Conventions:
    rmse / mae     -- per channel, pooled over all test frames; "overall" is the mean over channels
    pcc            -- per channel, computed within each utterance then averaged (as in the SPARC paper)
    pcc_pooled     -- per channel over the concatenated test set (as in earlier MNGU0 inversion work)
    vel_rmse       -- RMSE of first differences, in mm/s
    articulator    -- Euclidean (x, y) error per sensor, pooled
    *_ci           -- 95% bootstrap interval over test utterances
"""

import numpy as np

from .mngu0 import ARTICULATORS, CHANNELS, FT_SR

PHONE_CLASSES = {
    "labial": {"p", "b", "m", "m!", "f", "v", "w"},
    "coronal": {"t", "d", "n", "n!", "s", "z", "l", "l!", "lw", "T", "D", "S", "Z", "tS", "dZ", "r"},
    "dorsal": {"k", "g", "N", "j"},
    "glottal": {"h"},
    "silence": {"#"},
}


def phone_class(label):
    for name, members in PHONE_CLASSES.items():
        if label in members:
            return name
    return "vowel"


def _pcc(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _overall(preds, trues):
    P, Y = np.concatenate(preds), np.concatenate(trues)
    rmse = np.sqrt(((P - Y) ** 2).mean(0))
    pcc = np.nanmean([[_pcc(p[:, c], y[:, c]) for c in range(Y.shape[1])] for p, y in zip(preds, trues)], axis=0)
    return float(rmse.mean()), float(np.nanmean(pcc))


def bootstrap_ci(preds, trues, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(preds)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(_overall([preds[i] for i in idx], [trues[i] for i in idx]))
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {"rmse_ci": (float(lo[0]), float(hi[0])), "pcc_ci": (float(lo[1]), float(hi[1]))}


def ema_metrics(preds, trues, phones=None, n_boot=1000):
    """preds/trues: lists of (T, 12) mm arrays; phones: optional list of per-frame label lists."""
    P, Y = np.concatenate(preds), np.concatenate(trues)
    err = P - Y
    rmse = np.sqrt((err**2).mean(0))
    mae = np.abs(err).mean(0)
    pcc_utt = np.array([[_pcc(p[:, c], y[:, c]) for c in range(12)] for p, y in zip(preds, trues)])
    pcc = np.nanmean(pcc_utt, axis=0)
    pcc_pooled = np.array([_pcc(P[:, c], Y[:, c]) for c in range(12)])
    dP = np.concatenate([np.diff(p, axis=0) for p in preds]) * FT_SR
    dY = np.concatenate([np.diff(y, axis=0) for y in trues]) * FT_SR
    vel_rmse = np.sqrt(((dP - dY) ** 2).mean(0))
    vel_pcc = np.array([_pcc(dP[:, c], dY[:, c]) for c in range(12)])
    eucl = np.sqrt(err[:, 0::2] ** 2 + err[:, 1::2] ** 2)  # (N, 6) per-sensor distance

    out = {
        "n_utts": len(preds),
        "n_frames": int(len(P)),
        "rmse": float(rmse.mean()),
        "mae": float(mae.mean()),
        "pcc": float(np.nanmean(pcc)),
        "pcc_pooled": float(np.nanmean(pcc_pooled)),
        "vel_rmse": float(vel_rmse.mean()),
        "vel_pcc": float(np.nanmean(vel_pcc)),
        "per_channel": {
            ch: {"rmse": float(rmse[c]), "mae": float(mae[c]), "pcc": float(pcc[c]), "vel_rmse": float(vel_rmse[c])}
            for c, ch in enumerate(CHANNELS)
        },
        "per_articulator": {
            a: {"eucl_mean": float(eucl[:, i].mean()), "rmse_xy": float(np.sqrt((eucl[:, i] ** 2).mean() / 2))}
            for i, a in enumerate(ARTICULATORS)
        },
    }
    if n_boot:
        out.update(bootstrap_ci(preds, trues, n_boot=n_boot))
    if phones is not None:
        labels = np.array([phone_class(l) for ph in phones for l in ph])
        out["per_phone_class"] = {}
        for cls in sorted(set(labels)):
            m = labels == cls
            out["per_phone_class"][cls] = {
                "n_frames": int(m.sum()),
                "rmse": float(np.sqrt((err[m] ** 2).mean(0)).mean()),
                "per_articulator_eucl": {a: float(eucl[m, i].mean()) for i, a in enumerate(ARTICULATORS)},
            }
    return out
