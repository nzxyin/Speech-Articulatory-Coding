"""Lin's concordance correlation coefficient (CCC) for articulatory inversion, from per-utterance sufficient statistics.

Plan and recommendation rule: docs/ccc_metric_plan.md. For one series pair (prediction p, target y), per channel,

    CCC = 2 cov(p, y) / (var p + var y + (mean p - mean y)^2) = PCC x C_s x C_l

    correlation factor  PCC = cov / (s_p s_y)
    scale factor        C_s = 2 s_p s_y / (s_p^2 + s_y^2)
    location factor     C_l = (s_p^2 + s_y^2) / (s_p^2 + s_y^2 + (m_p - m_y)^2)

(population variances, as in `rigor.utt_stats`). All variants below need only the per-utterance sums that
`rigor.utt_stats` stores (n, sum p, sum p^2, sum y, sum y^2, sum p*y), so a gain on the predictions, a subset of
utterances or a bootstrap resample is evaluated without touching the frames again.

Variants (every one: per channel, then mean over the 12 channels, degenerate units dropped as NaN):

    utt           CCC per utterance and channel, mean over utterances and channels (the project's PCC convention)
    spk           the speaker's test frames concatenated, one CCC per channel
    spk_centered  as spk after removing each utterance's mean from p and y (isolates correlation and scale)

Arm-level decomposition: with A = mean PCC, B = mean(PCC C_s) / A, C = mean(PCC C_s C_l) / mean(PCC C_s) over the same
units, CCC = A B C holds exactly, so log CCC = log A + log B + log C and a difference of two arms splits into three
additive parts (`log_shares`). B is the PCC-weighted scale factor, C the PCC- and scale-weighted location factor.
"""

import numpy as np

EPS = 1e-8  # std threshold for a non-degenerate series, as in rigor.utt_stats
MIN_FRAMES_1S = 50  # 1 s at the 50 Hz frame rate
MIN_FRAMES_3S = 150  # exploratory second cut: the cached utterances are all longer than 1 s


def _b(n, like):
    n = np.asarray(n, dtype=np.float64)
    return n.reshape(n.shape + (1,) * (like.ndim - n.ndim))


def components(n, psum, psq, ysum, ysq, pysum):
    """Per-unit PCC, scale factor, location factor and CCC from sums (arrays broadcast over leading axes).

    Units where either series has std <= EPS are NaN in all four outputs."""
    n = _b(n, np.asarray(psum))
    mp, my = psum / n, ysum / n
    vp, vy = np.maximum(psq / n - mp ** 2, 0.0), np.maximum(ysq / n - my ** 2, 0.0)
    cov = pysum / n - mp * my
    sp, sy = np.sqrt(vp), np.sqrt(vy)
    ok = (sp > EPS) & (sy > EPS)
    spy = np.where(ok, sp * sy, np.nan)
    tot = np.where(ok, vp + vy, np.nan)
    pcc = cov / spy
    cs = 2 * spy / tot
    cl = tot / (tot + (mp - my) ** 2)
    ccc = 2 * cov / (tot + (mp - my) ** 2)
    return {"pcc": pcc, "cs": cs, "cl": cl, "ccc": ccc}


def ccc_direct(p, y):
    """Reference implementation on frames (per channel); NaN for a degenerate series."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mp, my = p.mean(0), y.mean(0)
    vp, vy = p.var(0), y.var(0)
    cov = ((p - mp) * (y - my)).mean(0)
    ok = (np.sqrt(vp) > EPS) & (np.sqrt(vy) > EPS)
    return np.where(ok, 2 * cov / (vp + vy + (mp - my) ** 2), np.nan)


def _decompose(c):
    """(A, B, C, ccc, plain C_s, plain C_l) from per-unit components (NaN-aware means over all units)."""
    pcc, cs, cl, ccc = c["pcc"], c["cs"], c["cl"], c["ccc"]
    if not np.isfinite(ccc).any():
        return (np.nan,) * 6
    A = np.nanmean(pcc)
    pc = pcc * cs
    B = np.nanmean(pc) / A if A != 0 else np.nan
    C = np.nanmean(pc * cl) / np.nanmean(pc) if np.nanmean(pc) != 0 else np.nan
    return float(A), float(B), float(C), float(np.nanmean(ccc)), float(np.nanmean(cs)), float(np.nanmean(cl))


def _sub(st, idx):
    return st if idx is None else {k: v[idx] for k, v in st.items()}


def ccc_utt(st, idx=None, min_frames=0):
    """Mean per-utterance concordance correlation and its decomposition (utterances with >= min_frames frames)."""
    st = _sub(st, idx)
    keep = st["n"] >= min_frames
    c = components(st["n"][keep], st["psum"][keep], st["psq"][keep], st["ysum"][keep], st["ysq"][keep], st["pysum"][keep])
    A, B, C, ccc, cs, cl = _decompose(c)
    return {"ccc": ccc, "pcc": A, "scale_w": B, "location_w": C, "cs": cs, "cl": cl, "n_utt": int(keep.sum())}


def ccc_spk(st, idx=None):
    """Concatenated-per-speaker concordance correlation (one per channel, mean over channels) and decomposition."""
    st = _sub(st, idx)
    c = components(st["n"].sum(), st["psum"].sum(0), st["psq"].sum(0), st["ysum"].sum(0), st["ysq"].sum(0), st["pysum"].sum(0))
    A, B, C, ccc, cs, cl = _decompose(c)
    return {"ccc": ccc, "pcc": A, "scale_w": B, "location_w": C, "cs": cs, "cl": cl}


def ccc_spk_centered(st, idx=None):
    """Concatenated-per-speaker CCC after removing each utterance's own mean from p and y (location factor is 1)."""
    st = _sub(st, idx)
    n = st["n"][:, None]
    vp = (st["psq"] - st["psum"] ** 2 / n).sum(0)
    vy = (st["ysq"] - st["ysum"] ** 2 / n).sum(0)
    cv = (st["pysum"] - st["psum"] * st["ysum"] / n).sum(0)
    N = st["n"].sum()
    sp, sy = np.sqrt(np.maximum(vp / N, 0)), np.sqrt(np.maximum(vy / N, 0))
    ok = (sp > EPS) & (sy > EPS)
    pcc = np.where(ok, cv / np.where(ok, np.sqrt(vp * vy), np.nan), np.nan)
    cs = np.where(ok, 2 * np.sqrt(vp * vy) / np.where(ok, vp + vy, np.nan), np.nan)
    ccc = pcc * cs
    return {"ccc": float(np.nanmean(ccc)), "pcc": float(np.nanmean(pcc)), "cs": float(np.nanmean(cs))}


def metrics_all(st, idx=None):
    """Every metric of the study from one set of per-utterance statistics (optionally a utterance subset)."""
    from .rigor import metrics_from

    st = _sub(st, idx)
    m = metrics_from(st)
    u, u1, s, sc = ccc_utt(st), ccc_utt(st, min_frames=MIN_FRAMES_1S), ccc_spk(st), ccc_spk_centered(st)
    u3 = ccc_utt(st, min_frames=MIN_FRAMES_3S)
    n = st["n"][:, None]
    mp, my = st["psum"] / n, st["ysum"] / n
    sy = np.sqrt(np.maximum(st["ysq"] / n - my ** 2, 0))
    off = np.abs(mp - my) / np.where(sy > EPS, sy, np.nan)
    m.update({
        "pcc_concat": s["pcc"],
        "ccc_utt": u["ccc"], "ccc_utt_1s": u1["ccc"], "ccc_utt_3s": u3["ccc"], "n_utt_3s": u3["n_utt"], "ccc_spk": s["ccc"], "ccc_spk_centered": sc["ccc"],
        "utt_corr": u["pcc"], "utt_scale_w": u["scale_w"], "utt_location_w": u["location_w"], "utt_cs": u["cs"], "utt_cl": u["cl"],
        "utt_1s_corr": u1["pcc"], "utt_1s_scale_w": u1["scale_w"], "utt_1s_location_w": u1["location_w"],
        "spk_corr": s["pcc"], "spk_scale_w": s["scale_w"], "spk_location_w": s["location_w"], "spk_cs": s["cs"], "spk_cl": s["cl"],
        "spkc_corr": sc["pcc"], "spkc_cs": sc["cs"],
        "offset_sd": float(np.nanmean(off)), "n_utt_1s": u1["n_utt"], "n_utt": len(st["n"]),
    })
    return m


def apply_gain(st, g):
    """Statistics of g * p against the same targets (exact: nothing but sums enters any metric)."""
    n = st["n"][:, None]
    vp = st["psq"] - st["psum"] ** 2 / n
    vy = st["ysq"] - st["ysum"] ** 2 / n
    cv = st["pysum"] - st["psum"] * st["ysum"] / n
    out = dict(st)
    out["psum"], out["psq"], out["pysum"] = g * st["psum"], g * g * st["psq"], g * st["pysum"]
    out["sse"] = g * g * st["psq"] - 2 * g * st["pysum"] + st["ysq"]
    out["csse"] = np.maximum(g * g * vp - 2 * g * cv + vy, 0.0)
    out["emean"] = (g * st["psum"] - st["ysum"]) / n
    return out


def log_shares(a, b):
    """Difference a - b of two arms' (A, B, C) factors in log space: dict of the three parts and their sum."""
    la = np.log([a["corr"], a["scale_w"], a["location_w"]])
    lb = np.log([b["corr"], b["scale_w"], b["location_w"]])
    d = la - lb
    return {"correlation": float(d[0]), "scale": float(d[1]), "location": float(d[2]), "total": float(d.sum())}
