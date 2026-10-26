"""Metric suite of the speaker-encoder screen: Phase 1 definitions (``sparc.spkemb.layers.analyze``) on embeddings.

Inputs are numpy arrays in Phase 1 row order: ``X [N, d]`` embeddings of the full utterances, ``Y [N, k]`` targets,
``logf0 [N]``, ``spk [N]`` speaker labels, ``is_test [N]`` bool. ``X1`` / ``X2`` are embeddings of the first 1 s / 2 s.
Every number comes with a speaker-level SE (per-speaker SD / sqrt(n) for per-speaker means; leave-one-speaker-out
jackknife for pooled statistics).
"""

import numpy as np


def ridge_fit(X, Y, alphas=(1e-1, 1, 10, 100, 1e3, 1e4)):
    """Ridge with alpha chosen on a 20 percent holdout of the training rows (Phase 1 ``ridge_fit``)."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    va, tr = idx[: len(X) // 5], idx[len(X) // 5 :]
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-6
    ym = Y[tr].mean(0)

    def solve(rows, a):
        Xz = (X[rows] - xm) / xs
        return np.linalg.solve(Xz.T @ Xz + a * np.eye(X.shape[1]), Xz.T @ (Y[rows] - ym))

    best = min(alphas, key=lambda a: (((X[va] - xm) / xs @ solve(tr, a) + ym - Y[va]) ** 2).mean())
    W = solve(idx, best)
    return lambda Z: (Z - xm) / xs @ W + ym


def score(P, Y, ysd):
    rmse = np.sqrt(((P - Y) ** 2).mean(0)) / ysd
    pcc = [np.corrcoef(P[:, j], Y[:, j])[0, 1] for j in range(Y.shape[1]) if Y[:, j].std() > 0]
    return float(rmse.mean()), float(np.nanmean(pcc))


def partners(spk):
    """Phase 1 pairing: each utterance with the next utterance of its speaker (cyclic, in row order)."""
    order = {}
    for i, s in enumerate(spk):
        order.setdefault(s, []).append(i)
    partner = np.zeros(len(spk), dtype=int)
    for ids in order.values():
        for j, i in enumerate(ids):
            partner[i] = ids[(j + 1) % len(ids)]
    return partner, order


def jackknife_se(stat, speakers):
    """Leave-one-speaker-out jackknife SE of ``stat(keep_mask)`` over the given speaker array (one entry per row)."""
    names = sorted(set(speakers))
    vals = np.array([stat(speakers != s) for s in names], dtype=np.float64)
    n = len(names)
    return float(np.sqrt((n - 1) / n * ((vals - vals.mean()) ** 2).sum()))


def _pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def equal_error_rate(scores_t, scores_n):
    """EER of target / non-target score arrays (higher = more similar)."""
    s = np.concatenate([scores_t, scores_n])
    lab = np.concatenate([np.ones(len(scores_t)), np.zeros(len(scores_n))])
    o = np.argsort(-s, kind="stable")
    lab = lab[o]
    tp = np.cumsum(lab) / lab.sum()
    fp = np.cumsum(1 - lab) / (1 - lab).sum()
    fnr = 1 - tp
    i = int(np.argmin(np.abs(fnr - fp)))
    return float((fnr[i] + fp[i]) / 2)


def _eer_from_cos(C, spk):
    same = spk[:, None] == spk[None]
    iu = np.triu_indices(len(spk), 1)
    s, m = C[iu], same[iu]
    return equal_error_rate(s[m], s[~m])


def evaluate(X, Y, logf0, spk, is_test, X1=None, X2=None) -> dict:
    """All metrics of one embedding set; keys ``<metric>`` and ``<metric>_se``."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    spk = np.asarray(spk).astype(str)
    is_te = np.asarray(is_test, bool)
    te = np.where(is_te)[0]
    test_spk = sorted(set(spk[te]))
    ysd = Y[~is_te].std(0) + 1e-9
    partner, order = partners(spk)
    res = {}
    # M1: A's embedding -> a different utterance of the same speaker
    f = ridge_fit(X[~is_te], Y[partner[~is_te]])
    P = f(X[is_te])
    res["m1_rmse"], res["m1_pcc"] = score(P, Y[partner[is_te]], ysd)
    E = (P - Y[partner[te]]) / ysd
    per = [np.sqrt((E[spk[te] == s] ** 2).mean(0)).mean() for s in test_spk]
    res["m1_rmse_se"] = float(np.std(per, ddof=1) / np.sqrt(len(per)))
    # M4: short references (probe fitted on full-utterance embeddings)
    for tag, Xs in (("1s", X1), ("2s", X2)):
        if Xs is None:
            continue
        Ps = f(np.asarray(Xs, np.float64)[is_te])
        Es = (Ps - Y[partner[te]]) / ysd
        res[f"m4_rmse_{tag}"] = score(Ps, Y[partner[is_te]], ysd)[0]
        pers = [np.sqrt((Es[spk[te] == s] ** 2).mean(0)).mean() for s in test_spk]
        res[f"m4_rmse_{tag}_se"] = float(np.std(pers, ddof=1) / np.sqrt(len(pers)))
    # M2: 5-shot identification (cosine to centroid of the first 5 utterances) and EER
    Xt, st = X[te], spk[te]
    Xn = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-9)
    cents = np.stack([Xn[np.where(st == s)[0][:5]].mean(0) for s in test_spk])
    acc = []
    for k, s in enumerate(test_spk):
        rest = np.where(st == s)[0][5:]
        if len(rest):
            acc.append(np.mean(np.argmax(Xn[rest] @ cents.T, 1) == k))
    res["id_acc"] = float(np.mean(acc))
    res["id_acc_se"] = float(np.std(acc, ddof=1) / np.sqrt(len(acc)))
    C = Xn @ Xn.T
    res["eer"] = _eer_from_cos(C, st)
    res["eer_se"] = jackknife_se(lambda keep: _eer_from_cos(C[np.ix_(keep, keep)], st[keep]), st)
    # M3: content leakage, speaker-mean-removed embedding -> speaker-mean-removed own targets
    spk_mean_X = {s: X[ids].mean(0) for s, ids in order.items()}
    spk_mean_Y = {s: Y[ids].mean(0) for s, ids in order.items()}
    Xd = X - np.stack([spk_mean_X[s] for s in spk])
    Yd = Y - np.stack([spk_mean_Y[s] for s in spk])
    g = ridge_fit(Xd[~is_te], Yd[~is_te])
    Pd = g(Xd[is_te])
    ysd_d = Yd[~is_te].std(0) + 1e-9
    res["m3_leak_pcc"] = score(Pd, Yd[is_te], ysd_d)[1]
    res["m3_leak_pcc_se"] = jackknife_se(lambda keep: score(Pd[keep], Yd[is_te][keep], ysd_d)[1], st)
    # M5: median F0
    ok = np.isfinite(logf0)
    h = ridge_fit(X[~is_te & ok], logf0[~is_te & ok, None])
    ph = h(X[is_te & ok])[:, 0]
    th = logf0[is_te & ok]
    res["f0_pcc"] = _pcc(ph, th)
    sf0 = spk[is_te & ok]
    res["f0_pcc_se"] = jackknife_se(lambda keep: _pcc(ph[keep], th[keep]), sf0)
    return res
