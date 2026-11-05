import numpy as np
import pytest

from sparc.compression import ccc, rigor


def _data(seed=0, U=6, C=12):
    rng = np.random.default_rng(seed)
    Y = [rng.normal(size=(int(rng.integers(30, 120)), C)) * rng.uniform(0.5, 1.5, C) + rng.normal(size=C) * 0.3
         for _ in range(U)]
    P = [0.7 * y + rng.normal(size=y.shape) * 0.6 + rng.normal(size=C) * 0.2 for y in Y]
    return P, Y


def _lin(p, y):  # reference: Lin's definition with numpy var/cov
    return 2 * np.cov(p, y, bias=True)[0, 1] / (p.var() + y.var() + (p.mean() - y.mean()) ** 2)


def test_identical_series_is_one():
    P, Y = _data()
    st = rigor.utt_stats(Y, Y)
    assert np.isclose(ccc.ccc_utt(st)["ccc"], 1.0)
    assert np.isclose(ccc.ccc_spk(st)["ccc"], 1.0)
    assert np.isclose(ccc.ccc_spk_centered(st)["ccc"], 1.0)


def test_closed_form_affine():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(200, 1))
    y = y - y.mean()  # zero-mean target: the mean difference is then exactly b
    for a, b in ((1.0, 0.0), (0.5, 0.0), (2.0, 0.3), (-0.8, -0.2)):
        p = a * y + b
        c = ccc.components(np.array(200.0), p.sum(0), (p ** 2).sum(0), y.sum(0), (y ** 2).sum(0), (p * y).sum(0))
        sy2 = y.var()
        want = 2 * a * sy2 / (a * a * sy2 + sy2 + b * b)
        assert np.isclose(c["ccc"][0], want)
        assert np.isclose(c["ccc"][0], _lin(p[:, 0], y[:, 0]))
        if a == 1.0 and b == 0.0:
            assert np.isclose(c["cs"][0] * c["cl"][0], 1.0) and np.isclose(c["pcc"][0], 1.0)


def test_identity_ccc_equals_pcc_times_factors_and_bound():
    P, Y = _data(2)
    st = rigor.utt_stats(P, Y)
    c = ccc.components(st["n"], st["psum"], st["psq"], st["ysum"], st["ysq"], st["pysum"])
    assert np.allclose(c["ccc"], c["pcc"] * c["cs"] * c["cl"])
    assert np.allclose(c["pcc"], st["pcc"])  # sums reproduce rigor's centered PCC
    assert (c["ccc"] <= np.abs(c["pcc"]) + 1e-12).all()
    assert (c["cs"] <= 1 + 1e-12).all() and (c["cl"] <= 1 + 1e-12).all()


def test_log_decomposition_sums_to_log_ccc():
    P, Y = _data(3)
    st = rigor.utt_stats(P, Y)
    for f in (lambda s: ccc.ccc_utt(s), lambda s: ccc.ccc_spk(s)):
        r = f(st)
        assert np.isclose(np.log(r["pcc"]) + np.log(r["scale_w"]) + np.log(r["location_w"]), np.log(r["ccc"]))
    a, b = ccc.ccc_utt(st), ccc.ccc_utt(rigor.utt_stats([0.8 * p for p in P], Y))
    fa = {"corr": a["pcc"], "scale_w": a["scale_w"], "location_w": a["location_w"]}
    fb = {"corr": b["pcc"], "scale_w": b["scale_w"], "location_w": b["location_w"]}
    s = ccc.log_shares(fa, fb)
    assert np.isclose(s["total"], np.log(a["ccc"]) - np.log(b["ccc"]))
    assert abs(s["correlation"]) < 1e-12  # a gain leaves the correlation factor alone


def test_symmetric_and_matches_reference():
    P, Y = _data(4)
    for p, y in zip(P, Y):
        assert np.allclose(ccc.ccc_direct(p, y), ccc.ccc_direct(y, p))
        assert np.allclose(ccc.ccc_direct(p, y), [_lin(p[:, c], y[:, c]) for c in range(p.shape[1])])
    st = rigor.utt_stats(P, Y)
    ref = np.mean([ccc.ccc_direct(p, y) for p, y in zip(P, Y)])
    assert np.isclose(ccc.ccc_utt(st)["ccc"], ref)
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    assert np.isclose(ccc.ccc_spk(st)["ccc"], np.mean(ccc.ccc_direct(Pc, Yc)))
    Pm = np.concatenate([p - p.mean(0) for p in P])
    Ym = np.concatenate([y - y.mean(0) for y in Y])
    assert np.isclose(ccc.ccc_spk_centered(st)["ccc"], np.mean(ccc.ccc_direct(Pm, Ym)))
    assert np.isclose(ccc.metrics_all(st)["pcc_concat"], rigor.pcc_concat(P, Y))


def test_invariance_and_penalty():
    P, Y = _data(5)
    base = ccc.ccc_utt(rigor.utt_stats(P, Y))
    # PCC is untouched by a gain or an offset, CCC is not: both lower it
    gained = ccc.ccc_utt(rigor.utt_stats([0.5 * p for p in P], Y))
    shifted = ccc.ccc_utt(rigor.utt_stats([p + 1.0 for p in P], Y))
    assert np.isclose(gained["pcc"], base["pcc"]) and np.isclose(shifted["pcc"], base["pcc"])
    assert gained["ccc"] < base["ccc"] and shifted["ccc"] < base["ccc"]
    assert np.isclose(shifted["scale_w"], base["scale_w"])  # an offset only moves the location factor
    assert shifted["location_w"] < base["location_w"]
    # a common shift of p and y changes nothing (CCC depends on the difference of means only)
    both = ccc.ccc_utt(rigor.utt_stats([p + 2.0 for p in P], [y + 2.0 for y in Y]))
    assert np.isclose(both["ccc"], base["ccc"])
    # speaker-centered variant ignores per-utterance offsets
    sc0 = ccc.ccc_spk_centered(rigor.utt_stats(P, Y))["ccc"]
    sc1 = ccc.ccc_spk_centered(rigor.utt_stats([p + i for i, p in enumerate(P)], Y))["ccc"]
    assert np.isclose(sc0, sc1)


def test_nan_handling_matches_utt_stats():
    P, Y = _data(6)
    P[1][:, 3] = 0.3  # constant prediction channel in one utterance
    Y[2][:, 5] = -1.0
    st = rigor.utt_stats(P, Y)
    c = ccc.components(st["n"], st["psum"], st["psq"], st["ysum"], st["ysq"], st["pysum"])
    assert np.array_equal(np.isnan(c["pcc"]), np.isnan(st["pcc"]))
    assert np.isnan(c["ccc"][1, 3]) and np.isnan(c["ccc"][2, 5])
    assert np.isfinite(ccc.ccc_utt(st)["ccc"])
    assert ccc.ccc_utt(st)["n_utt"] == 6
    assert ccc.ccc_utt(st, min_frames=10 ** 6)["n_utt"] == 0
    assert np.isnan(ccc.ccc_utt(st, min_frames=10 ** 6)["ccc"])


def test_apply_gain_matches_direct_scoring():
    P, Y = _data(7)
    st = rigor.utt_stats(P, Y)
    for g in (0.5, 0.86, 1.0, 1.3):
        a = rigor.metrics_from(ccc.apply_gain(st, g))
        b = rigor.metrics_from(rigor.utt_stats([g * p for p in P], Y))
        for k in ("rmse_z", "rmse_centered", "pcc", "bias_share", "std_ratio", "rmse_spk_centered"):
            assert np.isclose(a[k], b[k]), k
        ma, mb = ccc.metrics_all(ccc.apply_gain(st, g)), ccc.metrics_all(rigor.utt_stats([g * p for p in P], Y))
        for k in ("ccc_utt", "ccc_spk", "ccc_spk_centered", "utt_scale_w", "utt_location_w"):
            assert np.isclose(ma[k], mb[k]), k


def test_subset_index_equals_subset_data():
    P, Y = _data(8)
    st = rigor.utt_stats(P, Y)
    idx = np.array([0, 2, 2, 5])
    sub = ccc.metrics_all(rigor.utt_stats([P[i] for i in idx], [Y[i] for i in idx]))
    got = ccc.metrics_all(st, idx)
    for k in ("ccc_utt", "ccc_spk", "rmse_z", "pcc"):
        assert np.isclose(sub[k], got[k])


# ---------------------------------------------------------------- ccc_analysis helpers

def _arm(seed, folds=3, shrink=0.7):
    from sparc.compression import ccc_analysis as ca

    rng = np.random.default_rng(seed)
    arm = {}
    for f in range(folds):
        Y = [rng.normal(size=(80, 12)) for _ in range(5)]
        P = [shrink * y + 0.5 * rng.normal(size=y.shape) for y in Y]
        arm[f"h{f}"] = [([f"u{i}" for i in range(5)], ca.prep(rigor.utt_stats(P, Y)))]
    return arm


def test_core4_matches_metrics_all():
    from sparc.compression import ccc_analysis as ca

    P, Y = _data(9)
    st = ca.prep(rigor.utt_stats(P, Y))
    idx = np.array([0, 1, 3, 3, 5])
    v, m = ca.core4(st, idx), ccc.metrics_all(st, idx)
    assert np.allclose(v, [m["rmse_z"], m["pcc"], m["ccc_utt"], m["ccc_spk"]])


def test_q5_label_cases():
    from sparc.compression import ccc_analysis as ca

    mk = lambda mean, p, se: {"mean": mean, "sign_flip_p": p, "std_effect": se}  # noqa: E731
    assert ca.q5_label(mk(0.1, 0.01, 2.0), mk(0.1, 0.01, 2.0)) == "unchanged"
    assert ca.q5_label(mk(0.1, 0.01, 2.0), mk(-0.1, 0.01, 2.0)) == "reversed"
    assert ca.q5_label(mk(0.1, 0.01, 2.0), mk(0.02, 0.4, 0.3)).startswith("weakened (no longer")
    assert ca.q5_label(mk(0.1, 0.01, 2.0), mk(0.1, 0.02, 1.0)).startswith("weakened (smaller")
    assert ca.q5_label(mk(0.01, 0.6, 0.2), mk(0.1, 0.01, 2.0)).startswith("now significant")
    assert ca.q5_label(mk(0.0, 1.0, np.nan), mk(-0.1, 0.01, 2.0)).startswith("reference metric cannot")


def test_gain_tables_mse_gain_below_ccc_gain():
    from sparc.compression import ccc_analysis as ca

    arm = _arm(0, shrink=0.7)
    cur, lfo, orc = ca.gain_tables(arm)
    g = {m: np.mean([x[m] for x in orc.values()]) for m in ("rmse_z", "ccc_utt", "ccc_spk")}
    assert g["rmse_z"] < 1.0  # MSE-optimal damping of a noisy prediction
    assert g["ccc_spk"] > g["rmse_z"] and g["ccc_utt"] > g["rmse_z"]  # a scale-agreement metric wants a larger gain
    assert all(np.ptp(c["pcc"]) < 1e-12 for c in cur.values())  # PCC is flat in the gain
    h = "h0"
    oth = [k for k in arm if k != h]
    want = sum(arm[o][0][1]["pysum"].sum() for o in oth) / sum(arm[o][0][1]["psq"].sum() for o in oth)
    assert np.isclose(lfo[(h, 0)]["rmse_z"], want)
