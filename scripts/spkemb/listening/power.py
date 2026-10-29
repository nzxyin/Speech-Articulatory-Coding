"""Power calculator: listeners needed for 80 % power to detect a preference rate of 0.60 against 0.50.

    python scripts/spkemb/listening/power.py --dir <dir> --pilot-analysis <dir>/analysis_pilot/analysis.json
    python scripts/spkemb/listening/power.py --dir <dir> --sd-listener 0.6 --sd-speaker 0.3        # manual variances

Model: the primary (VCTK) trials of the main design (``design.make_design`` with the real items, so the speaker and item
structure and the listener x item assignment are the ones used), preference ~ Bernoulli(logistic(b0 + u_listener +
v_speaker [+ w_item])) with ``b0`` set so that the MARGINAL preference rate equals ``--p-alt`` (0.60); the null uses 0.50 with
the same variances. Test per simulated dataset: two-way cluster-robust (listener, speaker; Cameron-Gelbach-Miller) t test of
the mean preference against 0.5, df = min(#listeners, #speakers) - 1, alpha = 0.05 two-sided (the cluster-robust test
is a fast stand-in for the GLMM interval of ``analyze.py``; the type I error under the null is reported by the same
simulation so the stand-in is calibrated). Variances come from a pilot (``--pilot-analysis``); with only ~5 pilot
listeners the listener sd is imprecise, so by default the upper 80 % confidence value (chi-square, df = listeners - 1 /
speakers - 1) is used (``--no-conservative`` to use the point estimates). The returned count is inflated for the expected
listener exclusion rate (``--exclusion``, default from the pilot).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design import make_design  # noqa: E402


def marginal_to_intercept(p: float, sd_total: float) -> float:
    """b0 with E[logistic(b0 + N(0, sd_total^2))] = p (Gauss-Hermite)."""
    x, w = np.polynomial.hermite_e.hermegauss(60)
    w = w / w.sum()
    f = lambda b: float((w / (1 + np.exp(-(b + sd_total * x)))).sum()) - p  # noqa: E731
    lo, hi = -5.0, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def cluster_t(y: np.ndarray, li: np.ndarray, si: np.ndarray, mu0: float = 0.5) -> tuple[float, float]:
    """Mean, two-way cluster-robust (CGM) standard error."""
    e = y - y.mean()
    n = len(y)

    def var_by(g: np.ndarray) -> float:
        s = np.bincount(g, weights=e)
        return float((s**2).sum()) / n**2

    key = li * (si.max() + 1) + si
    v = var_by(li) + var_by(si) - var_by(key)
    if v <= 0:
        v = var_by(li)
    return float(y.mean()), float(np.sqrt(v))


def power_at(items: dict, n_listeners: int, p: float, sd_l: float, sd_s: float, sd_i: float, n_sim: int, n_vctk: int, seed: int, **dk) -> dict:
    rng = np.random.default_rng([seed, n_listeners])
    sd_tot = float(np.sqrt(sd_l**2 + sd_s**2 + sd_i**2))
    b0 = marginal_to_intercept(p, sd_tot)
    rej_alt = rej_null = 0
    for k in range(n_sim):
        _, rows = make_design(items, n_listeners, int(rng.integers(1 << 30)), n_vctk=n_vctk, **dk)
        rows = [r for r in rows if r["pool"] == "vctk" and r["check"] == ""]
        l_idx = {l: i for i, l in enumerate(sorted({r["listener"] for r in rows}))}
        s_idx = {s: i for i, s in enumerate(sorted({r["speaker"] for r in rows}))}
        i_idx = {s: i for i, s in enumerate(sorted({r["item"] for r in rows}))}
        li = np.array([l_idx[r["listener"]] for r in rows])
        si = np.array([s_idx[r["speaker"]] for r in rows])
        ii = np.array([i_idx[r["item"]] for r in rows])
        u, v, w = rng.normal(0, sd_l, len(l_idx)), rng.normal(0, sd_s, len(s_idx)), rng.normal(0, sd_i, len(i_idx))
        df = min(len(l_idx), len(s_idx)) - 1
        for alt in (True, False):
            b = b0 if alt else marginal_to_intercept(0.5, sd_tot)
            y = (rng.random(len(rows)) < 1 / (1 + np.exp(-(b + u[li] + v[si] + w[ii])))).astype(float)
            m, se = cluster_t(y, li, si)
            p_val = 2 * stats.t.sf(abs((m - 0.5) / se), df) if se > 0 else 1.0
            if p_val < 0.05:
                if alt and m > 0.5:
                    rej_alt += 1
                elif not alt:
                    rej_null += 1
    return {"n_listeners": n_listeners, "power": rej_alt / n_sim, "type1": rej_null / n_sim, "mc_se": float(np.sqrt(rej_alt / n_sim * (1 - rej_alt / n_sim) / n_sim))}


def upper_sd(sd: float, df: int, conf: float = 0.8) -> float:
    """Upper one-sided ``conf`` bound of a standard deviation (chi-square)."""
    return float(sd * np.sqrt(df / stats.chi2.ppf(1 - conf, df))) if df > 0 else sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pilot-analysis")
    ap.add_argument("--sd-listener", type=float)
    ap.add_argument("--sd-speaker", type=float)
    ap.add_argument("--sd-item", type=float, default=0.0, help="item-within-speaker sd (not identified by the pilot GLMM; default 0)")
    ap.add_argument("--p-alt", type=float, default=0.60)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--grid", default="6,8,10,12,16,20,24,30,36,48,60,80,120")
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--n-vctk", type=int, default=24, help="VCTK items per listener in the main design")
    ap.add_argument("--exclusion", type=float, default=None)
    ap.add_argument("--no-conservative", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    items = json.loads((Path(a.dir) / "items.json").read_text())["items"]
    sd_l, sd_s, excl = a.sd_listener, a.sd_speaker, a.exclusion
    note = "manual variances"
    if a.pilot_analysis:
        vc = json.loads(Path(a.pilot_analysis).read_text())["variance_components"]
        sd_l, sd_s = vc["sd_listener"], vc["sd_speaker"]
        excl = vc["exclusion_rate"] if excl is None else excl
        note = f"pilot ({vc['n_listeners']} listeners, {vc['n_speakers']} speakers): sd listener {sd_l:.3f}, speaker {sd_s:.3f}"
        if not a.no_conservative:
            sd_l, sd_s = upper_sd(sd_l, vc["n_listeners"] - 1), upper_sd(sd_s, vc["n_speakers"] - 1)
            note += f"; upper-80% values used: {sd_l:.3f}, {sd_s:.3f}"
    if sd_l is None or sd_s is None:
        sys.exit("give --pilot-analysis or both --sd-listener and --sd-speaker")
    excl = excl or 0.0
    res = []
    found = None
    for n in (int(x) for x in a.grid.split(",")):
        r = power_at(items, n, a.p_alt, sd_l, sd_s, a.sd_item, a.n_sim, a.n_vctk, a.seed)
        res.append(r)
        print(f"N={n:3d} power={r['power']:.3f} (se {r['mc_se']:.3f}) type1={r['type1']:.3f}", flush=True)
        if found is None and r["power"] >= a.power:
            found = n
            break
    summary = {"variances": note, "sd_listener": sd_l, "sd_speaker": sd_s, "sd_item": a.sd_item, "p_alt": a.p_alt, "target_power": a.power, "n_sim": a.n_sim,
               "n_vctk_items_per_listener": a.n_vctk, "grid": res, "listeners_needed": found,
               "listeners_to_recruit": int(np.ceil(found / (1 - excl))) if found else None, "exclusion_rate": excl}
    if found is None:
        summary["note"] = ("target power not reached on the grid: with a fixed set of test speakers the speaker variance caps the power "
                           "however many listeners are added; more speakers (items from more VCTK speakers) or a larger effect are needed")
    print(json.dumps({k: v for k, v in summary.items() if k != "grid"}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
