"""Phase 8.3 analysis of listening responses (pilot or main).

Environment with statsmodels: ``/data/user_data/xoy/venvs/p8-listening/bin/python`` (pandas, scipy, statsmodels).

    python scripts/spkemb/listening/analyze.py --dir /data/user_data/xoy/spkemb/phase8/listening --set pilot \
        --responses '<dir>/responses/pilot/*.json' --out <dir>/analysis_pilot --nat-margin 0.25

Inputs: the private design ``trials/<set>_design.csv``, ``items.json`` and the response files exported by the page.

Steps
1. Attention checks: identical pair passes if both ratings differ by <= 1; mismatch passes if the same-speaker output is chosen
   and rated higher in similarity. A listener failing a share >= ``--max-check-fail`` (default 0.5) of its checks is excluded.
   Excluded listeners are listed; all models use the remaining listeners and the non-check trials.
2. Primary pool = VCTK test items. LibriTTS-R (control) and GLOBE (descriptive) are reported separately, not pooled.
3. Preference (fitted code more similar to the reference than the standard code): logistic mixed model with crossed random
   intercepts for listener and speaker (``BinomialBayesMixedGLM``, variational Bayes: posterior mean and sd; this
   approximation is known to understate uncertainty, so a two-way cluster bootstrap over listeners and speakers is reported
   next to it). The rule uses the GLMM interval AND the bootstrap interval; both must exclude 0.5.
4. Ratings (1-5): linear mixed model of the within-trial difference fit - standard, crossed random intercepts for listener
   and speaker (similarity and naturalness), plus a cumulative-link model of the single ratings (fixed arm effect,
   sensitivity only, ignores clustering) and a Wilcoxon test on listener means.
5. Naturalness: not rated lower than ``--nat-margin`` points: lower 95% bound of fit - standard > -margin.
Writes ``analysis.json`` (also the variance components used by ``power.py``) and ``analysis.md``.
"""

import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_responses(pattern: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        for r in d["responses"]:
            rows.append({"listener": d["listener"], **{k: r[k] for k in ("trial_id", "sim_a", "sim_b", "nat_a", "nat_b", "pref")}, "rt_s": r.get("rt_s")})
    return pd.DataFrame(rows)


def merge(design: pd.DataFrame, resp: pd.DataFrame) -> pd.DataFrame:
    df = design.merge(resp.drop(columns=["listener"]), on="trial_id", how="inner")
    chosen_arm = np.where(df["pref"] == "A", df["a_arm"], df["b_arm"])
    df["chosen"] = chosen_arm
    return df


def check_table(df: pd.DataFrame, max_fail: float) -> tuple[pd.DataFrame, list[str]]:
    c = df[df["check"] != ""].copy()
    if c.empty:
        return c.assign(passed=[]), []
    ident = (c["check"] == "identical") & ((c["sim_a"] - c["sim_b"]).abs() <= 1) & ((c["nat_a"] - c["nat_b"]).abs() <= 1)
    a_is_match = c["a_arm"] == "std"
    sim_match = np.where(a_is_match, c["sim_a"], c["sim_b"])
    sim_other = np.where(a_is_match, c["sim_b"], c["sim_a"])
    mism = (c["check"] == "mismatch") & (c["chosen"] == "std") & (sim_match > sim_other)
    c["passed"] = ident | mism
    fail = c.groupby("listener")["passed"].apply(lambda s: 1.0 - s.mean())
    return c, sorted(fail[fail >= max_fail].index)


def long_diffs(df: pd.DataFrame) -> pd.DataFrame:
    t = df[df["check"] == ""].copy()
    fit_a = t["a_arm"] == "fit"
    t["sim_fit"] = np.where(fit_a, t["sim_a"], t["sim_b"])
    t["sim_std"] = np.where(fit_a, t["sim_b"], t["sim_a"])
    t["nat_fit"] = np.where(fit_a, t["nat_a"], t["nat_b"])
    t["nat_std"] = np.where(fit_a, t["nat_b"], t["nat_a"])
    t["sim_diff"] = t["sim_fit"] - t["sim_std"]
    t["nat_diff"] = t["nat_fit"] - t["nat_std"]
    t["pref_fit"] = (t["chosen"] == "fit").astype(int)
    return t


def glmm_preference(t: pd.DataFrame) -> dict:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    vc = {"listener": "0 + C(listener)", "speaker": "0 + C(speaker)"}
    m = BinomialBayesMixedGLM.from_formula("pref_fit ~ 1", vc, t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.fit_vb()
    mu, sd = float(r.fe_mean[0]), float(r.fe_sd[0])
    expit = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    sd_l, sd_s = (float(np.exp(v)) for v in r.vcp_mean)
    return {"logit_mean": mu, "logit_sd": sd, "p_conditional": float(expit(mu)), "ci95_p": [float(expit(mu - 1.96 * sd)), float(expit(mu + 1.96 * sd))],
            "sd_listener": sd_l, "sd_speaker": sd_s}


def lmm_diff(t: pd.DataFrame, col: str) -> dict:
    import statsmodels.formula.api as smf

    d = t.assign(g=1)
    vc = {"listener": "0 + C(listener)", "speaker": "0 + C(speaker)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = smf.mixedlm(f"{col} ~ 1", d, groups="g", vc_formula=vc).fit(reml=True)
            mean, se = float(r.params["Intercept"]), float(r.bse["Intercept"])
            sds = {k: float(np.sqrt(v)) for k, v in zip(r.model.exog_vc.names, np.atleast_1d(r.vcomp))}
        except Exception as e:  # singular fits: fall back to listener-clustered mean
            m = d.groupby("listener")[col].mean()
            mean, se, sds = float(m.mean()), float(m.std(ddof=1) / np.sqrt(len(m))), {"error": str(e)}
    return {"mean": mean, "se": se, "ci95": [mean - 1.96 * se, mean + 1.96 * se], "sd_components": sds}


def cluster_bootstrap(t: pd.DataFrame, col: str, n_boot: int, seed: int = 0) -> list[float]:
    """Two-way cluster bootstrap of the mean: resample listeners and speakers independently, keep the trials in both."""
    rng = np.random.default_rng(seed)
    ls, ss = np.array(sorted(t["listener"].unique())), np.array(sorted(t["speaker"].unique()))
    piv = t.pivot_table(index="listener", columns="speaker", values=col, aggfunc=["sum", "count"])
    s_, c_ = piv["sum"].reindex(index=ls, columns=ss).fillna(0).to_numpy(), piv["count"].reindex(index=ls, columns=ss).fillna(0).to_numpy()
    out = np.empty(n_boot)
    for b in range(n_boot):
        li, si = rng.integers(len(ls), size=len(ls)), rng.integers(len(ss), size=len(ss))
        n = c_[np.ix_(li, si)].sum()
        out[b] = s_[np.ix_(li, si)].sum() / n if n else np.nan
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def ordinal_arm_effect(t: pd.DataFrame, kind: str) -> dict:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    d = pd.concat([pd.DataFrame({"y": t[f"{kind}_fit"], "arm": 1}), pd.DataFrame({"y": t[f"{kind}_std"], "arm": 0})])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = OrderedModel(d["y"].astype(int), d[["arm"]], distr="logit").fit(method="bfgs", disp=False)
    return {"log_odds_fit_vs_std": float(r.params["arm"]), "se": float(r.bse["arm"]), "note": "single ratings, no clustering: sensitivity only"}


def pool_report(t: pd.DataFrame, n_boot: int, model: bool) -> dict:
    out: dict = {"n_trials": int(len(t)), "n_listeners": int(t["listener"].nunique()), "n_speakers": int(t["speaker"].nunique()), "n_items": int(t["item"].nunique()),
                 "pref_fit_rate": float(t["pref_fit"].mean()), "sim_fit": float(t["sim_fit"].mean()), "sim_std": float(t["sim_std"].mean()),
                 "nat_fit": float(t["nat_fit"].mean()), "nat_std": float(t["nat_std"].mean())}
    lm = t.groupby("listener")[["sim_diff", "nat_diff"]].mean()
    for c in ("sim_diff", "nat_diff"):
        try:
            out[f"wilcoxon_{c}_p"] = float(stats.wilcoxon(lm[c]).pvalue) if (lm[c] != 0).any() else 1.0
        except ValueError:
            out[f"wilcoxon_{c}_p"] = None
    if model and t["listener"].nunique() >= 3 and t["speaker"].nunique() >= 3:
        out["pref_glmm"] = glmm_preference(t)
        out["pref_bootstrap_ci95"] = cluster_bootstrap(t, "pref_fit", n_boot)
        for c in ("sim_diff", "nat_diff"):
            out[f"{c}_lmm"] = lmm_diff(t, c)
            out[f"{c}_bootstrap_ci95"] = cluster_bootstrap(t, c, n_boot)
        out["ordinal_similarity"] = ordinal_arm_effect(t, "sim")
        out["ordinal_naturalness"] = ordinal_arm_effect(t, "nat")
    return out


def decide(primary: dict, margin: float) -> dict:
    g, b = primary["pref_glmm"]["ci95_p"], primary["pref_bootstrap_ci95"]
    pref_ok = g[0] > 0.5 and b[0] > 0.5
    nat_lo = min(primary["nat_diff_lmm"]["ci95"][0], primary["nat_diff_bootstrap_ci95"][0])
    nat_ok = nat_lo > -margin
    return {"preference_interval_excludes_0.5": bool(pref_ok), "preference_below_0.5": bool(g[1] < 0.5 and b[1] < 0.5), "nat_margin": margin,
            "naturalness_lower_bound": float(nat_lo), "naturalness_not_lower_than_margin": bool(nat_ok), "audible": bool(pref_ok and nat_ok)}


def analyze(design_csv: str, responses: str, out: Path, max_fail: float = 0.5, nat_margin: float = 0.25, n_boot: int = 2000) -> dict:
    design = pd.read_csv(design_csv, keep_default_na=False)
    resp = load_responses(responses)
    df = merge(design, resp)
    checks, excluded = check_table(df, max_fail)
    n_all = df["listener"].nunique()
    kept = df[~df["listener"].isin(excluded)]
    t = long_diffs(kept)
    res: dict = {"n_listeners_total": int(n_all), "excluded_listeners": excluded, "check_pass_rate": float(checks["passed"].mean()) if len(checks) else None,
                 "check_pass_rate_by_type": {k: float(v) for k, v in checks.groupby("check")["passed"].mean().items()} if len(checks) else {}}
    res["pools"] = {p: pool_report(t[t["pool"] == p], n_boot, model=(p == "vctk")) for p in sorted(t["pool"].unique())}
    if "vctk" in res["pools"] and "pref_glmm" in res["pools"]["vctk"]:
        res["decision"] = decide(res["pools"]["vctk"], nat_margin)
        v = res["pools"]["vctk"]
        res["variance_components"] = {"sd_listener": v["pref_glmm"]["sd_listener"], "sd_speaker": v["pref_glmm"]["sd_speaker"],
                                      "n_listeners": v["n_listeners"], "n_speakers": v["n_speakers"], "trials_per_listener": v["n_trials"] / max(v["n_listeners"], 1),
                                      "check_fail_rate": 1 - (res["check_pass_rate"] if res["check_pass_rate"] is not None else 1.0),
                                      "exclusion_rate": len(excluded) / max(n_all, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    (out / "analysis.md").write_text(render(res))
    return res


def render(res: dict) -> str:
    L = [f"# Listening test analysis", "",
         f"Listeners {res['n_listeners_total']}, excluded {len(res['excluded_listeners'])} ({', '.join(res['excluded_listeners']) or 'none'}); "
         f"attention-check pass rate {res['check_pass_rate']} by type {res['check_pass_rate_by_type']}", ""]
    for p, r in res["pools"].items():
        L += [f"## {p} ({r['n_trials']} trials, {r['n_listeners']} listeners, {r['n_speakers']} speakers)", "",
              f"- preference for the fitted code: {r['pref_fit_rate']:.3f}", f"- similarity fit/standard: {r['sim_fit']:.2f} / {r['sim_std']:.2f}; naturalness: {r['nat_fit']:.2f} / {r['nat_std']:.2f}"]
        if "pref_glmm" in r:
            g = r["pref_glmm"]
            L += [f"- GLMM (listener, speaker random): P(fit) {g['p_conditional']:.3f}, 95% [{g['ci95_p'][0]:.3f}, {g['ci95_p'][1]:.3f}]; sd listener {g['sd_listener']:.2f}, speaker {g['sd_speaker']:.2f} (logit)",
                  f"- two-way cluster bootstrap of the rate: [{r['pref_bootstrap_ci95'][0]:.3f}, {r['pref_bootstrap_ci95'][1]:.3f}]"]
            for c, nm in (("sim_diff", "similarity"), ("nat_diff", "naturalness")):
                m = r[f"{c}_lmm"]
                L.append(f"- {nm} difference (fit - standard), LMM: {m['mean']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]; bootstrap [{r[f'{c}_bootstrap_ci95'][0]:+.3f}, {r[f'{c}_bootstrap_ci95'][1]:+.3f}]")
        L.append("")
    if "decision" in res:
        L += ["## Rule", "", "```", json.dumps(res["decision"], indent=1), "```", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--responses", required=True, help="glob of exported response json files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-check-fail", type=float, default=0.5)
    ap.add_argument("--nat-margin", type=float, default=0.25, help="naturalness non-inferiority margin in rating points (to be fixed before the main test)")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    res = analyze(f"{a.dir}/trials/{a.set}_design.csv", a.responses, Path(a.out), a.max_check_fail, a.nat_margin, a.n_boot)
    print(render(res))


if __name__ == "__main__":
    main()
