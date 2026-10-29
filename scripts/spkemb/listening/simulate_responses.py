"""Simulate listener responses for a trial design (testing the analysis and the page export format).

    python scripts/spkemb/listening/simulate_responses.py --dir <dir> --set pilot --out <dir>/sim/pilot --p-fit 0.6 \
        --sd-listener 0.5 --sd-speaker 0.3 --nat-effect 0.0 --careless 1

Writes one ``responses_<set>_<listener>.json`` per listener in the page's export format. The preference is a logistic
mixed model with the given marginal-conditional intercept (``--p-fit`` is the conditional probability at zero random
effects), ratings follow a latent mean plus listener and speaker offsets; ``--careless`` listeners answer at random.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def simulate(design: pd.DataFrame, p_fit: float, sd_l: float, sd_s: float, sim_effect: float, nat_effect: float, careless: int, seed: int) -> dict[str, dict]:
    rng = np.random.default_rng([seed, 4242])
    ls, ss = sorted(design["listener"].unique()), sorted(design["speaker"].unique())
    u = dict(zip(ls, rng.normal(0, sd_l, len(ls))))
    v = dict(zip(ss, rng.normal(0, sd_s, len(ss))))
    bad = set(ls[:careless])
    mu = np.log(p_fit / (1 - p_fit))
    out: dict[str, dict] = {}
    rate = lambda m: int(np.clip(np.round(m + rng.normal(0, 0.8)), 1, 5))  # noqa: E731
    for lid, g in design.groupby("listener"):
        resp = []
        lo = rng.normal(0, 0.4)  # listener's rating bias
        for _, r in g.sort_values("pos").iterrows():
            if lid in bad:
                a, b = (int(rng.integers(1, 6)) for _ in range(2))
                na, nb = (int(rng.integers(1, 6)) for _ in range(2))
                pref = "A" if rng.random() < 0.5 else "B"
            elif r["check"] == "identical":
                a = rate(3.0 + lo)
                b, na = a, rate(3.5 + lo)
                nb = na
                pref = "A" if rng.random() < 0.5 else "B"
            elif r["check"] == "mismatch":
                a_match = r["a_arm"] == "std"
                m, o = rate(3.5 + lo), rate(1.5 + lo)
                a, b = (m, o) if a_match else (o, m)
                na, nb = rate(3.5), rate(3.5)
                pref = "A" if (a_match == (rng.random() < 0.95)) else "B"
            else:
                fit_a = r["a_arm"] == "fit"
                m_std, m_fit = 3.0 + lo + v[r["speaker"]], 3.0 + lo + v[r["speaker"]] + sim_effect
                a, b = (rate(m_fit), rate(m_std)) if fit_a else (rate(m_std), rate(m_fit))
                n_std, n_fit = 3.6 + lo, 3.6 + lo + nat_effect
                na, nb = (rate(n_fit), rate(n_std)) if fit_a else (rate(n_std), rate(n_fit))
                p = 1 / (1 + np.exp(-(mu + u[lid] + v[r["speaker"]])))
                fit_chosen = rng.random() < p
                pref = "A" if (fit_chosen == fit_a) else "B"
            resp.append({"trial_id": r["trial_id"], "sim_a": a, "sim_b": b, "nat_a": na, "nat_b": nb, "pref": pref, "plays": {"ref": 1, "a": 1, "b": 1},
                         "listened": {"ref": True, "a": True, "b": True}, "rt_s": 20.0, "t": ""})
        out[lid] = {"schema": 1, "set": "sim", "listener": lid, "started": "", "finished": "", "user_agent": "sim", "responses": resp}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p-fit", type=float, default=0.6)
    ap.add_argument("--sd-listener", type=float, default=0.5)
    ap.add_argument("--sd-speaker", type=float, default=0.3)
    ap.add_argument("--sim-effect", type=float, default=0.15)
    ap.add_argument("--nat-effect", type=float, default=0.0)
    ap.add_argument("--careless", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    design = pd.read_csv(f"{a.dir}/trials/{a.set}_design.csv", keep_default_na=False)
    res = simulate(design, a.p_fit, a.sd_listener, a.sd_speaker, a.sim_effect, a.nat_effect, a.careless, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for lid, d in res.items():
        (out / f"responses_{a.set}_{lid}.json").write_text(json.dumps(d))
    print(f"wrote {len(res)} response files to {out}")


if __name__ == "__main__":
    main()
