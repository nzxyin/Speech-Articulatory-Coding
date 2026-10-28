"""Report of the out-of-domain encoder screen (docs/spkemb_ood_screen.md): per-corpus metrics and the selection rule.

Usage: ``python -m sparc.vocoders.spkscreen.ood_report --out <dir> [--split val|test]`` writes ``ood.json`` and ``ood.md``
there (``ood_val.*`` / ``ood_test.*`` with ``--split``). ``--split`` restricts every statistic to the validation or test
speakers of ``docs/spkemb_validation_protocol.md`` (``eval/splits.py``; probes stay fitted on the same non-test rows). A
corpus without speakers in the split (GLOBE real speakers have no test split) is skipped and the out-of-domain statistic
then uses the remaining corpus alone. Variants are selected on ``--split val`` only; ``test`` reports the chosen one.

Per-corpus statistics reuse ``precond``: the ridge probe is fitted once per run on the non-test held-out rows
(LibriTTS-R dev/test and VCTK speakers outside the 62 test speakers) and the statistic is recomputed on the speakers of a
corpus; paired differences are seed-paired and carry a leave-one-speaker-out jackknife SE. The out-of-domain statistic
is the mean of the VCTK and the GLOBE real-speaker (``globe_spk``) values; its SE is sqrt(SE_vctk^2 + SE_globe^2) / 2 (the
two corpora have disjoint speakers) and the t interval uses the smaller corpus's speaker count minus 1 degrees of freedom.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

from sparc.vocoders.eval import splits as SP
from sparc.vocoders.spkscreen import cache as C
from sparc.vocoders.spkscreen import precond as P

SEEDS = (0, 1, 2)
MAIN = ["ood_gruconv", "ood_gruconv_asp", "ood_sv", "ood_sv_l6", "ood_center", "ood_aug"]
CONTROLS = {"ood_gruconv": "ood_framemlp", "ood_gruconv_asp": "ood_framemlp", "ood_sv_l6": "ood_width_ctl"}
ALL = ["baseline", *MAIN, "ood_framemlp", "ood_width_ctl"]
M3_LIMIT = 0.05


def load_pool(root: Path, name: str):
    import pandas as pd

    man = pd.read_parquet(root / name / "manifest.parquet")
    tg = np.concatenate([np.load(C.chunk_dir(root, name, c) / "targets.npy") for c in range(C.n_chunks(len(man)))])
    return {"spk": man["speaker"].values.astype(str), "targets": tg}


def combine(pv: dict, pg: dict) -> dict:
    d = (pv["d"] + pg["d"]) / 2
    se = 0.5 * float(np.hypot(pv["se"], pg["se"]))
    df = min(pv["n"], pg["n"]) - 1
    t = sps.t.ppf(0.975, df)
    return {"d": d, "se": se, "lo": d - t * se, "hi": d + t * se,
            "p": float(2 * sps.t.sf(abs(d / se), df)) if se > 0 else float("nan")}


def combine_all(parts: list[dict]) -> dict:
    """Mean of the out-of-domain corpora present in the split (see ``combine``); one corpus alone is returned as is."""
    if len(parts) == 1:
        return dict(parts[0])
    return combine(parts[0], parts[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=C.ROOT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=("all", "val", "test"), default="all")
    p.add_argument("--splits-file", default=None)
    a = p.parse_args()
    root = a.root
    H = P.load_heldout(root)
    corp = np.array([s.split(":")[0].replace("libri_dev", "libri").replace("libri_test", "libri") for s in H["spk"]])
    rows = {"all": None, "libri": corp == "libri", "vctk": corp == "vctk"}
    pools = {n: load_pool(root, n) for n in ("globe_spk", "globe")}
    pool_masks = {n: None for n in pools}
    if a.split != "all":
        sp = SP.load(a.splits_file)
        sm = SP.screen_masks(H["spk"], sp, a.split, "heldout")
        rows = {dn: (sm if r is None else r & sm) for dn, r in rows.items()}
        pool_masks = {n: SP.screen_masks(G["spk"], sp, a.split, n) for n, G in pools.items()}
        rows = {dn: r for dn, r in rows.items() if r.sum()}
        pools = {n: G for n, G in pools.items() if pool_masks[n].sum()}
    S: dict[str, dict[str, list]] = {}
    for v in ALL:
        for s in SEEDS:
            d = root / "runs" / v / f"seed{s}"
            z = np.load(d / "embeddings.npz")
            X, X1, X2 = (z[k].astype(np.float64) for k in ("full", "s1", "s2"))
            for dn, r in rows.items():
                S.setdefault(dn, {}).setdefault(v, []).append(P.heldout_stats(X, X1, X2, H, r, pooled=(dn == "all")))
            for pn, G in pools.items():
                zg = np.load(d / f"{pn}_embeddings.npz")
                m = pool_masks[pn]
                Gs = {"spk": G["spk"], "targets": G["targets"], **{k: zg[k].astype(np.float64) for k in ("full", "s1", "s2")}}
                if m is not None:
                    Gs = {k: v_[m] for k, v_ in Gs.items()}
                S.setdefault(pn, {}).setdefault(v, []).append(P.globe_stats(X, X1, X2, H, Gs))
        print(v, "done", flush=True)
    res = {"abs": {}, "diff": {}, "rule": {}}
    for dn, per_v in S.items():
        res["abs"][dn] = {v: {m: [float(r[m][0]) for r in runs] for m in runs[0] if m != "speakers"} for v, runs in per_v.items()}
        res["abs"][dn]["_n_speakers"] = len(per_v["baseline"][0]["speakers"])
    res["split"] = a.split
    doms = [dn for dn in ("libri", "vctk", "globe_spk", "globe") if dn in S]

    def pair(a_, b_, dn, m):
        return P.paired(S[dn][a_], S[dn][b_], m)

    ood = {}
    for v in MAIN:
        for ref in {"baseline", CONTROLS.get(v, "baseline")}:
            key = f"{ref}|{v}"
            res["diff"][key] = {dn: {m: pair(ref, v, dn, m) for m in ("m1", "eer", "m1_1s", "m1_2s")}
                                for dn in doms}
            res["diff"][key]["ood"] = {m: combine_all([res["diff"][key][dn][m] for dn in ("vctk", "globe_spk") if dn in doms])
                                       for m in ("m1", "eer", "m1_1s", "m1_2s")}
            res["diff"][key]["all_m3"] = pair(ref, v, "all", "m3")
            res["diff"][key]["all_m1"] = pair(ref, v, "all", "m1")
        ood[v] = res["diff"][f"baseline|{v}"]["ood"]["m1"]["p"]
    holm = dict(zip(MAIN, P.holm(np.array([ood[v] for v in MAIN]))))
    for v in MAIN:
        b = res["diff"][f"baseline|{v}"]
        ctrl = CONTROLS.get(v, "baseline")
        c = res["diff"][f"{ctrl}|{v}"]
        m3 = -b["all_m3"]["d"]  # paired(): sign * (baseline - variant); m3 sign +1, so variant - baseline = -d
        res["rule"][v] = {
            "ood_m1_gain": b["ood"]["m1"]["d"], "ood_m1_ci": [b["ood"]["m1"]["lo"], b["ood"]["m1"]["hi"]],
            "p_holm": float(holm[v]), "beats_baseline": bool(b["ood"]["m1"]["lo"] > 0 and holm[v] < 0.05),
            "control": ctrl, "gain_vs_control": c["ood"]["m1"]["d"], "control_ci": [c["ood"]["m1"]["lo"], c["ood"]["m1"]["hi"]],
            "beats_control": bool(c["ood"]["m1"]["lo"] > 0), "m3_change": float(m3), "m3_ok": bool(m3 <= M3_LIMIT)}
        r = res["rule"][v]
        r["promising"] = bool(r["beats_baseline"] and r["beats_control"] and r["m3_ok"])
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.split == "all" else f"_{a.split}"
    (a.out / f"ood{tag}.json").write_text(json.dumps(res, indent=1))
    (a.out / f"ood{tag}.md").write_text(markdown(res))
    print(markdown(res))


def mean(res, dn, v, m):
    if dn not in res["abs"]:
        return float("nan"), float("nan")
    x = res["abs"][dn][v][m]
    return float(np.mean(x)), float(np.std(x, ddof=1))


def markdown(res) -> str:
    L = ["# Out-of-domain encoder screen: results", "",
         "M1 = RMSE in target-std units (lower is better), mean over 3 seeds (sd over seeds); EER in percent.", "",
         "| variant | LibriTTS-R M1 | VCTK M1 | GLOBE real M1 | GLOBE pseudo M1 | LibriTTS EER | VCTK EER | GLOBE real EER | M3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for v in ALL:
        cells = [f"{mean(res, d, v, 'm1')[0]:.3f} ({mean(res, d, v, 'm1')[1]:.3f})" for d in ("libri", "vctk", "globe_spk", "globe")]
        cells += [f"{100 * mean(res, d, v, 'eer')[0]:.1f}" for d in ("libri", "vctk", "globe_spk")]
        cells += [f"{mean(res, 'all', v, 'm3')[0]:.3f}"]
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    L += ["", "Out-of-domain M1 gain over the baseline (mean of VCTK and GLOBE real speakers; positive = better), "
          "95% interval, Holm-adjusted p over the six variants; gain over the control; M3 change vs baseline:", "",
          "| variant | OOD M1 gain [95% CI] | p (Holm) | control | gain vs control [95% CI] | OOD EER gain (pts) | M3 change | promising |",
          "|---|---|---|---|---|---|---|---|"]
    for v, r in res["rule"].items():
        e = res["diff"][f"baseline|{v}"]["ood"]["eer"]
        L.append(f"| {v} | {r['ood_m1_gain']:+.4f} [{r['ood_m1_ci'][0]:+.4f}, {r['ood_m1_ci'][1]:+.4f}] | {r['p_holm']:.3f} | "
                 f"{r['control']} | {r['gain_vs_control']:+.4f} [{r['control_ci'][0]:+.4f}, {r['control_ci'][1]:+.4f}] | "
                 f"{100 * e['d']:+.2f} | {r['m3_change']:+.3f} | {'yes' if r['promising'] else 'no'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
