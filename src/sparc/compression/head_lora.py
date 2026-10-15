"""Do LoRA's gains stack on a strong sequence head? (LOSO, WavLM Large k12)

Arms on the held-out speaker (rmse_z / PCC in its z units; per fold the seed mean, then mean over the 7 folds):
    probe              ridge readout of the frozen encoder (depth_sweep: alpha and shift per fold)
    frozen <head>      head trained on frozen cached features (head_screen: lr on training-speaker validation, 3 seeds)
    LoRA <head>        LoRA r8 q/v + the head (train.py, head lr 3e-4, smoother off for the sequence heads, 3 seeds)
    blend <head>       0.75 x LoRA <head> + 0.25 x probe (lora_readout; weight chosen on training-speaker validation)
for head in {causal conv 9 (the original recipe), BiMamba, BiGRU -> conv 9}. Paired differences over folds:
mean [95% t CI] folds better (+ = better).

Usage: python -m sparc.compression.head_lora
"""

import json

import numpy as np
from scipy import stats as sstats

from . import datasets

B = datasets.XLSR_EMA
MODEL, K = "wavlm-large", 12
SPK = datasets.ALL_SPEAKERS
HEADS = {  # label: (head_screen name, LoRA run template, lora_readout tag)
    "causal conv 9": ("conv_causal9", "prefix12_independent_r8_q_proj+v_proj_conv_causal_s{s}", "k12_cc"),
    "BiMamba": ("mamba", "prefix12_independent_r8_q_proj+v_proj_mamba_nosm_s{s}", "k12_mamba"),
    "BiGRU -> conv 9": ("gruconv9", "prefix12_independent_r8_q_proj+v_proj_gruconv_k9_nosm_s{s}", "k12_gruconv"),
}
METRICS = (("rmse_z", -1), ("pcc", 1))


def probe():
    from .depth_sweep import load_all
    d = load_all(MODEL)
    return {h: {m: d[f"{K}/{h}"]["unseen"][m] for m, _ in METRICS} for h in SPK if f"{K}/{h}" in d}


def frozen(name):
    f = B / "analysis_heads" / MODEL / f"k{K}_{name}.json"
    if not f.exists():
        return {}
    r = json.loads(f.read_text())
    out = {}
    for h in SPK:
        if h in r and "seeds" in r[h]:
            ok = [x for x in r[h]["seeds"].values() if "failed" not in x]
            out[h] = {m: float(np.mean([x["test_unseen"][m] for x in ok])) for m, _ in METRICS}
    return out


def lora(tmpl):
    out = {}
    for h in SPK:
        rs = []
        for s in (0, 1, 2):
            f = datasets.adapt_root(f"ema_loso_{h}") / MODEL / tmpl.format(s=s) / "results.json"
            if f.exists():
                rs.append(json.loads(f.read_text())["test_unseen"])
        if rs:
            out[h] = {m: float(np.mean([x[m] for x in rs])) for m, _ in METRICS} | {"n_seeds": len(rs)}
    return out


def blend(tag):
    out = {}
    for h in SPK:
        f = B / "analysis_readout" / MODEL / tag / f"{h}.json"
        if f.exists():
            r = json.loads(f.read_text())
            if r:
                out[h] = {m: float(np.mean([x["test_unseen"]["blend"][m] for x in r.values()])) for m, _ in METRICS}
    return out


def paired(a, b, m, sign):
    folds = [h for h in SPK if h in a and h in b]
    if len(folds) < 2:
        return "-"
    d = np.array([sign * (a[h][m] - b[h][m]) for h in folds])
    hw = sstats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.4f} [{d.mean() - hw:+.4f}, {d.mean() + hw:+.4f}] {(d > 0).sum()}/{len(d)}"


def main():
    arms = {"probe": probe()}
    for label, (screen, tmpl, tag) in HEADS.items():
        arms[f"frozen {label}"] = frozen(screen)
        arms[f"LoRA {label}"] = lora(tmpl)
        arms[f"blend {label}"] = blend(tag)
    L = ["# LoRA x sequence heads (LOSO, WavLM Large k12; held-out rmse_z / PCC, mean over folds)\n"]
    L.append("| arm | folds | rmse_z | PCC |")
    L.append("|---|---|---|---|")
    for a, r in arms.items():
        if r:
            L.append(f"| {a} | {len(r)} | {np.mean([v['rmse_z'] for v in r.values()]):.4f} | "
                     f"{np.mean([v['pcc'] for v in r.values()]):.4f} |")
    L.append("\n## Paired comparisons over folds (+ = first arm better)\n")
    L.append("| comparison | rmse_z | PCC |")
    L.append("|---|---|---|")
    comps = []
    for label in HEADS:
        comps += [(f"LoRA {label}", f"frozen {label}"), (f"blend {label}", f"LoRA {label}"),
                  (f"LoRA {label}", "probe"), (f"blend {label}", "probe")]
    for label in ("BiMamba", "BiGRU -> conv 9"):
        comps += [(f"LoRA {label}", "LoRA causal conv 9"), (f"blend {label}", "blend causal conv 9"),
                  (f"frozen {label}", "LoRA causal conv 9")]
    for a, b in comps:
        if arms.get(a) and arms.get(b):
            L.append(f"| {a} vs {b} | " + " | ".join(paired(arms[a], arms[b], m, s) for m, s in METRICS) + " |")
    out = B / "analysis_heads" / f"lora_heads_{MODEL}_k{K}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
