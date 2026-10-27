"""Prints the markdown tables of the precondition checks from ``precond.json`` (see ``precond.py``)."""

import json
import sys

import numpy as np

NAMES = {"baseline": "baseline", "seg_vus3": "voiced / unvoiced / silence split", "seg_vuv2": "voiced / non-voiced split",
         "ms_l1_l6": "layer 1 + layer 6", "pool_trimmed": "trimmed mean", "seg_mel_on_vuv2": "two-way split + log-mel branch",
         "cap_rep3": "baseline vector x3 (3072-d)", "cap_rep2": "baseline vector x2 (2048-d)"}
MET = [("m1", "M1"), ("eer", "EER"), ("id", "5-shot ID"), ("m3", "M3"), ("f0", "F0 PCC"), ("m1_1s", "M1 1 s"), ("m1_2s", "M1 2 s")]


def cell(y, holm=False):
    s = f"{y['d']:+.4f} ({y['se']:.4f}) [{y['lo']:+.4f}, {y['hi']:+.4f}]"
    if "improved" in y:
        s += f" {y['improved']}/{y['n']}"
    return s


def main(path):
    r = json.load(open(path))
    out = []
    for dom, title in (("all", "all 62 held-out speakers"), ("libri", "LibriTTS-R speakers"), ("vctk", "VCTK speakers"),
                       ("globe", "GLOBE speakers")):
        if dom not in r["abs"]:
            continue
        out += [f"\n#### Absolute values, {title} ({r['abs'][dom]['_n_speakers']} speakers, mean of 3 seeds)\n",
                "| variant | " + " | ".join(n for k, n in MET if k in next(v for k, v in r["abs"][dom].items() if k[0] != "_")) + " |",
                "|---|" + "---|" * len([k for k, _ in MET if k in r["abs"][dom]["baseline"]])]
        for v, x in r["abs"][dom].items():
            if v[0] == "_":
                continue
            out.append(f"| {NAMES[v]} | " + " | ".join(f"{np.mean(x[k]):.4f}" for k, _ in MET if k in x) + " |")
        out += [f"\n#### Paired differences, {title}: positive = second variant better; estimate (jackknife SE) [95% CI] "
                "speakers improved\n",
                "| comparison | " + " | ".join(n for k, n in MET if k in r["diff"][dom]["baseline|seg_vus3"]) + " | Holm p (M1) |",
                "|---|" + "---|" * (1 + len([k for k, _ in MET if k in r["diff"][dom]["baseline|seg_vus3"]]))]
        for c, x in r["diff"][dom].items():
            a, b = c.split("|")
            h = x["m1"].get("p_holm")
            out.append(f"| {NAMES[b]} vs {NAMES[a]} | " + " | ".join(cell(x[k]) for k, _ in MET if k in x) + " | "
                       + (f"{h:.3f}" if h is not None else f"raw {x['m1']['p']:.3f}") + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
