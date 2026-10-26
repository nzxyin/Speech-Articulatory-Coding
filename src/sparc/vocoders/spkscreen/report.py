"""Summary table and selection rule of the speaker-encoder screen (docs/spkemb_encoder_screen.md).

Usage::

    python -m sparc.vocoders.spkscreen.report --baseline baseline [--variants a b c] [--out summary.md]

Without ``--variants`` every run directory under ``<root>/runs`` except the baseline and the en+ reference is a variant.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sparc.vocoders.spkscreen import cache as C

COLS = [("m1_rmse", "M1 RMSE"), ("eer", "EER"), ("id_acc", "5-shot ID"), ("m3_leak_pcc", "M3 leak"),
        ("f0_pcc", "F0 PCC"), ("m4_rmse_1s", "M1 1 s"), ("m4_rmse_2s", "M1 2 s")]


def load_runs(root: Path, name: str) -> list[dict]:
    runs = [json.loads(f.read_text()) for f in sorted((root / "runs" / name).glob("seed*/result.json"))]
    return sorted(runs, key=lambda r: r["seed"])


def paired(base: list[dict], var: list[dict], key: str, sign: float) -> tuple[float, float]:
    """Mean and SE of ``sign * (base - var)`` over the seeds both have; SE = SD / sqrt(n)."""
    b = {r["seed"]: r[key] for r in base}
    d = np.array([sign * (b[r["seed"]] - r[key]) for r in var if r["seed"] in b])
    if len(d) < 2:
        return float("nan"), float("nan")
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def verdict(base, var) -> dict:
    """Pre-registered rule: M1 better by > 2 SE, EER not worse by > 2 SE, M3 not higher by > 0.05."""
    m1, m1_se = paired(base, var, "m1_rmse", 1.0)  # positive = variant better (lower RMSE)
    eer, eer_se = paired(base, var, "eer", 1.0)  # positive = variant better (lower EER)
    m3, _ = paired(base, var, "m3_leak_pcc", -1.0)  # variant minus baseline
    ok = bool(m1 > 2 * m1_se and eer >= -2 * eer_se and m3 <= 0.05)
    return {"d_m1": m1, "d_m1_se": m1_se, "d_eer": eer, "d_eer_se": eer_se, "d_m3": m3, "promising": ok}


def summarize(root: Path, baseline: str, variants: list[str] | None = None) -> str:
    base = load_runs(root, baseline)
    if variants is None:
        variants = sorted(p.name for p in (root / "runs").iterdir() if p.is_dir() and p.name not in (baseline, "en_plus"))
    lines = ["| variant | seeds | params | " + " | ".join(n for _, n in COLS) + " | train s | ms/utt |",
             "|---|---|---|" + "---|" * len(COLS) + "---|---|"]

    def row(name, runs):
        cells = []
        for k, _ in COLS:
            v = np.array([r[k] for r in runs if k in r])
            cells.append(f"{v.mean():.4f} ({v.std(ddof=1):.4f})" if len(v) > 1 else (f"{v[0]:.4f}" if len(v) else "-"))
        p = runs[0].get("encoder_params")
        t = np.mean([r.get("train_seconds", np.nan) for r in runs])
        ms = np.mean([r.get("embed_ms_per_utt", np.nan) for r in runs])
        lines.append(f"| {name} | {len(runs)} | {p if p is not None else '-'} | " + " | ".join(cells) + f" | {t:.0f} | {ms:.1f} |")

    row(baseline + " (baseline)", base)
    ref = root / "runs" / "en_plus" / "result.json"
    if ref.exists():
        r = json.loads(ref.read_text())
        cells = [f"{r[k]:.4f} ({r[k + '_se']:.4f})" if k + "_se" in r else "-" for k, _ in COLS]
        lines.append("| shipped en+ 64-d (reference) | - | - | " + " | ".join(cells) + " | - | - |")
    results = {}
    for v in variants:
        runs = load_runs(root, v)
        if not runs:
            continue
        row(v, runs)
        results[v] = verdict(base, runs)
    lines += ["", "Cells: mean over seeds (SD over seeds). Paired differences against the baseline (same seed), positive = variant better "
              "for M1 and EER, M3 = variant minus baseline:", "",
              "| variant | d M1 (SE) | d EER (SE) | d M3 | promising |", "|---|---|---|---|---|"]
    for v, r in results.items():
        lines.append(f"| {v} | {r['d_m1']:.4f} ({r['d_m1_se']:.4f}) | {r['d_eer']:.4f} ({r['d_eer_se']:.4f}) | {r['d_m3']:.3f} | "
                     f"{'yes' if r['promising'] else 'no'} |")
    ranked = sorted((v for v, r in results.items() if r["promising"]), key=lambda v: (-results[v]["d_m1"], -results[v]["d_eer"]))
    lines += ["", "Promising, ranked by M1 gain then EER gain: " + (", ".join(ranked) if ranked else "none")]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--root", type=Path, default=C.ROOT)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    text = summarize(a.root, a.baseline, a.variants)
    (a.out or a.root / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
