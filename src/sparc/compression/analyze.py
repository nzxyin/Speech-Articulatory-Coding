"""Collect probe, compute and adaptation results into tables and the two main figures.

    fig_perf_vs_flops   -- test RMSE and PCC vs inference GFLOPs per second of audio: every prefix
                           truncation of every model (ridge probe on its last layer), the full models,
                           and adapted runs placed at the FLOPs of their retained layer count
    fig_perf_vs_ratio   -- same metrics vs retained fraction of each model's full inference FLOPs
    compute_matched.csv -- for each FLOPs budget, each model's best prefix within budget (chosen on
                           validation RMSE) and its test metrics
    summary.md          -- the tables above in markdown

Usage: python -m sparc.compression.analyze [--out DIR]
"""

import argparse
import csv
import json
from pathlib import Path

from .extract import OUT_ROOT
from .train import ADAPT_ROOT

COLORS = {"xlsr-1b": "#2a78d6", "xlsr-300m": "#eb6834", "wavlm-large": "#1baf7a", "xlsr-2b": "#eda100"}
LABELS = {"xlsr-1b": "XLS-R 1B", "xlsr-300m": "XLS-R 300M", "wavlm-large": "WavLM Large", "xlsr-2b": "XLS-R 2B"}
MARKERS = {"none": "s", "independent": "^", "shared_a": "D", "shared_gated": "P"}


def load(root=OUT_ROOT, adapt_root=ADAPT_ROOT):
    models = {}
    for d in sorted(Path(root).iterdir()):
        if not (d / "probe_results.json").exists() or not (d / "compute_prefix.json").exists():
            continue
        probe = json.loads((d / "probe_results.json").read_text())
        comp = {len(p["layers"]): p for p in json.loads((d / "compute_prefix.json").read_text())["prefixes"]}
        L = max(comp)
        rows = []
        for k_str, r in probe["layers"].items():
            k = int(k_str)
            if k == 0:
                continue
            c = comp[k]
            rows.append({"model": d.name, "layers": k, "gflops": c["gflops_per_audio_s"],
                         "params_m": c["params"]["total"] / 1e6, "flops_frac": c["flops"] / comp[L]["flops"],
                         "rtf": c.get("rtf"), "act_mb": c.get("activation_bytes", 0) / 2**20,
                         "valid_rmse": r["valid"]["rmse"], "test_rmse": r["test"]["rmse"],
                         "test_rmse_ci": r["test"].get("rmse_ci"), "test_pcc": r["test"]["pcc"],
                         "test_vel_rmse": r["test"]["vel_rmse"], "shift": r["shift"]})
        models[d.name] = {"L": L, "rows": sorted(rows, key=lambda x: x["layers"]), "comp": comp,
                          "best": probe["best_layer_by_valid_rmse"]}
    adapted = []
    if Path(adapt_root).exists():
        for f in Path(adapt_root).glob("*/*/results.json"):
            r = json.loads(f.read_text())
            m = f.parent.parent.name
            if m not in models:
                continue
            n = len(r["config"]["layers"])
            c = models[m]["comp"][n]
            adapted.append({"model": m, "run": f.parent.name, "variant": r["config"]["variant"],
                            "head": r["config"]["head"], "layers": r["config"]["layers"], "n_layers": n,
                            "gflops": c["gflops_per_audio_s"], "flops_frac": c["flops"] / models[m]["comp"][models[m]["L"]]["flops"],
                            "lora_params": r["n_lora_params"], "head_params": r["n_head_params"],
                            "valid_rmse": r["best_valid_rmse"], "test_rmse": r["test"]["rmse"],
                            "test_pcc": r["test"]["pcc"], "test_vel_rmse": r["test"]["vel_rmse"]})
    return models, adapted


def compute_matched(models, budgets):
    out = []
    for name, budget in budgets:
        for m, d in models.items():
            ok = [r for r in d["rows"] if r["gflops"] <= budget * 1.001]
            if not ok:
                continue
            best = min(ok, key=lambda r: r["valid_rmse"])
            out.append({"budget": name, "budget_gflops": budget, **best})
    return out


def figures(models, adapted, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3d3d3a", "xtick.color": "#5f5e5a",
                         "ytick.color": "#5f5e5a", "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6})
    for xkey, xlabel, fname in [("gflops", "Inference GFLOPs per second of audio", "fig_perf_vs_flops"),
                                ("flops_frac", "Retained fraction of the full model's inference FLOPs",
                                 "fig_perf_vs_ratio")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, ykey, ylabel in [(axes[0], "test_rmse", "Test RMSE (mm), lower is better"),
                                 (axes[1], "test_pcc", "Test PCC, higher is better")]:
            for m, d in models.items():
                xs = [r[xkey] for r in d["rows"]]
                ys = [r[ykey] for r in d["rows"]]
                col = COLORS.get(m, "#5f5e5a")
                ax.plot(xs, ys, color=col, lw=2, marker="o", ms=3.5, label=f"{LABELS.get(m, m)} prefix 1..k (best k={d['best']})")
                full = d["rows"][-1]
                ax.plot(full[xkey], full[ykey], "o", ms=9, color=col, mec="white", mew=2)
                best = next(r for r in d["rows"] if r["layers"] == d["best"])
                ax.plot(best[xkey], best[ykey], "*", ms=14, color=col, mec="white", mew=1.5)
            for a in adapted:
                ax.plot(a[xkey], a[ykey], MARKERS.get(a["variant"], "x"), ms=8, color=COLORS.get(a["model"]),
                        mec="white", mew=1.5, alpha=0.95)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        from matplotlib.lines import Line2D

        extra = [Line2D([], [], ls="", marker="*", ms=12, color="#5f5e5a", label="best layer (validation)"),
                 Line2D([], [], ls="", marker="o", ms=8, color="#5f5e5a", label="full model")]
        variants = {a["variant"] for a in adapted}
        extra += [Line2D([], [], ls="", marker=MARKERS[v], ms=8, color="#5f5e5a", label=f"adapted: {v}")
                  for v in MARKERS if v in variants]
        fig.legend(handles + extra, labels + [h.get_label() for h in extra], loc="lower center",
                   ncol=min(4, len(handles) + len(extra)), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{fname}.{ext}", dpi=200)
        plt.close(fig)


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def md_table(rows, cols):
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[k]) for _, (k, fmt) in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_ROOT.parent / "analysis"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models, adapted = load()
    write_csv([r for d in models.values() for r in d["rows"]], out / "probe_by_layer.csv")
    write_csv(adapted, out / "adapted.csv")

    budgets = []
    if "xlsr-1b" in models:
        g1 = models["xlsr-1b"]["rows"][-1]["gflops"]
        budgets += [("XLS-R 1B full", g1), ("75% of 1B", 0.75 * g1), ("50% of 1B", 0.5 * g1)]
    if "xlsr-300m" in models:
        g3 = models["xlsr-300m"]["rows"][-1]["gflops"]
        budgets += [("XLS-R 300M full", g3), ("2/3 of 300M", 2 / 3 * g3), ("1/3 of 300M", g3 / 3)]
    matched = compute_matched(models, budgets)
    write_csv(matched, out / "compute_matched.csv")
    figures(models, adapted, out)

    f2, f3, f4 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}"), (lambda v: f"{v:.4f}")
    s = (lambda v: str(v))
    parts = ["# SSL compression for speech-to-EMA (MNGU0) -- results\n"]
    for m, d in models.items():
        b = next(r for r in d["rows"] if r["layers"] == d["best"])
        full = d["rows"][-1]
        parts.append(f"- **{LABELS.get(m, m)}**: best layer {d['best']} of {d['L']} by validation; test RMSE "
                     f"{b['test_rmse']:.3f} mm (95% CI {b['test_rmse_ci'][0]:.3f}-{b['test_rmse_ci'][1]:.3f}), "
                     f"PCC {b['test_pcc']:.4f}, at {b['gflops']:.1f} GFLOPs/s ({b['flops_frac']:.0%} of full); "
                     f"full model last layer {full['test_rmse']:.3f} mm.")
    parts.append("\n## Compute-matched comparison (best prefix within budget, chosen on validation)\n")
    parts.append(md_table(matched, [("budget", ("budget", s)), ("GFLOPs/s cap", ("budget_gflops", f2)),
                                    ("model", ("model", s)), ("k", ("layers", s)), ("GFLOPs/s", ("gflops", f2)),
                                    ("params (M)", ("params_m", f2)), ("test RMSE", ("test_rmse", f3)),
                                    ("test PCC", ("test_pcc", f4)), ("vel RMSE", ("test_vel_rmse", f2))]))
    if adapted:
        parts.append("\n## Adapted runs\n")
        parts.append(md_table(sorted(adapted, key=lambda a: (a["model"], a["n_layers"], a["test_rmse"])),
                              [("model", ("model", s)), ("run", ("run", s)), ("GFLOPs/s", ("gflops", f2)),
                               ("LoRA params", ("lora_params", s)), ("valid RMSE", ("valid_rmse", f3)),
                               ("test RMSE", ("test_rmse", f3)), ("test PCC", ("test_pcc", f4))]))
    (out / "summary.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
