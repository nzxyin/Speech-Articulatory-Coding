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
import re
from pathlib import Path

import numpy as np

from .extract import OUT_ROOT
from .train import ADAPT_ROOT

COLORS = {"xlsr-1b": "#2a78d6", "xlsr-300m": "#eb6834", "wavlm-large": "#1baf7a", "xlsr-2b": "#eda100"}
LABELS = {"xlsr-1b": "XLS-R 1B", "xlsr-300m": "XLS-R 300M", "wavlm-large": "WavLM Large", "xlsr-2b": "XLS-R 2B"}


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
        if len(rows) < L:  # probe sweep still running
            continue
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
                            "test_pcc": r["test"]["pcc"], "test_vel_rmse": r["test"]["vel_rmse"],
                            "pool": r["config"].get("pool", "static" if r["config"].get("layer_pool") else "none"),
                            "pool_norm": r["config"].get("pool_norm", "layer"),
                            "per_articulator": r["config"].get("per_articulator", False),
                            "hidden": r["config"].get("hidden", 256),
                            "probe_init": not r["config"].get("no_probe_init", False),
                            "pool_weights": r.get("pool_weights_valid")})
            rob = f.parent / "robustness.json"
            if rob.exists():
                adapted[-1]["robustness"] = {k: v["rmse"] for k, v in json.loads(rob.read_text()).items()}
    return models, adapted


def load_pruned(models, adapt_root=ADAPT_ROOT):
    """Component-pruning runs (components.py): adapt/<model>/pruned/<run>/results.json."""
    out = []
    for f in sorted(Path(adapt_root).glob("*/pruned/*/results.json")):
        r = json.loads(f.read_text())
        m = f.parent.parent.parent.name
        if m not in models or not r.get("cost"):
            continue
        full = models[m]["comp"][models[m]["L"]]["flops"]
        out.append({"model": m, "run": f.parent.name, "keep": r["keep"], "importance": r["importance"],
                    "n_layers": len(r["config"]["layers"]), "heads": f"{r['heads_kept']}/{r['heads_total']}",
                    "neurons": f"{r['neurons_kept']}/{r['neurons_total']}",
                    "params_m": r["cost"]["params"]["total"] / 1e6, "gflops": r["cost"]["gflops_per_audio_s"],
                    "flops_frac": r["cost"]["flops"] / full, "rtf": r["cost"].get("rtf"),
                    "valid_rmse": r["best_valid_rmse"], "test_rmse": r["test"]["rmse"], "test_pcc": r["test"]["pcc"],
                    "test_rmse_ci": r["test"].get("rmse_ci")})
    return out


def pool_summary(weights, top=3):
    """Short text summary of mean pool weights (groups x layers): top layers per group (1-indexed retained layer)."""
    if not weights:
        return ""
    names = ["TD", "TB", "TT", "LI", "UL", "LL"] if len(weights) == 6 else ["all"]
    parts = []
    for name, w in zip(names, weights):
        order = sorted(range(len(w)), key=lambda i: -w[i])[:top]
        parts.append(f"{name}: " + ", ".join(f"L{i + 1} {w[i]:.2f}" for i in order))
    return "; ".join(parts)


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


def best_adapted(adapted, model):
    """Per retained-layer count, the adapted run of `model` with the lowest validation RMSE (prefixes only)."""
    by_n = {}
    for a in adapted:
        if a["model"] != model or a["layers"] != list(range(1, a["n_layers"] + 1)):
            continue
        if a["n_layers"] not in by_n or a["valid_rmse"] < by_n[a["n_layers"]]["valid_rmse"]:
            by_n[a["n_layers"]] = a
    return [by_n[n] for n in sorted(by_n)]


def figures(models, adapted, out, pruned=()):
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
            for m in models:
                best = best_adapted(adapted, m)
                if best:
                    ax.plot([a[xkey] for a in best], [a[ykey] for a in best], color=COLORS.get(m, "#5f5e5a"),
                            lw=2, ls="--", marker="^", ms=8, mec="white", mew=1.5)
                taylor = sorted((p for p in pruned if p["model"] == m and p["importance"] == "taylor"),
                                key=lambda p: p[xkey])
                if taylor:
                    ax.plot([p[xkey] for p in taylor], [p[ykey] for p in taylor], color=COLORS.get(m, "#5f5e5a"),
                            lw=2, ls=":", marker="D", ms=7, mec="white", mew=1.5)
                for p in pruned:
                    if p["model"] == m and p["importance"] != "taylor":
                        ax.plot(p[xkey], p[ykey], "D", ms=7, mfc="none", mec=COLORS.get(m, "#5f5e5a"), mew=1.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        from matplotlib.lines import Line2D

        extra = [Line2D([], [], ls="", marker="*", ms=12, color="#5f5e5a", label="best layer (validation)"),
                 Line2D([], [], ls="", marker="o", ms=8, color="#5f5e5a", label="full model")]
        if adapted:
            extra.append(Line2D([], [], ls="--", lw=2, marker="^", ms=8, color="#5f5e5a",
                                label="best adapted run per depth (validation)"))
        if pruned:
            extra.append(Line2D([], [], ls=":", lw=2, marker="D", ms=7, color="#5f5e5a",
                                label="head + FFN pruned (open: random control)"))
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
    write_csv([{k: v for k, v in a.items() if k != "robustness"} for a in adapted], out / "adapted.csv")

    budgets = []
    if "xlsr-1b" in models:
        g1 = models["xlsr-1b"]["rows"][-1]["gflops"]
        budgets += [("XLS-R 1B full", g1), ("75% of 1B", 0.75 * g1), ("50% of 1B", 0.5 * g1)]
    if "xlsr-300m" in models:
        g3 = models["xlsr-300m"]["rows"][-1]["gflops"]
        budgets += [("XLS-R 300M full", g3), ("2/3 of 300M", 2 / 3 * g3), ("1/3 of 300M", g3 / 3)]
    matched = compute_matched(models, budgets)
    write_csv(matched, out / "compute_matched.csv")
    pruned = load_pruned(models)
    write_csv(pruned, out / "pruned.csv")
    figures(models, adapted, out, pruned)

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
    groups = {}
    for a in adapted:
        groups.setdefault((a["model"], re.sub(r"_s\d+$", "", a["run"])), []).append(a)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        parts.append("\n## Seed variation (configurations with several seeds)\n")
        parts.append("| model | configuration | seeds | GFLOPs/s | test RMSE mean +/- std | test PCC mean +/- std |")
        parts.append("|---|---|---|---|---|---|")
        for (m, cfg), v in sorted(multi.items()):
            r = np.array([a["test_rmse"] for a in v])
            p = np.array([a["test_pcc"] for a in v])
            parts.append(f"| {m} | {cfg} | {len(v)} | {v[0]['gflops']:.1f} | {r.mean():.3f} +/- {r.std(ddof=1):.3f} "
                         f"| {p.mean():.4f} +/- {p.std(ddof=1):.4f} |")
    for m in models:
        for pf in sorted((Path(OUT_ROOT) / m).glob("prune_from*_to*.json")):
            ev = json.loads(pf.read_text()).get("eval", {})
            sizes = sorted({v["n_layers"] for v in ev.values()}, reverse=True)
            if not sizes:
                continue
            parts.append(f"\n## Non-contiguous selection, {LABELS.get(m, m)} ({pf.stem}; ridge probe, test RMSE / PCC)\n")
            parts.append("| layers kept | prefix | greedy | block influence | greedy subset |")
            parts.append("|---|---|---|---|---|")
            for n in sizes:
                cell = lambda k: (f"{ev[k]['test']['rmse']:.3f} / {ev[k]['test']['pcc']:.4f}" if k in ev else "-")
                g = ev.get(f"greedy_{n}", {}).get("layers", [])
                parts.append(f"| {n} | {cell(f'prefix_{n}')} | {cell(f'greedy_{n}')} | {cell(f'bi_{n}')} | {g} |")
    if adapted:
        rob = [a for a in adapted if "robustness" in a]
        if rob:
            conds = list(rob[0]["robustness"])
            parts.append("\n## Robustness (test RMSE, mm)\n")
            parts.append("| model | run | " + " | ".join(conds) + " |")
            parts.append("|---|---|" + "---|" * len(conds))
            for a in sorted(rob, key=lambda a: (a["model"], a["run"])):
                parts.append(f"| {a['model']} | {a['run']} | " + " | ".join(f"{a['robustness'][c]:.3f}" for c in conds) + " |")
    pooled = [a for a in adapted if a["pool"] != "none" or not a["probe_init"]]
    if pooled:
        parts.append("\n## Layer pooling (same retained prefix; compare with the matching unpooled run)\n")
        parts.append("| model | arm | pool | layer norm | per articulator | head hidden | head params | test RMSE | "
                     "test PCC | unpooled RMSE | top pool weights (valid) |")
        parts.append("|---|---|---|---|---|---|---|---|---|---|---|")
        def is_causal(run):
            return "_causal" in run

        for a in sorted(pooled, key=lambda a: (a["model"], a["variant"], a["pool"], a["per_articulator"], a["run"])):
            arm = "frozen + linear" if a["variant"] == "none" and a["head"] == "linear" else f"{a['variant']} + {a['head']}"
            base = next((b for b in adapted
                         if b["model"] == a["model"] and b["layers"] == a["layers"] and b["variant"] == a["variant"]
                         and b["head"] == a["head"] and b["pool"] == "none" and b["probe_init"]
                         and b["hidden"] == 256 and b["run"].endswith("_s0") and is_causal(b["run"]) == is_causal(a["run"])),
                        None)
            cells = [a["model"], arm, a["pool"] if a["pool"] != "none" else "none (random init)",
                     a["pool_norm"] if a["pool"] != "none" else "-", "yes" if a["per_articulator"] else "no",
                     str(a["hidden"]) if a["head"] != "linear" else "-", str(a["head_params"]),
                     f"{a['test_rmse']:.3f}", f"{a['test_pcc']:.4f}", f"{base['test_rmse']:.3f}" if base else "-",
                     pool_summary(a["pool_weights"])]
            parts.append("| " + " | ".join(cells) + " |")
    if pruned:
        parts.append("\n## Component (head + FFN neuron) pruning\n")
        parts.append(md_table(sorted(pruned, key=lambda p: (p["model"], -p["keep"], p["importance"])),
                              [("model", ("model", s)), ("keep", ("keep", s)), ("importance", ("importance", s)),
                               ("heads", ("heads", s)), ("FFN neurons", ("neurons", s)), ("params (M)", ("params_m", f2)),
                               ("GFLOPs/s", ("gflops", f2)), ("valid RMSE", ("valid_rmse", f3)),
                               ("test RMSE", ("test_rmse", f3)), ("test PCC", ("test_pcc", f4))]))
    (out / "summary.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
