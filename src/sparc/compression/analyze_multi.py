"""Tables and figures for the multi-speaker EMA experiments (dataset ema_multi).

  zero-shot   MNGU0-trained models on all seven new speakers (crossspeaker.py)
  probes      per-layer ridge probes trained on the five training speakers: seen-speaker test (text-disjoint)
              and unseen-speaker test (usc_F1, 5emo_kf), with compute from the MNGU0 compute profiles
  adapted     train.py runs (LoRA / heads / pooling) and components.py pruned runs, seen vs unseen speakers
  figure      seen / unseen test RMSE vs GFLOPs per second of audio

Usage: python -m sparc.compression.analyze_multi [--out DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import datasets
from .analyze import COLORS, LABELS, md_table

FEAT = datasets.features_root("ema_multi")
ADAPT = datasets.adapt_root("ema_multi")
COMPUTE = datasets.features_root("mngu0")


def compute_table(model):
    p = COMPUTE / model / "compute_prefix.json"
    if not p.exists():
        return {}
    return {len(x["layers"]): x for x in json.loads(p.read_text())["prefixes"]}


def probe_rows():
    rows = []
    for d in sorted(FEAT.iterdir()) if FEAT.exists() else []:
        f = d / "probe_results.json"
        if not f.exists():
            continue
        pr = json.loads(f.read_text())
        comp = compute_table(d.name)
        for k, r in pr["layers"].items():
            k = int(k)
            if k == 0 or "test_unseen" not in r:
                continue
            rows.append({"model": d.name, "layers": k, "gflops": comp.get(k, {}).get("gflops_per_audio_s", np.nan),
                         "valid_z": r.get("valid_rmse_fit_units"), "test_rmse": r["test"]["rmse"],
                         "test_pcc": r["test"]["pcc"], "unseen_rmse": r["test_unseen"]["rmse"],
                         "unseen_pcc": r["test_unseen"]["pcc"], "best": k == pr["best_layer_by_valid_rmse"]})
    return rows


def adapted_rows():
    rows = []
    if not ADAPT.exists():
        return rows
    for f in sorted(ADAPT.glob("*/*/results.json")) + sorted(ADAPT.glob("*/pruned/*/results.json")):
        r = json.loads(f.read_text())
        pruned = f.parent.parent.name == "pruned"
        model = (f.parent.parent.parent if pruned else f.parent.parent).name
        comp = compute_table(model)
        n = len(r["config"]["layers"])
        g = r["cost"]["gflops_per_audio_s"] if pruned and r.get("cost") else comp.get(n, {}).get("gflops_per_audio_s", np.nan)
        rows.append({"model": model, "run": f.parent.name, "kind": "pruned" if pruned else "adapted",
                     "variant": r["config"].get("variant"), "head": r["config"].get("head"), "n_layers": n,
                     "keep": r.get("keep"), "gflops": g, "test_rmse": r["test"]["rmse"], "test_pcc": r["test"]["pcc"],
                     "unseen_rmse": r.get("test_unseen", {}).get("rmse", np.nan),
                     "unseen_pcc": r.get("test_unseen", {}).get("pcc", np.nan),
                     "per_speaker": {**r["test"].get("per_speaker", {}), **r.get("test_unseen", {}).get("per_speaker", {})}})
    return rows


def zero_shot_rows():
    f = FEAT / "crossspeaker_zero_shot.json"
    if not f.exists():
        return []
    z = json.loads(f.read_text())
    return [{"model": name, **res["mean"], "per_speaker": {s: v for s, v in res.items() if s != "mean"}}
            for name, res in z.items()]


def corpus_of(speaker):
    return "USC-TIMIT" if speaker.startswith("usc") else "EMA_5EMO"


def corpus_mean(per_speaker, key):
    """{corpus: mean over that corpus's speakers (equal weight per speaker)} of per_speaker[spk][key]."""
    out = {}
    for c in ("USC-TIMIT", "EMA_5EMO"):
        vals = [v[key] for s, v in per_speaker.items() if corpus_of(s) == c and key in v]
        if vals:
            out[c] = float(np.mean(vals))
    return out


def corpus_breakdown(zs, adapted):
    """USC-TIMIT vs EMA_5EMO tables (speaker-averaged within each corpus); seeds averaged for adapted runs."""
    parts = []
    if zs:
        parts.append("\n## Per corpus: zero-shot (MNGU0-trained), PCC\n")
        parts.append("| model | USC zero-shot | 5EMO zero-shot | USC calibrated | 5EMO calibrated | USC calibrated RMSE | 5EMO calibrated RMSE |")
        parts.append("|---|---|---|---|---|---|---|")
        for z in zs:
            a, b, c = (corpus_mean(z["per_speaker"], k) for k in ("zero_shot_pcc", "linear_pcc", "linear_rmse"))
            parts.append(f"| {z['model']} | {a['USC-TIMIT']:.3f} | {a['EMA_5EMO']:.3f} | {b['USC-TIMIT']:.3f} | "
                         f"{b['EMA_5EMO']:.3f} | {c['USC-TIMIT']:.2f} | {c['EMA_5EMO']:.2f} |")
    groups = {}
    for a in adapted:
        key = (a["model"], a["run"][:-3] if a["kind"] == "adapted" and a["run"][-3:-1] == "_s" else a["run"])
        groups.setdefault(key, []).append(a)
    if groups:
        parts.append("\n## Per corpus: multi-speaker trained runs, RMSE mm / PCC (unseen: usc_F1, 5emo_kf)\n")
        parts.append("| model | run | n | USC seen | 5EMO seen | USC unseen | 5EMO unseen |")
        parts.append("|---|---|---|---|---|---|---|")
        for (m, run), v in sorted(groups.items()):
            cells = []
            for split, c in [("seen", "USC-TIMIT"), ("seen", "EMA_5EMO"), ("unseen", "USC-TIMIT"), ("unseen", "EMA_5EMO")]:
                ps = [{s: x for s, x in a["per_speaker"].items() if (s in HELD) == (split == "unseen")} for a in v]
                r = [corpus_mean(p, "rmse").get(c, np.nan) for p in ps]
                p = [corpus_mean(q, "pcc").get(c, np.nan) for q in ps]
                cells.append(f"{np.mean(r):.3f} / {np.mean(p):.3f}")
            parts.append(f"| {m} | {run} | {len(v)} | " + " | ".join(cells) + " |")
    return parts


HELD = set(datasets.HELD_OUT)


def figure(probes, adapted, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3d3d3a", "xtick.color": "#5f5e5a",
                         "ytick.color": "#5f5e5a", "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, title in [(axes[0], "test_rmse", "Seen speakers (text-disjoint test)"),
                           (axes[1], "unseen_rmse", "Unseen speakers (usc_F1, 5emo_kf)")]:
        for m in sorted({r["model"] for r in probes}):
            rs = sorted([r for r in probes if r["model"] == m], key=lambda r: r["layers"])
            col = COLORS.get(m, "#5f5e5a")
            ax.plot([r["gflops"] for r in rs], [r[key] for r in rs], color=col, lw=2, marker="o", ms=3.5,
                    label=f"{LABELS.get(m, m)} probe prefix 1..k")
            b = [r for r in rs if r["best"]]
            if b:
                ax.plot(b[0]["gflops"], b[0][key], "*", ms=14, color=col, mec="white", mew=1.5)
            ad = sorted([a for a in adapted if a["model"] == m and a["kind"] == "adapted"], key=lambda a: a["gflops"])
            if ad:  # best adapted per depth (by seen test is not allowed: use the run's own selection, plot all)
                ax.plot([a["gflops"] for a in ad], [a[key] for a in ad], "^", ms=7, color=col, mec="white", mew=1.2)
            pr = sorted([a for a in adapted if a["model"] == m and a["kind"] == "pruned"], key=lambda a: a["gflops"])
            if pr:
                ax.plot([a["gflops"] for a in pr], [a[key] for a in pr], ":", lw=2, marker="D", ms=7, color=col,
                        mec="white", mew=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Inference GFLOPs per second of audio")
    axes[0].set_ylabel("Test RMSE (mm), lower is better")
    handles, labels = axes[0].get_legend_handles_labels()
    extra = [Line2D([], [], ls="", marker="*", ms=12, color="#5f5e5a", label="best probe layer (validation)"),
             Line2D([], [], ls="", marker="^", ms=8, color="#5f5e5a", label="adapted runs"),
             Line2D([], [], ls=":", lw=2, marker="D", ms=7, color="#5f5e5a", label="head + FFN pruned")]
    fig.legend(handles + extra, labels + [h.get_label() for h in extra], loc="lower center", ncol=3, frameon=False,
               fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_multi_rmse_vs_flops.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(datasets.XLSR_EMA / "analysis_multi"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zs, probes, adapted = zero_shot_rows(), probe_rows(), adapted_rows()
    f2, f3, f4, s = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}"), (lambda v: f"{v:.4f}"), str
    parts = ["# Multi-speaker EMA (USC-TIMIT EMA + USC EMA_5EMO) -- results\n"]
    if zs:
        parts.append("## Zero-shot: MNGU0-trained models on the seven new speakers (mean over speakers)\n")
        parts.append(md_table(zs, [("model", ("model", s)), ("zero-shot PCC", ("zero_shot_pcc", f3)),
                                   ("affine RMSE mm", ("affine_rmse", f3)), ("affine PCC", ("affine_pcc", f3)),
                                   ("linear RMSE mm", ("linear_rmse", f3)), ("linear PCC", ("linear_pcc", f3))]))
        spk = sorted(zs[0]["per_speaker"])
        parts.append("\nZero-shot PCC per speaker:\n")
        parts.append("| model | " + " | ".join(spk) + " |")
        parts.append("|---|" + "---|" * len(spk))
        for z in zs:
            parts.append(f"| {z['model']} | " + " | ".join(f"{z['per_speaker'][x]['zero_shot_pcc']:.3f}" for x in spk) + " |")
    if probes:
        best = [r for r in probes if r["best"]]
        parts.append("\n## Probes trained on the five training speakers: best layer by validation\n")
        parts.append(md_table(best, [("model", ("model", s)), ("k", ("layers", s)), ("GFLOPs/s", ("gflops", f2)),
                                     ("seen RMSE mm", ("test_rmse", f3)), ("seen PCC", ("test_pcc", f4)),
                                     ("unseen RMSE mm", ("unseen_rmse", f3)), ("unseen PCC", ("unseen_pcc", f4))]))
    if adapted:
        parts.append("\n## Adapted and pruned runs\n")
        parts.append(md_table(sorted(adapted, key=lambda a: (a["model"], a["kind"], a["gflops"])),
                              [("model", ("model", s)), ("run", ("run", s)), ("GFLOPs/s", ("gflops", f2)),
                               ("seen RMSE", ("test_rmse", f3)), ("seen PCC", ("test_pcc", f4)),
                               ("unseen RMSE", ("unseen_rmse", f3)), ("unseen PCC", ("unseen_pcc", f4))]))
    groups = {}
    for a in adapted:
        if a["kind"] == "adapted" and a["run"][-3:-1] == "_s":
            groups.setdefault((a["model"], a["run"][:-3]), []).append(a)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        parts.append("\n## Seed variation (mean +/- std over seeds)\n")
        parts.append("| model | configuration | seeds | GFLOPs/s | seen RMSE | seen PCC | unseen RMSE | unseen PCC |")
        parts.append("|---|---|---|---|---|---|---|---|")
        for (m, cfg), v in sorted(multi.items()):
            ms = lambda k: f"{np.mean([x[k] for x in v]):.3f} +/- {np.std([x[k] for x in v], ddof=1):.3f}"  # noqa: E731
            parts.append(f"| {m} | {cfg} | {len(v)} | {v[0]['gflops']:.1f} | {ms('test_rmse')} | {ms('test_pcc')} | "
                         f"{ms('unseen_rmse')} | {ms('unseen_pcc')} |")
    pf = [(d.name, f) for d in (sorted(FEAT.iterdir()) if FEAT.exists() else [])
          for f in sorted(d.glob("prune_from*_to*.json"))]
    for model, f in pf:
        ev = json.loads(f.read_text()).get("eval", {})
        sizes = sorted({v["n_layers"] for v in ev.values()}, reverse=True)
        if not sizes:
            continue
        parts.append(f"\n## Non-contiguous selection, {LABELS.get(model, model)} ({f.stem}; ridge probe, seen / unseen RMSE mm)\n")
        parts.append("| layers kept | prefix | greedy | block influence | greedy subset |")
        parts.append("|---|---|---|---|---|")
        for n in sizes:
            cell = lambda k: (f"{ev[k]['test']['rmse']:.3f} / {ev[k]['test_unseen']['rmse']:.3f}"  # noqa: E731
                              if k in ev and "test_unseen" in ev[k] else "-")
            parts.append(f"| {n} | {cell(f'prefix_{n}')} | {cell(f'greedy_{n}')} | {cell(f'bi_{n}')} | "
                         f"{ev.get(f'greedy_{n}', {}).get('layers', [])} |")
    parts += corpus_breakdown(zs, adapted)
    if probes:
        figure(probes, adapted, out)
    (out / "summary.md").write_text("\n".join(parts) + "\n")
    (out / "rows.json").write_text(json.dumps({"zero_shot": zs, "probes": probes, "adapted": adapted}, indent=1))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
