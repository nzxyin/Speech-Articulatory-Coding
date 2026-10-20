"""Speaker-embedding Phase 3 resynthesis tables: UTMOS, Whisper WER and speaker similarity of T1/T2/T3 per arm.

Reuses the stages and statistics of the vocoder evaluation (``stages.py``, ``aggregate.py``) and only redirects their
outputs: every arm (``--label``) is scored in its own eval root ``<TABLES>/<label>/work`` (results, embeddings, DONE
markers), the ground-truth embeddings are computed once under ``<TABLES>/_gt``. Nothing is written under the vocoder
comparison's ``runs`` or ``eval`` directories (they are only read).

Speaker similarity is defined as in docs/vocoders/EVALUATION.md 4.4, for each SV model (ECAPA, WavLM-SV):
``sim_same`` = cosine to the gt embedding of the same utterance; ``sim_spk`` = cosine to the mean L2-normalized gt
embedding of the speaker's other test utterances, excluding the target and its T2 reference. ``*_mean`` columns are the
mean of the two models, per utterance (the pre-registered Phase 3 rule uses it).

Commands (run through ``scripts/spkemb/p3_job.sh -m sparc.vocoders.eval.p3_tables ...`` on a GPU node):

* ``predict --label b3``            synthesize T1/T2/T3 of test.clean with the arm's final checkpoint (``sparc-predict``
  path, skip-existing, final checkpoint required); predictions land in ``<run>/predictions/test.clean/<cond>/``.
* ``score --label b3 [--stages gt,spk,utmos,asr] [--conditions T1,T2,T3]``   resumable chunked scoring, then
  ``per_utt.parquet``; shard by ``--stages``/``--conditions`` to keep jobs short.
* ``finalize --label b3``           (CPU) rebuild ``per_utt.parquet`` from the stored parts.
* ``table --labels b0 b1 ... [--pairs b1:b0,...] [--reference-csv table.csv --ref-map b0=a100k_hifigan]``  (CPU).

Labels: ``b0`` = ablation_100k hifigan baseline (read-only run), ``b1``..``b6`` = ``phase3/runs/p3_bN/hifigan``; any other
label needs ``--run <runs_root>/<experiment>/<vocoder>``.
"""

import argparse
import logging
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from sparc.vocoders.eval.aggregate import (
    METRICS,
    Metric,
    fmt,
    load_embeddings,
    paired_cell,
    similarity_columns,
    summarize_cell,
)
from sparc.vocoders.eval.io import (
    EXIT_STOPPED,
    StopFlag,
    StopRequested,
    base_columns,
    condition_ids,
    read_parquet_parts,
    write_csv,
    write_parquet,
    write_text,
)

logger = logging.getLogger(__name__)

TABLES = Path("/data/user_data/xoy/spkemb/phase3/eval/tables")
P3_RUNS = Path("/data/user_data/xoy/spkemb/phase3/runs")
B0_RUNS = Path("/data/user_data/xoy/sparc-vocoders/runs")
READ_ONLY_ROOT = Path("/data/user_data/xoy/sparc-vocoders")
SPLIT = "test.clean"
CONDITIONS = ("T1", "T2", "T3")
SV_MODELS = ("ecapa", "wavlm_sv")
GT = "gt"
GT_DIR = "_gt"
SUMMARY_DIR = "_summary"
DEFAULT_PAIRS = "b1:b0,b2:b1,b3:b2,b4:b3,b5:b3"
BOOT = {"n_boot": 10000, "seed": 0, "ci": 0.95}  # the settings of the existing evaluation (eval.aggregate)

_METRIC = {m.key: m for m in METRICS}
SIM_COLUMNS = [f"sim_{kind}_{m}" for m in (*SV_MODELS, "mean") for kind in ("same", "spk")]
TABLE_METRICS: tuple[Metric, ...] = (
    _METRIC["utmos"],
    _METRIC["wer"],
    _METRIC["cer"],
    *(Metric(c, c.replace("_", " ").replace("sim ", "Sim "), "quality", 3) for c in SIM_COLUMNS),
)
PAIRED_KEYS = ("sim_same_mean", "sim_spk_mean", *SIM_COLUMNS[:4], "utmos", "wer")
PAIRED_COLUMNS = ["a", "b", "condition", "metric", "n_utt", "n_speakers", "diff", "lo", "hi", "ci_excludes_zero", "frac_speakers_same_direction"]
ASR_COLUMNS = ("word_errors", "word_ref_len", "char_errors", "char_ref_len")


# ----------------------------------------------------------------------------------------------- run registry


@dataclass(frozen=True)
class RunSpec:
    label: str
    runs_root: Path
    experiment: str
    vocoder: str

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.experiment / self.vocoder

    def system(self) -> dict:
        return {
            "kind": "vocoder",
            "sr": 24000,
            "conditions": list(CONDITIONS),
            "head": "refit",
            "experiment": self.experiment,
            "vocoder": self.vocoder,
            "ckpt": None,
            "overrides": [],
            "label": self.label,
        }


def resolve_run(label: str, run: str | Path | None = None) -> RunSpec:
    """``b0`` -> the 100k baseline, ``bN`` -> ``phase3/runs/p3_bN/hifigan``; ``run`` = ``<runs_root>/<experiment>/<vocoder>``."""
    if run is not None:
        p = Path(run)
        return RunSpec(label, p.parent.parent, p.parent.name, p.name)
    if label == "b0":
        return RunSpec(label, B0_RUNS, "ablation_100k", "hifigan")
    m = re.fullmatch(r"b([1-9])", label)
    if m:
        return RunSpec(label, P3_RUNS, f"p3_b{m.group(1)}", "hifigan")
    raise ValueError(f"unknown label {label!r}: use b0, b1..b9 or pass --run <runs_root>/<experiment>/<vocoder>")


def label_dir(label: str, tables: Path = TABLES) -> Path:
    return Path(tables) / label


# ----------------------------------------------------------------------------------------------- contexts


def make_cfg(eval_root: Path, runs_root: Path, systems: dict[str, dict]) -> DictConfig:
    """The packaged ``eval_config`` with the given roots and system registry (no CLI, no Hydra app)."""
    from sparc.vocoders.eval.systems import _quote, compose_config

    cfg = compose_config(
        "eval_config",
        ["stage=aggregate", "eval_systems=ablations", f"paths.eval_root={_quote(eval_root)}", f"paths.runs_root={_quote(runs_root)}"],
    )
    OmegaConf.set_struct(cfg, False)
    cfg.eval.systems = OmegaConf.create(systems)
    cfg.eval.split = SPLIT
    cfg.eval.limit = None
    cfg.eval.spk_models = list(SV_MODELS)
    return cfg


GT_SPEC = {"kind": "gt", "sr": 24000, "conditions": ["T1"], "head": "refit", "label": "ground truth"}


def arm_context(run: RunSpec, stop: StopFlag | None = None, tables: Path = TABLES):
    from sparc.vocoders.eval.stages import EvalContext

    return EvalContext(make_cfg(label_dir(run.label, tables) / "work", run.runs_root, {GT: GT_SPEC, run.label: run.system()}), stop)


def gt_context(stop: StopFlag | None = None, tables: Path = TABLES):
    from sparc.vocoders.eval.stages import EvalContext

    return EvalContext(make_cfg(label_dir(GT_DIR, tables), B0_RUNS, {GT: GT_SPEC}), stop)


# ----------------------------------------------------------------------------------------------- pure assembly


def condition_frame(
    table: pd.DataFrame,
    condition: str,
    ids: Sequence[str],
    utmos: pd.DataFrame,
    asr: pd.DataFrame,
    sys_emb: dict[str, dict[str, np.ndarray]],
    gt_emb: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """One row per id of ``condition``: base columns, UTMOS, ASR counts and the similarities of every SV model.

    ``sys_emb[model]`` / ``gt_emb[model]`` map id -> embedding. ``sim_*_mean`` is the per-utterance mean of the models.
    """
    expected = condition_ids(table, condition, ids)
    frame = base_columns(table, expected, condition)
    frame["t2_ref"] = table.set_index("id").loc[expected, "ref_id"].to_numpy()
    if len(utmos):
        frame = frame.merge(utmos[["id", "utmos"]].drop_duplicates("id"), on="id", how="left")
    else:
        frame["utmos"] = np.nan
    if len(asr):
        frame = frame.merge(asr[["id", *ASR_COLUMNS]].drop_duplicates("id"), on="id", how="left")
    else:
        for c in ASR_COLUMNS:
            frame[c] = np.nan
    for model in SV_MODELS:
        if sys_emb.get(model) and gt_emb.get(model):
            same, spk = similarity_columns(frame, sys_emb[model], gt_emb[model], False)
        else:
            same = spk = np.full(len(frame), np.nan)
        frame[f"sim_same_{model}"] = same
        frame[f"sim_spk_{model}"] = spk
    for kind in ("same", "spk"):
        frame[f"sim_{kind}_mean"] = frame[[f"sim_{kind}_{m}" for m in SV_MODELS]].mean(axis=1, skipna=False)
    return frame


def completeness(frame: pd.DataFrame) -> dict[str, int]:
    """Number of missing values per reported column (0 = complete)."""
    return {c: int(frame[c].isna().sum()) for c in ("utmos", "word_errors", *SIM_COLUMNS) if c in frame}


# ----------------------------------------------------------------------------------------------- table and paired


def parse_pairs(text: str) -> list[tuple[str, str]]:
    out = []
    for item in text.split(","):
        if item.strip():
            a, b = item.strip().split(":")
            out.append((a, b))
    return out


def summary_table(cells: dict[tuple[str, str], pd.DataFrame], boot: dict | None = None) -> pd.DataFrame:
    """Mean (corpus ratio for WER/CER) and speaker-bootstrap CI95 per label and condition."""
    boot = boot or BOOT
    rows = []
    for (label, condition), frame in cells.items():
        s = summarize_cell(frame, boot["n_boot"], boot["seed"], boot["ci"], TABLE_METRICS)
        row: dict[str, Any] = {"system": label, "condition": condition, "n_utt": s["n_utt"], "n_speakers": s["n_speakers"]}
        for m in TABLE_METRICS:
            row |= {m.key: s[m.key]["point"], f"{m.key}_lo": s[m.key]["lo"], f"{m.key}_hi": s[m.key]["hi"], f"{m.key}_n": s[m.key]["n"]}
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(
    cells: dict[tuple[str, str], pd.DataFrame], pairs: Sequence[tuple[str, str]], conditions: Sequence[str] = CONDITIONS, boot: dict | None = None
) -> pd.DataFrame:
    """``a - b`` of :data:`PAIRED_KEYS` per pair and condition: ids paired, speakers resampled (39 clusters)."""
    boot = boot or BOOT
    metrics = tuple(m for m in TABLE_METRICS if m.key in PAIRED_KEYS)
    rows = []
    for a, b in pairs:
        for condition in conditions:
            if (a, condition) not in cells or (b, condition) not in cells:
                continue
            result = paired_cell(cells[(a, condition)], cells[(b, condition)], boot["n_boot"], boot["seed"], boot["ci"], metrics)
            for m in metrics:
                s = result.get(m.key)
                if s is None:
                    continue
                rows.append(
                    {
                        "a": a,
                        "b": b,
                        "condition": condition,
                        "metric": m.key,
                        "n_utt": s["n"],
                        "n_speakers": result["n_speakers"],
                        "diff": s["point"],
                        "lo": s["lo"],
                        "hi": s["hi"],
                        "ci_excludes_zero": bool(np.isfinite(s["lo"]) and (s["lo"] > 0 or s["hi"] < 0)),
                        "frac_speakers_same_direction": s["frac_same_direction"],
                    }
                )
    return pd.DataFrame(rows, columns=PAIRED_COLUMNS if not rows else None)


def render_table_md(table: pd.DataFrame) -> str:
    cols = [("utmos", 3), ("wer", 4), ("sim_same_ecapa", 3), ("sim_spk_ecapa", 3), ("sim_same_wavlm_sv", 3), ("sim_spk_wavlm_sv", 3), ("sim_same_mean", 3), ("sim_spk_mean", 3)]
    out = ["# Phase 3 resynthesis, LibriTTS-R test.clean", "", "Mean over utterances (WER: corpus level) with speaker-cluster bootstrap CI95 (10,000 resamples, seed 0). Each arm is a single training run.", ""]
    for condition in sorted(set(table["condition"])):
        sub = table[table["condition"] == condition]
        out += [f"## {condition}", "", "| arm | n | " + " | ".join(c for c, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
        for r in sub.to_dict("records"):
            cells = [fmt(r[c], r[f"{c}_lo"], r[f"{c}_hi"], d) for c, d in cols]
            out.append(f"| {r['system']} | {r['n_utt']} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def render_paired_md(paired: pd.DataFrame) -> str:
    out = ["# Paired differences a - b (speaker-level paired bootstrap CI95; * = CI excludes 0)", ""]
    for (a, b, condition), sub in paired.groupby(["a", "b", "condition"], sort=False):
        out += [f"## {a} - {b}, {condition} (n_utt {int(sub['n_utt'].max())}, {int(sub['n_speakers'].max())} speakers)", "", "| metric | diff [CI95] | speakers same direction |", "|---|---|---|"]
        for r in sub.to_dict("records"):
            digits = 4 if r["metric"] == "wer" else 3
            star = " *" if r["ci_excludes_zero"] else ""
            frac = r["frac_speakers_same_direction"]
            out.append(f"| {r['metric']} | {r['diff']:+.{digits}f} [{r['lo']:+.{digits}f}, {r['hi']:+.{digits}f}]{star} | {'n/a' if not np.isfinite(frac) else f'{frac:.2f}'} |")
        out.append("")
    return "\n".join(out)


def compare_with_reference(table: pd.DataFrame, reference: pd.DataFrame, ref_map: dict[str, str], tol: float = 5e-4) -> pd.DataFrame:
    """Rows ``label, condition, metric, mine, reference, diff`` for the metrics the two tables share (ECAPA = primary model
    of the existing tables, i.e. ``sim_same``/``sim_spk`` there and ``*_ecapa`` here)."""
    names = {"utmos": "utmos", "wer": "wer", "cer": "cer", "sim_same_ecapa": "sim_same", "sim_spk_ecapa": "sim_spk"}
    ref = reference[reference["stratum"] == "all"] if "stratum" in reference else reference
    rows = []
    for label, ref_name in ref_map.items():
        for condition in sorted(set(table[table["system"] == label]["condition"])):
            mine = table[(table["system"] == label) & (table["condition"] == condition)]
            theirs = ref[(ref["system"] == ref_name) & (ref["condition"] == condition)]
            if mine.empty or theirs.empty:
                continue
            for k_mine, k_ref in names.items():
                a, b = float(mine[k_mine].iloc[0]), float(theirs[k_ref].iloc[0])
                rows.append({"label": label, "condition": condition, "metric": k_mine, "mine": a, "reference": b, "diff": a - b, "ok": bool(abs(a - b) <= tol)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- IO commands


def check_writable_target(run: RunSpec) -> None:
    if READ_ONLY_ROOT in (run.runs_root, *run.runs_root.parents):
        raise SystemExit(f"{run.run_dir} is under the read-only {READ_ONLY_ROOT}: refusing to synthesize into it")


def cmd_predict(args: argparse.Namespace) -> int:
    from sparc.vocoders.eval.stages import run_stage

    run = resolve_run(args.label, args.run)
    check_writable_target(run)
    stop = StopFlag()
    with stop.installed():
        ctx = arm_context(run, stop, args.tables)
        try:
            run_stage(ctx, "synth", run.label, "all")
        except StopRequested as e:
            logger.warning("stopped: %s", e)
            return EXIT_STOPPED
    return 0


def _run_stages(ctx, system: str, stages: Sequence[str], conditions: Sequence[str]) -> None:
    from sparc.vocoders.eval.stages import run_stage

    for stage in stages:
        for condition in conditions:
            run_stage(ctx, stage, system, condition)


def cmd_score(args: argparse.Namespace) -> int:
    run = resolve_run(args.label, args.run)
    stages = [s for s in args.stages.split(",") if s]
    conditions = [c for c in args.conditions.split(",") if c]
    bad = [s for s in stages if s not in ("gt", "spk", "utmos", "asr")] + [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"bad --stages/--conditions: {bad}")
    stop = StopFlag()
    with stop.installed():
        try:
            if "gt" in stages or "spk" in stages:  # ground-truth embeddings of the shared root (no-op when finished)
                gctx = gt_context(stop, args.tables)
                _run_stages(gctx, GT, ["gt", "spk"], ["T1"])
            ctx = arm_context(run, stop, args.tables)
            _run_stages(ctx, run.label, [s for s in stages if s != "gt"], conditions)
        except StopRequested as e:
            logger.warning("stopped: %s", e)
            return EXIT_STOPPED
    finalize(run, args.tables)
    return 0


def finalize(run: RunSpec, tables: Path = TABLES) -> Path:
    """Writes ``<TABLES>/<label>/per_utt.parquet`` (all conditions) from the stored parts; reports missing values."""
    ctx = arm_context(run, None, tables)
    gctx = gt_context(None, tables)
    table, ids = ctx.table, ctx.ids
    gt_emb = {m: load_embeddings(gctx.paths.embeddings_dir(GT, "T1", m)) for m in SV_MODELS}
    frames = []
    for condition in CONDITIONS:
        sys_emb = {m: load_embeddings(ctx.paths.embeddings_dir(run.label, condition, m)) for m in SV_MODELS}
        utmos = read_parquet_parts(ctx.paths.results_dir(run.label, condition, "utmos"))
        asr = read_parquet_parts(ctx.paths.results_dir(run.label, condition, "asr"))
        frame = condition_frame(table, condition, ids, utmos, asr, sys_emb, gt_emb)
        frame.insert(0, "condition", condition)
        frame.insert(0, "label", run.label)
        miss = {k: v for k, v in completeness(frame).items() if v}
        logger.info("%s %s: %d utterances, missing %s", run.label, condition, len(frame), miss or "nothing")
        frames.append(frame)
    out = label_dir(run.label, tables) / "per_utt.parquet"
    write_parquet(out, pd.concat(frames, ignore_index=True))
    return out


def load_cells(labels: Sequence[str], tables: Path = TABLES, require_complete: bool = True) -> dict[tuple[str, str], pd.DataFrame]:
    cells = {}
    for label in labels:
        path = label_dir(label, tables) / "per_utt.parquet"
        if not path.is_file():
            raise SystemExit(f"{path} is missing: run `score --label {label}` first")
        frame = pd.read_parquet(path)
        for condition, sub in frame.groupby("condition", sort=False):
            gaps = {k: v for k, v in completeness(sub).items() if v}
            if gaps and require_complete:
                # utterances a model could not score are NaN in the means; the T2 reference-less speaker has no T2 row
                logger.warning("%s %s: missing values %s", label, condition, gaps)
            cells[(label, str(condition))] = sub.reset_index(drop=True)
    return cells


def cmd_table(args: argparse.Namespace) -> int:
    cells = load_cells(args.labels, args.tables)
    out = Path(args.out) if args.out else label_dir(SUMMARY_DIR, args.tables)
    table = summary_table(cells)
    write_csv(out / "table.csv", table)
    write_text(out / "table.md", render_table_md(table))
    pairs = [p for p in parse_pairs(args.pairs) if p[0] in args.labels and p[1] in args.labels]
    paired = paired_summary(cells, pairs)
    write_csv(out / "paired.csv", paired)
    write_text(out / "paired.md", render_paired_md(paired))
    if args.reference_csv:
        ref_map = dict(item.split("=") for item in args.ref_map.split(",")) if args.ref_map else {"b0": "a100k_hifigan"}
        cmp = compare_with_reference(table, pd.read_csv(args.reference_csv), ref_map)
        write_csv(out / "reference_check.csv", cmp)
        print(cmp.to_string(index=False))
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tables", type=Path, default=TABLES)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("predict", "score", "finalize"):
        s = sub.add_parser(name)
        s.add_argument("--label", required=True)
        s.add_argument("--run", default=None, help="<runs_root>/<experiment>/<vocoder> (default from the label)")
        if name == "score":
            s.add_argument("--stages", default="spk,utmos,asr")
            s.add_argument("--conditions", default=",".join(CONDITIONS))
    t = sub.add_parser("table")
    t.add_argument("--labels", nargs="+", required=True)
    t.add_argument("--pairs", default=DEFAULT_PAIRS)
    t.add_argument("--out", default=None)
    t.add_argument("--reference-csv", default=None)
    t.add_argument("--ref-map", default=None, help="label=system,... of the reference csv (default b0=a100k_hifigan)")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args = build_parser().parse_args(argv)
    if args.command == "finalize":
        print(finalize(resolve_run(args.label, args.run), args.tables))
        return
    code = {"predict": cmd_predict, "score": cmd_score, "table": cmd_table}[args.command](args)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
