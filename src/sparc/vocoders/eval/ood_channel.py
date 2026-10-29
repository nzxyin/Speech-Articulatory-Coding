"""Phase 8.1: channel ceiling and input path of the out-of-domain speaker-similarity gap (no training).

Question (``docs/speaker_embedding_plan.md`` "Phase 8", 8.1): own-speaker resynthesis reaches only ~0.7 of the
ground-truth same-speaker similarity on VCTK / GLOBE. Is that mainly the *channel* (what any vocoder of this kind does to
the recording, so even a perfect voice would score low), the *voice* (the speaker code / generator), or does the *content
path* (the articulatory inverter's features) carry or limit speaker identity?

Own-speaker items of the cloning plans (the standard own-speaker setting, K = 5 references; GLOBE: all of its 2), the
speaker's other utterances being the scoring references (held-out ones where they exist, as :mod:`ood_gap`). Conditions:

* ``std``         standard resynthesis of the target utterance (code from the references) with the ``p5_fv`` HiFi-GAN;
* ``gt``          the recording itself (ground-truth level);
* (i) channel ceiling, ``chan_*``: the real recording of the target utterance through a separate mel-input vocoder, the
  Vocos mel copy synthesis (``charactr/vocos-mel-24khz``, the reference of the vocoder comparison). Variants:
  ``chan_vocos`` (the recording at the evaluation level) and ``chan_vocos_lm`` (the recording first scaled to the RMS of
  the standard resynthesis, i.e. loudness matched to the generator's output). No articulatory bottleneck is involved, so
  the recording's channel and Vocos's rendering are all that is left;
* (ii) own-speaker control, ``ctrl_*``: the target's own content features with F0 and loudness taken from *another*
  utterance of the same speaker. ``ctrl_contour``: the other utterance's voiced ln-F0 sequence stretched over the target's
  voiced frames and its loudness curve stretched over the target's frames; ``ctrl_level``: the target's contours with the
  other utterance's median F0 and median loudness level. The target's voicing, periodicity and the 12 articulatory
  channels are kept. (The generator divides F0 by the utterance median, so the median-level change of ``ctrl_level``
  acts through loudness only.) If a swap of this kind moves similarity, the standard score depends on the prosody and level
  of the very utterance, not on the voice;
* (iii) content-path control, ``swap_content``: the content features (all 15 channels) of an utterance of a *different*
  speaker of the same pool and split, with F0 range matched to the target speaker's references (median kept, as the
  generator is speaker-relative) and the target speaker's standard code. If the content features carry the content
  speaker's identity, similarity to the target falls; if the standard own-speaker score is limited by the target's own
  content features, it rises.

Metrics: cosine to the speaker's other ground-truth utterances with ECAPA, WavLM-SV and CAM++ (``holdout``); the
fraction of the ground-truth same-speaker similarity (``frac_of_gt``); speaker-level bootstrap intervals; UTMOS. The
primary similarity is the mean over the three verifiers per speaker.

Selection (validation splits only, ``docs/spkemb_validation_splits.json``): the channel variant is the one with the
highest validation fraction of ground truth (the ceiling the channel allows; this is the choice least favourable to a
"channel" reading) and the control variant of (ii) the one with the lowest validation similarity (the strongest
perturbation, least favourable to "(ii) does not move"). Test items render and report only the chosen variants.

Interpretation rule (pre-registered, VCTK validation half, fixed before any number): the gap is mainly channel if the
channel-ceiling similarity is <= 0.85 of ground truth, mainly voice if >= 0.95, mixed in between. The content path limits
the gap only if condition (iii) differs from standard by >= 0.25 of the remaining gap (G - A: ground truth minus standard)
with the bootstrap interval of the difference excluding 0 while (ii) does not (difference below 0.25 of the gap or
interval including 0).

Usage (GPU)::

    python -m sparc.vocoders.eval.ood_channel run --run <run> --out <out> --split val --pool vctk
    python -m sparc.vocoders.eval.ood_channel select --out <out>              # CPU, writes choice.json
    python -m sparc.vocoders.eval.ood_channel run --run <run> --out <out> --split test --pool vctk
    python -m sparc.vocoders.eval.ood_channel summarize --out <out>           # CPU
"""

import argparse
import json
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sparc.vocoders.constants import F0_CHANNEL, F0_MAX_HZ, F0_MIN_HZ, LOUDNESS_CHANNEL, SAMPLE_RATE
from sparc.vocoders.eval.ood_gap import GAIN_DB, PHASE4_ROOT, STORED_ITEMS, Context, boot_stat, render_code, stable_seed, stored_check

logger = logging.getLogger(__name__)

SPLITS = "docs/spkemb_validation_splits.json"
MODELS = ("ecapa", "wavlm_sv", "holdout")
VERIFIER_LABEL = {"ecapa": "ECAPA", "wavlm_sv": "WavLM-SV", "holdout": "CAM++", "mean": "mean of the three"}
POOLS = {"val": {"libritts_dev": "libritts_dev", "vctk": "vctk", "globe_spk": "globe_spk"},
         "test": {"libritts": "libritts", "vctk": "vctk", "globe": "globe"}}
SPLIT_KEY = {"libritts_dev": "libritts_dev", "vctk": "vctk", "globe_spk": "globe_spk", "libritts": "libritts", "globe": "globe"}
CHANNEL_VARIANTS = ("chan_vocos", "chan_vocos_lm")
CONTROL_VARIANTS = ("ctrl_contour", "ctrl_level")
CONTENT = "swap_content"
COND_ORDER = ("std", "gt", *CHANNEL_VARIANTS, *CONTROL_VARIANTS, CONTENT)
LABEL = {"std": "standard resynthesis", "gt": "ground truth (recording)", "chan_vocos": "(i) recording through Vocos mel copy synthesis",
         "chan_vocos_lm": "(i) same, recording loudness-matched to the standard output", "ctrl_contour": "(ii) F0 and loudness contour of another own utterance",
         "ctrl_level": "(ii) median F0 / loudness level of another own utterance", CONTENT: "(iii) other speaker's content, target F0 range and code"}
VOCOS_REPO = "models--charactr--vocos-mel-24khz"
VOCOS_WEIGHTS_SHA = "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a"
VOCOS_SNAPSHOT = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
HUB = "/data/user_data/xoy/sparc-vocoders/hf_hub"  # private pinned cache (the shared /data/hf_cache blob was purged)
GAP_RULE = {"channel": 0.85, "voice": 0.95, "content_fraction": 0.25}


# ------------------------------------------------------------------------------------------------ pure helpers


def stretch(x: np.ndarray, n: int) -> np.ndarray:
    """Linear resampling of a 1-D sequence to ``n`` points (the first and last samples stay fixed)."""
    x = np.asarray(x, dtype=np.float64)
    if n <= 0:
        return np.zeros(0)
    if len(x) == 1:
        return np.full(n, x[0])
    return np.interp(np.linspace(0.0, len(x) - 1.0, n), np.arange(len(x)), x)


def voiced_mask(feats: np.ndarray) -> np.ndarray:
    from sparc.vocoders.eval.clone import voiced

    return voiced(feats)


def swap_contour(feats: np.ndarray, other: np.ndarray) -> np.ndarray:
    """``feats [15, T]`` with F0 and loudness taken from ``other`` (another utterance of the same speaker): the other's voiced
    ln-F0 sequence stretched over this utterance's voiced frames, its loudness curve stretched over all frames."""
    out = np.array(feats, dtype=np.float32, copy=True)
    v, vo = voiced_mask(feats), voiced_mask(other)
    if v.any() and vo.any():
        ln = np.log(other[F0_CHANNEL][vo].astype(np.float64))
        out[F0_CHANNEL, v] = np.clip(np.exp(stretch(ln, int(v.sum()))), F0_MIN_HZ, F0_MAX_HZ).astype(np.float32)
    out[LOUDNESS_CHANNEL] = stretch(other[LOUDNESS_CHANNEL], feats.shape[1]).astype(np.float32)
    return out


def swap_level(feats: np.ndarray, other: np.ndarray) -> np.ndarray:
    """``feats`` with the voiced median F0 and the voiced median loudness level of ``other`` (contours of ``feats`` kept)."""
    from sparc.vocoders.eval.clone import f0_stats, loudness_stat, match_f0, match_loudness

    own_med, own_rng = f0_stats([feats])
    other_med = f0_stats([other])[0]
    out = match_f0(feats, other_med, own_rng, clip=(1.0, 1.0)) if np.isfinite(own_med) and np.isfinite(other_med) else np.array(feats, copy=True)
    return match_loudness(out, loudness_stat([other]))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))) + 1e-20))


def pick_items(plan: dict[str, Any], speakers: set[str]) -> list[dict[str, Any]]:
    """All own items of the given speakers, sorted by ``own_id``."""
    return [o for o in sorted(plan["own"], key=lambda o: o["own_id"]) if o["speaker"] in speakers]


def donor_item(items: Sequence[dict[str, Any]], o: dict[str, Any]) -> dict[str, Any]:
    """A seeded-random own item of a different speaker (its content utterance gives the content features)."""
    others = [x for x in items if x["speaker"] != o["speaker"]]
    return others[int(np.random.default_rng(stable_seed("channel-donor", o["own_id"])).integers(len(others)))]


# ------------------------------------------------------------------------------------------------ Vocos


class VocosCopy:
    """``charactr/vocos-mel-24khz`` copy synthesis, loaded from the pinned hub-cache blobs (checked by SHA-256)."""

    def __init__(self, device):
        import hashlib

        import torch
        from vocos import Vocos

        snap = Path(HUB) / VOCOS_REPO / "snapshots" / VOCOS_SNAPSHOT
        weights, config = snap / "pytorch_model.bin", snap / "config.yaml"
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != VOCOS_WEIGHTS_SHA:
            raise ValueError(f"{weights} has sha256 {digest}")
        model = Vocos.from_hparams(str(config))
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = model.eval().to(device)
        self.device = device
        self.weights_sha256 = digest

    def synth(self, wav24: np.ndarray) -> np.ndarray:
        import torch

        wav = np.ascontiguousarray(wav24, dtype=np.float32)
        with torch.inference_mode():
            audio = torch.from_numpy(wav)[None].to(self.device)
            out = self.model.decode(self.model.feature_extractor(audio))[0].float().cpu().numpy()
        return out[: len(wav)] if len(out) >= len(wav) else np.concatenate([out, np.zeros(len(wav) - len(out), dtype=out.dtype)])


# ------------------------------------------------------------------------------------------------ run


def run_pool(args: argparse.Namespace) -> None:
    import soundfile as sf
    import torch

    from sparc.vocoders.eval.clone import Matching, apply_matching

    plan_name = POOLS[args.split][args.pool]
    ctx = Context(args.run, plan_name, args.root, args.device, MODELS)
    speakers = set(json.loads(Path(args.splits).read_text())["cloning"][args.split][SPLIT_KEY[plan_name]])
    items = pick_items(ctx.plan, speakers)
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out) / args.split / args.pool
    channel, control = list(CHANNEL_VARIANTS), list(CONTROL_VARIANTS)
    if args.split == "test":
        choice = json.loads((Path(args.out) / "choice.json").read_text())
        channel, control = [choice["channel"]], [choice["control"]]
    vocos = VocosCopy(ctx.device)
    start = time.time()
    for k, o in enumerate(items):
        if k % args.nshards != args.shard:
            continue
        row_path = out / "rows" / f"{o['own_id']}.json"
        if row_path.is_file():
            continue
        it = ctx.own_item(o)
        z_a = ctx.codes(it)[0]
        target = ctx.utterance(it.content_id)
        other = ctx.utterance(it.code_ids[0])["feats"]
        donor = donor_item(items, o)
        d_feats = ctx.utterance(donor["content_id"])["feats"]
        stats = Matching.from_features(ctx.conds[plan_name].clean_features(it.code_ids))
        d_feats = apply_matching(d_feats, stats, f0=True, loud=False, ema=False, f0_relative=ctx.renderer.rel_f0)
        std = render_code(ctx.module, ctx.device, target["feats"], z_a)
        audio = np.asarray(target["audio"], dtype=np.float32)
        wavs: dict[str, np.ndarray] = {"std": std, "gt": audio}
        if "chan_vocos" in channel:
            wavs["chan_vocos"] = vocos.synth(audio)
        if "chan_vocos_lm" in channel:
            wavs["chan_vocos_lm"] = vocos.synth(audio * np.float32(rms(std) / rms(audio)))
        if "ctrl_contour" in control:
            wavs["ctrl_contour"] = render_code(ctx.module, ctx.device, swap_contour(target["feats"], other), z_a)
        if "ctrl_level" in control:
            wavs["ctrl_level"] = render_code(ctx.module, ctx.device, swap_level(target["feats"], other), z_a)
        wavs[CONTENT] = render_code(ctx.module, ctx.device, d_feats, z_a)
        conds = list(wavs)
        ws = [wavs[c] for c in conds]
        sims = ctx.sims(ctx.embed(ws), o, it.code_ids)
        mos = ctx.utmos(ws)
        row: dict[str, Any] = {"pool": args.pool, "split": args.split, "own_id": o["own_id"], "speaker": o["speaker"], "content_id": o["content_id"],
                               "donor_speaker": donor["speaker"], "dur_s": o["dur_s"], "n_code_refs": len(it.code_ids),
                               "n_held": len(ctx.reference_sets(o, it.code_ids)[1])}
        for j, c in enumerate(conds):
            row[f"{c}/utmos"] = float(mos[j])
            for key, v in sims.items():
                row[f"{c}/{key}"] = float(v[j])
        if args.save_wav:
            for c in conds:
                p = out / "wav" / c / f"{o['own_id']}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), wavs[c], SAMPLE_RATE, subtype="PCM_16")
        row_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text(json.dumps(row))
        logger.info("%s/%s %s done (%.0f s)", args.split, args.pool, o["own_id"], time.time() - start)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ analysis


def load_rows(out: Path, split: str, pool: str) -> pd.DataFrame:
    paths = sorted((out / split / pool / "rows").glob("*.json"))
    return pd.DataFrame([json.loads(p.read_text()) for p in paths])


def with_sim(df: pd.DataFrame, conds: Sequence[str]) -> pd.DataFrame:
    """``<cond>/<verifier>/sim`` = held-out references where they exist else all, and ``<cond>/mean/sim`` = verifier mean."""
    df = df.copy()
    for c in conds:
        for m in MODELS:
            df[f"{c}/{m}/sim"] = df[f"{c}/{m}/held"].where(df[f"{c}/{m}/held"].notna(), df[f"{c}/{m}/all"])
        df[f"{c}/mean/sim"] = df[[f"{c}/{m}/sim" for m in MODELS]].mean(axis=1)
    return df


def conditions(df: pd.DataFrame) -> list[str]:
    return [c for c in COND_ORDER if f"{c}/utmos" in df]


def speaker_table(df: pd.DataFrame, c: str, v: str) -> pd.DataFrame:
    """Per speaker: the condition's similarity ``x``, standard ``a``, ground truth ``g``."""
    ps = df.groupby("speaker")[[f"{c}/{v}/sim", f"std/{v}/sim", f"gt/{v}/sim"]].mean()
    ps.columns = ["x", "a", "g"]
    return ps


def summarize_pool(df: pd.DataFrame, n_boot: int = 4000) -> dict[str, Any]:
    conds = conditions(df)
    df = with_sim(df, conds)
    res: dict[str, Any] = {"n_items": int(len(df)), "n_speakers": int(df["speaker"].nunique()), "held_out_refs": bool(df["n_held"].min() > 0)}
    for v in (*MODELS, "mean"):
        t: dict[str, Any] = {}
        for c in conds:
            ps = speaker_table(df, c, v)
            t[c] = {"sim": boot_stat(ps[["x"]], lambda d: d["x"], n_boot), "frac_of_gt": boot_stat(ps, lambda d: d["x"] / d["g"], n_boot),
                    "diff_std": boot_stat(ps, lambda d: d["x"] - d["a"], n_boot),
                    "diff_std_over_gap": boot_stat(ps, lambda d: (d["x"] - d["a"]) / (d["g"] - d["a"]) if d["g"] - d["a"] > 1e-9 else math.nan, n_boot)}
        res[v] = t
    res["utmos"] = {c: float(df[f"{c}/utmos"].mean()) for c in conds}
    return res


def select_variants(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Channel variant: highest validation fraction of ground truth (mean verifier, per speaker, pools equally weighted).
    Control variant: lowest validation similarity (strongest perturbation)."""
    scores: dict[str, dict[str, float]] = {"channel": {}, "control": {}}
    for kind, variants in (("channel", CHANNEL_VARIANTS), ("control", CONTROL_VARIANTS)):
        for c in variants:
            vals = []
            for df in frames.values():
                if f"{c}/utmos" not in df:
                    continue
                d = with_sim(df, ["std", "gt", c])
                ps = speaker_table(d, c, "mean")
                vals.append(float((ps["x"] / ps["g"]).mean()) if kind == "channel" else float(ps["x"].mean()))
            scores[kind][c] = float(np.mean(vals))
    return {"channel": max(scores["channel"], key=scores["channel"].get), "control": min(scores["control"], key=scores["control"].get), "scores": scores}


def apply_rule(res: dict[str, Any], channel: str, control: str) -> dict[str, Any]:
    """The pre-registered rule on one pool's summary (mean of the three verifiers)."""
    t = res["mean"]
    frac = t[channel]["frac_of_gt"]
    verdict = "mainly channel" if frac["mean"] <= GAP_RULE["channel"] else "mainly voice" if frac["mean"] >= GAP_RULE["voice"] else "mixed"
    gap = {"std_frac_of_gt": t["std"]["frac_of_gt"], "std_sim": t["std"]["sim"]["mean"], "gt_sim": t["gt"]["sim"]["mean"]}

    def moves(c: str) -> dict[str, Any]:
        gain = t[c]["diff_std_over_gap"]
        d = t[c]["diff_std"]
        return {"diff": d, "of_gap": gain, "moves": bool(abs(gain["mean"]) >= GAP_RULE["content_fraction"] and (d["lo"] > 0 or d["hi"] < 0))}

    iii, ii = moves(CONTENT), moves(control)
    return {"verdict": verdict, "channel_variant": channel, "channel_frac_of_gt": frac, **gap, "iii": iii, "ii": ii,
            "content_path_limits_gap": bool(iii["moves"] and not ii["moves"])}


# ------------------------------------------------------------------------------------------------ report


def fmt(d: dict[str, float] | None, digits: int = 3) -> str:
    if d is None or not np.isfinite(d.get("mean", math.nan)):
        return "n/a"
    lo, hi = d.get("lo", math.nan), d.get("hi", math.nan)
    return f"{d['mean']:.{digits}f}" + (f" [{lo:.{digits}f}, {hi:.{digits}f}]" if np.isfinite(lo) else "")


def render_markdown(results: dict[str, Any], choice: dict[str, Any], rules: dict[str, Any]) -> str:
    lines = ["# Phase 8.1 channel ceiling and input path", "", f"Chosen on validation: channel `{choice['channel']}`, control `{choice['control']}` "
             f"(scores {json.dumps(choice['scores'])}).", ""]
    for key, res in results.items():
        lines += [f"## {key} ({res['n_items']} items, {res['n_speakers']} speakers; similarity to "
                  f"{'held-out references' if res['held_out_refs'] else 'all references (none held out)'})", ""]
        for v in ("mean", *MODELS):
            t = res[v]
            lines += [f"### {VERIFIER_LABEL[v]}", "", "| condition | similarity | fraction of ground truth | difference to standard | as fraction of the standard gap | UTMOS |",
                      "|---|---|---|---|---|---|"]
            for c in conditions_in(res, v):
                diff = "" if c in ("std", "gt") else fmt(t[c]["diff_std"])
                gap = "" if c in ("std", "gt") else fmt(t[c]["diff_std_over_gap"], 2)
                lines.append(f"| {LABEL[c]} | {fmt(t[c]['sim'])} | {fmt(t[c]['frac_of_gt'])} | {diff} | {gap} | {res['utmos'][c]:.2f} |")
            lines.append("")
        if key in rules:
            lines += ["Rule: " + json.dumps(rules[key], default=float), ""]
        if res.get("check"):
            lines += ["Standard vs stored own-speaker rows: " + json.dumps(res["check"]), ""]
    return "\n".join(lines) + "\n"


def conditions_in(res: dict[str, Any], v: str) -> list[str]:
    return [c for c in COND_ORDER if c in res[v]]


def run_select(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    frames = {p: load_rows(out, "val", p) for p in POOLS["val"] if (out / "val" / p / "rows").is_dir()}
    choice = select_variants(frames)
    (out / "choice.json").write_text(json.dumps(choice, indent=2))
    print(json.dumps(choice, indent=2))
    return choice


def run_summarize(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    choice = json.loads((out / "choice.json").read_text())
    results, rules = {}, {}
    for split in ("val", "test"):
        for pool in POOLS[split]:
            if not (out / split / pool / "rows").is_dir():
                continue
            df = load_rows(out, split, pool)
            res = summarize_pool(df, args.n_boot)
            res["check"] = stored_check(df.rename(columns=lambda c: c.replace("std/", "a_refs/")), POOLS[split][pool])
            key = f"{split}/{pool}"
            results[key] = res
            rules[key] = apply_rule(res, choice["channel"], choice["control"])
    (out / "summary.json").write_text(json.dumps({"choice": choice, "pools": results, "rules": rules}, indent=2, default=float))
    (out / "summary.md").write_text(render_markdown(results, choice, rules))
    print((out / "summary.md").read_text())
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--split", required=True, choices=("val", "test"))
    s.add_argument("--pool", required=True)
    s.add_argument("--root", default=PHASE4_ROOT)
    s.add_argument("--splits", default=SPLITS)
    s.add_argument("--device", default="auto")
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--save-wav", action="store_true")
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=4000)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.cmd == "run" and args.pool not in POOLS[args.split]:
        raise SystemExit(f"pool {args.pool} not in {sorted(POOLS[args.split])} for split {args.split}")
    {"run": run_pool, "select": run_select, "summarize": run_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
