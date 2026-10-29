"""Phase 8.2: constrained speaker-code fitting through the frozen generator (pre-registered in ``docs/speaker_embedding_plan.md``).

Builds on :mod:`ood_svfit` (same frozen ``p5_fv`` HiFi-GAN, same own-speaker items, same scorers). The 64-d speaker code is
fitted with Adam on a loss through the differentiable ECAPA-TDNN, using the speaker's *other* utterances only (the target
embedding is the mean ECAPA embedding of the speaker's code-reference utterances, never the target utterance):

    loss = -cos(ECAPA(synth crop), e_refs) + lam * |dz|^2 / |z0|^2 + nat * D_mel(synth crop, standard synth crop)

Arms (each one is a setting of ``lam`` and ``nat``):

* ``std``  the standard code (no fit);
* ``l2``   ``nat = 0``, ``lam > 0``: ECAPA-loss fit with an L2 penalty toward the standard code;
* ``nat``  ``lam > 0``, ``nat > 0``: the same plus the naturalness term;
* ``unc``  ``lam = 0``, ``nat = 0``: the unconstrained fit (reference).

**Naturalness term (one formulation, fixed before any tuning): log-mel statistics.** ``D_mel`` is the mean squared difference
between the per-band mean and per-band standard deviation over time of the natural-log mel spectrogram (24 kHz, 80 bands,
n_fft 1024, hop 480, 0 to 12 kHz) of the fitted output and of the standard output (code ``z0``) on the same content crop,
with the same noise seed. It does not use WavLM features. It is cheap (one extra no-grad forward per step), differentiable,
and penalizes exactly the global spectral changes (tilt, band energy, dynamics) that the unconstrained fits used to raise
the verifier (and that UTMOS punishes), while leaving frame-level detail free.

Verifiers. ECAPA is only the loss. Selection uses CAM++ only (``holdout``). WeSpeaker ResNet34 is held out from loss and
selection and is the verifier the advance rule is judged on; CAM++ closure is reported as the selection verifier. No third
verifier with documented non-VoxCeleb training data, trained without LibriTTS, VCTK and Common Voice, is available offline
(checked: ``/data/hf_cache``, ``/data/models``, the spkemb hf caches hold only VoxCeleb-trained speaker models: ECAPA, CAM++,
WeSpeaker ResNet34, pyannote embedding, WavLM-base-plus-sv).

Validation (``sweep`` then ``select``): per arm the (lr, lam, nat, steps) with the highest mean CAM++ similarity over the
validation pools (LibriTTS-R dev-clean, VCTK validation half, GLOBE real speakers ``globe_spk``) such that in every pool the
UTMOS drop against the standard code is <= 0.05 and the pooled WER rises by <= 0.5 points. The setting that is judged is the
better (CAM++) of the ``l2`` and ``nat`` arms. Test (``run`` for own-speaker items, ``swap`` for swap items) runs once.

Usage (GPU)::

    python -m sparc.vocoders.eval.ood_svfit_p8 sweep --run <run> --out <out> --pool vctk --shard 0 --nshards 6
    python -m sparc.vocoders.eval.ood_svfit_p8 select --out <out>
    python -m sparc.vocoders.eval.ood_svfit_p8 run --run <run> --out <out> --pool vctk --shard 0 --nshards 4
    python -m sparc.vocoders.eval.ood_svfit_p8 swap --run <run> --out <out> --pool vctk --shard 0 --nshards 4
    python -m sparc.vocoders.eval.ood_svfit_p8 summarize --out <out>
"""

import argparse
import itertools
import json
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sparc.vocoders.constants import HOP, SAMPLE_RATE
from sparc.vocoders.eval.ood_gap import GAIN_DB, PHASE4_ROOT, boot_stat, render_code, stable_seed
from sparc.vocoders.eval.ood_svfit import CROP_SV, SPLITS, Item, SVContext, pick_items, split_speakers

logger = logging.getLogger(__name__)

JUDGES = ("holdout", "wespeaker")  # CAM++ (selection), WeSpeaker ResNet34 (held out from loss and selection)
SELECT_MODEL = "holdout"
MODELS = ("ecapa", "holdout", "wespeaker")
VAL_POOLS = {"libritts_dev": ("libritts_dev", 12, 1), "vctk": ("vctk", 20, 1), "globe_spk": ("globe_spk", 12, 1)}  # (plan, speakers, items)
TEST_POOLS = {"libritts": ("libritts", 20, 2), "vctk": ("vctk", 20, 2), "globe": ("globe", 40, 1)}  # globe: pseudo-speaker test half (own items)
SWAP_POOLS = {"libritts": "libritts", "vctk": "vctk", "globe_spk": "globe_spk_test"}  # swap items; GLOBE real test speakers (1 reference each)
N_SWAPS = 80
LRS = (0.01, 0.03, 0.1)
LAMS_L2 = (0.3, 1.0, 3.0, 10.0)
LAMS_NAT = (0.3, 3.0)
NATS = (1.0, 3.0, 10.0, 30.0)
SNAPS = (25, 50, 100, 200, 400)
GUARD = {"utmos": 0.05, "wer_points": 0.5}
ARMS = ("unc", "l2", "nat")
CHOSEN_FROM = ("l2", "nat")
ADVANCE = {"closure_lo": 0.25, "utmos_drop": 0.10, "wer_points": 0.5, "own_drop": 0.01, "source_rise": 0.02}
N_MEL, N_FFT = 80, 1024


def arm_configs() -> list[tuple[str, float, float, float]]:
    """``(arm, lr, lam, nat)`` of the validation grid."""
    cfgs = [("unc", lr, 0.0, 0.0) for lr in LRS]
    cfgs += [("l2", lr, lam, 0.0) for lr, lam in itertools.product(LRS, LAMS_L2)]
    cfgs += [("nat", lr, lam, n) for lr, lam, n in itertools.product(LRS, LAMS_NAT, NATS)]
    return cfgs


# ------------------------------------------------------------------------------------------------ pure maths


def mel_stats_np(logmel: np.ndarray) -> np.ndarray:
    """``[bands]`` mean and std over time of a log-mel ``[bands, T]`` concatenated (reference implementation of the term)."""
    return np.concatenate([logmel.mean(-1), logmel.std(-1, ddof=1)])


def mel_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared difference of two ``mel_stats_np`` vectors."""
    return float(np.mean((a - b) ** 2))


# ------------------------------------------------------------------------------------------------ fitting


def fit_code(
    ctx: SVContext, z0: np.ndarray, target: dict[str, Any], e_target: np.ndarray, lr: float, lam: float, nat: float,
    snapshots: Sequence[int], seed: int,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Adam on an offset of the code minimizing ``-cos + lam * |dz|^2/|z0|^2 + nat * D_mel`` (see the module doc).

    ``target`` has the content features ``feats [15, T]`` and ``utt_f0``. Returns ``{step: code [64]}`` and the mean
    ECAPA cosine over the first / last 10 steps and the mean mel distance over the last 10 steps (0 if ``nat == 0``)."""
    import torch
    import torch.nn.functional as F
    import torchaudio

    from sparc.vocoders.training.callbacks import EVAL_SEED, fixed_torch_rng

    module, device = ctx.module, ctx.device
    ecapa = ctx.scorers.get("ecapa").model
    z0_t = torch.from_numpy(np.asarray(z0, dtype=np.float32)).to(device)
    e_t = F.normalize(torch.from_numpy(np.asarray(e_target, dtype=np.float32)).to(device), dim=-1)
    feats = torch.from_numpy(np.ascontiguousarray(target["feats"])).to(device)
    n_frames = int(feats.shape[1])
    crop = min(CROP_SV, n_frames)
    f0 = torch.tensor([float(target["utt_f0"])], dtype=torch.float32, device=device)
    delta = torch.zeros_like(z0_t, requires_grad=True)
    scale = float(z0_t.pow(2).sum())
    mel = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MEL, f_min=0.0, f_max=SAMPLE_RATE / 2).to(device)

    def stats(wav: "torch.Tensor") -> "torch.Tensor":
        lm = torch.log(mel(wav[:, 0].float()).clamp_min(1e-5))  # [1, bands, frames]
        return torch.cat([lm.mean(-1), lm.std(-1)], dim=-1)[0]

    opt = torch.optim.Adam([delta], lr=lr)
    rng = np.random.default_rng([seed, 23])
    snaps = sorted({int(s) for s in snapshots})
    out: dict[int, np.ndarray] = {}
    coss: list[float] = []
    nats: list[float] = []
    for step in range(1, snaps[-1] + 1):
        s0 = int(rng.integers(0, n_frames - crop + 1))
        batch = {"features": feats[None, :, s0 : s0 + crop], "utt_f0": f0}
        ref_stats = None
        if nat > 0:
            with torch.no_grad(), fixed_torch_rng(device, EVAL_SEED):
                ref_stats = stats(module.synthesize(batch, z=z0_t[None]))
        with torch.enable_grad(), fixed_torch_rng(device, EVAL_SEED):
            z = z0_t + delta
            wav = module.synthesize(batch, z=z[None])
            w16 = torchaudio.functional.resample(wav[:, 0].float(), SAMPLE_RATE, 16000)
            emb = F.normalize(ecapa.encode_batch(w16, normalize=False)[:, 0], dim=-1)[0]
            cos = (emb * e_t).sum()
            loss = -cos + lam * delta.pow(2).sum() / scale
            d_mel = torch.zeros((), device=device)
            if ref_stats is not None:
                d_mel = (stats(wav) - ref_stats).pow(2).mean()
                loss = loss + nat * d_mel
            opt.zero_grad()
            loss.backward()
            opt.step()
        coss.append(float(cos.detach()))
        nats.append(float(d_mel.detach()))
        if step in snaps:
            with torch.no_grad():
                out[step] = (z0_t + delta).float().cpu().numpy()
    return out, {"cos_start": float(np.mean(coss[:10])), "cos_end": float(np.mean(coss[-10:])), "mel_dist": float(np.mean(nats[-10:]))}


# ------------------------------------------------------------------------------------------------ items and scoring


class SwapItem:
    """A swap pair of the cloning plan under the chosen inference variant (5 references, F0 and loudness matching)."""

    def __init__(self, ctx: SVContext, s: dict[str, Any]):
        from sparc.vocoders.eval.clone import Matching, all_variants, apply_matching, build_items, median_f0_hz

        self.s = s
        self.it = build_items({**ctx.plan, "own": [], "swaps": [s]}, all_variants(5), ["k5_f0_loud"])[0]
        self.z_a, self.raw, self.ref_f0 = ctx.codes(self.it)
        feats = ctx.renderer.features_fn(ctx.store, self.it.content_id, GAIN_DB)
        stats = Matching.from_features(ctx.conds[ctx.pool].clean_features(self.it.match_ids))
        feats = apply_matching(feats, stats, self.it.f0, self.it.loud, self.it.ema, f0_relative=ctx.renderer.rel_f0)
        self.target = {"feats": feats, "utt_f0": median_f0_hz(feats)}
        g = ctx.gt_embeddings(list(self.it.code_ids))
        from sparc.vocoders.eval.ood_gap import unit_rows

        self.e_refs = unit_rows(unit_rows(np.stack([g[u]["ecapa"] for u in self.it.code_ids])).mean(0))


def score_wavs(ctx: SVContext, speaker: str, content_id: str, code_ids: Sequence[str], wavs: list[np.ndarray], source: str | None = None) -> list[dict[str, float]]:
    """Per waveform: ``<model>/sim`` (held-out references of ``speaker`` where they exist, else all), ``<model>/all``, UTMOS,
    WER counts and, when ``source`` is given, ``<model>/src`` (similarity to the source speaker's references)."""
    emb = ctx.embed(wavs)
    sims = ctx.sims(emb, {"speaker": speaker}, code_ids)
    ssims = ctx.sims(emb, {"speaker": source}, []) if source is not None else None
    mos = ctx.utmos(wavs)
    wer = ctx.wer(wavs, content_id)
    rows = []
    for j in range(len(wavs)):
        r: dict[str, float] = {"utmos": float(mos[j]), "wer_err": float(wer[j][0]), "wer_len": float(wer[j][1])}
        for m in MODELS:
            held, allv = float(sims[f"{m}/held"][j]), float(sims[f"{m}/all"][j])
            r[f"{m}/sim"] = held if np.isfinite(held) else allv
            r[f"{m}/all"] = allv
            if ssims is not None:
                r[f"{m}/src"] = float(ssims[f"{m}/all"][j])
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------ sweep (validation)


def _val_items(args: argparse.Namespace, pool: str) -> list[dict[str, Any]]:
    plan_name, n, per = VAL_POOLS[pool]
    plan = json.loads((Path(args.root) / "plans" / f"{plan_name}.json").read_text())
    return pick_items(type("P", (), {"plan": plan})(), split_speakers("val", plan_name), n, per, seed=11)  # type: ignore[arg-type]


def run_sweep(args: argparse.Namespace) -> None:
    import torch

    out = Path(args.out) / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rows_{args.pool}_shard{args.shard:02d}.parquet"
    rows: list[dict[str, Any]] = pd.read_parquet(path).to_dict("records") if path.is_file() else []
    done = {r["own_id"] for r in rows}
    mine = _val_items(args, args.pool)[args.shard :: args.nshards][: args.max_items]
    configs, snaps_grid = (([("nat", 0.03, 1.0, 3.0), ("l2", 0.03, 1.0, 0.0), ("unc", 0.03, 0.0, 0.0)], (5, 10)) if args.smoke else (arm_configs(), SNAPS))
    ctx = SVContext(args.run, VAL_POOLS[args.pool][0], args.root, args.device)
    start = time.time()
    for o in mine:
        if o["own_id"] in done:
            continue
        item = Item(ctx, o)
        w_std = render_code(ctx.module, ctx.device, item.target["feats"], item.z_a)
        base = score_wavs(ctx, o["speaker"], o["content_id"], item.it.code_ids, [w_std])[0]
        rows.append({"pool": args.pool, "own_id": o["own_id"], "speaker": o["speaker"], "arm": "std", "lr": 0.0, "lam": 0.0, "nat": 0.0, "step": 0, **base})
        for arm, lr, lam, nat in configs:
            snaps, stats = fit_code(ctx, item.z_a, item.target, item.e_others, lr, lam, nat, snaps_grid, stable_seed("p8sweep", o["own_id"], args.pool, arm, lr, lam, nat))
            steps = sorted(snaps)
            sc = score_wavs(ctx, o["speaker"], o["content_id"], item.it.code_ids, [render_code(ctx.module, ctx.device, item.target["feats"], snaps[s]) for s in steps])
            for s, r in zip(steps, sc):
                shift = float(np.linalg.norm(snaps[s] - item.z_a) / max(np.linalg.norm(item.z_a), 1e-9))
                rows.append({"pool": args.pool, "own_id": o["own_id"], "speaker": o["speaker"], "arm": arm, "lr": lr, "lam": lam, "nat": nat, "step": s, "shift": shift, **stats, **r})
        logger.info("sweep %s %s done (%.0f s)", args.pool, o["own_id"], time.time() - start)
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_parquet(path)


def select_choice(rows: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Per arm the setting with the highest mean CAM++ similarity (pools equally weighted) whose UTMOS drop is <= 0.05 and pooled
    WER rise <= 0.5 points in every pool; ``steps = 0`` (standard code) if no guard-passing setting beats the standard code. The
    unconstrained reference falls back to the best setting overall (flagged) when none passes. ``chosen`` is the better
    (CAM++) of the l2 and nat arms."""
    key = f"{SELECT_MODEL}/sim"
    std = rows[rows["arm"] == "std"]
    fit = rows[rows["arm"] != "std"]
    std_pool = {p: {"score": float(g[key].mean()), "utmos": float(g["utmos"].mean()), "wer": 100 * g["wer_err"].sum() / max(g["wer_len"].sum(), 1)} for p, g in std.groupby("pool")}
    std_score = float(np.mean([v["score"] for v in std_pool.values()]))
    keys = ["arm", "lr", "lam", "nat", "step"]
    tab = []
    for k, g in fit.groupby(keys):
        row: dict[str, Any] = dict(zip(keys, k))
        ok, scores = True, []
        for p, gp in g.groupby("pool"):
            sp = std_pool[p]
            drop = sp["utmos"] - float(gp["utmos"].mean())
            wer_up = 100 * gp["wer_err"].sum() / max(gp["wer_len"].sum(), 1) - sp["wer"]
            row[f"{p}/utmos_drop"], row[f"{p}/wer_up"], row[f"{p}/sim"] = drop, wer_up, float(gp[key].mean())
            ok &= drop <= GUARD["utmos"] and wer_up <= GUARD["wer_points"]
            scores.append(float(gp[key].mean()))
        row.update({"score": float(np.mean(scores)), "ok": bool(ok) and len(scores) == len(std_pool), "shift": float(g["shift"].mean()),
                    "wespeaker": float(g.groupby("pool")["wespeaker/sim"].mean().mean()), "ecapa": float(g.groupby("pool")["ecapa/sim"].mean().mean())})
        tab.append(row)
    tab = pd.DataFrame(tab)
    choice: dict[str, Any] = {"standard": {"score": std_score, "pools": std_pool}}
    for arm, g in tab.groupby("arm"):
        cand = g[g["ok"]].sort_values(["score", "step"], ascending=[False, True])
        best, flagged = (cand.iloc[0] if len(cand) else None), False
        if best is None and arm == "unc":
            best, flagged = g.sort_values("score", ascending=False).iloc[0], True
        use = best is not None and float(best["score"]) > std_score
        entry: dict[str, Any] = {"steps": 0, "lr": 0.0, "lam": 0.0, "nat": 0.0}
        if use:
            entry = {"lr": float(best["lr"]), "lam": float(best["lam"]), "nat": float(best["nat"]), "steps": int(best["step"])}
        entry.update({"guard_ok": bool(best is not None and bool(best["ok"])), "reference_flagged": flagged, "n_guard_ok": int(g["ok"].sum()), "n_configs": int(len(g)),
                      "dev_score": float(best["score"]) if best is not None else math.nan, "dev_wespeaker": float(best["wespeaker"]) if best is not None else math.nan,
                      "dev_ecapa": float(best["ecapa"]) if best is not None else math.nan,
                      "dev_pools": {p: {k: float(best[f"{p}/{k}"]) for k in ("sim", "utmos_drop", "wer_up")} for p in std_pool if best is not None}})
        choice[arm] = entry
    cands = [(a, choice[a]) for a in CHOSEN_FROM if a in choice and choice[a]["steps"] > 0]
    choice["chosen"] = max(cands, key=lambda t: t[1]["dev_score"])[0] if cands else "standard"
    return choice, tab


def run_select(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows = pd.concat([pd.read_parquet(p) for p in sorted((out / "sweep").glob("rows_*_shard*.parquet"))], ignore_index=True)
    choice, tab = select_choice(rows)
    choice["n_items"] = {p: int(g["own_id"].nunique()) for p, g in rows.groupby("pool")}
    (out / "choice.json").write_text(json.dumps(choice, indent=2))
    (out / "sweep_table.md").write_text(tab.sort_values(["arm", "score"], ascending=[True, False]).round(4).to_string(index=False) + "\n")
    print(json.dumps(choice, indent=2))


# ------------------------------------------------------------------------------------------------ test: own-speaker items


def _write_wav(path: Path, w: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), w, SAMPLE_RATE, subtype="PCM_16")


def run_pool(args: argparse.Namespace) -> None:
    plan_name, n, per = TEST_POOLS[args.pool]
    ctx = SVContext(args.run, plan_name, args.root, args.device)
    choice = json.loads((Path(args.out) / "choice.json").read_text())
    items = pick_items(ctx, split_speakers("test", plan_name), n, per, seed=0)[: args.max_items]
    out = Path(args.out) / "test" / args.pool
    start = time.time()
    for k, o in enumerate(items):
        if k % args.nshards != args.shard:
            continue
        row_path = out / "rows" / f"{o['own_id']}.json"
        if row_path.is_file():
            continue
        item = Item(ctx, o)
        zs: dict[str, np.ndarray] = {"std": item.z_a}
        info: dict[str, float] = {}
        for arm in ARMS:
            ch = choice[arm]
            if ch["steps"] == 0:
                zs[arm] = item.z_a
                continue
            snaps, stats = fit_code(ctx, item.z_a, item.target, item.e_others, ch["lr"], ch["lam"], ch["nat"], [ch["steps"]], stable_seed("p8test", o["own_id"], args.pool, arm))
            zs[arm] = snaps[ch["steps"]]
            info.update({f"{arm}/cos_start": stats["cos_start"], f"{arm}/cos_end": stats["cos_end"], f"{arm}/mel_dist": stats["mel_dist"],
                         f"{arm}/shift": float(np.linalg.norm(zs[arm] - item.z_a) / max(np.linalg.norm(item.z_a), 1e-9))})
        conds = list(zs)
        wavs = [render_code(ctx.module, ctx.device, item.target["feats"], zs[c]) for c in conds] + [np.asarray(item.target["audio"], dtype=np.float32)]
        sc = score_wavs(ctx, o["speaker"], o["content_id"], item.it.code_ids, wavs)
        row: dict[str, Any] = {"pool": args.pool, "own_id": o["own_id"], "speaker": o["speaker"], "content_id": o["content_id"], "n_code_refs": len(item.it.code_ids),
                               "n_held": len(ctx.reference_sets(o, item.it.code_ids)[1]), **info}
        for c, r in zip([*conds, "gt"], sc):
            for key, v in r.items():
                row[f"{c}/{key}"] = v
        for c, w in zip(conds, wavs):
            _write_wav(Path(args.audio) / "own" / args.pool / c / f"{o['own_id']}.wav", w)
        row_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text(json.dumps(row))
        logger.info("%s %s done (%.0f s)", args.pool, o["own_id"], time.time() - start)


# ------------------------------------------------------------------------------------------------ test: swap items


def swap_selection(plan: dict[str, Any], speakers: set[str], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """A seeded subset of ``n`` swap pairs whose target is in ``speakers`` (identical on every shard)."""
    cand = sorted((s for s in plan["swaps"] if s["target"] in speakers), key=lambda s: s["swap_id"])
    pick = np.random.default_rng([seed, 59]).permutation(len(cand))[:n]
    return [cand[i] for i in sorted(pick)]


def run_swap(args: argparse.Namespace) -> None:
    plan_name = SWAP_POOLS[args.pool]
    ctx = SVContext(args.run, plan_name, args.root, args.device)
    choice = json.loads((Path(args.out) / "choice.json").read_text())
    arm = choice["chosen"]
    ch = choice.get(arm, {"steps": 0})
    swaps = swap_selection(ctx.plan, split_speakers("test", plan_name), N_SWAPS)[: args.max_items]
    out = Path(args.out) / "test_swap" / args.pool
    start = time.time()
    for k, s in enumerate(swaps):
        if k % args.nshards != args.shard:
            continue
        row_path = out / "rows" / f"{s['swap_id']}.json"
        if row_path.is_file():
            continue
        item = SwapItem(ctx, s)
        z_fit, info = item.z_a, {}
        if ch["steps"] > 0:
            snaps, stats = fit_code(ctx, item.z_a, item.target, item.e_refs, ch["lr"], ch["lam"], ch["nat"], [ch["steps"]], stable_seed("p8swap", s["swap_id"], args.pool))
            z_fit = snaps[ch["steps"]]
            info = {"cos_start": stats["cos_start"], "cos_end": stats["cos_end"], "mel_dist": stats["mel_dist"],
                    "shift": float(np.linalg.norm(z_fit - item.z_a) / max(np.linalg.norm(item.z_a), 1e-9))}
        wavs = [render_code(ctx.module, ctx.device, item.target["feats"], z) for z in (item.z_a, z_fit)]
        sc = score_wavs(ctx, s["target"], s["content_id"], item.it.code_ids, wavs, source=s["source"])
        row: dict[str, Any] = {"pool": args.pool, "swap_id": s["swap_id"], "target": s["target"], "source": s["source"], "arm": arm, "n_code_refs": len(item.it.code_ids), **info}
        for c, r in zip(("std", "fit"), sc):
            for key, v in r.items():
                row[f"{c}/{key}"] = v
        for c, w in zip(("std", "fit"), wavs):
            _write_wav(Path(args.audio) / "swap" / args.pool / c / f"{s['swap_id']}.wav", w)
        row_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text(json.dumps(row))
        logger.info("swap %s %s done (%.0f s)", args.pool, s["swap_id"], time.time() - start)


# ------------------------------------------------------------------------------------------------ summarize and rule


def summarize_pool(df: pd.DataFrame, arms: Sequence[str], n_boot: int = 4000) -> dict[str, Any]:
    """Closure (fraction of the standard-to-ground-truth gap closed), gain and guards of every arm in a pool (own items)."""
    res: dict[str, Any] = {"n_items": int(len(df)), "n_speakers": int(df["speaker"].nunique()), "held_out_refs": bool(df["n_held"].min() > 0), "models": {}}
    for m in MODELS:
        t: dict[str, Any] = {}
        for c in ["std", *arms, "gt"]:
            ps = df.groupby("speaker")[[f"{c}/{m}/sim", f"gt/{m}/sim", f"std/{m}/sim"]].mean()
            ps.columns = ["x", "g", "a"]
            t[c] = {"sim": boot_stat(ps[["x"]], lambda d: d["x"], n_boot), "frac_of_gt": boot_stat(ps, lambda d: d["x"] / d["g"], n_boot),
                    "gain": boot_stat(ps, lambda d: d["x"] - d["a"], n_boot),
                    "closure": boot_stat(ps, lambda d: (d["x"] - d["a"]) / (d["g"] - d["a"]) if d["g"] - d["a"] > 1e-9 else math.nan, n_boot)}
        res["models"][m] = t
    res["utmos"] = {c: float(df[f"{c}/utmos"].mean()) for c in ["std", *arms, "gt"]}
    res["wer"] = {c: float(100 * df[f"{c}/wer_err"].sum() / max(df[f"{c}/wer_len"].sum(), 1)) for c in ["std", *arms, "gt"]}
    res["fit"] = {k: float(df[k].mean()) for k in df.columns if k.endswith(("/cos_start", "/cos_end", "/shift", "/mel_dist")) and df[k].notna().any()}
    return res


def summarize_swaps(df: pd.DataFrame, n_boot: int = 4000) -> dict[str, Any]:
    """Fitted minus standard similarity to the target and to the source speaker, per verifier (speaker-level bootstrap on targets)."""
    res: dict[str, Any] = {"n_swaps": int(len(df)), "n_targets": int(df["target"].nunique()), "arm": str(df["arm"].iloc[0]), "models": {}}
    for m in MODELS:
        e: dict[str, Any] = {}
        for what in ("sim", "src"):
            ps = df.assign(d=df[f"fit/{m}/{what}"] - df[f"std/{m}/{what}"], a=df[f"std/{m}/{what}"], x=df[f"fit/{m}/{what}"]).groupby("target")[["d", "a", "x"]].mean()
            e[what] = {"std": float(ps["a"].mean()), "fit": float(ps["x"].mean()), "diff": boot_stat(ps[["d"]], lambda d: d["d"], n_boot)}
        res["models"][m] = e
    res["utmos"] = {c: float(df[f"{c}/utmos"].mean()) for c in ("std", "fit")}
    res["wer"] = {c: float(100 * df[f"{c}/wer_err"].sum() / max(df[f"{c}/wer_len"].sum(), 1)) for c in ("std", "fit")}
    return res


def advance_rule(pools: dict[str, Any], swaps: dict[str, Any], arm: str, judges: Sequence[str] = ("wespeaker",)) -> dict[str, Any]:
    """The pre-registered rule for ``arm`` on test. ``judges`` are the held-out verifiers (WeSpeaker); the CAM++ closure is
    reported as well (selection verifier) but does not decide. Every failed guard is a failure (no re-tuning)."""
    lim = ADVANCE
    out: dict[str, Any] = {"arm": arm, "checks": {}}
    v = pools.get("vctk")
    clos = {m: (v["models"][m][arm]["closure"] if v else {}) for m in MODELS}
    out["vctk_closure"] = {m: clos[m] for m in MODELS}
    out["checks"]["closure_lower_bound_>=0.25_each_held_out_verifier"] = bool(v) and all(np.isfinite(clos[m].get("lo", math.nan)) and clos[m]["lo"] >= lim["closure_lo"] for m in judges)
    out["checks"]["closure_lower_bound_>=0.25_on_CAM++_too"] = bool(v) and np.isfinite(clos["holdout"].get("lo", math.nan)) and clos["holdout"]["lo"] >= lim["closure_lo"]
    drops = {p: r["utmos"]["std"] - r["utmos"][arm] for p, r in pools.items()}
    wers = {p: r["wer"][arm] - r["wer"]["std"] for p, r in pools.items()}
    out["utmos_drop"], out["wer_up"] = drops, wers
    out["checks"]["utmos_drop_<=0.1_every_pool"] = all(d <= lim["utmos_drop"] for d in drops.values())
    out["checks"]["wer_<=+0.5_every_pool"] = all(w <= lim["wer_points"] for w in wers.values())
    lib = pools.get("libritts")
    own_drop = {m: (lib["models"][m]["std"]["sim"]["mean"] - lib["models"][m][arm]["sim"]["mean"]) if lib else math.nan for m in JUDGES}
    out["libritts_own_drop"] = own_drop
    out["checks"]["libritts_own_speaker_drop_<=0.01"] = bool(lib) and all(d <= lim["own_drop"] for d in own_drop.values())
    rise = {p: {m: r["models"][m]["src"]["diff"]["mean"] for m in JUDGES} for p, r in swaps.items()}
    out["swap_source_rise"] = rise
    out["checks"]["swap_source_similarity_rise_<=0.02"] = bool(swaps) and all(x <= lim["source_rise"] for p in rise.values() for x in p.values())
    # The same quality guards on the swap items of every pool (added to the evaluation after the first summary; it can only add failures).
    sw_drop = {p: r["utmos"]["std"] - r["utmos"]["fit"] for p, r in swaps.items()}
    sw_wer = {p: r["wer"]["fit"] - r["wer"]["std"] for p, r in swaps.items()}
    out["swap_utmos_drop"], out["swap_wer_up"] = sw_drop, sw_wer
    out["checks"]["utmos_drop_<=0.1_swap_pools"] = bool(swaps) and all(d <= lim["utmos_drop"] for d in sw_drop.values())
    out["checks"]["wer_<=+0.5_swap_pools"] = bool(swaps) and all(w <= lim["wer_points"] for w in sw_wer.values())
    main = [k for k in out["checks"] if not k.endswith("CAM++_too")]
    out["advances"] = all(out["checks"][k] for k in main)
    return out


def fmt(d: dict[str, float] | None, digits: int = 3) -> str:
    if d is None or not np.isfinite(d.get("mean", math.nan)):
        return "n/a"
    lo, hi = d.get("lo", math.nan), d.get("hi", math.nan)
    return f"{d['mean']:.{digits}f}" + (f" [{lo:.{digits}f}, {hi:.{digits}f}]" if np.isfinite(lo) else "")


NAMES = {"std": "standard code", "l2": "ECAPA fit + L2 penalty", "nat": "ECAPA fit + L2 + naturalness (log-mel statistics)", "unc": "unconstrained ECAPA fit (reference)", "gt": "ground truth"}
MODEL_NAMES = {"holdout": "CAM++ (selection verifier)", "wespeaker": "WeSpeaker ResNet34 (held out)", "ecapa": "ECAPA (the fitted loss)"}


def render_markdown(choice: dict[str, Any], pools: dict[str, Any], swaps: dict[str, Any], rule: dict[str, Any] | None) -> str:
    lines = ["# Phase 8.2 constrained speaker-code fitting (test, run once)", "",
             "Validation choice: " + json.dumps({a: {k: choice[a][k] for k in ("lr", "lam", "nat", "steps")} for a in ARMS if a in choice}) + f"; judged setting: **{choice.get('chosen')}**.", ""]
    for pool, res in pools.items():
        lines += [f"## {pool} own-speaker items ({res['n_items']} items, {res['n_speakers']} speakers; similarity to "
                  f"{'held-out references' if res['held_out_refs'] else 'all references (none held out)'})", ""]
        arms = [a for a in ("std", *ARMS, "gt") if a in res["utmos"]]
        for m in ("wespeaker", "holdout", "ecapa"):
            t = res["models"][m]
            lines += [f"### {MODEL_NAMES[m]}", "", "| condition | similarity | fraction of ground truth | gain over standard | closure |", "|---|---|---|---|---|"]
            for c in arms:
                lines.append(f"| {NAMES[c]} | {fmt(t[c]['sim'])} | {fmt(t[c]['frac_of_gt'])} | {fmt(t[c]['gain']) if c not in ('gt', 'std') else ''} | {fmt(t[c]['closure']) if c not in ('gt', 'std') else ''} |")
            lines.append("")
        lines += ["Quality (UTMOS / WER %): " + ", ".join(f"{NAMES[c]} {res['utmos'][c]:.2f} / {res['wer'][c]:.1f}" for c in arms), ""]
    for pool, res in swaps.items():
        lines += [f"## {pool} swap items ({res['n_swaps']} swaps, {res['n_targets']} targets; setting {res['arm']})", "", "| verifier | to target: standard | fitted | diff | to source: standard | fitted | diff |", "|---|---|---|---|---|---|---|"]
        for m in ("wespeaker", "holdout", "ecapa"):
            e = res["models"][m]
            lines.append(f"| {MODEL_NAMES[m]} | {e['sim']['std']:.3f} | {e['sim']['fit']:.3f} | {fmt(e['sim']['diff'])} | {e['src']['std']:.3f} | {e['src']['fit']:.3f} | {fmt(e['src']['diff'])} |")
        lines += ["", f"UTMOS standard {res['utmos']['std']:.2f}, fitted {res['utmos']['fit']:.2f}; WER % standard {res['wer']['std']:.1f}, fitted {res['wer']['fit']:.1f}", ""]
    if rule:
        lines += ["## Advance rule", "", f"Judged setting: {rule['arm']}. Advances: **{rule['advances']}**.", "", "| check | pass |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rule["checks"].items()]
        lines += ["", "VCTK closure: " + ", ".join(f"{MODEL_NAMES[m]} {fmt(rule['vctk_closure'][m])}" for m in MODELS)]
        lines += ["UTMOS drop per pool (own-speaker items): " + json.dumps({k: round(v, 3) for k, v in rule["utmos_drop"].items()}),
                  "WER rise per pool (own-speaker items): " + json.dumps({k: round(v, 2) for k, v in rule["wer_up"].items()}),
                  "UTMOS drop per pool (swap items): " + json.dumps({k: round(v, 3) for k, v in rule["swap_utmos_drop"].items()}),
                  "WER rise per pool (swap items): " + json.dumps({k: round(v, 2) for k, v in rule["swap_wer_up"].items()}),
                  "LibriTTS own-speaker drop: " + json.dumps({k: round(v, 4) for k, v in rule["libritts_own_drop"].items()}),
                  "Swap source-similarity rise: " + json.dumps({p: {m: round(x, 4) for m, x in d.items()} for p, d in rule["swap_source_rise"].items()}), ""]
    return "\n".join(lines) + "\n"


def run_summarize(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    choice = json.loads((out / "choice.json").read_text())
    pools, swaps = {}, {}
    for pool in [p for p in TEST_POOLS if (out / "test" / p / "rows").is_dir()]:
        df = pd.DataFrame([json.loads(p.read_text()) for p in sorted((out / "test" / pool / "rows").glob("*.json"))])
        pools[pool] = summarize_pool(df, ARMS, args.n_boot)
    for pool in [p for p in SWAP_POOLS if (out / "test_swap" / p / "rows").is_dir()]:
        df = pd.DataFrame([json.loads(p.read_text()) for p in sorted((out / "test_swap" / pool / "rows").glob("*.json"))])
        swaps[pool] = summarize_swaps(df, args.n_boot)
    rule = advance_rule(pools, swaps, choice["chosen"]) if choice["chosen"] in ARMS else None
    (out / "summary.json").write_text(json.dumps({"choice": choice, "pools": pools, "swaps": swaps, "rule": rule}, indent=2, default=float))
    (out / "summary.md").write_text(render_markdown(choice, pools, swaps, rule))
    print((out / "summary.md").read_text())
    return {"pools": pools, "swaps": swaps, "rule": rule}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, pools in (("sweep", VAL_POOLS), ("run", TEST_POOLS), ("swap", SWAP_POOLS)):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--root", default=PHASE4_ROOT)
        s.add_argument("--device", default="auto")
        s.add_argument("--shard", type=int, default=0)
        s.add_argument("--nshards", type=int, default=1)
        s.add_argument("--pool", required=True, choices=tuple(pools))
        s.add_argument("--max-items", type=int, default=None, help="smoke tests only")
        if name == "sweep":
            s.add_argument("--smoke", action="store_true", help="3 configs, 10 steps")
        if name in ("run", "swap"):
            s.add_argument("--audio", default="/data/user_data/xoy/spkemb/phase8/svfit/audio")
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=4000)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    {"sweep": run_sweep, "select": run_select, "run": run_pool, "swap": run_swap, "summarize": run_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
