"""Speaker-verification-driven code fitting through the frozen generator (decisive follow-up of :mod:`ood_gap`).

:mod:`ood_gap` fitted the speaker code on a mel + STFT reconstruction loss, which is nearly flat in the code, so it cannot
show that "no code renders the voice". Here the code is fitted to *maximize speaker similarity* of the synthesized
audio, with the differentiable ECAPA-TDNN as the fitting loss, through the frozen generator of the ``p5_fv`` run.

Every condition resynthesizes the speaker's own held-out utterance (the ``own`` items of the cloning plans) from its
content features, speaker-relative pitch as in the standard path:

* ``a_refs``   standard code from the speaker's reference utterances (K = 5; GLOBE: all of the 2 to 7);
* ``e_z`` / ``e_ffn``  the 64-d code (offset added to the standard code) or an offset on the speaker network's pooled
  input vectors (shared by the K rows, through the frozen FFN) fitted with Adam to maximize ECAPA cosine of the
  synthesized crops to the mean ECAPA embedding of the speaker's code-reference utterances (never the target
  utterance), + ``lam`` * squared distance to the standard code (relative to ``|z0|^2`` for the code, in units of the input
  std for the FFN input) + ``w_rec`` * the training reconstruction loss on crops of the references (quality anchor);
* ``e_z_self`` / ``e_ffn_self``  the same, but the target embedding is the ECAPA embedding of the target utterance's own
  ground-truth audio (an upper bound: the code is told the exact answer under the fitted judge);
* ``f_swap``   code-sensitivity check: the same content rendered with the standard code of a different speaker of the
  same pool (sizes the code's leverage on every verifier);
* ``gt``       ground-truth audio of the utterance.

Scoring uses only speaker models that are not in the fitting loss: CAM++ (``holdout``) and the WeSpeaker ResNet34 (the
OOD-screen encoder input, never a judge until now). ECAPA is reported only as the fitted objective. Similarity = cosine
to the mean embedding of the speaker's reference utterances that did not give the code (``held``; none for GLOBE,
which then uses all of its references, inside which the code references lie). UTMOS and Whisper WER are the quality
guard.

Selection (validation speakers only; ``docs/spkemb_validation_splits.json``): per (mode, objective) the (lr, lam, w_rec,
steps) maximizing the mean of CAM++ and WeSpeaker similarity over the validation items subject to a mean UTMOS drop
<= 0.1 and a pooled WER increase <= 1 point against the standard code. The test items are then run once.

Interpretation rule (fixed before any run). With A the speaker-averaged held-out-verifier similarity of the standard code,
E that of the fitted (e, references-only) condition, and G that of ground truth:
closure = (E - A) / (G - A). If closure > 0.5 (bootstrap interval over speakers excluding 0) at acceptable quality, the
code space can express the voice and the encoder or inference is the bottleneck. If E stays near A (closure < 0.2) even
for the self-target upper bound at acceptable quality, the generator or its other inputs (articulatory-trained content
features, pitch, loudness) are the limit. Between the two: partial, reported as such. Fits that raise a verifier only by
degrading quality (guard failure) do not count.

Usage (GPU)::

    python -m sparc.vocoders.eval.ood_svfit sweep --run <run> --out <out> --shard 0 --nshards 6
    python -m sparc.vocoders.eval.ood_svfit select --out <out>                     # CPU, writes choice.json
    python -m sparc.vocoders.eval.ood_svfit run --run <run> --out <out> --pool vctk --shard 0 --nshards 4
    python -m sparc.vocoders.eval.ood_svfit summarize --out <out>                  # CPU
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
from sparc.vocoders.eval.ood_gap import (
    GAIN_DB,
    PHASE4_ROOT,
    Context,
    boot_stat,
    render_code,
    sample_crops,
    stable_seed,
    unit_rows,
)

logger = logging.getLogger(__name__)

SPLITS = "docs/spkemb_validation_splits.json"
MODELS = ("ecapa", "holdout", "wespeaker")
JUDGES = ("holdout", "wespeaker")  # never in the fitting loss
HOLDOUT_SPEC = "campplus:"
CROP_SV = 200  # frames (4 s) of the target content per fitting step
CROP_REC = 64
REC_BATCH = 4
VAL_POOLS = {"libritts_dev": ("libritts_dev", 8, 1), "vctk": ("vctk", 10, 1), "globe_spk": ("globe_spk", 10, 1)}  # (plan, speakers, items)
TEST_POOLS = {"libritts": ("libritts", 20, 2), "vctk": ("vctk", 20, 2), "globe": ("globe", 40, 1)}  # globe: pseudo-speaker test half
GRID = {"mode": ("z", "ffn"), "obj": ("others", "self"), "lr": (0.01, 0.03), "lam": (0.0, 0.3, 3.0), "w_rec": (0.0, 0.1)}
SNAPS = (25, 50, 100, 200)
GUARD = {"utmos": 0.10, "wer_points": 1.0}


# ------------------------------------------------------------------------------------------------ verifiers


class WeSpeakerEmbedder:
    """WeSpeaker ResNet34 (VoxCeleb2) embedder on 16 kHz numpy waveforms (see ``spkscreen.ood_sv``)."""

    def __init__(self, device):
        import torch

        from sparc.vocoders.spkscreen import ood_sv

        self.device = torch.device(device)
        self.net = ood_sv.load_resnet().to(self.device).eval()
        self.fbank = ood_sv.fbank

    def embed(self, wavs16: list[np.ndarray], errors: list[str] | None = None) -> np.ndarray:
        import torch

        rows = []
        for w in wavs16:
            with torch.inference_mode():
                f = self.fbank(torch.from_numpy(np.asarray(w, dtype=np.float32)).to(self.device))
                v = self.net(f[None], torch.tensor([len(f)], device=self.device))[0].float().cpu().numpy()
            rows.append(v / max(float(np.linalg.norm(v)), 1e-12))
        return np.stack(rows)


class _Scorers:
    def __init__(self, inner, we):
        self.inner, self.we, self.base = inner, we, inner.base

    def get(self, key: str):
        return self.we if key == "wespeaker" else self.inner.get(key)


class SVContext(Context):
    """:class:`Context` scoring with ECAPA (fitted), CAM++ and WeSpeaker ResNet34, plus Whisper for WER."""

    def __init__(self, run: str, plan_name: str, root: str, device: str):
        super().__init__(run, plan_name, root, device, ("ecapa", "holdout"))
        self.models = list(MODELS)
        self.scorers = _Scorers(self.scorers, WeSpeakerEmbedder(self.device))

    def wer(self, wavs24: list[np.ndarray], content_id: str) -> list[tuple[int, int]]:
        from sparc.vocoders.eval.interp import real_wer
        from sparc.vocoders.eval.io import to_16k

        wavs = [to_16k(np.asarray(w, dtype=np.float32), SAMPLE_RATE) for w in wavs24]
        hyp = self.scorers.base.get("asr").transcribe(wavs, errors=[])
        ref = self.pool_obj.text(content_id)
        return [real_wer(ref, h) for h in hyp]

    def score(self, o: dict[str, Any], code_ids: Sequence[str], wavs: list[np.ndarray]) -> list[dict[str, float]]:
        """Per waveform: ``<model>/sim`` (held-out references where they exist, else all), ``<model>/all``, utmos, WER counts."""
        sims = self.sims(self.embed(wavs), o, code_ids)
        mos = self.utmos(wavs)
        wer = self.wer(wavs, o["content_id"])
        rows = []
        for j in range(len(wavs)):
            r: dict[str, float] = {"utmos": float(mos[j]), "wer_err": float(wer[j][0]), "wer_len": float(wer[j][1])}
            for m in self.models:
                held, allv = float(sims[f"{m}/held"][j]), float(sims[f"{m}/all"][j])
                r[f"{m}/sim"] = held if np.isfinite(held) else allv
                r[f"{m}/all"] = allv
            rows.append(r)
        return rows


# ------------------------------------------------------------------------------------------------ fitting


def fit_sv(
    ctx: SVContext, mode: str, z0: np.ndarray, raw: np.ndarray, ref_f0: float | None, target: dict[str, Any], e_target: np.ndarray,
    rec_data: Sequence[dict[str, Any]], lr: float, lam: float, w_rec: float, snapshots: Sequence[int], seed: int,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Adam on an offset of the code (``z``) or of the speaker network's input vectors (``ffn``) minimizing
    ``-cos(ECAPA(synth crop), e_target) + lam * penalty + w_rec * reconstruction``. ``target`` has the content ``feats``
    ``[15, T]`` and its ``utt_f0``. Returns ``{step: code [64]}`` and the mean ECAPA cosine over the first / last 10 steps."""
    import torch
    import torch.nn.functional as F
    import torchaudio

    from sparc.vocoders.training.callbacks import EVAL_SEED, fixed_torch_rng

    module, device = ctx.module, ctx.device
    ecapa = ctx.scorers.get("ecapa").model
    cw = module.cfg.loss
    w_mel, w_mss = float(cw.mel_weight), float(cw.mss_weight)
    z0_t = torch.from_numpy(np.asarray(z0, dtype=np.float32)).to(device)
    e_t = F.normalize(torch.from_numpy(np.asarray(e_target, dtype=np.float32)).to(device), dim=-1)
    feats = torch.from_numpy(np.ascontiguousarray(target["feats"])).to(device)
    n_frames = int(feats.shape[1])
    crop = min(CROP_SV, n_frames)
    f0 = torch.tensor([float(target["utt_f0"])], dtype=torch.float32, device=device)
    if mode == "z":
        delta = torch.zeros_like(z0_t, requires_grad=True)
        step_lr = lr
        scale = float(z0_t.pow(2).sum())
    else:
        raw_t = torch.from_numpy(np.asarray(raw, dtype=np.float32)).to(device)
        in_std = module.speaker.std.float().to(device)
        delta = torch.zeros(raw_t.shape[1], device=device, requires_grad=True)
        step_lr = lr * float(in_std.mean())
        mask = torch.ones(1, raw_t.shape[0], dtype=torch.bool, device=device)

    def code() -> torch.Tensor:
        if mode == "z":
            return z0_t + delta
        z, _ = module.speaker.embed((raw_t + delta)[None], mask)
        pitch = getattr(module, "pitch_code", None)
        if pitch is not None:
            z = z + pitch(torch.tensor([math.nan if ref_f0 is None else ref_f0], dtype=torch.float32, device=device))
        return z[0]

    def penalty() -> torch.Tensor:
        return delta.pow(2).sum() / scale if mode == "z" else (delta / in_std).pow(2).mean()

    rec = None
    if w_rec > 0 and rec_data:
        rc = min(CROP_REC, min(int(d["feats"].shape[1]) for d in rec_data))
        rec = ([torch.from_numpy(np.ascontiguousarray(d["feats"])).to(device) for d in rec_data],
               [torch.from_numpy(np.ascontiguousarray(d["audio"], dtype=np.float32)).to(device) for d in rec_data],
               torch.tensor([float(d["utt_f0"]) for d in rec_data], dtype=torch.float32, device=device), rc)
    opt = torch.optim.Adam([delta], lr=step_lr)
    rng = np.random.default_rng([seed, 23])
    snaps = sorted({int(s) for s in snapshots})
    out: dict[int, np.ndarray] = {}
    coss: list[float] = []
    for step in range(1, snaps[-1] + 1):
        s0 = int(rng.integers(0, n_frames - crop + 1))
        with torch.enable_grad(), fixed_torch_rng(device, EVAL_SEED):
            z = code()
            wav = module.synthesize({"features": feats[None, :, s0 : s0 + crop], "utt_f0": f0}, z=z[None])
            w16 = torchaudio.functional.resample(wav[:, 0].float(), SAMPLE_RATE, 16000)
            emb = F.normalize(ecapa.encode_batch(w16, normalize=False)[:, 0], dim=-1)[0]
            cos = (emb * e_t).sum()
            loss = -cos + lam * penalty()
            if rec is not None:
                lens = [int(f.shape[1]) for f in rec[0]]
                picks = sample_crops(lens, rec[3], REC_BATCH, rng)
                fr = torch.stack([rec[0][u][:, s : s + rec[3]] for u, s in picks])
                au = torch.stack([rec[1][u][s * HOP : (s + rec[3]) * HOP] for u, s in picks])[:, None]
                rw = module.synthesize({"features": fr, "utt_f0": rec[2][[u for u, _ in picks]]}, z=z[None].expand(len(picks), -1))
                loss = loss + w_rec * (w_mel * module.mel_loss(rw, au) + w_mss * module.mss_loss(rw, au))
            opt.zero_grad()
            loss.backward()
            opt.step()
        coss.append(float(cos.detach()))
        if step in snaps:
            with torch.no_grad():
                out[step] = code().float().cpu().numpy()
    return out, {"cos_start": float(np.mean(coss[:10])), "cos_end": float(np.mean(coss[-10:]))}


# ------------------------------------------------------------------------------------------------ items


def split_speakers(split: str, key: str, path: str = SPLITS) -> set[str]:
    return set(json.loads(Path(path).read_text())["cloning"][split][key])


def pick_items(ctx: SVContext, speakers: set[str], n_speakers: int, per_speaker: int, seed: int) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for o in sorted(ctx.plan["own"], key=lambda o: o["own_id"]):
        if o["speaker"] in speakers:
            by.setdefault(o["speaker"], []).append(o)
    names = sorted(by)
    chosen = [names[i] for i in np.random.default_rng([seed, 41]).permutation(len(names))[:n_speakers]]
    return [o for s in sorted(chosen) for o in by[s][:per_speaker]]


class Item:
    """Everything fitting and scoring need for one own-speaker item."""

    def __init__(self, ctx: SVContext, o: dict[str, Any]):
        self.o = o
        self.it = ctx.own_item(o)
        self.z_a, self.raw, self.ref_f0 = ctx.codes(self.it)
        self.refs = [ctx.utterance(u) for u in self.it.code_ids]
        self.target = ctx.utterance(self.it.content_id)
        g = ctx.gt_embeddings(list(self.it.code_ids) + [self.it.content_id])
        self.e_others = unit_rows(unit_rows(np.stack([g[u]["ecapa"] for u in self.it.code_ids])).mean(0))
        self.e_self = g[self.it.content_id]["ecapa"]

    def objective(self, obj: str) -> np.ndarray:
        return self.e_others if obj == "others" else self.e_self


def render(ctx: SVContext, item: Item, z: np.ndarray) -> np.ndarray:
    return render_code(ctx.module, ctx.device, item.target["feats"], z)


# ------------------------------------------------------------------------------------------------ sweep (validation)


def run_sweep(args: argparse.Namespace) -> None:
    import torch

    out = Path(args.out) / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.pool, o) for o in _val_items(args, args.pool)]
    mine = jobs[args.shard :: args.nshards]
    rows, start = [], time.time()
    ctxs: dict[str, SVContext] = {}
    configs = list(itertools.product(GRID["mode"], GRID["obj"], GRID["lr"], GRID["lam"], GRID["w_rec"]))
    for pool, o in mine:
        if pool not in ctxs:
            ctxs[pool] = SVContext(args.run, VAL_POOLS[pool][0], args.root, args.device)
        ctx = ctxs[pool]
        item = Item(ctx, o)
        base = ctx.score(o, item.it.code_ids, [render(ctx, item, item.z_a)])[0]
        rows.append({"pool": pool, "own_id": o["own_id"], "speaker": o["speaker"], "mode": "std", "obj": "", "lr": 0.0, "lam": 0.0, "w_rec": 0.0, "step": 0, **base})
        for mode, obj, lr, lam, w_rec in configs:
            snaps, stats = fit_sv(ctx, mode, item.z_a, item.raw, item.ref_f0, item.target, item.objective(obj), item.refs, lr, lam, w_rec, SNAPS,
                                  stable_seed("sweep", o["own_id"], pool, mode, obj, lr, lam, w_rec))
            steps = sorted(snaps)
            sc = ctx.score(o, item.it.code_ids, [render(ctx, item, snaps[s]) for s in steps])
            for s, r in zip(steps, sc):
                shift = float(np.linalg.norm(snaps[s] - item.z_a) / max(np.linalg.norm(item.z_a), 1e-9))
                rows.append({"pool": pool, "own_id": o["own_id"], "speaker": o["speaker"], "mode": mode, "obj": obj, "lr": lr, "lam": lam, "w_rec": w_rec,
                             "step": s, "shift": shift, **stats, **r})
        logger.info("sweep %s %s done (%.0f s)", pool, o["own_id"], time.time() - start)
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_parquet(out / f"rows_{args.pool}_shard{args.shard:02d}.parquet")


def _val_items(args: argparse.Namespace, pool: str) -> list[dict[str, Any]]:
    """Validation items of a pool, selected from the plan file alone (identical on every shard)."""
    plan_name, n, per = VAL_POOLS[pool]
    plan = json.loads((Path(args.root) / "plans" / f"{plan_name}.json").read_text())
    ctx = type("P", (), {"plan": plan})()
    return pick_items(ctx, split_speakers("val", plan_name), n, per, seed=11)  # type: ignore[arg-type]


def select_choice(rows: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Per (mode, objective) the setting with the best mean (CAM++, WeSpeaker) similarity, pools equally weighted, that
    meets the guard (mean UTMOS drop <= 0.1, pooled WER increase <= 1 point against the standard code); none if nothing
    beats the standard code."""
    rows = rows.copy()
    rows["score"] = rows[[f"{m}/sim" for m in JUDGES]].mean(axis=1)
    std = rows[rows["mode"] == "std"]
    std_score = float(std.groupby("pool")["score"].mean().mean())
    std_utmos = float(std["utmos"].mean())
    std_wer = 100 * std["wer_err"].sum() / max(std["wer_len"].sum(), 1)
    fit = rows[rows["mode"] != "std"]
    keys = ["mode", "obj", "lr", "lam", "w_rec", "step"]
    tab = []
    for k, g in fit.groupby(keys):
        tab.append({**dict(zip(keys, k)), "score": float(g.groupby("pool")["score"].mean().mean()), "utmos": float(g["utmos"].mean()),
                    "wer": 100 * g["wer_err"].sum() / max(g["wer_len"].sum(), 1), "ecapa": float(g.groupby("pool")["ecapa/sim"].mean().mean()),
                    "shift": float(g["shift"].mean()), "n": int(len(g))})
    tab = pd.DataFrame(tab)
    tab["utmos_drop"] = std_utmos - tab["utmos"]
    tab["wer_up"] = tab["wer"] - std_wer
    tab["ok"] = (tab["utmos_drop"] <= GUARD["utmos"]) & (tab["wer_up"] <= GUARD["wer_points"])
    choice: dict[str, Any] = {"standard": {"score": std_score, "utmos": std_utmos, "wer": std_wer}}
    for (mode, obj), g in tab.groupby(["mode", "obj"]):
        cand = g[g["ok"]].sort_values(["score", "step"], ascending=[False, True])
        best = cand.iloc[0] if len(cand) else None
        use = best is not None and float(best["score"]) > std_score
        entry = {"steps": 0, "lr": 0.0, "lam": 0.0, "w_rec": 0.0}
        if use:
            entry = {k: (int(best[k]) if k == "steps" else float(best[k])) for k in ("lr", "lam", "w_rec")} | {"steps": int(best["step"])}
        entry.update({"dev_score": float(best["score"]) if best is not None else math.nan, "dev_utmos_drop": float(best["utmos_drop"]) if best is not None else math.nan,
                      "dev_wer_up": float(best["wer_up"]) if best is not None else math.nan, "dev_ecapa": float(best["ecapa"]) if best is not None else math.nan,
                      "n_guard_ok": int(g["ok"].sum()), "n_configs": int(len(g))})
        choice.setdefault(mode, {})[obj] = entry
    return choice, tab


def run_select(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows = pd.concat([pd.read_parquet(p) for p in sorted((out / "sweep").glob("rows_shard*.parquet"))], ignore_index=True)
    choice, tab = select_choice(rows)
    choice["n_items"] = {p: int(g["own_id"].nunique()) for p, g in rows.groupby("pool")}
    (out / "choice.json").write_text(json.dumps(choice, indent=2))
    (out / "sweep_table.md").write_text(tab.sort_values(["mode", "obj", "score"], ascending=[True, True, False]).round(4).to_string(index=False) + "\n")
    print(json.dumps(choice, indent=2))


# ------------------------------------------------------------------------------------------------ test run


def run_pool(args: argparse.Namespace) -> None:
    import soundfile as sf

    plan_name, n, per = TEST_POOLS[args.pool]
    ctx = SVContext(args.run, plan_name, args.root, args.device)
    choice = json.loads((Path(args.out) / "choice.json").read_text())
    items = pick_items(ctx, split_speakers("test", plan_name), n, per, seed=0)
    out = Path(args.out) / "test" / args.pool
    start = time.time()
    for k, o in enumerate(items):
        if k % args.nshards != args.shard:
            continue
        row_path = out / "rows" / f"{o['own_id']}.json"
        if row_path.is_file():
            continue
        item = Item(ctx, o)
        donor = _donor(ctx, items, o)
        d_item = Item(ctx, donor)
        zs: dict[str, np.ndarray] = {"a_refs": item.z_a, "f_swap": d_item.z_a}
        info: dict[str, float] = {}
        for mode in ("z", "ffn"):
            for obj, tag in (("others", ""), ("self", "_self")):
                ch = choice[mode][obj]
                cond = f"e_{mode}{tag}"
                if ch["steps"] == 0:
                    zs[cond] = item.z_a
                    continue
                snaps, stats = fit_sv(ctx, mode, item.z_a, item.raw, item.ref_f0, item.target, item.objective(obj), item.refs, ch["lr"], ch["lam"], ch["w_rec"],
                                      [ch["steps"]], stable_seed("test", o["own_id"], args.pool, cond))
                zs[cond] = snaps[ch["steps"]]
                info.update({f"{cond}/cos_start": stats["cos_start"], f"{cond}/cos_end": stats["cos_end"],
                             f"{cond}/shift": float(np.linalg.norm(zs[cond] - item.z_a) / max(np.linalg.norm(item.z_a), 1e-9))})
        conds = list(zs)
        wavs = [render(ctx, item, zs[c]) for c in conds] + [np.asarray(item.target["audio"], dtype=np.float32)]
        sc = ctx.score(o, item.it.code_ids, wavs)
        row: dict[str, Any] = {"pool": args.pool, "own_id": o["own_id"], "speaker": o["speaker"], "donor": donor["speaker"], "content_id": o["content_id"],
                               "n_code_refs": len(item.it.code_ids), "n_held": len(ctx.reference_sets(o, item.it.code_ids)[1]), **info}
        for c, r in zip([*conds, "gt"], sc):
            for key, v in r.items():
                row[f"{c}/{key}"] = v
        # donor-speaker similarity of the swapped rendering (leverage: does it move toward the code's speaker?)
        dsims = ctx.sims(ctx.embed([wavs[conds.index("f_swap")]]), donor, [])
        for m in MODELS:
            row[f"f_swap/{m}/donor_sim"] = float(dsims[f"{m}/all"][0])
        for c, w in zip(conds, wavs):
            p = out / "wav" / c / f"{o['own_id']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), w, SAMPLE_RATE, subtype="PCM_16")
        row_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text(json.dumps(row))
        logger.info("%s %s done (%.0f s)", args.pool, o["own_id"], time.time() - start)


def _donor(ctx: SVContext, items: list[dict[str, Any]], o: dict[str, Any]) -> dict[str, Any]:
    """A seeded-random own item of a different speaker of the same pool (its standard code is the swap code)."""
    others = [x for x in items if x["speaker"] != o["speaker"]]
    return others[int(np.random.default_rng(stable_seed("donor", o["own_id"])).integers(len(others)))]


# ------------------------------------------------------------------------------------------------ summarize


COND_ORDER = ("a_refs", "e_z", "e_ffn", "e_z_self", "e_ffn_self", "f_swap")
LABEL = {"a_refs": "(a) standard code", "e_z": "(e) SV-fit 64-d code, refs target", "e_ffn": "(e) SV-fit FFN input, refs target",
         "e_z_self": "(e) SV-fit 64-d code, self target (upper bound)", "e_ffn_self": "(e) SV-fit FFN input, self target (upper bound)",
         "f_swap": "(f) other speaker's standard code", "gt": "ground truth"}


def summarize_pool(df: pd.DataFrame, n_boot: int = 4000) -> dict[str, Any]:
    conds = [c for c in COND_ORDER if f"{c}/utmos" in df]
    res: dict[str, Any] = {"n_items": int(len(df)), "n_speakers": int(df["speaker"].nunique()), "held_out_refs": bool(df["n_held"].min() > 0)}
    for m in MODELS:
        t: dict[str, Any] = {}
        for c in [*conds, "gt"]:
            ps = df.groupby("speaker")[[f"{c}/{m}/sim", f"gt/{m}/sim", f"a_refs/{m}/sim"]].mean()
            ps.columns = ["x", "g", "a"]
            t[c] = {"sim": boot_stat(ps[["x"]], lambda d: d["x"], n_boot), "frac_of_gt": boot_stat(ps, lambda d: d["x"] / d["g"], n_boot),
                    "gain_over_a": boot_stat(ps, lambda d: d["x"] - d["a"], n_boot),
                    "closure": boot_stat(ps, lambda d: (d["x"] - d["a"]) / (d["g"] - d["a"]) if d["g"] - d["a"] > 1e-9 else math.nan, n_boot)}
        res[m] = t
    # held-out judges together (mean of CAM++ and WeSpeaker) per speaker
    t = {}
    for c in [*conds, "gt"]:
        ps = pd.DataFrame({"x": df.groupby("speaker")[[f"{c}/{m}/sim" for m in JUDGES]].mean().mean(axis=1),
                           "g": df.groupby("speaker")[[f"gt/{m}/sim" for m in JUDGES]].mean().mean(axis=1),
                           "a": df.groupby("speaker")[[f"a_refs/{m}/sim" for m in JUDGES]].mean().mean(axis=1)})
        t[c] = {"sim": boot_stat(ps[["x"]], lambda d: d["x"], n_boot), "frac_of_gt": boot_stat(ps, lambda d: d["x"] / d["g"], n_boot),
                "gain_over_a": boot_stat(ps, lambda d: d["x"] - d["a"], n_boot),
                "closure": boot_stat(ps, lambda d: (d["x"] - d["a"]) / (d["g"] - d["a"]) if d["g"] - d["a"] > 1e-9 else math.nan, n_boot)}
    res["judges"] = t
    res["utmos"] = {c: float(df[f"{c}/utmos"].mean()) for c in [*conds, "gt"]}
    res["wer"] = {c: float(100 * df[f"{c}/wer_err"].sum() / max(df[f"{c}/wer_len"].sum(), 1)) for c in [*conds, "gt"]}
    res["donor_sim"] = {m: float(df[f"f_swap/{m}/donor_sim"].mean()) for m in MODELS if f"f_swap/{m}/donor_sim" in df}
    res["fit"] = {k: float(df[k].mean()) for k in df.columns if k.endswith(("/cos_start", "/cos_end", "/shift")) and df[k].notna().any()}
    return res


def fmt(d: dict[str, float] | None, digits: int = 3) -> str:
    if d is None or not np.isfinite(d.get("mean", math.nan)):
        return "n/a"
    lo, hi = d.get("lo", math.nan), d.get("hi", math.nan)
    return f"{d['mean']:.{digits}f}" + (f" [{lo:.{digits}f}, {hi:.{digits}f}]" if np.isfinite(lo) else "")


def render_markdown(results: dict[str, Any], choice: dict[str, Any]) -> str:
    lines = ["# SV-driven code fitting through the frozen generator (test items)", "",
             "Fitted judge ECAPA is reported only as the objective; CAM++ and WeSpeaker ResNet34 are the held-out verifiers.", "",
             "Choice (validation): " + json.dumps({m: {o: {k: v for k, v in e.items() if k in ("lr", "lam", "w_rec", "steps")} for o, e in d.items()}
                                                  for m, d in choice.items() if m in ("z", "ffn")}), ""]
    for pool, res in results.items():
        lines += [f"## {pool} ({res['n_items']} items, {res['n_speakers']} speakers; similarity to "
                  f"{'held-out references' if res['held_out_refs'] else 'all references (none held out)'})", ""]
        for m in ("judges", "holdout", "wespeaker", "ecapa"):
            t = res[m]
            name = {"judges": "mean of CAM++ and WeSpeaker (held-out)", "holdout": "CAM++ (held-out)", "wespeaker": "WeSpeaker ResNet34 (held-out)",
                    "ecapa": "ECAPA (the fitted objective, not a held-out judge)"}[m]
            lines += [f"### {name}", "", "| condition | similarity | fraction of ground truth | gain over (a) | closure of (a)-to-GT gap |", "|---|---|---|---|---|"]
            for c in [*COND_ORDER, "gt"]:
                if c in t:
                    lines.append(f"| {LABEL[c]} | {fmt(t[c]['sim'])} | {fmt(t[c]['frac_of_gt'])} | {fmt(t[c]['gain_over_a']) if c != 'gt' else ''} | "
                                 f"{fmt(t[c]['closure']) if c not in ('gt', 'a_refs') else ''} |")
            lines.append("")
        lines += ["Quality guard (UTMOS / WER %): " + ", ".join(f"{LABEL[c]} {res['utmos'][c]:.2f} / {res['wer'][c]:.1f}" for c in [*COND_ORDER, "gt"] if c in res["utmos"]), ""]
        if res.get("donor_sim"):
            lines += ["(f) similarity of the swapped rendering to the code's donor speaker: " + json.dumps({k: round(v, 3) for k, v in res["donor_sim"].items()}), ""]
    return "\n".join(lines) + "\n"


def run_summarize(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    choice = json.loads((out / "choice.json").read_text())
    results = {}
    for pool in [p for p in TEST_POOLS if (out / "test" / p / "rows").is_dir()]:
        df = pd.DataFrame([json.loads(p.read_text()) for p in sorted((out / "test" / pool / "rows").glob("*.json"))])
        results[pool] = summarize_pool(df, args.n_boot)
    (out / "summary.json").write_text(json.dumps({"choice": choice, "pools": results}, indent=2, default=float))
    (out / "summary.md").write_text(render_markdown(results, choice))
    print((out / "summary.md").read_text())
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sweep", "run"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--root", default=PHASE4_ROOT)
        s.add_argument("--device", default="auto")
        s.add_argument("--shard", type=int, default=0)
        s.add_argument("--nshards", type=int, default=1)
        s.add_argument("--pool", required=True, choices=tuple(TEST_POOLS) if name == "run" else tuple(VAL_POOLS))
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=4000)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    {"sweep": run_sweep, "select": run_select, "run": run_pool, "summarize": run_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
