"""Out-of-domain speaker-similarity gap diagnosis on a trained vocoder (no vocoder training).

Question: own-speaker resynthesis reaches only ~0.6 of the ground-truth same-speaker similarity on VCTK / GLOBE (against
~0.9 on LibriTTS-R) even without a swap. Is the speaker *code* the bottleneck (the encoder maps the references to a poor
point), or can the *generator / conditioning* not render these voices at all?

Every condition resynthesizes the speaker's own held-out utterance (the ``own`` items of the cloning plans, K = 5
references; GLOBE pseudo-speakers only have 2):

* ``a_refs``    code from the speaker's reference utterances (the standard own-speaker setting; reproduces ``own_t3`` of
  :mod:`sparc.vocoders.eval.clone` on the same items);
* ``b_oracle``  code from the target utterance itself (K = 1);
* ``c_z`` / ``c_ffn``  test-time fitting from (a) on the references only (never the target): the 64-d code itself
  (``c_z``) or an additive offset on the speaker network's pooled input vectors (``c_ffn``, shared by the K references,
  passed through the frozen FFN) minimizes the training reconstruction loss (``mel_weight * mel + mss_weight * mss`` of
  the run's config, no adversarial terms; random 64-frame crops of the references, the references' own utterance F0 for
  the speaker-relative pitch input) through the frozen generator with Adam;
* ``d_z`` / ``d_ffn``  the same fitting on the target utterance itself (upper bound: can any code make this generator
  render this voice?);
* ``gt``  ground-truth audio of the same utterance (``gt_same`` of the cloning evaluation: copy-synthesis ceiling).

Metric: cosine of the output's SV embedding to the mean embedding of the speaker's reference utterances that neither gave
the code nor were fitted (``held``; NaN for GLOBE, whose pseudo-speakers have no other utterance) and to all of the
speaker's references (``all``, as ``sim_target`` of the cloning evaluation; the fitted references are inside it). ``sim``
is ``held`` where it exists, otherwise ``all``. The ground truth is scored against the same reference sets.

Interpretation rule (fixed before the runs). With A, C, D the speaker-averaged similarities of (a), (c), (d) and G the
ground-truth level:

* closure = (C - A) / (D - A) above 0.5 with a bootstrap interval excluding 0 for (c) over (a): the code / encoder is
  the bottleneck (fixable by representation);
* (D - A) small but D / G stays well below 1 (about 0.8 or less): even the best code cannot make the generator render the
  voice, so the conditioning or generator is the bottleneck;
* (B - A) large (the oracle code from the target utterance far above the references' code): the references carry
  session / content information the encoder over-uses (or, equivalently, the target utterance is more informative about
  itself than the other utterances are).

Fitting steps and learning rate are chosen on a separate LibriTTS-R dev set (``tune``), never on the test items.

Usage (GPU)::

    python -m sparc.vocoders.eval.ood_gap tune --run <run> --out <out> --shard 0 --nshards 4
    python -m sparc.vocoders.eval.ood_gap tune-select --out <out>            # writes <out>/choice.json (CPU)
    python -m sparc.vocoders.eval.ood_gap run --run <run> --out <out> --pool vctk --shard 0 --nshards 4
    python -m sparc.vocoders.eval.ood_gap summarize --out <out>              # CPU
"""

import argparse
import dataclasses
import hashlib
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

logger = logging.getLogger(__name__)

PHASE4_ROOT = "/data/user_data/xoy/spkemb/phase4"
STORED_ITEMS = "/data/user_data/xoy/spkemb/phase5/eval/p5_fv/items.parquet"
MODELS = ("ecapa", "wavlm_sv", "holdout")
HOLDOUT_SPEC = "campplus:"
POOLS = ("libritts", "vctk", "globe")
DEV_POOL = "libritts_dev"
COND_ORDER = ("a_refs", "b_oracle", "c_z", "c_ffn", "d_z", "d_ffn", "d_z_long", "d_ffn_long")
LONG_FIT = {"z": (0.03, 800), "ffn": (0.01, 800)}  # fixed (lr, steps) of the unregularized upper-bound fits (d_*_long), not tuned
CROP_FRAMES = 64
FIT_BATCH = 8
GAIN_DB = -3.0
DEFAULT_SIZES = {"libritts": (20, 5), "vctk": (20, 5), "globe": (40, 1)}  # (speakers, max items per speaker)
TUNE_SPEAKERS, TUNE_PER_SPEAKER = 8, 2
TUNE_STEPS = (0, 25, 50, 100, 200, 400, 800)
TUNE_LRS = {"z": (0.001, 0.003, 0.01, 0.03, 0.1, 0.3), "ffn": (0.001, 0.003, 0.01, 0.03, 0.1)}  # ffn: in units of the input std


# ------------------------------------------------------------------------------------------------ pure helpers


def stable_seed(*parts: Any) -> int:
    """A reproducible 31-bit seed from the parts (not Python's salted ``hash``)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31)


def select_own_items(plan: dict[str, Any], n_speakers: int, per_speaker: int, seed: int = 0) -> list[dict[str, Any]]:
    """Own items of ``n_speakers`` seeded-random speakers (all if fewer), at most ``per_speaker`` each, by ``own_id``."""
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for o in sorted(plan["own"], key=lambda o: o["own_id"]):
        by_speaker.setdefault(o["speaker"], []).append(o)
    speakers = sorted(by_speaker)
    order = np.random.default_rng([seed, 31]).permutation(len(speakers))
    chosen = [speakers[i] for i in order[:n_speakers]]
    return [o for s in sorted(chosen) for o in by_speaker[s][:per_speaker]]


def sample_crops(lengths: Sequence[int], crop: int, batch: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """``batch`` ``(utterance index, start frame)`` pairs: uniform utterance, uniform start inside it (every utterance
    must be at least ``crop`` frames long)."""
    out = []
    for _ in range(batch):
        u = int(rng.integers(len(lengths)))
        out.append((u, int(rng.integers(0, lengths[u] - crop + 1))))
    return out


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cosine_to_mean(emb: np.ndarray, ref_emb: np.ndarray | None) -> np.ndarray:
    """Cosine of each row of ``emb`` to the unit mean of the unit rows of ``ref_emb`` (NaN without references)."""
    if ref_emb is None or len(ref_emb) == 0:
        return np.full(len(emb), np.nan)
    mean = unit_rows(unit_rows(ref_emb).mean(0))
    return unit_rows(emb) @ mean


def speaker_level(df: pd.DataFrame, col: str, by: str = "speaker") -> pd.Series:
    """Per-speaker mean of the finite values of ``col``."""
    v = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return v.groupby(df[by]).mean().dropna()


def boot_stat(per_speaker: pd.DataFrame, fn, n_boot: int = 4000, seed: int = 0) -> dict[str, float]:
    """``fn(mean of the resampled speakers' rows)`` with a speaker-level percentile bootstrap 95% interval.
    ``per_speaker`` has one row per speaker (columns are the quantities ``fn`` combines, a dict of column means)."""
    rng = np.random.default_rng([seed, 5])
    n = len(per_speaker)
    point = fn(per_speaker.mean().to_dict())
    if n < 2:
        return {"mean": point, "lo": math.nan, "hi": math.nan, "n": n}
    vals = []
    arr = per_speaker.to_numpy()
    cols = list(per_speaker.columns)
    for _ in range(n_boot):
        m = arr[rng.integers(0, n, n)].mean(0)
        vals.append(fn(dict(zip(cols, m))))
    vals = np.asarray(vals, dtype=float)
    ok = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (math.nan, math.nan))
    return {"mean": float(point), "lo": float(lo), "hi": float(hi), "n": n}


def closure(a: float, c: float, d: float) -> float:
    """Fraction of the gap from (a) to (d) that (c) closes (NaN when (d) is not above (a))."""
    return (c - a) / (d - a) if d - a > 1e-9 else math.nan


# ------------------------------------------------------------------------------------------------ fitting


def fit_embedding(
    module, device, data: Sequence[dict[str, Any]], mode: str, z0: np.ndarray, raw: np.ndarray | None, ref_f0: float | None,
    lr: float, snapshots: Sequence[int], seed: int, batch: int = FIT_BATCH, crop: int = CROP_FRAMES,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Adam on an offset of the speaker code (``mode='z'``: added to ``z0 [64]``) or of the speaker network's raw input
    vectors (``mode='ffn'``: one offset ``[D]`` added to each of the K rows of ``raw [K, D]``, then the frozen FFN, mean
    over references and the pitch code) to minimize ``mel_weight * mel + mss_weight * mss`` of the frozen generator on
    random crops of ``data`` (dicts with ``feats [15, T]``, ``audio [N]``, ``utt_f0``). Returns ``{step: code [64]}`` for
    the requested steps (0 = the initial code) and the mean loss over the first / last 10 steps."""
    import torch

    from sparc.vocoders.training.callbacks import EVAL_SEED, fixed_torch_rng

    w = module.cfg.loss
    w_mel, w_mss = float(w.mel_weight), float(w.mss_weight)
    crop = min(crop, min(int(d["feats"].shape[1]) for d in data))
    feats = [torch.from_numpy(np.ascontiguousarray(d["feats"])).to(device) for d in data]
    audio = [torch.from_numpy(np.ascontiguousarray(d["audio"], dtype=np.float32)).to(device) for d in data]
    utt_f0 = torch.tensor([float(d["utt_f0"]) for d in data], dtype=torch.float32, device=device)
    lengths = [int(f.shape[1]) for f in feats]
    z0_t = torch.from_numpy(np.asarray(z0, dtype=np.float32)).to(device)
    if mode == "z":
        delta = torch.zeros_like(z0_t, requires_grad=True)
        step_lr = lr
    elif mode == "ffn":
        raw_t = torch.from_numpy(np.asarray(raw, dtype=np.float32)).to(device)
        delta = torch.zeros(raw_t.shape[1], device=device, requires_grad=True)
        step_lr = lr * float(module.speaker.std.float().mean())
        mask = torch.ones(1, raw_t.shape[0], dtype=torch.bool, device=device)
    else:
        raise ValueError(mode)

    def code() -> torch.Tensor:
        if mode == "z":
            return z0_t + delta
        z, _ = module.speaker.embed((raw_t + delta)[None], mask)
        pitch = getattr(module, "pitch_code", None)
        if pitch is not None:
            z = z + pitch(torch.tensor([math.nan if ref_f0 is None else ref_f0], dtype=torch.float32, device=device))
        return z[0]

    opt = torch.optim.Adam([delta], lr=step_lr)
    rng = np.random.default_rng([seed, 17])
    snaps = sorted({int(s) for s in snapshots})
    out: dict[int, np.ndarray] = {}
    with torch.no_grad():
        if 0 in snaps:
            out[0] = code().float().cpu().numpy()
    losses: list[float] = []
    for step in range(1, (snaps[-1] if snaps else 0) + 1):
        picks = sample_crops(lengths, crop, batch, rng)
        f = torch.stack([feats[u][:, s : s + crop] for u, s in picks])
        a = torch.stack([audio[u][s * HOP : (s + crop) * HOP] for u, s in picks])[:, None]
        with torch.enable_grad(), fixed_torch_rng(device, EVAL_SEED):
            z = code()
            batch_in = {"features": f, "utt_f0": utt_f0[[u for u, _ in picks]]}
            wav = module.synthesize(batch_in, z=z[None].expand(len(picks), -1))
            loss = w_mel * module.mel_loss(wav, a) + w_mss * module.mss_loss(wav, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(float(loss.detach()))
        if step in snaps:
            with torch.no_grad():
                out[step] = code().float().cpu().numpy()
    stats = {"loss_start": float(np.mean(losses[:10])) if losses else math.nan, "loss_end": float(np.mean(losses[-10:])) if losses else math.nan}
    return out, stats


def render_code(module, device, feats: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Waveform ``[n]`` of content features ``[15, T]`` rendered with code ``z [64]`` (as ``ItemRenderer.render``)."""
    import torch

    from sparc.vocoders.eval.clone import median_f0_hz
    from sparc.vocoders.training.callbacks import EVAL_SEED, fixed_torch_rng

    batch = {"features": torch.from_numpy(np.ascontiguousarray(feats))[None].to(device)}
    if getattr(module, "rel_f0", False):
        batch["utt_f0"] = torch.tensor([median_f0_hz(feats)], dtype=torch.float32, device=device)
    with torch.inference_mode(), fixed_torch_rng(device, EVAL_SEED):
        wav = module.synthesize(batch, z=torch.from_numpy(np.asarray(z, dtype=np.float32))[None].to(device))
    return wav[0, 0].float().cpu().numpy()


# ------------------------------------------------------------------------------------------------ context


class Context:
    """Loaded run, pool stores, reference conditions and scorers shared by ``tune`` and ``run``."""

    def __init__(self, run: str, pool: str, root: str, device: str, models: Sequence[str]):
        from sparc.vocoders.eval.clone import CloneScorers, ItemRenderer, RefConditions
        from sparc.vocoders.eval.clone_pools import Pool
        from sparc.vocoders.eval.interp import load_run, pick_device, setup_offline_env

        setup_offline_env()
        self.module, cfg, self.info = load_run(run, None, device)
        self.device = next(self.module.parameters()).device
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.layer = str(cfg.speaker.layer)
        self.pool = pool
        self.plan = json.loads((Path(root) / "plans" / f"{pool}.json").read_text())
        self.pool_obj = Pool(pool, root)
        self.store = self.pool_obj.store(self.layer)
        self.conds = {pool: RefConditions(self.pool_obj, self.store, self.layer, GAIN_DB, self.device, None)}
        self.renderer = ItemRenderer(self.module, {pool: self.store}, self.conds, GAIN_DB, self.device)
        self.models = list(models)
        self.scorers = CloneScorers(_device(device), HOLDOUT_SPEC if "holdout" in models else None)
        self._gt: dict[str, dict[str, np.ndarray]] = {}

    def own_item(self, o: dict[str, Any]):
        from sparc.vocoders.eval.clone import Variant, build_items, own_variants

        plan = {**self.plan, "own": [o], "swaps": []}
        return build_items(plan, own_variants(5), ["own_t3"])[0]

    def utterance(self, uid: str) -> dict[str, Any]:
        """Fit data of one utterance: features, audio at the evaluation gain and its median voiced F0."""
        from sparc.vocoders.eval.clone import median_f0_hz
        from sparc.vocoders.eval.interp import utterance_features
        from sparc.vocoders.eval.io import gt_audio

        feats = utterance_features(self.store, uid, GAIN_DB)
        audio = gt_audio(self.store, self.store.index_of(uid), GAIN_DB)
        n = feats.shape[1] * HOP
        return {"feats": feats, "audio": np.asarray(audio[:n], dtype=np.float32), "utt_f0": median_f0_hz(feats)}

    def embed(self, wavs24: list[np.ndarray]) -> dict[str, np.ndarray]:
        from sparc.vocoders.eval.io import to_16k

        wavs = [to_16k(np.asarray(w, dtype=np.float32), SAMPLE_RATE) for w in wavs24]
        out = {m: unit_rows(self.scorers.get(m).embed(wavs, errors=[])) for m in self.models}
        return out

    def utmos(self, wavs24: list[np.ndarray]) -> np.ndarray:
        from sparc.vocoders.eval.io import to_16k

        wavs = [to_16k(np.asarray(w, dtype=np.float32), SAMPLE_RATE) for w in wavs24]
        return np.asarray(self.scorers.base.get("utmos").score(wavs, []), dtype=np.float64)

    def gt_embeddings(self, uids: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
        from sparc.vocoders.eval.clone import _gt16

        todo = [u for u in uids if u not in self._gt]
        if todo:
            wavs = [_gt16(self.store, u, GAIN_DB) for u in todo]
            emb = {m: unit_rows(self.scorers.get(m).embed(wavs, errors=[])) for m in self.models}
            for i, u in enumerate(todo):
                self._gt[u] = {m: emb[m][i] for m in self.models}
        return {u: self._gt[u] for u in uids}

    def reference_sets(self, o: dict[str, Any], code_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        from sparc.vocoders.eval.clone import holdout_refs

        refs = list(self.plan["refs"][o["speaker"]])
        return refs, holdout_refs(refs, code_ids)

    def sims(self, emb: dict[str, np.ndarray], o: dict[str, Any], code_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """``{<model>/held, <model>/all}`` cosines for each row of ``emb[model]``."""
        refs, held = self.reference_sets(o, code_ids)
        g = self.gt_embeddings(refs)
        out = {}
        for m in self.models:
            ref_all = np.stack([g[u][m] for u in refs])
            ref_held = np.stack([g[u][m] for u in held]) if held else None
            out[f"{m}/all"] = cosine_to_mean(emb[m], ref_all)
            out[f"{m}/held"] = cosine_to_mean(emb[m], ref_held)
        return out

    def codes(self, it) -> tuple[np.ndarray, np.ndarray, float | None]:
        """``(z of the references, raw vectors [K, D], reference F0)`` of an item's code references."""
        from sparc.vocoders.eval.clone import reference_f0_hz

        z, _ = self.renderer.code(it)
        raw = np.stack([self.store.speaker_vector(self.store.index_of(u)) for u in it.code_ids]).astype(np.float32)
        ref_f0 = reference_f0_hz(self.conds[self.pool].clean_features(it.code_ids)) if self.renderer.rel_f0 else None
        return z, raw, ref_f0


def _device(device: str):
    from sparc.vocoders.eval.interp import pick_device

    return pick_device(device)


# ------------------------------------------------------------------------------------------------ tune


def run_tune(args: argparse.Namespace) -> None:
    """Fits the references of dev items from the standard code over a grid of learning rates (snapshots at several step
    counts) and records held-out similarity to the speaker's other references (never the fitted ones)."""
    import torch

    ctx = Context(args.run, DEV_POOL, args.root, args.device, ("ecapa", "wavlm_sv"))
    items = select_own_items(ctx.plan, TUNE_SPEAKERS, TUNE_PER_SPEAKER, seed=1)
    out = Path(args.out) / "tune"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    mine = [o for i, o in enumerate(items) if i % args.nshards == args.shard]
    start = time.time()
    for o in mine:
        it = ctx.own_item(o)
        z0, raw, ref_f0 = ctx.codes(it)
        data = [ctx.utterance(u) for u in it.code_ids]
        content = ctx.utterance(it.content_id)["feats"]
        for mode in ("z", "ffn"):
            for lr in TUNE_LRS[mode]:
                t0 = time.time()
                snaps, stats = fit_embedding(ctx.module, ctx.device, data, mode, z0, raw, ref_f0, lr, TUNE_STEPS, stable_seed("tune", o["own_id"], mode, lr))
                steps = sorted(snaps)
                wavs = [render_code(ctx.module, ctx.device, content, snaps[s]) for s in steps]
                sims = ctx.sims(ctx.embed(wavs), o, it.code_ids)
                for j, s in enumerate(steps):
                    rows.append({"own_id": o["own_id"], "speaker": o["speaker"], "mode": mode, "lr": lr, "step": s, **{k: float(v[j]) for k, v in sims.items()}, **stats})
                logger.info("tune %s %s lr %g: %.1f s (%.0f s total)", o["own_id"], mode, lr, time.time() - t0, time.time() - start)
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_parquet(out / f"rows_shard{args.shard:02d}.parquet")


def select_choice(rows: pd.DataFrame) -> dict[str, Any]:
    """Per mode the ``(lr, step)`` maximizing the mean (over dev items) of the held-out similarity averaged over ECAPA and
    WavLM-SV (ties: fewer steps). Step 0 (no fitting) is allowed: if nothing beats it, the choice is 0 steps."""
    rows = rows.copy()
    rows["score"] = (rows["ecapa/held"] + rows["wavlm_sv/held"]) / 2
    choice: dict[str, Any] = {}
    for mode, g in rows.groupby("mode"):
        base = float(g[g["step"] == 0]["score"].mean())
        table = g.groupby(["lr", "step"])["score"].mean().reset_index().sort_values(["score", "step"], ascending=[False, True])
        best = table.iloc[0]
        use = best if float(best["score"]) > base else None
        choice[mode] = {"lr": float(best["lr"]) if use is not None else 0.0, "steps": int(best["step"]) if use is not None else 0,
                        "dev_score": float(best["score"]), "dev_score_no_fit": base, "n_items": int(g["own_id"].nunique())}
    return choice


def run_tune_select(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    rows = pd.concat([pd.read_parquet(p) for p in sorted((out / "tune").glob("rows_shard*.parquet"))], ignore_index=True)
    choice = select_choice(rows)
    (out / "choice.json").write_text(json.dumps(choice, indent=2))
    grid = rows.assign(score=(rows["ecapa/held"] + rows["wavlm_sv/held"]) / 2).groupby(["mode", "lr", "step"])["score"].mean().unstack("step")
    (out / "tune_grid.md").write_text(grid.round(4).to_string() + "\n")
    print(json.dumps(choice, indent=2))
    return choice


# ------------------------------------------------------------------------------------------------ run


def run_pool(args: argparse.Namespace) -> None:
    """Renders and scores the six conditions for this shard's items of one pool (resumable per item)."""
    import soundfile as sf

    ctx = Context(args.run, args.pool, args.root, args.device, MODELS)
    choice = json.loads((Path(args.out) / "choice.json").read_text()) if args.choice is None else json.loads(Path(args.choice).read_text())
    n_spk, per = (args.n_speakers or DEFAULT_SIZES[args.pool][0]), (args.max_per_speaker or DEFAULT_SIZES[args.pool][1])
    items = select_own_items(ctx.plan, n_spk, per, seed=0)
    out = Path(args.out) / args.pool
    (out / "wav").mkdir(parents=True, exist_ok=True)
    mine = [o for i, o in enumerate(items) if i % args.nshards == args.shard]
    start = time.time()
    for o in mine:
        row_path = out / "rows" / f"{o['own_id']}.json"
        if row_path.is_file():
            continue
        it = ctx.own_item(o)
        z_a, raw, ref_f0 = ctx.codes(it)
        z_b, _, _ = ctx.codes(dataclasses.replace(it, code_ids=[it.content_id]))
        refs_data = [ctx.utterance(u) for u in it.code_ids]
        target = ctx.utterance(it.content_id)
        zs: dict[str, np.ndarray] = {"a_refs": z_a, "b_oracle": z_b}
        info: dict[str, float] = {}
        for mode in ("z", "ffn"):
            ch = choice[mode]
            for tag, data in (("c", refs_data), ("d", [target])):
                cond = f"{tag}_{mode}"
                if ch["steps"] == 0:
                    zs[cond] = z_a
                    continue
                snaps, stats = fit_embedding(ctx.module, ctx.device, data, mode, z_a, raw, ref_f0, ch["lr"], [ch["steps"]], stable_seed("run", o["own_id"], cond))
                zs[cond] = snaps[ch["steps"]]
                info.update({f"{cond}/loss_start": stats["loss_start"], f"{cond}/loss_end": stats["loss_end"]})
                info[f"{cond}/code_shift"] = float(np.linalg.norm(zs[cond] - z_a) / max(np.linalg.norm(z_a), 1e-9))
        for mode in ("z", "ffn"):
            cond = f"d_{mode}_long"
            lr, steps = LONG_FIT[mode]
            snaps, stats = fit_embedding(ctx.module, ctx.device, [target], mode, z_a, raw, ref_f0, lr, [steps], stable_seed("run", o["own_id"], cond))
            zs[cond] = snaps[steps]
            info.update({f"{cond}/loss_start": stats["loss_start"], f"{cond}/loss_end": stats["loss_end"],
                         f"{cond}/code_shift": float(np.linalg.norm(zs[cond] - z_a) / max(np.linalg.norm(z_a), 1e-9))})
        info["b_oracle/code_shift"] = float(np.linalg.norm(z_b - z_a) / max(np.linalg.norm(z_a), 1e-9))
        conds = list(zs)
        wavs = [render_code(ctx.module, ctx.device, target["feats"], zs[c]) for c in conds]
        gt_wav = np.asarray(target["audio"], dtype=np.float32)
        allw = wavs + [gt_wav]
        emb = ctx.embed(allw)
        mos = ctx.utmos(allw)
        sims = ctx.sims(emb, o, it.code_ids)
        row: dict[str, Any] = {"pool": args.pool, "own_id": o["own_id"], "speaker": o["speaker"], "content_id": o["content_id"], "dur_s": o["dur_s"],
                               "n_code_refs": len(it.code_ids), "n_held": len(ctx.reference_sets(o, it.code_ids)[1]), **info}
        for j, c in enumerate([*conds, "gt"]):
            row[f"{c}/utmos"] = float(mos[j])
            for k, v in sims.items():
                row[f"{c}/{k}"] = float(v[j])
        for c, w in zip(conds, wavs):
            p = out / "wav" / c / f"{o['own_id']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), w, SAMPLE_RATE, subtype="PCM_16")
        row_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text(json.dumps(row))
        logger.info("%s %s done (%.0f s)", args.pool, o["own_id"], time.time() - start)


# ------------------------------------------------------------------------------------------------ summarize


def load_rows(out: Path, pool: str) -> pd.DataFrame:
    return pd.DataFrame([json.loads(p.read_text()) for p in sorted((out / pool / "rows").glob("*.json"))])


def with_sim(df: pd.DataFrame, conds: Sequence[str]) -> pd.DataFrame:
    """Adds ``<cond>/<model>/sim`` = held where finite else all (per pool the held-out set exists for every item or none)."""
    df = df.copy()
    for c in conds:
        for m in MODELS:
            if f"{c}/{m}/all" in df:
                df[f"{c}/{m}/sim"] = df[f"{c}/{m}/held"].where(df[f"{c}/{m}/held"].notna(), df[f"{c}/{m}/all"])
    return df


def summarize_pool(df: pd.DataFrame, n_boot: int = 4000) -> dict[str, Any]:
    conds = [c for c in COND_ORDER if f"{c}/utmos" in df]
    df = with_sim(df, [*conds, "gt"])
    res: dict[str, Any] = {"n_items": int(len(df)), "n_speakers": int(df["speaker"].nunique()), "held_out_refs": bool(df["n_held"].min() > 0)}
    for m in MODELS:
        if f"gt/{m}/sim" not in df:
            continue
        table: dict[str, Any] = {}
        for c in [*conds, "gt"]:
            ps = df.groupby("speaker")[[f"{c}/{m}/sim", f"gt/{m}/sim"]].mean()
            ps.columns = ["x", "g"]
            table[c] = {"sim": boot_stat(ps[["x"]], lambda d: d["x"], n_boot), "frac_of_gt": boot_stat(ps, lambda d: d["x"] / d["g"], n_boot)}
        for c in conds:
            ps = df.groupby("speaker")[[f"{c}/{m}/sim", f"a_refs/{m}/sim"]].mean()
            ps.columns = ["x", "a"]
            table[c]["gain_over_a"] = boot_stat(ps, lambda d: d["x"] - d["a"], n_boot)
        for mode in ("z", "ffn"):
            for dn in (f"d_{mode}", f"d_{mode}_long"):
                if f"c_{mode}/utmos" in df and f"{dn}/utmos" in df:
                    ps = df.groupby("speaker")[[f"a_refs/{m}/sim", f"c_{mode}/{m}/sim", f"{dn}/{m}/sim"]].mean()
                    ps.columns = ["a", "c", "d"]
                    table[f"closure_{dn}"] = boot_stat(ps, lambda d: closure(d["a"], d["c"], d["d"]), n_boot)
        res[m] = table
    res["utmos"] = {c: float(df[f"{c}/utmos"].mean()) for c in [*conds, "gt"]}
    res["fit"] = {k: float(df[k].mean()) for k in df.columns if k.endswith(("/loss_start", "/loss_end", "/code_shift")) and df[k].notna().any()}
    return res


def stored_check(df: pd.DataFrame, pool: str, stored_path: str = STORED_ITEMS) -> dict[str, Any]:
    """Recomputed (a) against the stored ``own_t3`` rows of the cloning evaluation on the same items (``<model>/sim_target``
    = all references; ``sim_spk`` = held-out references; UTMOS)."""
    if not Path(stored_path).is_file():
        return {}
    st = pd.read_parquet(stored_path)
    st = st[(st["pool"] == pool) & (st["variant"] == "own_t3")].set_index("pair_id")
    j = df.set_index("own_id").join(st, how="inner", rsuffix="_stored")
    out: dict[str, Any] = {"n_matched": int(len(j))}
    for m in MODELS:
        for mine, theirs in ((f"a_refs/{m}/all", f"{m}/sim_target"), (f"a_refs/{m}/held", f"{m}/sim_spk")):
            if mine in j and theirs in j and j[theirs].notna().any():
                d = (j[mine] - j[theirs]).abs()
                out[f"{mine} vs {theirs}"] = {"mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max())}
    if "utmos_stored" in j or "utmos" in st:
        col = "utmos_stored" if "utmos_stored" in j else "utmos"
        out["utmos"] = {"mean_abs_diff": float((j["a_refs/utmos"] - j[col]).abs().mean())}
    return out


RULE = """Interpretation rule (fixed before the runs). With A, C, D the speaker-averaged similarities of (a), (c), (d) and G the
ground-truth level:

* closure = (C - A) / (D - A) above 0.5 with a bootstrap interval excluding 0 for (c) over (a): the code / encoder is
  the bottleneck (fixable by representation);
* (D - A) small but D / G stays well below 1 (about 0.8 or less): even the best code cannot make the generator render the
  voice, so the conditioning or generator is the bottleneck;
* (B - A) large (the oracle code from the target utterance far above the references' code): the references carry
  session / content information the encoder over-uses (or, equivalently, the target utterance is more informative about
  itself than the other utterances are).

"""


def fmt(d: dict[str, float], digits: int = 3) -> str:
    if d is None or not np.isfinite(d.get("mean", math.nan)):
        return "n/a"
    lo, hi = d.get("lo", math.nan), d.get("hi", math.nan)
    return f"{d['mean']:.{digits}f}" + (f" [{lo:.{digits}f}, {hi:.{digits}f}]" if np.isfinite(lo) else "")


def render_markdown(results: dict[str, Any], choice: dict[str, Any]) -> str:
    lines = ["# Out-of-domain speaker-similarity gap diagnosis", "", "## " + RULE.strip(), "",
             f"Fitting choice (picked on {DEV_POOL}): {json.dumps({k: {'lr': v['lr'], 'steps': v['steps']} for k, v in choice.items()})}", ""]
    for pool, res in results.items():
        lines += [f"## {pool} ({res['n_items']} items, {res['n_speakers']} speakers; similarity to "
                  f"{'held-out references' if res['held_out_refs'] else 'all references (none held out)'})", ""]
        for m in MODELS:
            if m not in res:
                continue
            t = res[m]
            lines += [f"### {m}", "", "| condition | similarity | fraction of ground truth | gain over (a) | UTMOS |", "|---|---|---|---|---|"]
            for c in [*COND_ORDER, "gt"]:
                if c in t:
                    gain = fmt(t[c].get("gain_over_a")) if c != "gt" else ""
                    lines.append(f"| {c} | {fmt(t[c]['sim'])} | {fmt(t[c]['frac_of_gt'])} | {gain} | {res['utmos'].get(c, math.nan):.2f} |")
            for k in [k for k in t if k.startswith("closure_")]:
                lines.append(f"\nClosure of the (a)-to-({k[8:]}) gap by (c, same mode): {fmt(t[k])}")
            lines.append("")
        if res.get("check"):
            lines += ["Reproduction of the stored own-speaker numbers (recomputed (a) vs `own_t3`, same items): " + json.dumps(res["check"]), ""]
    return "\n".join(lines) + "\n"


def run_summarize(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    choice = json.loads((out / "choice.json").read_text())
    results = {}
    for pool in [p for p in args.pools.split(",") if (out / p / "rows").is_dir()]:
        df = load_rows(out, pool)
        results[pool] = summarize_pool(df, args.n_boot)
        results[pool]["check"] = stored_check(df, pool)
    (out / "summary.json").write_text(json.dumps({"choice": choice, "pools": results}, indent=2, default=float))
    (out / "summary.md").write_text(render_markdown(results, choice))
    print((out / "summary.md").read_text())
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("tune", "run"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--root", default=PHASE4_ROOT)
        s.add_argument("--device", default="auto")
        s.add_argument("--shard", type=int, default=0)
        s.add_argument("--nshards", type=int, default=1)
        if name == "run":
            s.add_argument("--pool", required=True, choices=POOLS)
            s.add_argument("--choice", default=None)
            s.add_argument("--n-speakers", type=int, default=0)
            s.add_argument("--max-per-speaker", type=int, default=0)
    s = sub.add_parser("tune-select")
    s.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--pools", default=",".join(POOLS))
    s.add_argument("--n-boot", type=int, default=4000)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    {"tune": run_tune, "tune-select": run_tune_select, "run": run_pool, "summarize": run_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
