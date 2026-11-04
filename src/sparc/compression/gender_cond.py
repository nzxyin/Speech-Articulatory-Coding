"""Does speaker-gender conditioning help the multi-speaker articulatory inverter? (pre-registration:
docs/gender_conditioning_prereg.md; every arm, draw, threshold and the decision rule are fixed there.)

Reference system: frozen WavLM Large layer 12 cached features, head `gruconv9` (BiGRU -> centered conv 9), no output
smoother, leave-one-speaker-out over the seven EMA speakers, the fold's own alignment shift (depth_sweep), learning rate
from {1e-4, 3e-4, 1e-3, 3e-3} on the validation sentences of the arm's training speakers (seed 0), then seeds 1 and 2.

Arms (see the pre-registration for the full text):
    A0   pooled, unconditioned (reference recipe through this code path)
    A0b  second independent A0 (seeds 10-12): empirical noise floor
    A1   pooled, oracle gender label through zero-initialized FiLM tables
    A2   per-gender model: trained on the same-gender training speakers only
    A3   count- and corpus-matched control for A2 (mixed-gender subsets, fixed draws)
    A4   pseudo-gender control for A1 (random non-gender partitions, fixed draws)
    A4c  corpus-label control for A1 (USC vs 5EMO)
    ridge analogues R0-R4 (sufficient-statistics ridge, real and kf/jr-swapped gender assignment)

Usage:
    python -m sparc.compression.gender_cond draws [--write]
    python -m sparc.compression.gender_cond run --arm A1 --fold usc_M1 [--part 0 --parts 2] [--root DIR]
    python -m sparc.compression.gender_cond ridge [--swap]
    python -m sparc.compression.gender_cond status
    python -m sparc.compression.gender_cond analyze
"""

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import datasets, head_screen
from .head_screen import FT_SR, LRS, NO_DECAY, SEEDS, batches, fold_shift, load_features, make_head, zm
from .probe import ALPHAS, aligned, load_targets
from .rigor import metrics_from, utt_stats
from .seqheads import CascadeHead

ROOT = datasets.XLSR_EMA / "analysis_gender"
MODEL, LAYER, HEAD = "wavlm-large", 12, "gruconv9"
SPEAKERS = tuple(datasets.ALL_SPEAKERS)
GENDER = {"usc_M1": "M", "usc_M3": "M", "5emo_kf": "M", "usc_F1": "F", "usc_F5": "F", "5emo_jn": "F", "5emo_jr": "F"}
GENDER_SWAP = {**GENDER, "5emo_kf": "F", "5emo_jr": "M"}  # sensitivity: the one inferred label flipped (ridge only)
CORPUS = {s: s.split("_")[0] for s in SPEAKERS}
DRAW_SEED = 20261104
ARM_ID = {"A3": 3, "A4": 4}
D_A3, D_A4 = 6, 8
A0B_SEEDS = (10, 11, 12)
ARMS = ("A0", "A0b", "A1", "A2", "A3", "A4", "A4c")
HIDDEN, KERNEL = 256, 9
EPOCHS = 40
STORED_HEADS = datasets.XLSR_EMA / "analysis_heads" / MODEL / f"k{LAYER}_{HEAD}.json"
DRAWS_REPO = Path(__file__).resolve().parents[3] / "docs" / "gender_conditioning_draws.json"


# ---------------------------------------------------------------- draws

def a2_set(h, gender=GENDER):
    """Training speakers of the held-out speaker's gender (the per-gender model's training set)."""
    return [s for s in SPEAKERS if s != h and gender[s] == gender[h]]


def a3_candidates(h, gender=GENDER):
    """Subsets of the six training speakers with A2's size, at least one speaker of each gender, and the same number of
    speakers from h's corpus as A2's set (so the control differs from A2 in gender composition, not in corpus overlap)."""
    T = [s for s in SPEAKERS if s != h]
    a2 = a2_set(h, gender)
    n, k = len(a2), sum(CORPUS[s] == CORPUS[h] for s in a2)
    return [c for c in itertools.combinations(T, n)
            if 0 < sum(gender[s] == gender[h] for s in c) < n and sum(CORPUS[s] == CORPUS[h] for s in c) == k]


def a4_candidates(h, gender=GENDER):
    """Groups S of training speakers that join h under one label so that h's group has the real gender group's size,
    excluding the real gender partition."""
    T = [s for s in SPEAKERS if s != h]
    size = sum(gender[s] == gender[h] for s in SPEAKERS)
    real = set(a2_set(h, gender))
    return [S for S in itertools.combinations(T, size - 1) if set(S) != real]


def partition_labels(h, S):
    return {s: int(s == h or s in S) for s in SPEAKERS}


def gender_labels(gender=GENDER):
    return {s: int(gender[s] == "M") for s in SPEAKERS}


def corpus_labels(h):
    return {s: int(CORPUS[s] == CORPUS[h]) for s in SPEAKERS}


def make_draws():
    """All fixed draws, deterministic (numpy default_rng, one SeedSequence child per (arm, fold))."""
    out = {"seed": DRAW_SEED, "D_A3": D_A3, "D_A4": D_A4, "gender": GENDER, "corpus": CORPUS, "folds": {}}
    for fi, h in enumerate(SPEAKERS):
        T = [s for s in SPEAKERS if s != h]
        a2 = a2_set(h)
        mates = {s for s in SPEAKERS if CORPUS[s] == CORPUS[h] and s != h}
        c3 = a3_candidates(h)
        rng = np.random.default_rng(np.random.SeedSequence([DRAW_SEED, ARM_ID["A3"], fi]))
        perm = rng.permutation(len(c3))
        a3 = [{"j": j, "train": list(c3[perm[j % len(c3)]]), "key": "+".join(c3[perm[j % len(c3)]]),
               "n_corpus_mates": sum(CORPUS[s] == CORPUS[h] for s in c3[perm[j % len(c3)]]),
               "n_same_gender": sum(GENDER[s] == GENDER[h] for s in c3[perm[j % len(c3)]])} for j in range(D_A3)]
        c4 = a4_candidates(h)
        rng = np.random.default_rng(np.random.SeedSequence([DRAW_SEED, ARM_ID["A4"], fi]))
        perm = rng.permutation(len(c4))
        a4 = []
        for j in range(D_A4):
            S = c4[perm[j]]
            a4.append({"j": j, "group_with_h": list(S), "labels": partition_labels(h, S),
                       "overlap_gender": len(set(S) & set(a2)), "overlap_corpus": len(set(S) & mates),
                       "corpus_equal": set(S) == mates})
        out["folds"][h] = {"A2_train": a2, "A3_n_candidates": len(c3), "A3_n_unique": len({d["key"] for d in a3}), "A3": a3,
                           "A4_n_candidates": len(c4), "A4": a4, "A4c_labels": corpus_labels(h)}
    return out


# ---------------------------------------------------------------- head with label-conditioned FiLM

def film(x, table, g):
    """x (B, T, H) -> x * (1 + gamma[g]) + beta[g]; table (K, 2, H); g (B,) long. Identity at zero init."""
    t = table[g]
    return x * (1 + t[:, 0, None, :]) + t[:, 1, None, :]


class FiLMGRUConv(CascadeHead):
    """gruconv head with two zero-initialized FiLM sites (after the front end, after the BiGRU). The tables are created
    after every other parameter, so the random initialization stream is that of CascadeHead; with g=None, or at zero
    init, the output equals CascadeHead's."""

    def __init__(self, dim, kernel=KERNEL, hidden=HIDDEN, n_out=12, n_groups=2):
        super().__init__(dim, "gru_conv", kernel=kernel, hidden=hidden, n_out=n_out, smooth=False)
        self.film_front = nn.Parameter(torch.zeros(n_groups, 2, hidden))
        self.film_rnn = nn.Parameter(torch.zeros(n_groups, 2, hidden))

    def forward(self, h, pad_mask=None, g=None):
        x = self.front(h)
        if g is not None:
            x = film(x, self.film_front, g)
        x = self._gru(x, pad_mask)
        if g is not None:
            x = film(x, self.film_rnn, g)
        return self.out(self._conv(x, pad_mask))


def make_head_g(labelled, dim):
    return FiLMGRUConv(dim) if labelled else make_head(HEAD, dim, False)


def param_groups_g(head):
    """head_screen.param_groups plus: FiLM tables never decayed (they are named film_*; no capital C in the names)."""
    decay, no = [], []
    for n, p in head.named_parameters():
        (no if ("film" in n or any(k in n for k in NO_DECAY)) else decay).append(p)
    return [{"params": decay, "weight_decay": 0.01}, {"params": no, "weight_decay": 0.0}]


def _fwd(head, xb, mask, gb):
    return head(xb, mask) if gb is None else head(xb, mask, gb)


def _label_chunks(g, order, bs):
    for i in range(0, len(order), bs):
        yield None if g is None else torch.as_tensor(np.asarray([g[j] for j in order[i:i + bs]]), dtype=torch.long)


@torch.no_grad()
def predict_g(head, X, ym, ys, device, g=None, bs=16):
    """head_screen.predict with per-utterance labels g (same order as X); g=None for unconditioned heads."""
    head.eval()
    order = list(range(len(X)))
    out = []
    for (xb, _, mask), gb in zip(batches(X, [np.zeros((len(x), 12)) for x in X], order, bs), _label_chunks(g, order, bs)):
        y = _fwd(head, xb.to(device), mask.to(device), None if gb is None else gb.to(device)).float().cpu().numpy()
        for b in range(len(y)):
            n = int((~mask[b]).sum())
            out.append(y[b, :n] * ys + ym)
    return out


def train_one_g(labelled, lr, data, ym, ys, device, seed=0, epochs=None, patience=8, bs=8):
    """Copy of head_screen.train_one with per-utterance labels. data['train'] = (X, Y, g), data['valid'] likewise; g is
    None for an unconditioned head. Returns (head, best validation rmse_z, info)."""
    epochs = EPOCHS if epochs is None else epochs
    torch.manual_seed(seed)
    random.seed(seed)
    Xtr, Ytr, gtr = data["train"]
    Xva, Yva, gva = data["valid"]
    dim = Xtr[0].shape[1]
    head = make_head_g(labelled, dim).to(device)
    Ytr_s = [(y - ym) / ys for y in Ytr]
    opt = torch.optim.AdamW(param_groups_g(head), lr=lr)
    steps = math.ceil(len(Xtr) / bs)
    total = epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / steps) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    best, best_state, bad, n_ep, n_step = float("inf"), None, 0, 0, 0
    for _ in range(epochs):
        head.train()
        order = list(range(len(Xtr)))
        random.shuffle(order)
        for (xb, yb, mask), gb in zip(batches(Xtr, Ytr_s, order, bs), _label_chunks(gtr, order, bs)):
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            y = _fwd(head, xb, mask, None if gb is None else gb.to(device))
            valid = (~mask).float()[..., None]
            loss = (((y - yb) ** 2) * valid).sum() / (valid.sum() * 12)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            n_step += 1
        n_ep += 1
        v = zm(predict_g(head, Xva, ym, ys, device, gva), Yva)["rmse_z"]
        if not math.isfinite(v):
            break
        if best_state is None or v < best - 1e-5:
            best, best_state, bad = v, {k: t.detach().clone() for k, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    info = {"epochs": n_ep, "steps": n_step}
    if best_state is None:
        return None, float("inf"), info
    head.load_state_dict(best_state)
    return head, best, info


# ---------------------------------------------------------------- data

def _split_dict(X, Y, stems, spk):
    return {"X": [np.asarray(x, dtype=np.float32) for x in X], "Y": [y.astype(np.float64) for y in Y],
            "stems": list(stems), "spk": list(spk)}


def subset(split, speakers):
    keep = [i for i, s in enumerate(split["spk"]) if s in set(speakers)]
    return {k: [v[i] for i in keep] for k, v in split.items()}


def utt_labels(split, label_map):
    return None if label_map is None else [label_map[s] for s in split["spk"]]


class Ctx:
    """Cached features and the current fold's aligned data (one fold at a time)."""

    def __init__(self, device=None, model=MODEL, layer=LAYER, limit=None):
        from .depth_sweep import load_all

        self.limit = limit  # smoke runs: at most this many utterances per speaker and split
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.layer = model, layer
        self.feats, self.index = load_features(model, layer, "zp")
        self.depth = load_all(model)
        self.utts = None
        self._fold = (None, None)

    def fold(self, h):
        if self._fold[0] == h:
            return self._fold[1]
        ds = datasets.EMAMulti(held_out=(h,), norm_by="norm_group")
        sp = {k: [s for s in v if s in self.index] for k, v in ds.splits().items()}
        if self.utts is None:
            self.utts = load_targets(sorted(set(self.index) & set(ds.rows)), ds)
        shift = fold_shift(self.model, self.layer, h, self.depth)
        fd = {"shift": shift}
        for split in ("train", "valid", "test", "test_unseen"):
            X, Y, keep = aligned(self.feats, self.index, self.utts, sp[split], shift)
            assert len(X) == len(keep)
            fd[split] = _split_dict(X, Y, keep, [ds.speaker(s) for s in keep])
            if self.limit:
                seen, idx = {}, []
                for i, s in enumerate(fd[split]["spk"]):
                    seen[s] = seen.get(s, 0) + 1
                    if seen[s] <= self.limit:
                        idx.append(i)
                fd[split] = {k: [v[i] for i in idx] for k, v in fd[split].items()}
        self._fold = (h, fd)
        return fd


# ---------------------------------------------------------------- runs

@dataclass
class Spec:
    arm: str
    fold: str
    key: str
    train: tuple
    labels: dict = None
    seeds: tuple = SEEDS
    meta: dict = field(default_factory=dict)


def specs_for(arm, h, draws):
    T = tuple(s for s in SPEAKERS if s != h)
    fd = draws["folds"][h]
    if arm == "A0":
        return [Spec(arm, h, h, T)]
    if arm == "A0b":
        return [Spec(arm, h, h, T, seeds=A0B_SEEDS)]
    if arm == "A1":
        return [Spec(arm, h, h, T, labels=gender_labels())]
    if arm == "A2":
        return [Spec(arm, h, h, tuple(fd["A2_train"]))]
    if arm == "A3":
        seen, out = set(), []
        for d in fd["A3"]:
            if d["key"] not in seen:
                seen.add(d["key"])
                out.append(Spec(arm, h, f"{h}__{d['key']}", tuple(d["train"]), meta={"draw_j": d["j"]}))
        return out
    if arm == "A4":
        return [Spec(arm, h, f"{h}__d{d['j']}", T, labels=d["labels"], meta={k: d[k] for k in ("j", "overlap_gender", "overlap_corpus", "corpus_equal")}) for d in fd["A4"]]
    if arm == "A4c":
        return [Spec(arm, h, h, T, labels=fd["A4c_labels"])]
    raise ValueError(arm)


def score(P, Y):
    z = zm(P, Y)
    c = metrics_from(utt_stats(P, Y))
    return {"rmse_z": z["rmse_z"], "pcc": float(z["pcc"]), "rmse_centered": c["rmse_centered"]}


def evaluate_g(head, fd, spec, ym, ys, device, wrong=False):
    h = spec.fold
    lab = spec.labels
    un = fd["test_unseen"]
    gu = None if lab is None else [(1 - lab[h]) if wrong else lab[h]] * len(un["X"])
    out = {"unseen": score(predict_g(head, un["X"], ym, ys, device, gu), un["Y"])}
    if wrong:
        return out["unseen"]
    te = fd["test"]
    P = predict_g(head, te["X"], ym, ys, device, utt_labels(te, lab))
    out["seen"] = {s: score([P[i] for i in idx], [te["Y"][i] for i in idx])
                   for s in sorted(set(te["spk"])) for idx in [[i for i, x in enumerate(te["spk"]) if x == s]]}
    return out


def is_complete(res, spec):
    """All seeds present and none failed (a failed seed is retried on the next run)."""
    sd = res.get("seeds", {})
    return all(str(s) in sd and "failed" not in sd[str(s)] for s in spec.seeds)


def run_one(spec, ctx, root=ROOT):
    path = Path(root) / spec.arm / f"{spec.key}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if is_complete(res, spec):
        return res
    fd = ctx.fold(spec.fold)
    assert res.get("shift", fd["shift"]) == fd["shift"]
    if res:  # a stored file must come from the same spec; a changed spec or draw needs a new root, not silent reuse
        for k, want in (("train", list(spec.train)), ("labels", spec.labels), ("arm", spec.arm), ("fold", spec.fold)):
            if k in res and res[k] != want:
                raise RuntimeError(f"{path}: stored {k} {res[k]!r} differs from the current spec {want!r}")
    tr, va = subset(fd["train"], spec.train), subset(fd["valid"], spec.train)
    assert len(tr["X"]) > 0 and len(tr["X"]) == len(tr["Y"]) == len(tr["spk"]) and set(tr["spk"]) <= set(spec.train)
    assert spec.fold not in set(tr["spk"]) | set(va["spk"])
    labelled = spec.labels is not None
    data = {"train": (tr["X"], tr["Y"], utt_labels(tr, spec.labels)), "valid": (va["X"], va["Y"], utt_labels(va, spec.labels))}
    Yall = np.concatenate(tr["Y"])
    ym, ys = Yall.mean(0), Yall.std(0)
    dev = ctx.device
    res.update({"arm": spec.arm, "fold": spec.fold, "key": spec.key, "train": list(spec.train), "shift": fd["shift"],
                "labels": spec.labels, "meta": spec.meta, "n_train_utts": len(tr["X"]), "n_valid_utts": len(va["X"])})
    res.setdefault("seeds", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    s0 = spec.seeds[0]
    if "lr" not in res:
        t0 = time.time()
        trials, best_head = {}, None
        for lr in LRS:
            head, v, info = train_one_g(labelled, lr, data, ym, ys, dev, seed=s0)
            trials[lr] = (v, info)
            if head is not None and v <= min(t[0] for t in trials.values()):
                best_head = (lr, head)
        if best_head is None:
            raise RuntimeError(f"{spec.arm} {spec.key}: every learning rate diverged")
        lr = min(trials, key=lambda k: trials[k][0])
        head = best_head[1]
        rec = {"valid_rmse_z": trials[lr][0], **trials[lr][1], **evaluate_g(head, fd, spec, ym, ys, dev)}
        if spec.arm == "A1":
            rec["wrong"] = evaluate_g(head, fd, spec, ym, ys, dev, wrong=True)
        res.update({"lr": lr, "lr_valid": {str(k): t[0] for k, t in trials.items()}})
        res["seeds"][str(s0)] = rec
        res["secs"] = res.get("secs", 0) + time.time() - t0
        path.write_text(json.dumps(res, indent=1))
    for s in spec.seeds[1:]:
        if str(s) in res["seeds"] and "failed" not in res["seeds"][str(s)]:
            continue
        t0 = time.time()
        head, v, info = train_one_g(labelled, res["lr"], data, ym, ys, dev, seed=s)
        if head is None:
            res["seeds"][str(s)] = {"failed": True}
        else:
            rec = {"valid_rmse_z": v, **info, **evaluate_g(head, fd, spec, ym, ys, dev)}
            if spec.arm == "A1":
                rec["wrong"] = evaluate_g(head, fd, spec, ym, ys, dev, wrong=True)
            res["seeds"][str(s)] = rec
        res["secs"] = res.get("secs", 0) + time.time() - t0
        path.write_text(json.dumps(res, indent=1))
    u = [x["unseen"] for x in res["seeds"].values() if "failed" not in x]
    print(f"{spec.arm} {spec.key}: lr {res['lr']} held-out {np.mean([x['rmse_z'] for x in u]):.4f} / "
          f"{np.mean([x['pcc'] for x in u]):.4f} ({len(u)} seeds, {res['secs']:.0f} s)", flush=True)
    return res


def load_draws(root=ROOT):
    d = make_draws()
    p = Path(root) / "draws.json"
    if p.exists():
        stored = json.loads(p.read_text())
        if stored != json.loads(json.dumps(d)):
            raise RuntimeError(f"{p} differs from make_draws(); draws are fixed in advance")
    return json.loads(json.dumps(d))


def run_arm_fold(arm, h, part=0, parts=1, root=ROOT, device=None, smoke=False):
    global EPOCHS, LRS
    if smoke:  # pipeline check only: tiny epochs, one learning rate, one seed; never into the real output directory
        assert Path(root) != ROOT
        EPOCHS, LRS = 2, (1e-3,)
    elif not (ROOT / "draws.json").exists():
        raise SystemExit(f"{ROOT / 'draws.json'} missing: write and commit the draws first (draws --write)")
    draws = load_draws()
    ctx = Ctx(device, limit=12 if smoke else None)
    specs = specs_for(arm, h, draws)[part::parts]
    for sp in specs:
        if smoke:
            sp.seeds = sp.seeds[:1]
        run_one(sp, ctx, root)


def all_specs(draws):
    return [sp for arm in ARMS for h in SPEAKERS for sp in specs_for(arm, h, draws)]


def missing(root=ROOT, ridge=True):
    draws = load_draws()
    out = []
    for sp in all_specs(draws):
        p = Path(root) / sp.arm / f"{sp.key}.json"
        if not (p.exists() and is_complete(json.loads(p.read_text()), sp)):
            out.append((sp.arm, sp.fold, sp.key))
    if ridge:
        for name in ("ridge", "ridge_swap"):
            p = Path(root) / f"{name}.json"
            have = json.loads(p.read_text()) if p.exists() else {}
            out += [(name, h, h) for h in SPEAKERS if h not in have]
    return out


# ---------------------------------------------------------------- ridge analogues (sufficient statistics)

def speaker_stats(X, Y, device):
    """Per-speaker sufficient statistics of a set of (frames, feature / target) arrays, float64."""
    x = torch.from_numpy(np.concatenate(X)).to(device, torch.float64)
    y = torch.from_numpy(np.concatenate(Y)).to(device, torch.float64)
    return {"n": x.shape[0], "sx": x.sum(0), "sxx": x.T @ x, "sy": y.sum(0), "sxy": x.T @ y, "syy": (y ** 2).sum(0)}


def combine(stats, speakers, blocks):
    """Sum per-speaker statistics over `speakers` for the feature map [w_a(s) x for each block a]; blocks is a list of
    {speaker: 0/1}. One block = plain features; two blocks [all, g] = the interaction [x, g x]."""
    d = next(iter(stats.values()))["sx"].shape[0]
    B = len(blocks)
    dev, dt = next(iter(stats.values()))["sx"].device, torch.float64
    out = {"n": 0, "sx": torch.zeros(B * d, device=dev, dtype=dt), "sxx": torch.zeros(B * d, B * d, device=dev, dtype=dt),
           "sy": 0, "sxy": torch.zeros(B * d, 12, device=dev, dtype=dt), "syy": 0}
    for s in speakers:
        st = stats[s]
        out["n"] += st["n"]
        out["sy"] = out["sy"] + st["sy"]
        out["syy"] = out["syy"] + st["syy"]
        for a, wa in enumerate(blocks):
            if not wa[s]:
                continue
            out["sx"][a * d:(a + 1) * d] += st["sx"]
            out["sxy"][a * d:(a + 1) * d] += st["sxy"]
            for b, wb in enumerate(blocks):
                if wb[s]:
                    out["sxx"][a * d:(a + 1) * d, b * d:(b + 1) * d] += st["sxx"]
    return out


class SuffRidge:
    """probe.Ridge (standardized features and targets, ddof=1, one eigendecomposition for all alphas) from sufficient
    statistics; the interaction block columns are standardized like every other column."""

    def __init__(self, st):
        N = st["n"]
        self.xm = st["sx"] / N
        var = (torch.diagonal(st["sxx"]) - N * self.xm ** 2) / (N - 1)
        self.xs = var.clamp_min(0).sqrt().clamp_min(1e-6)
        self.ym = st["sy"] / N
        self.ys = ((st["syy"] - N * self.ym ** 2) / (N - 1)).clamp_min(0).sqrt()
        gram = (st["sxx"] - N * torch.outer(self.xm, self.xm)) / torch.outer(self.xs, self.xs)
        xty = (st["sxy"] - N * torch.outer(self.xm, self.ym)) / self.xs[:, None] / self.ys[None, :]
        ev, self.V = torch.linalg.eigh(gram)
        self.ev = ev.clamp_min(0)
        self.vxy = self.V.T @ xty

    def weights(self, alpha):
        W = (self.V @ (self.vxy / (self.ev + alpha)[:, None])) / self.xs[:, None] * self.ys[None, :]
        return W, self.ym - self.xm @ W


def stats_rmse(st, W, b):
    """Pooled-frames-per-channel rmse_z (mean over channels) of the linear map on the frames summarized by st."""
    N = st["n"]
    sse = (st["syy"] - 2 * (W * st["sxy"]).sum(0) - 2 * b * st["sy"] + ((st["sxx"] @ W) * W).sum(0)
           + 2 * b * (st["sx"] @ W) + N * b ** 2)
    return float(torch.sqrt(sse.clamp_min(0) / N).mean())


def fit_stats(st_tr, st_va):
    """Ridge strength on validation (as lora_readout.fit_ridge); returns (W, b, alpha)."""
    r = SuffRidge(st_tr)
    best = None
    for a in ALPHAS:
        W, b = r.weights(float(a))
        e = stats_rmse(st_va, W, b)
        if best is None or e < best[0]:
            best = (e, W, b, float(a))
    return best[1], best[2], best[3]


def ridge_predict(X, w_blocks, W, b, device):
    d = X[0].shape[1]
    out = []
    for x in X:
        xt = torch.from_numpy(np.asarray(x, dtype=np.float64)).to(device)
        y = b.clone().expand(len(xt), -1).clone()
        for a, wa in enumerate(w_blocks):
            if wa:
                y += xt @ W[a * d:(a + 1) * d]
        out.append(y.cpu().numpy())
    return out


def ridge_fold(fd, h, gender, device):
    """All ridge variants for one fold under a gender assignment. Returns {variant: ...}."""
    T = [s for s in SPEAKERS if s != h]
    st_tr, st_va = {}, {}
    for s in T:
        for split, store in (("train", st_tr), ("valid", st_va)):
            sub = subset(fd[split], [s])
            store[s] = speaker_stats(sub["X"], sub["Y"], device)
    un = fd["test_unseen"]
    plain = [{s: 1 for s in SPEAKERS}]

    def fit_eval(S, blocks, w_h):
        W, b, alpha = fit_stats(combine(st_tr, S, blocks), combine(st_va, S, blocks))
        # ridge targets are the pooled-training z targets: predictions are in the same z units as the heads' outputs
        P = ridge_predict(un["X"], w_h, W, b, device)
        return {"alpha": alpha, **score(P, un["Y"])}

    def inter(lab):
        return [{s: 1 for s in SPEAKERS}, lab]

    same = {s: int(gender[s] == gender[h]) for s in SPEAKERS}  # 1 = h's gender group
    out = {"R0": fit_eval(T, plain, [1]), "R2": fit_eval(a2_set(h, gender), plain, [1]),
           "R1": fit_eval(T, inter(same), [1, same[h]])}
    r3 = {"+".join(c): fit_eval(list(c), plain, [1]) for c in a3_candidates(h, gender)}
    r4 = {"+".join(S): fit_eval(T, inter(partition_labels(h, S)), [1, 1]) for S in a4_candidates(h, gender)}
    avg = lambda d: {m: float(np.mean([v[m] for v in d.values()])) for m in ("rmse_z", "pcc", "rmse_centered")}
    out["R3"] = {"mean": avg(r3), "cands": r3}
    out["R4"] = {"mean": avg(r4), "cands": r4}
    return out


def ridge_arms(swap=False, root=ROOT, device=None):
    ctx = Ctx(device)
    gender = GENDER_SWAP if swap else GENDER
    path = Path(root) / ("ridge_swap.json" if swap else "ridge.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    for h in SPEAKERS:
        if h in res:
            continue
        res[h] = ridge_fold(ctx.fold(h), h, gender, ctx.device)
        res[h]["shift"] = ctx.fold(h)["shift"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1))
        print(f"ridge{' swap' if swap else ''} {h}: R0 {res[h]['R0']['rmse_z']:.4f} / {res[h]['R0']['pcc']:.4f}", flush=True)


# ---------------------------------------------------------------- decision rule and analysis

METRICS = ("rmse_z", "pcc", "rmse_centered")
MALE = [s for s in SPEAKERS if GENDER[s] == "M"]
FEMALE = [s for s in SPEAKERS if GENDER[s] == "F"]
FLOOR = {"rmse_z": 0.005, "pcc": 0.003}


def majority(n):
    return n // 2 + 1


def gain(x, ref):
    """Positive = x better than ref. x, ref: {'rmse_z','pcc','rmse_centered'} dicts."""
    return {"rmse_z": ref["rmse_z"] - x["rmse_z"], "pcc": x["pcc"] - ref["pcc"], "rmse_centered": ref["rmse_centered"] - x["rmse_centered"]}


def judge(gains, tau, G):
    """gains: {fold: gain dict} for the folds of one gender. Returns wins/losses (both-metric beyond tau), means,
    equivalence status."""
    f = list(gains)
    wins = sum(gains[h]["rmse_z"] > tau["rmse_z"] and gains[h]["pcc"] > tau["pcc"] for h in f)
    losses = sum(gains[h]["rmse_z"] < -tau["rmse_z"] and gains[h]["pcc"] < -tau["pcc"] for h in f)
    mean = {m: float(np.mean([gains[h][m] for h in f])) for m in METRICS}
    within = all(abs(mean[m]) <= tau[m] for m in ("rmse_z", "pcc"))
    no_big = all(abs(gains[h][m]) <= 2 * tau[m] for h in f for m in ("rmse_z", "pcc"))
    return {"wins": int(wins), "losses": int(losses), "n": len(f), "majority": majority(len(f)), "mean": mean,
            "equivalent": bool(within and no_big)}


def verdict(j_ctrl, j_ref, rank_above, rank_below, tau):
    """Pre-registered per-gender verdict from the two contrasts (vs unconditioned/pooled reference and vs control),
    the replicate ranks (gain above / below every replicate on both metrics) and the centered gate."""
    helps = (j_ref["wins"] >= j_ref["majority"] and j_ref["mean"]["rmse_z"] > 0 and j_ref["mean"]["pcc"] > 0
             and j_ctrl["wins"] >= j_ctrl["majority"] and j_ctrl["mean"]["rmse_z"] > 0 and j_ctrl["mean"]["pcc"] > 0
             and rank_above and j_ref["mean"]["rmse_centered"] > 0 and j_ctrl["mean"]["rmse_centered"] > 0)
    hurts = (j_ref["losses"] >= j_ref["majority"] and j_ref["mean"]["rmse_z"] < 0 and j_ref["mean"]["pcc"] < 0
             and j_ctrl["losses"] >= j_ctrl["majority"] and j_ctrl["mean"]["rmse_z"] < 0 and j_ctrl["mean"]["pcc"] < 0
             and rank_below and j_ref["mean"]["rmse_centered"] < 0 and j_ctrl["mean"]["rmse_centered"] < 0)
    if helps:
        return "helps"
    if hurts:
        return "hurts"
    if j_ref["equivalent"] and j_ctrl["equivalent"]:
        return "no detectable effect"
    return "inconclusive"


def sign_flip_p(d):
    """Exact two-sided sign-flip p of the mean of d (all 2^n sign patterns)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    signs = np.array(list(itertools.product((-1, 1), repeat=n)))
    obs = abs(d.sum())
    return float((np.abs(signs @ d) >= obs - 1e-12).mean())


def _seed_vals(rec, part="unseen", seeds=None):
    return [rec["seeds"][str(s)][part] for s in (seeds or SEEDS) if "failed" not in rec["seeds"].get(str(s), {"failed": 1})]


def mean_metrics(vals):
    return {m: float(np.mean([v[m] for v in vals])) for m in METRICS}


def load_arm(arm, h, draws, root=ROOT):
    """Run records of one (arm, fold), keyed by run key."""
    out = {}
    for sp in specs_for(arm, h, draws):
        p = Path(root) / arm / f"{sp.key}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        r = json.loads(p.read_text())
        if not is_complete(r, sp):
            raise RuntimeError(f"incomplete: {p}")
        out[sp.key] = (sp, r)
    return out


def fold_values(root=ROOT):
    """{arm: {fold: value}}: A0/A0b/A1/A2/A4c = seed-mean metrics; A3/A4 = {'ctrl': mean over draws, 'reps': per-draw list}."""
    draws = load_draws(root)
    V = {a: {} for a in ARMS}
    for h in SPEAKERS:
        for arm in ("A0", "A0b", "A1", "A2", "A4c"):
            (sp, r), = load_arm(arm, h, draws, root).values()
            V[arm][h] = {"seed": [x for x in _seed_vals(r, seeds=sp.seeds)], "mean": mean_metrics(_seed_vals(r, seeds=sp.seeds)), "rec": r}
        runs = load_arm("A3", h, draws, root)
        a3 = draws["folds"][h]["A3"]
        reps = [mean_metrics(_seed_vals(runs[f"{h}__{d['key']}"][1])) for d in a3]
        first = {}  # control value: mean over distinct subsets, equal weight (repeated slots do not re-weight it)
        for d, r in zip(a3, reps):
            first.setdefault(d["key"], r)
        uniq = list(first.values())
        V["A3"][h] = {"reps": reps, "n_unique": len(uniq), "ctrl": {m: float(np.mean([r[m] for r in uniq])) for m in METRICS},
                      "runs": runs}
        runs = load_arm("A4", h, draws, root)
        reps = [mean_metrics(_seed_vals(runs[f"{h}__d{d['j']}"][1])) for d in draws["folds"][h]["A4"]]
        V["A4"][h] = {"reps": reps, "ctrl": {m: float(np.mean([r[m] for r in reps])) for m in METRICS}, "runs": runs}
    return V, draws


def tau_from(V):
    sd = {m: [np.std([s[m] for s in V["A0"][h]["seed"]], ddof=1) for h in SPEAKERS] for m in ("rmse_z", "pcc")}
    return {m: max(float(np.sqrt(np.mean(np.square(sd[m])))), FLOOR[m]) for m in sd}, sd


def check_pipeline(V, tol_r=0.010, tol_p=0.005):
    stored = json.loads(STORED_HEADS.read_text())
    s_r = float(np.mean([head_screen.seed_mean(stored[h], "test_unseen", "rmse_z") for h in SPEAKERS]))
    s_p = float(np.mean([head_screen.seed_mean(stored[h], "test_unseen", "pcc") for h in SPEAKERS]))
    n_r = float(np.mean([V["A0"][h]["mean"]["rmse_z"] for h in SPEAKERS]))
    n_p = float(np.mean([V["A0"][h]["mean"]["pcc"] for h in SPEAKERS]))
    shifts = all(V["A0"][h]["rec"]["shift"] == stored[h]["shift"] for h in SPEAKERS)
    lr_agree = {h: (V["A0"][h]["rec"]["lr"], stored[h]["lr"]) for h in SPEAKERS}
    ok = abs(n_r - s_r) <= tol_r and abs(n_p - s_p) <= tol_p and shifts
    return {"ok": bool(ok), "stored": [s_r, s_p], "new": [n_r, n_p], "shifts_identical": shifts, "lr": lr_agree}


def gender_folds(G):
    return MALE if G == "M" else FEMALE


def analyze_primary(V, tau, G):
    F = gender_folds(G)
    g_ref = {h: gain(V["A1"][h]["mean"], V["A0"][h]["mean"]) for h in F}
    g_ctl = {h: gain(V["A1"][h]["mean"], V["A4"][h]["ctrl"]) for h in F}
    j_ref, j_ctl = judge(g_ref, tau, G), judge(g_ctl, tau, G)
    d = len(V["A4"][F[0]]["reps"])
    rep_gain = [{m: float(np.mean([gain(V["A4"][h]["reps"][j], V["A0"][h]["mean"])[m] for h in F])) for m in ("rmse_z", "pcc")} for j in range(d)]
    a1 = {m: j_ref["mean"][m] for m in ("rmse_z", "pcc")}
    above = all(a1[m] > r[m] for r in rep_gain for m in a1)
    below = all(a1[m] < r[m] for r in rep_gain for m in a1)
    c4c = {h: gain(V["A1"][h]["mean"], V["A4c"][h]["mean"]) for h in F}
    j_c = judge(c4c, tau, G)
    v = verdict(j_ctl, j_ref, above, below, tau)
    dist_corpus = j_c["wins"] >= j_c["majority"]
    return {"verdict": v, "vs_A0": j_ref, "vs_A4": j_ctl, "rank_above_all": above, "rank_below_all": below,
            "replicate_gains": rep_gain, "vs_A4c": j_c, "beats_corpus_label": bool(dist_corpus),
            "fold_gain_vs_A0": g_ref, "fold_gain_vs_A4": g_ctl}


def analyze_secondary(V, tau, G):
    F = gender_folds(G)
    g_ctl = {h: gain(V["A2"][h]["mean"], V["A3"][h]["ctrl"]) for h in F}
    g_ref = {h: gain(V["A2"][h]["mean"], V["A0"][h]["mean"]) for h in F}
    j_ctl, j_ref = judge(g_ctl, tau, G), judge(g_ref, tau, G)
    d = len(V["A3"][F[0]]["reps"])
    rep_gain = [{m: float(np.mean([gain(V["A3"][h]["reps"][j], V["A0"][h]["mean"])[m] for h in F])) for m in ("rmse_z", "pcc")} for j in range(d)]
    a2 = {m: j_ref["mean"][m] for m in ("rmse_z", "pcc")}
    above = all(a2[m] > r[m] for r in rep_gain for m in a2)
    below = all(a2[m] < r[m] for r in rep_gain for m in a2)
    # secondary rule: the control contrast is the criterion; A2 vs A0 is reported (speaker count confounds it)
    v = verdict(j_ctl, j_ctl, above, below, tau)
    return {"verdict": v, "vs_A3": j_ctl, "vs_A0": j_ref, "rank_above_all": above, "rank_below_all": below,
            "replicate_gains": rep_gain, "fold_gain_vs_A3": g_ctl}


def calibration(V, tau):
    """Apply the primary rule to each pseudo-gender replicate j as a fake treatment (vs A0, vs the mean of the other
    replicates, ranked against the other replicates). Returns pass counts per gender."""
    out = {}
    d = len(V["A4"][SPEAKERS[0]]["reps"])
    for G in ("M", "F"):
        F = gender_folds(G)
        n12 = full = 0
        for j in range(d):
            others = [k for k in range(d) if k != j]
            g_ref = {h: gain(V["A4"][h]["reps"][j], V["A0"][h]["mean"]) for h in F}
            g_ctl = {h: gain(V["A4"][h]["reps"][j], {m: float(np.mean([V["A4"][h]["reps"][k][m] for k in others])) for m in METRICS}) for h in F}
            jr, jc = judge(g_ref, tau, G), judge(g_ctl, tau, G)
            first = (jr["wins"] >= jr["majority"] and jr["mean"]["rmse_z"] > 0 and jr["mean"]["pcc"] > 0
                     and jc["wins"] >= jc["majority"] and jc["mean"]["rmse_z"] > 0 and jc["mean"]["pcc"] > 0)
            reps = [{m: float(np.mean([gain(V["A4"][h]["reps"][k], V["A0"][h]["mean"])[m] for h in F])) for m in ("rmse_z", "pcc")} for k in others]
            above = all(jr["mean"][m] > r[m] for r in reps for m in ("rmse_z", "pcc"))
            n12 += first
            full += bool(verdict(jc, jr, above, False, tau) == "helps")
        out[G] = {"replicates": d, "pass_contrasts_1_2": int(n12), "pass_full_rule": int(full)}
    return out


def noise_floor(V, tau):
    """A0 vs the independent A0b rerun, per fold, and the paired seed-difference sd of A1 - A0."""
    rows = {}
    for h in SPEAKERS:
        g = gain(V["A0b"][h]["mean"], V["A0"][h]["mean"])
        rows[h] = {**g, "exceeds_tau_both": bool(abs(g["rmse_z"]) > tau["rmse_z"] and abs(g["pcc"]) > tau["pcc"])}
    paired = {m: float(np.sqrt(np.mean([np.var([a[m] - b[m] for a, b in zip(V["A1"][h]["seed"], V["A0"][h]["seed"])], ddof=1) for h in SPEAKERS])))
              for m in ("rmse_z", "pcc")}
    return {"folds": rows, "paired_seed_diff_sd_A1_A0": paired}


def seen_summary(V, draws, root=ROOT):
    """Seen speakers (text-disjoint test sentences of training speakers), descriptive: mean over folds and training
    speakers of each speaker gender; A2/A3 only over the speakers in their own training set, with A0 on the same
    (fold, speaker) pairs."""
    def collect(arm_runs, only_train):
        acc = {"M": [], "F": []}
        for h, recs in arm_runs:
            for sp, r in recs:
                for s, m in mean_seen(r, sp.seeds).items():
                    if only_train and s not in sp.train:
                        continue
                    acc[GENDER[s]].append((h, s, m))
        return acc

    def mean_seen(r, seeds):
        per = {}
        for s in {k for x in r["seeds"].values() for k in x.get("seen", {})}:
            per[s] = mean_metrics([r["seeds"][str(sd)]["seen"][s] for sd in seeds if str(sd) in r["seeds"] and "failed" not in r["seeds"][str(sd)]])
        return per

    def agg(acc, keys=None):
        out = {}
        for G, v in acc.items():
            v = [x for x in v if keys is None or (x[0], x[1]) in keys]
            out[G] = {m: float(np.mean([x[2][m] for x in v])) for m in ("rmse_z", "pcc", "rmse_centered")} | {"n": len(v)} if v else None
        return out

    res = {}
    for arm in ("A0", "A0b", "A1", "A4c", "A4"):
        runs = [(h, [v for v in (load_arm(arm, h, draws, root).values())]) for h in SPEAKERS]
        res[arm] = agg(collect(runs, False))
    for arm in ("A2", "A3"):
        runs = [(h, [v for v in (load_arm(arm, h, draws, root).values())]) for h in SPEAKERS]
        acc = collect(runs, True)
        keys = {(h, s) for G in acc for (h, s, _) in acc[G]}
        a0 = collect([(h, [v for v in load_arm("A0", h, draws, root).values()]) for h in SPEAKERS], False)
        res[arm] = {"arm": agg(acc), "A0_same_speakers": agg(a0, keys)}
    return res


def ridge_summary(root=ROOT):
    out = {}
    for name in ("ridge", "ridge_swap"):
        p = Path(root) / f"{name}.json"
        if not p.exists():
            continue
        R = json.loads(p.read_text())
        gender = GENDER_SWAP if name == "ridge_swap" else GENDER
        res = {}
        for G in ("M", "F"):
            F = [s for s in SPEAKERS if gender[s] == G]
            if not all(h in R for h in F):
                continue
            g = lambda a, b: {m: float(np.mean([gain(a(R[h]), b(R[h]))[m] for h in F])) for m in METRICS}
            res[G] = {"folds": F, "R1_vs_R0": g(lambda r: r["R1"], lambda r: r["R0"]),
                      "R1_vs_R4": g(lambda r: r["R1"], lambda r: r["R4"]["mean"]),
                      "R2_vs_R0": g(lambda r: r["R2"], lambda r: r["R0"]),
                      "R2_vs_R3": g(lambda r: r["R2"], lambda r: r["R3"]["mean"])}
        out[name] = res
    if "ridge" in out and "ridge_swap" in out:
        flag = {}
        for G in ("M", "F"):
            def ind(x):
                r = x.get(G)
                return None if r is None else bool(r["R1_vs_R4"]["rmse_z"] > 0 and r["R1_vs_R4"]["pcc"] > 0)
            flag[G] = ind(out["ridge"]) != ind(out["ridge_swap"])
        out["swap_flips_indicator"] = flag
    return out


def r0_check(root=ROOT):
    p = Path(root) / "ridge.json"
    if not p.exists():
        return None
    from .depth_sweep import load_all

    R, d = json.loads(p.read_text()), load_all(MODEL)
    if not all(h in R for h in SPEAKERS):
        return None
    return {"ridge_r0": [float(np.mean([R[h]["R0"]["rmse_z"] for h in SPEAKERS])), float(np.mean([R[h]["R0"]["pcc"] for h in SPEAKERS]))],
            "depth_sweep": [float(np.mean([d[f"{LAYER}/{h}"]["unseen"]["rmse_z"] for h in SPEAKERS])),
                            float(np.mean([d[f"{LAYER}/{h}"]["unseen"]["pcc"] for h in SPEAKERS]))]}


def corpus_excluded(V, draws, tau):
    """Primary rule's A1-vs-control contrast with the corpus-equal pseudo-partitions removed (descriptive)."""
    out = {}
    for G in ("M", "F"):
        F = gender_folds(G)
        g = {}
        for h in F:
            keep = [i for i, d in enumerate(draws["folds"][h]["A4"]) if not d["corpus_equal"]]
            if not keep:
                g = None
                break
            ctl = {m: float(np.mean([V["A4"][h]["reps"][i][m] for i in keep])) for m in METRICS}
            g[h] = gain(V["A1"][h]["mean"], ctl)
        out[G] = judge(g, tau, G) if g is not None else {"verdict": "not available: every pseudo-partition of a fold equals the corpus partition"}
    return out


def analyze(root=ROOT):
    V, draws = fold_values(root)
    tau, sd = tau_from(V)
    chk = check_pipeline(V)
    res = {"tau": tau, "a0_seed_sd": {m: dict(zip(SPEAKERS, map(float, v))) for m, v in sd.items()}, "pipeline_check": chk}
    if chk["ok"]:
        res["primary"] = {G: analyze_primary(V, tau, G) for G in ("M", "F")}
        res["secondary"] = {G: analyze_secondary(V, tau, G) for G in ("M", "F")}
        res["calibration"] = calibration(V, tau)
    res["noise_floor"] = noise_floor(V, tau)
    res["corpus_excluded_A1_vs_A4"] = corpus_excluded(V, draws, tau)
    res["seen"] = seen_summary(V, draws, root)
    res["ridge"] = ridge_summary(root)
    res["ridge_r0_check"] = r0_check(root)
    res["sign_flip_p"] = {"A1_vs_A0_rmse_z": sign_flip_p([gain(V["A1"][h]["mean"], V["A0"][h]["mean"])["rmse_z"] for h in SPEAKERS]),
                          "A1_vs_A4_rmse_z": sign_flip_p([gain(V["A1"][h]["mean"], V["A4"][h]["ctrl"])["rmse_z"] for h in SPEAKERS])}
    res["a1_wrong_label"] = {h: {"right": V["A1"][h]["mean"], "wrong": mean_metrics_wrong(V["A1"][h]["rec"])} for h in SPEAKERS}
    res["fold_values"] = {arm: {h: (V[arm][h]["mean"] if "mean" in V[arm][h] else V[arm][h]["ctrl"]) for h in SPEAKERS} for arm in ARMS}
    res["a4_overlap"] = {h: [{"j": d["j"], "overlap_gender": d["overlap_gender"], "overlap_corpus": d["overlap_corpus"],
                              "corpus_equal": d["corpus_equal"], "gain_vs_A0": gain(V["A4"][h]["reps"][d["j"]], V["A0"][h]["mean"])}
                             for d in draws["folds"][h]["A4"]] for h in SPEAKERS}
    root = Path(root)
    (root / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (root / "summary.md").write_text(render(res, V, draws))
    print((root / "summary.md").read_text())
    return res


def mean_metrics_wrong(rec):
    ok = [rec["seeds"][str(s)]["wrong"] for s in SEEDS if "failed" not in rec["seeds"].get(str(s), {"failed": 1})]
    return mean_metrics(ok)


def _f(x, k=4):
    return f"{x:+.{k}f}"


def render(res, V, draws):
    L = ["# Gender conditioning of the multi-speaker inverter (LOSO; WavLM Large layer 12, BiGRU->conv head)\n",
         "Pre-registration: `docs/gender_conditioning_prereg.md`. Fold value = mean over 3 seeds (A3/A4: also over draws); "
         "gain = positive when better (rmse_z: reference minus arm; PCC: arm minus reference). Descriptive consistency rule, "
         "not inference (3 male / 4 female folds).\n"]
    c = res["pipeline_check"]
    L.append(f"## Pipeline check (A0 against the stored head-screen result)\n\nA0 7-fold mean rmse_z / PCC {c['new'][0]:.4f} / {c['new'][1]:.4f}; "
             f"stored {c['stored'][0]:.4f} / {c['stored'][1]:.4f}; shifts identical: {c['shifts_identical']}; "
             f"**{'PASSED' if c['ok'] else 'FAILED: verdicts withheld, debug the pipeline before reading any arm'}**. "
             f"Learning rate (new, stored) per fold: {c['lr']}\n")
    L.append(f"tau (noise scale from the A0 seed sd, not an effect-size threshold): rmse_z {res['tau']['rmse_z']:.4f}, PCC {res['tau']['pcc']:.4f}\n")
    L.append("A3 control value = mean over distinct subsets (equal weight); distinct subsets per fold: "
             + ", ".join(f"{h} {V['A3'][h]['n_unique']} of {len(V['A3'][h]['reps'])} slots" for h in SPEAKERS) + "\n")
    L.append("## Fold values (held-out speaker, seed mean; A3/A4: mean over draws)\n")
    L.append("| fold | gender | " + " | ".join(f"{a} rmse_z / PCC / centered" for a in ARMS) + " |")
    L.append("|---|---|" + "---|" * len(ARMS))
    for h in SPEAKERS:
        L.append(f"| {h} | {GENDER[h]} | " + " | ".join(
            f"{res['fold_values'][a][h]['rmse_z']:.4f} / {res['fold_values'][a][h]['pcc']:.4f} / {res['fold_values'][a][h]['rmse_centered']:.4f}" for a in ARMS) + " |")
    if "primary" in res:
        L.append("\n## Primary: does telling the pooled model the gender help?\n")
        for G, name in (("M", "male held-out speakers (3 folds)"), ("F", "female held-out speakers (4 folds)")):
            p = res["primary"][G]
            L.append(f"**{name}: {p['verdict']}**" + ("" if p["verdict"] != "helps" or p["beats_corpus_label"] else
                                                  " (not distinguished from a corpus label: A1 does not beat A4c in a majority of folds)"))
            for nm, j in (("vs pooled model (A1 - A0)", p["vs_A0"]), ("vs random-grouping control (A1 - A4)", p["vs_A4"]), ("vs corpus label (A1 - A4c, descriptive)", p["vs_A4c"])):
                L.append(f"- {nm}: folds won {j['wins']}/{j['n']} (lost {j['losses']}); mean gain rmse_z {_f(j['mean']['rmse_z'])}, PCC {_f(j['mean']['pcc'])}, centered rmse_z {_f(j['mean']['rmse_centered'])}; equivalent within tau: {j['equivalent']}")
            L.append(f"- rank of A1's gain over A0 against the {len(p['replicate_gains'])} pseudo-gender replicates: above all {p['rank_above_all']}, below all {p['rank_below_all']}")
            L.append("- per fold gain vs A0 (rmse_z / PCC): " + "; ".join(f"{h} {_f(g['rmse_z'])} / {_f(g['pcc'])}" for h, g in p["fold_gain_vs_A0"].items()) + "\n")
        v = {G: res["primary"][G]["verdict"] for G in ("M", "F")}
        helped = [G for G in v if v[G] == "helps"]
        overall = ("conditioning helps both genders" if len(helped) == 2 else
                   f"conditioning helps one gender only ({'male' if helped[0] == 'M' else 'female'}); 3-4 folds support no per-gender test" if helped else
                   f"conditioning is not shown to help; male: {v['M']}, female: {v['F']} ('inconclusive' is not 'no effect')")
        L.append(f"**Overall: {overall}.**\n")
        L.append("## Secondary: separate per-gender models\n")
        for G, name in (("M", "male folds"), ("F", "female folds")):
            s = res["secondary"][G]
            L.append(f"**{name}: {s['verdict']}**")
            for nm, j in (("vs count- and corpus-matched mixed-gender pooled models (A2 - A3)", s["vs_A3"]), ("vs the pooled model with all six speakers (A2 - A0, speaker count confounds)", s["vs_A0"])):
                L.append(f"- {nm}: folds won {j['wins']}/{j['n']} (lost {j['losses']}); mean gain rmse_z {_f(j['mean']['rmse_z'])}, PCC {_f(j['mean']['pcc'])}, centered {_f(j['mean']['rmse_centered'])}")
            L.append(f"- A2's gain over A0 against {len(s['replicate_gains'])} matched-control replicate gains: above all {s['rank_above_all']}, below all {s['rank_below_all']}\n")
        L.append("## Operating characteristics: the primary rule applied to each random-grouping replicate as a fake treatment\n")
        for G, r in res["calibration"].items():
            L.append(f"- {'male' if G == 'M' else 'female'}: {r['pass_contrasts_1_2']}/{r['replicates']} replicates pass contrasts 1 and 2; {r['pass_full_rule']}/{r['replicates']} pass the full rule")
    nf = res["noise_floor"]
    L.append("\n## Noise floor: A0 against an independent A0 rerun (seeds 10-12)\n")
    L.append("| fold | rmse_z diff | PCC diff | beyond tau on both |")
    L.append("|---|---|---|---|")
    for h, r in nf["folds"].items():
        L.append(f"| {h} | {_f(r['rmse_z'])} | {_f(r['pcc'])} | {r['exceeds_tau_both']} |")
    pd_ = nf["paired_seed_diff_sd_A1_A0"]
    L.append(f"\nPaired seed-difference sd of A1 - A0 (same seed index): rmse_z {pd_['rmse_z']:.4f}, PCC {pd_['pcc']:.4f}.\n")
    L.append("## Descriptive only\n")
    L.append(f"- Sign-flip p over all 7 folds (rmse_z gain): A1 vs A0 {res['sign_flip_p']['A1_vs_A0_rmse_z']:.3f}, A1 vs A4 {res['sign_flip_p']['A1_vs_A4_rmse_z']:.3f} (minimum 2/128 = 0.016).")
    L.append("- A1 scored with the opposite gender label (held-out rmse_z / PCC, right vs wrong): " + "; ".join(
        f"{h} {w['right']['rmse_z']:.4f}/{w['right']['pcc']:.4f} vs {w['wrong']['rmse_z']:.4f}/{w['wrong']['pcc']:.4f}" for h, w in res["a1_wrong_label"].items()))
    L.append("- A1 vs A4 with corpus-equal partitions excluded: " + "; ".join(
        f"{G}: won {j['wins']}/{j['n']}, mean rmse_z {_f(j['mean']['rmse_z'])}, PCC {_f(j['mean']['pcc'])}" for G, j in res["corpus_excluded_A1_vs_A4"].items()))
    L.append("\nA4 draws (gain over A0 of each pseudo-gender replicate against its overlap with the real gender partition / corpus partition):\n")
    L.append("| fold | j | overlap gender | overlap corpus | corpus-equal | gain rmse_z | gain PCC |")
    L.append("|---|---|---|---|---|---|---|")
    for h, rows in res["a4_overlap"].items():
        for r in rows:
            L.append(f"| {h} | {r['j']} | {r['overlap_gender']} | {r['overlap_corpus']} | {r['corpus_equal']} | {_f(r['gain_vs_A0']['rmse_z'])} | {_f(r['gain_vs_A0']['pcc'])} |")
    L.append("\nSeen speakers (text-disjoint test sentences of training speakers; mean over folds and speakers, by the speaker's gender):\n")
    for arm, v in res["seen"].items():
        if "arm" in v:
            for nm, x in (("", v["arm"]), (" (A0 on the same speakers)", v["A0_same_speakers"])):
                L.append(f"- {arm}{nm}: " + "; ".join(f"{G} {m['rmse_z']:.4f}/{m['pcc']:.4f}" for G, m in x.items() if m))
        else:
            L.append(f"- {arm}: " + "; ".join(f"{G} {m['rmse_z']:.4f}/{m['pcc']:.4f}" for G, m in v.items() if m))
    if res["ridge"]:
        L.append("\nRidge analogues (mean gain over the gender's folds; a gender intercept is meaningless under per-speaker z targets):\n")
        L.append("| assignment | gender | R1-R0 rmse_z/PCC | R1-R4 | R2-R0 | R2-R3 |")
        L.append("|---|---|---|---|---|---|")
        for nm in ("ridge", "ridge_swap"):
            for G, r in res["ridge"].get(nm, {}).items():
                L.append(f"| {'real' if nm == 'ridge' else 'kf/jr swapped'} | {G} | " + " | ".join(
                    f"{_f(r[k]['rmse_z'])} / {_f(r[k]['pcc'])}" for k in ("R1_vs_R0", "R1_vs_R4", "R2_vs_R0", "R2_vs_R3")) + " |")
        if "swap_flips_indicator" in res["ridge"]:
            L.append(f"\nThe kf/jr swap flips the ridge 'conditioning beats random groupings' indicator for: {res['ridge']['swap_flips_indicator']} "
                     "(a flip marks the primary verdict fragile).")
    if res.get("ridge_r0_check"):
        k = res["ridge_r0_check"]
        L.append(f"\nR0 against the depth-sweep probe (k12): {k['ridge_r0'][0]:.4f} / {k['ridge_r0'][1]:.4f} vs {k['depth_sweep'][0]:.4f} / {k['depth_sweep'][1]:.4f}.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("draws", "run", "ridge", "status", "analyze"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--fold", choices=SPEAKERS)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if args.cmd == "draws":
        d = make_draws()
        txt = json.dumps(d, indent=1)
        if args.write:
            for p in (root / "draws.json", DRAWS_REPO):
                p.parent.mkdir(parents=True, exist_ok=True)
                if p.exists() and json.loads(p.read_text()) != json.loads(txt):
                    raise SystemExit(f"{p} exists and differs; draws are fixed in advance")
                p.write_text(txt + "\n")
        for h, f in d["folds"].items():
            print(h, "A3 candidates", f["A3_n_candidates"], "unique", f["A3_n_unique"], "A4 candidates", f["A4_n_candidates"],
                  "corpus-equal draws", sum(x["corpus_equal"] for x in f["A4"]))
    elif args.cmd == "run":
        run_arm_fold(args.arm, args.fold, args.part, args.parts, root, smoke=args.smoke)
    elif args.cmd == "ridge":
        ridge_arms(args.swap, root)
    elif args.cmd == "status":
        m = missing(root)
        print("MISSING", len(m))
        for x in m:
            print(*x)
        raise SystemExit(1 if m else 0)
    else:
        analyze(root)


if __name__ == "__main__":
    main()
