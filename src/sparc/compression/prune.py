"""Non-contiguous layer selection: does a chosen subset of layers beat the prefix of the same size?

Starting from the prefix 1..K, two selection methods produce one subset per size n < K:
    greedy -- backward elimination: repeatedly drop the layer whose removal gives the lowest
              validation RMSE of a ridge probe on the subset's output (probe fit on a train
              subsample for speed)
    bi     -- Block Influence (ShortGPT): drop the n_drop layers with the smallest
              1 - cos(input, output), measured once on the full prefix
Removing an intermediate layer changes everything downstream, so every candidate is scored with a
fresh forward pass, not with cached features.

Every selected subset, and the prefix of the same size, is then evaluated identically: ridge fit on
the full train split, alignment shift and alpha chosen on validation, test metrics reported.

Usage: python -m sparc.compression.prune <model> --start K --min N [--train-subsample 400]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import datasets, mngu0
from .encoders import load_full, normalize_wav
from .extract import OUT_ROOT, lowpass
from .metrics import ema_metrics
from .probe import ALPHAS, Ridge, rmse


class SubsetRunner:
    def __init__(self, name, device):
        self.model = load_full(name).to(device)
        self.device = device
        self.all_layers = self.model.encoder.layers
        self.final_norm = self.model.encoder.layer_norm
        self.L = len(self.all_layers)
        self.is_wavlm = self.model.config.model_type == "wavlm"

    def set(self, layers):
        self.model.encoder.layers = nn.ModuleList([self.all_layers[i - 1] for i in layers])
        self.model.encoder.layer_norm = self.final_norm if layers[-1] == self.L else nn.Identity()

    @torch.no_grad()
    def features(self, utts, layers):
        self.set(layers)
        out = []
        for u in utts:
            x = torch.from_numpy(normalize_wav(u.wav)).unsqueeze(0).to(self.device)
            h = self.model(x).last_hidden_state[0]  # fp32, like the cached features in extract.py
            out.append(lowpass(h.float().cpu().numpy()).astype(np.float32))
        return out

    @torch.no_grad()
    def block_influence(self, utts, layers):
        self.set(layers)
        bi = np.zeros(len(layers))
        for u in utts:
            x = torch.from_numpy(normalize_wav(u.wav)).unsqueeze(0).to(self.device)
            hs = self.model(x, output_hidden_states=True).hidden_states
            for j in range(len(layers)):
                bi[j] += (1 - torch.nn.functional.cosine_similarity(hs[j][0], hs[j + 1][0], dim=-1)).mean().item()
        return dict(zip(layers, bi / len(utts)))


def _aligned(feats, utts, shift):
    X, Y, keep = [], [], []
    for f, u in zip(feats, utts):
        x, y = mngu0.align(f, u.ema_mm, u.base_offset + shift)
        if len(x) >= 2:
            X.append(x)
            Y.append(y)
            keep.append(u)
    return X, Y, keep


def fit_select(tr_feats, tr_utts, va_feats, va_utts, shifts, device):
    best = None
    for shift in shifts:
        Xtr, Ytr, _ = _aligned(tr_feats, tr_utts, shift)
        Xva, Yva, _ = _aligned(va_feats, va_utts, shift)
        ridge = Ridge(Xtr, Ytr, device)
        for a in ALPHAS:
            W, b = ridge.weights(a)
            r = rmse(ridge.predict(W, b, Xva), Yva)
            if best is None or r < best[0]:
                best = (r, shift, float(a), W, b, ridge)
    return best


def full_eval(runner, layers, data, device, ds=None):
    ds = ds or datasets.get("mngu0")
    evals = [s for s in ("test", "test_unseen") if data.get(s)]
    feats = {s: runner.features(data[s], layers) for s in ["train", "valid"] + evals}
    r, shift, alpha, W, b, ridge = fit_select(feats["train"], data["train"], feats["valid"], data["valid"],
                                              ds.shift_candidates, device)
    out = {"layers": layers, "n_layers": len(layers), "shift": shift, "alpha": alpha, "valid_rmse": r}
    for split in evals:
        X, Y, us = _aligned(feats[split], data[split], shift)
        phones = [ds.frame_phones(u, u.base_offset + shift, len(y)) for u, y in zip(us, Y)]
        out[split] = ds.evaluate(us, ridge.predict(W, b, X), Y, phones=phones if any(phones) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dataset", default="mngu0", choices=datasets.DATASET_NAMES)
    ap.add_argument("--start", type=int, required=True, help="K: start from the prefix 1..K")
    ap.add_argument("--min", type=int, required=True, help="smallest subset size to reach")
    ap.add_argument("--train-subsample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    froot = datasets.features_root(args.dataset)
    out = Path(args.out or froot / args.model / f"prune_from{args.start}_to{args.min}.json")
    state = json.loads(out.read_text()) if out.exists() else {}

    ds = datasets.get(args.dataset)
    data = {s: [ds.load_utterance(x) for x in v] for s, v in ds.splits().items()}
    rng = random.Random(args.seed)
    sub_train = rng.sample(data["train"], min(args.train_subsample, len(data["train"])))
    runner = SubsetRunner(args.model, device)
    probe = json.loads((froot / args.model / "probe_results.json").read_text())["layers"]
    search_shifts = [probe[str(args.start)]["shift"]]  # fixed during search; re-chosen in full_eval
    protected = {1} if runner.is_wavlm else set()

    def score(layers):
        tr = runner.features(sub_train, layers)
        va = runner.features(data["valid"], layers)
        return fit_select(tr, sub_train, va, data["valid"], search_shifts, device)[0]

    def save():
        out.write_text(json.dumps(state, indent=1))

    # greedy backward elimination (resumable: the path is saved after every step)
    path = state.setdefault("greedy_path", [])
    current = path[-1]["layers"] if path else list(range(1, args.start + 1))
    if not path:
        path.append({"layers": current, "search_valid_rmse": score(current)})
        save()
    while len(current) > args.min:
        cands = []
        for l in current:
            if l in protected:
                continue
            trial = [x for x in current if x != l]
            cands.append((score(trial), l, trial))
        r, dropped, current = min(cands)
        path.append({"layers": current, "dropped": dropped, "search_valid_rmse": r,
                     "candidates": {str(l): c for c, l, _ in cands}})
        print(f"greedy: dropped {dropped} -> {len(current)} layers, search valid RMSE {r:.4f}", flush=True)
        save()

    # block influence on the full start prefix
    if "bi" not in state:
        bi = runner.block_influence(sub_train[:100], list(range(1, args.start + 1)))
        state["bi"] = {str(k): v for k, v in bi.items()}
        save()
    bi = {int(k): v for k, v in state["bi"].items()}
    order = [l for l in sorted(bi, key=bi.get) if l not in protected]  # least influential first

    # identical full evaluation of greedy, BI and prefix subsets at every size
    evals = state.setdefault("eval", {})
    for n in range(args.start, args.min - 1, -1):
        drop = set(order[: args.start - n])
        subsets = {"prefix": list(range(1, n + 1)),
                   "greedy": next(p["layers"] for p in path if len(p["layers"]) == n),
                   "bi": [l for l in range(1, args.start + 1) if l not in drop]}
        for method, layers in subsets.items():
            key = f"{method}_{n}"
            if key in evals:
                continue
            same = next((v for v in evals.values() if v["layers"] == layers), None)
            evals[key] = same or full_eval(runner, layers, data, device, ds)
            print(f"{key}: layers {layers} test RMSE {evals[key]['test']['rmse']:.4f} "
                  f"PCC {evals[key]['test']['pcc']:.4f}", flush=True)
            save()


if __name__ == "__main__":
    main()
