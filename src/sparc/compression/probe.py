"""Per-layer linear (ridge) EMA probes on cached SSL features.

For every cached layer, fits ridge regression from the (low-passed, standardized) features to the
12 EMA targets on the dataset's train split (datasets.py: MNGU0 in mm, or the multi-speaker corpora in
per-speaker z units). The alignment shift (dataset.shift_candidates) and the ridge strength are chosen on
the validation split; test (and, for multi-speaker data, test_unseen: held-out speakers) is evaluated once
with the chosen setting. The best layer is also chosen on validation RMSE, never on test.

Outputs under <features root>/<model>/:
    probe_results.json  -- per-layer {shift, alpha, valid / test (/ test_unseen) metrics}, best layer
    probe_heads.npz     -- per-layer weight (D, 12) / bias (12,) in raw-feature -> target-unit space

Usage: python -m sparc.compression.probe <model> [--dataset mngu0|ema_multi] [--layers ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import datasets, mngu0

ALPHAS = np.logspace(-1, 6, 15)


def load_targets(stems, ds=None):
    ds = ds or datasets.get("mngu0")
    return {s: ds.load_targets(s) for s in stems}


def aligned(feats, index, utts, stems, shift):
    X, Y, keep = [], [], []
    for s in stems:
        a, n = index[s]
        x, y = mngu0.align(feats[a : a + n], utts[s].ema_mm, utts[s].base_offset + shift)
        if len(x) < 2:
            continue
        X.append(x)
        Y.append(y)
        keep.append(s)
    return X, Y, keep


class Ridge:
    """Ridge on standardized features and targets, solved for many alphas via one eigendecomposition."""

    def __init__(self, X, Y, device):
        X = torch.from_numpy(np.concatenate(X)).to(device, torch.float64)
        Y = torch.from_numpy(np.concatenate(Y)).to(device, torch.float64)
        self.device = device
        self.xm, self.xs = X.mean(0), X.std(0).clamp_min(1e-6)
        self.ym, self.ys = Y.mean(0), Y.std(0)
        Xz = (X - self.xm) / self.xs
        Yz = (Y - self.ym) / self.ys
        evals, self.V = torch.linalg.eigh(Xz.T @ Xz)
        self.evals = evals.clamp_min(0)
        self.VtXtY = self.V.T @ (Xz.T @ Yz)

    def weights(self, alpha):
        Wz = self.V @ (self.VtXtY / (self.evals + alpha)[:, None])
        W = Wz / self.xs[:, None] * self.ys[None, :]  # raw features -> target units
        b = self.ym - self.xm @ W
        return W, b

    def predict(self, W, b, X):
        out = []
        for x in X:
            out.append((torch.from_numpy(x).to(self.device, torch.float64) @ W + b).cpu().numpy())
        return out


def rmse(P, Y):
    P, Y = np.concatenate(P), np.concatenate(Y)
    return float(np.sqrt(((P - Y) ** 2).mean(0)).mean())


def evaluate_split(ds, feats, index, utts, stems, shift, ridge, W, b, n_boot):
    X, Y, keep = aligned(feats, index, utts, stems, shift)
    us = [utts[s] for s in keep]
    phones = [ds.frame_phones(u, u.base_offset + shift, len(y)) for u, y in zip(us, Y)]
    phones = phones if any(phones) else None
    return ds.evaluate(us, ridge.predict(W, b, X), Y, phones=phones, n_boot=n_boot)


def probe_layer(feats, index, utts, splits, device, n_boot=1000, ds=None):
    ds = ds or datasets.get("mngu0")
    best = None
    for shift in ds.shift_candidates:
        Xtr, Ytr, _ = aligned(feats, index, utts, splits["train"], shift)
        Xva, Yva, _ = aligned(feats, index, utts, splits["valid"], shift)
        ridge = Ridge(Xtr, Ytr, device)
        for alpha in ALPHAS:
            W, b = ridge.weights(alpha)
            r = rmse(ridge.predict(W, b, Xva), Yva)
            if best is None or r < best["valid_rmse"]:
                best = {"shift": shift, "alpha": float(alpha), "valid_rmse": r, "W": W, "b": b, "ridge": ridge}
    shift, ridge, W, b = best["shift"], best["ridge"], best["W"], best["b"]
    result = {"shift": shift, "alpha": best["alpha"], "valid_rmse_fit_units": best["valid_rmse"]}
    for split in ("valid", "test", "valid_unseen", "test_unseen"):
        if splits.get(split):
            result[split] = evaluate_split(ds, feats, index, utts, splits[split], shift, ridge, W, b,
                                           n_boot if split.startswith("test") else 0)
    return result, W.cpu().numpy().astype(np.float32), b.cpu().numpy().astype(np.float32)


def run(name, root=None, layers=None, device=None, dataset="mngu0"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = datasets.get(dataset)
    d = Path(root or datasets.features_root(dataset)) / name
    meta = json.loads((d / "meta.json").read_text())
    index = json.loads((d / "index.json").read_text())
    splits = {k: [s for s in v if s in index] for k, v in ds.splits().items()}
    utts = load_targets([s for v in splits.values() for s in v], ds)
    layers = layers if layers is not None else range(meta["num_layers"] + 1)

    results, heads = {}, {}
    out_json = d / "probe_results.json"
    if out_json.exists():
        results = {int(k): v for k, v in json.loads(out_json.read_text())["layers"].items()}
    sel = lambda j: results[j].get("valid_rmse_fit_units", results[j]["valid"]["rmse"])  # noqa: E731
    for k in layers:
        if k in results:
            continue
        feats = np.load(d / f"layer_{k:02d}.npy", mmap_mode="r").astype(np.float32)
        res, W, b = probe_layer(feats, index, utts, splits, device, ds=ds)
        results[k] = res
        heads[f"W_{k:02d}"], heads[f"b_{k:02d}"] = W, b
        t = res["test"]
        extra = f" unseen_rmse={res['test_unseen']['rmse']:.3f} unseen_pcc={res['test_unseen']['pcc']:.4f}" \
            if "test_unseen" in res else ""
        print(f"{name} layer {k:2d}: shift={res['shift']} alpha={res['alpha']:.3g} "
              f"valid_rmse={res['valid']['rmse']:.3f} test_rmse={t['rmse']:.3f}mm test_pcc={t['pcc']:.4f}{extra}",
              flush=True)
        best_layer = min(results, key=sel)
        out_json.write_text(json.dumps({"model": meta["model"], "dataset": dataset, "best_layer_by_valid_rmse": best_layer,
                                        "layers": {str(j): results[j] for j in sorted(results)}}, indent=1))
        heads_path = d / "probe_heads.npz"
        prev = dict(np.load(heads_path)) if heads_path.exists() else {}
        prev.update(heads)
        np.savez(heads_path, **prev)
        heads = {}
    best_layer = min(results, key=sel)
    print(f"{name}: best layer by validation RMSE = {best_layer} "
          f"(test RMSE {results[best_layer]['test']['rmse']:.3f} mm, PCC {results[best_layer]['test']['pcc']:.4f})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("root", nargs="?", default=None)
    ap.add_argument("--dataset", default="mngu0", choices=("mngu0", "ema_multi"))
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    args = ap.parse_args()
    run(args.model, args.root, args.layers, dataset=args.dataset)


if __name__ == "__main__":
    main()
