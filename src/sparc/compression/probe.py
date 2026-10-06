"""Per-layer linear (ridge) EMA probes on cached SSL features.

For every cached layer, fits ridge regression from the (low-passed, standardized) features to the
12 EMA channels on the MNGU0 train split. The alignment shift (mngu0.SHIFT_CANDIDATES) and the
ridge strength are chosen on the validation split; the test split is evaluated once with the
chosen setting. The best layer is also chosen on validation RMSE, never on test.

Outputs under <features>/<model>/:
    probe_results.json  -- per-layer {shift, alpha, valid metrics, test metrics}, best layer
    probe_heads.npz     -- per-layer weight (D, 12) / bias (12,) in raw-feature -> mm space

Usage: python -m sparc.compression.probe <model> [features_root]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import mngu0
from .extract import OUT_ROOT
from .metrics import ema_metrics

ALPHAS = np.logspace(-1, 6, 15)


def load_targets(stems):
    out = {}
    for s in stems:
        phones = mngu0.load_phones(s)
        sil_end = phones[0][1] if phones and phones[0][2] == mngu0.SILENCE else 0.0
        out[s] = mngu0.Utterance(stem=s, wav=None, ema_mm=mngu0.load_ema_mm(s), sil_end=sil_end, phones=phones)
    return out


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
        W = Wz / self.xs[:, None] * self.ys[None, :]  # raw features -> mm
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


def probe_layer(feats, index, utts, splits, device, n_boot=1000):
    best = None
    for shift in mngu0.SHIFT_CANDIDATES:
        Xtr, Ytr, _ = aligned(feats, index, utts, splits["train"], shift)
        Xva, Yva, _ = aligned(feats, index, utts, splits["valid"], shift)
        ridge = Ridge(Xtr, Ytr, device)
        for alpha in ALPHAS:
            W, b = ridge.weights(alpha)
            r = rmse(ridge.predict(W, b, Xva), Yva)
            if best is None or r < best["valid_rmse"]:
                best = {"shift": shift, "alpha": float(alpha), "valid_rmse": r, "W": W, "b": b, "ridge": ridge}
    shift, ridge, W, b = best["shift"], best["ridge"], best["W"], best["b"]
    Xva, Yva, _ = aligned(feats, index, utts, splits["valid"], shift)
    Xte, Yte, te_stems = aligned(feats, index, utts, splits["test"], shift)
    phones = [mngu0.frame_phones(utts[s], utts[s].base_offset + shift, len(y)) for s, y in zip(te_stems, Yte)]
    result = {
        "shift": shift,
        "alpha": best["alpha"],
        "valid": ema_metrics(ridge.predict(W, b, Xva), Yva, n_boot=0),
        "test": ema_metrics(ridge.predict(W, b, Xte), Yte, phones=phones, n_boot=n_boot),
    }
    return result, W.cpu().numpy().astype(np.float32), b.cpu().numpy().astype(np.float32)


def run(name, root=OUT_ROOT, layers=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(root) / name
    meta = json.loads((d / "meta.json").read_text())
    index = json.loads((d / "index.json").read_text())
    splits = {k: [s for s in v if s in index] for k, v in mngu0.stems_by_split().items()}
    utts = load_targets([s for v in splits.values() for s in v])
    layers = layers if layers is not None else range(meta["num_layers"] + 1)

    results, heads = {}, {}
    out_json = d / "probe_results.json"
    if out_json.exists():
        results = {int(k): v for k, v in json.loads(out_json.read_text())["layers"].items()}
    for k in layers:
        if k in results:
            continue
        feats = np.load(d / f"layer_{k:02d}.npy", mmap_mode="r").astype(np.float32)
        res, W, b = probe_layer(feats, index, utts, splits, device)
        results[k] = res
        heads[f"W_{k:02d}"], heads[f"b_{k:02d}"] = W, b
        t = res["test"]
        print(
            f"{name} layer {k:2d}: shift={res['shift']} alpha={res['alpha']:.3g} "
            f"valid_rmse={res['valid']['rmse']:.3f} test_rmse={t['rmse']:.3f}mm test_pcc={t['pcc']:.4f}",
            flush=True,
        )
        best_layer = min(results, key=lambda j: results[j]["valid"]["rmse"])
        out_json.write_text(
            json.dumps({"model": meta["model"], "best_layer_by_valid_rmse": best_layer,
                        "layers": {str(j): results[j] for j in sorted(results)}}, indent=1)
        )
        heads_path = d / "probe_heads.npz"
        prev = dict(np.load(heads_path)) if heads_path.exists() else {}
        prev.update(heads)
        np.savez(heads_path, **prev)
        heads = {}
    best_layer = min(results, key=lambda j: results[j]["valid"]["rmse"])
    print(f"{name}: best layer by validation RMSE = {best_layer} "
          f"(test RMSE {results[best_layer]['test']['rmse']:.3f} mm, PCC {results[best_layer]['test']['pcc']:.4f})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("root", nargs="?", default=str(OUT_ROOT))
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    args = ap.parse_args()
    run(args.model, args.root, args.layers)


if __name__ == "__main__":
    main()
