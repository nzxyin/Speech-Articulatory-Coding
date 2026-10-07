"""Zero-shot generalization of MNGU0-trained models to the multi-speaker EMA corpora.

Every model was fit on MNGU0 (one speaker, its own sensor placement and coordinate frame). On each of the seven
new speakers (USC-TIMIT M1/F1/M3/F5, EMA_5EMO jn/jr/kf) it is evaluated on that speaker's test sentences:

  zero_shot_pcc   per-channel Pearson correlation of the raw prediction with the measured trajectory (MNGU0's x
                  axis points posterior, the new corpora's anterior, so predicted x is negated). Invariant to
                  per-channel scale and offset, i.e. to sensor placement and units.
  affine          per-channel scale + offset fitted on the speaker's training sentences (least squares, the
                  same alignment shift for all channels, chosen there), then RMSE in mm / PCC on test.
  linear          a full 12 -> 12 linear map + bias per speaker (absorbs coordinate rotation and cross-channel
                  differences), fitted and evaluated the same way: how much of the articulatory signal the
                  model's output carries, given a per-speaker readout calibration.

Outputs: <features_ema_multi>/crossspeaker_zero_shot.json and a printed table.
Usage: python -m sparc.compression.crossspeaker [--calib-utts 120]
"""

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from . import datasets, mngu0
from .components import apply_spec
from .encoders import load_subset, normalize_wav
from .extract import lowpass
from .lora import merge_lora
from .metrics import ema_metrics
from .train import build, forward, load_trainable

FEAT = datasets.features_root("mngu0")
ADAPT = datasets.adapt_root("mngu0")
X_CH = list(range(0, 12, 2))  # x channels: MNGU0 posterior-positive, new corpora anterior-positive


def mngu0_target_stats():
    ds = datasets.get("mngu0")
    Y = np.concatenate([ds.load_targets(s).ema_mm for s in ds.splits()["train"]])
    return Y.mean(0), Y.std(0)


class ProbeModel:
    """Frozen prefix 1..k encoder + low-pass + the MNGU0 ridge head of layer k (raw features -> mm)."""

    def __init__(self, model, k, device):
        heads = np.load(FEAT / model / "probe_heads.npz")
        self.W = torch.from_numpy(heads[f"W_{k:02d}"]).to(device, torch.float32)
        self.b = torch.from_numpy(heads[f"b_{k:02d}"]).to(device, torch.float32)
        self.enc = load_subset(model, range(1, k + 1)).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, wav):
        h = self.enc(torch.from_numpy(normalize_wav(wav)).unsqueeze(0).to(self.device)).last_hidden_state[0]
        h = torch.from_numpy(lowpass(h.float().cpu().numpy()).astype(np.float32)).to(self.device)
        return (h @ self.W + self.b).cpu().numpy()


class AdaptedModel:
    """A train.py run (encoder subset + LoRA + head), or a components.py pruned model; outputs mm (MNGU0)."""

    def __init__(self, run_dir, device, ym, ys, pruned=False):
        run_dir = Path(run_dir)
        res = json.loads((run_dir / "results.json").read_text())
        args = Namespace(**res["config"])
        args.dataset = "mngu0"
        args.bf16 = device == "cuda"
        model, head, _ = build(args, device)
        if pruned:
            merge_lora(model)
            st = torch.load(run_dir / "pruned_model.pt", weights_only=False)
            apply_spec(model, st["spec"])
            model.load_state_dict(st["encoder"])
            head.load_state_dict(st["head"])
        else:
            if args.head == "linear":
                mean, std = torch.load(run_dir / "input_stats.pt")
                head.set_input_stats(mean, std)
            load_trainable(model, head, torch.load(run_dir / "best_trainable.pt"))
        self.model, self.head, self.args, self.device = model.eval(), head.eval(), args, device
        self.ym, self.ys = ym, ys

    @torch.no_grad()
    def __call__(self, wav):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.args.bf16):
            y, _ = forward(self.model, self.head, [normalize_wav(wav)], self.device, self.args.layer_pool, train=False)
        return y[0].float().cpu().numpy() * self.ys + self.ym


class SparcModel:
    def __init__(self, device):
        from sparc import load_model
        import soundfile as sf

        self.coder, self.sf = load_model("en", device=device), sf
        self.tmp = Path(f"/tmp/_crossspeaker_{id(self)}.wav")

    def __call__(self, wav):
        self.sf.write(self.tmp, wav, 16000)
        return self.coder.inverter(str(self.tmp))["ema"][0]


def paired(pred, u, shift):
    """Pair predicted frames with target frames: target frame i <-> prediction frame base_offset + shift + i."""
    p, y = mngu0.align(pred, u.ema_mm, u.base_offset + shift)
    return p, y


def fit_affine(P, Y):
    """Per-channel y = a p + c."""
    a, c = np.zeros(12), np.zeros(12)
    for j in range(12):
        A = np.stack([P[:, j], np.ones(len(P))], 1)
        a[j], c[j] = np.linalg.lstsq(A, Y[:, j], rcond=None)[0]
    return lambda X: X * a + c


def fit_linear(P, Y, alpha=1e-3):
    A = np.concatenate([P, np.ones((len(P), 1))], 1)
    W = np.linalg.solve(A.T @ A + alpha * len(A) * np.eye(13), A.T @ Y)
    return lambda X: np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def evaluate_model(fn, calib, test, ds, to_mm):
    """calib / test: lists of Utterance (targets in the speaker's mm after to_mm)."""
    pc = [fn(u.wav) for u in calib]
    pt = [fn(u.wav) for u in test]
    for p in pc + pt:
        p[:, X_CH] *= -1
    best = None
    for shift in (-1, 0, 1):
        pairs = [paired(p, u, shift) for p, u in zip(pc, calib)]
        P = np.concatenate([a for a, _ in pairs])
        Y = np.concatenate([to_mm(u, b) for (_, b), u in zip(pairs, calib)])
        f = fit_affine(P, Y)
        err = np.sqrt(((f(P) - Y) ** 2).mean(0)).mean()
        if best is None or err < best[0]:
            best = (err, shift, P, Y)
    _, shift, P, Y = best
    aff, lin = fit_affine(P, Y), fit_linear(P, Y)
    tp = [paired(p, u, shift) for p, u in zip(pt, test)]
    raw = [a for a, _ in tp]
    true = [to_mm(u, b) for (_, b), u in zip(tp, test)]
    zs = ema_metrics(raw, true, n_boot=0)
    a = ema_metrics([aff(x) for x in raw], true, n_boot=0)
    li = ema_metrics([lin(x) for x in raw], true, n_boot=0)
    return {"shift": shift, "zero_shot_pcc": zs["pcc"], "zero_shot_pcc_per_channel": {k: v["pcc"] for k, v in zs["per_channel"].items()},
            "affine_rmse": a["rmse"], "affine_pcc": a["pcc"], "linear_rmse": li["rmse"], "linear_pcc": li["pcc"],
            "n_test": len(test)}


def models_to_run():
    return [
        ("sparc_en", "sparc", None),
        ("probe wavlm-large k9", "probe", ("wavlm-large", 9)),
        ("probe xlsr-300m k18", "probe", ("xlsr-300m", 18)),
        ("probe xlsr-1b k16", "probe", ("xlsr-1b", 16)),
        ("probe xlsr-2b k11", "probe", ("xlsr-2b", 11)),
        ("lora+cc wavlm-large k9", "adapted", ADAPT / "wavlm-large/prefix9_independent_r8_q_proj+v_proj_conv_causal_s0"),
        ("lora+cc xlsr-300m k18", "adapted", ADAPT / "xlsr-300m/prefix18_independent_r8_q_proj+v_proj_conv_causal_s0"),
        ("lora+cc xlsr-1b k16", "adapted", ADAPT / "xlsr-1b/prefix16_independent_r8_q_proj+v_proj_conv_causal_s0"),
        ("lora+cc xlsr-2b k11", "adapted", ADAPT / "xlsr-2b/prefix11_independent_r8_q_proj+v_proj_conv_causal_s0"),
        ("pruned xlsr-300m keep0.75", "pruned",
         ADAPT / "xlsr-300m/pruned/prefix18_independent_r8_q_proj+v_proj_conv_causal_s0_keep0.75_taylor_steps4_s0"),
        ("pruned xlsr-300m keep0.5", "pruned",
         ADAPT / "xlsr-300m/pruned/prefix18_independent_r8_q_proj+v_proj_conv_causal_s0_keep0.5_taylor_steps4_s0"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-utts", type=int, default=120)
    ap.add_argument("--max-test", type=int, default=None, help="debug: cap test utterances per speaker")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names (substring match)")
    ap.add_argument("--out", default=str(datasets.features_root("ema_multi") / "crossspeaker_zero_shot.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = datasets.EMAMulti(held_out=())  # every speaker is unseen for MNGU0-trained models
    by_spk = {}
    for stem, r in ds.rows.items():
        by_spk.setdefault(r["speaker"], {"train": [], "test": []})
        if r["_split"] in ("train", "test"):
            by_spk[r["speaker"]][r["_split"]].append(stem)
    rng = np.random.default_rng(0)
    data = {}
    for spk, d in sorted(by_spk.items()):
        calib = sorted(rng.choice(d["train"], min(args.calib_utts, len(d["train"])), replace=False))
        test = d["test"][: args.max_test] if args.max_test else d["test"]
        data[spk] = ([ds.load_utterance(s) for s in calib], [ds.load_utterance(s) for s in test])
        print(spk, "calibration", len(calib), "test", len(test), flush=True)
    ym, ys = mngu0_target_stats()
    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for name, kind, spec in models_to_run():
        if args.models and not any(m in name for m in args.models):
            continue
        if name in out:
            continue
        if kind == "sparc":
            fn = SparcModel(device)
        elif kind == "probe":
            fn = ProbeModel(*spec, device)
        else:
            fn = AdaptedModel(spec, device, ym, ys, pruned=kind == "pruned")
        res = {spk: evaluate_model(fn, calib, test, ds, ds.to_mm) for spk, (calib, test) in data.items()}
        res["mean"] = {k: float(np.mean([v[k] for s, v in res.items() if s != "mean"]))
                       for k in ("zero_shot_pcc", "affine_rmse", "affine_pcc", "linear_rmse", "linear_pcc")}
        out[name] = res
        print(f"{name:32s} zero-shot PCC {res['mean']['zero_shot_pcc']:.3f}  affine RMSE {res['mean']['affine_rmse']:.3f} mm "
              f"PCC {res['mean']['affine_pcc']:.3f}  linear RMSE {res['mean']['linear_rmse']:.3f} mm PCC {res['mean']['linear_pcc']:.3f}",
              flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        del fn
        torch.cuda.empty_cache() if device == "cuda" else None


if __name__ == "__main__":
    main()
