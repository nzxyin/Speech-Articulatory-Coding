"""Does held-out-speaker performance hold for short and long inputs? (LOSO, WavLM Large k layers)

For each fold (held-out speaker h) the systems are trained as in head_screen / depth_sweep (training speakers only):
ridge probe (alpha and shift of the fold, depth_sweep), frozen heads conv_causal9, conv9, gru, gruconv9, mamba
(learning rate chosen in head_screen, seed 0, retrained here on the cached features), and the LoRA + causal conv
checkpoint (seed 0, LOSO frontier). Every system is then run on h's test audio from scratch (encoder forward, 10 Hz
low-pass, head) under four input conditions and always scored on the same EMA frames:
    native  -- each test utterance alone (as everywhere else; checks the reproduction);
    long    -- h's test utterances concatenated (in a fixed order) into recordings of >= 20 s (~20-30 s);
    chunk1  -- each utterance cut into independent 1.0 s pieces (+400 samples of lookahead so every frame exists);
    chunk05 -- the same with 0.5 s pieces.
Frame bookkeeping: each utterance's audio is trimmed to a multiple of 320 samples, so its frames sit at a fixed offset
inside a concatenation and inside its own chunks. Metrics: rmse_z (pooled frames) and per-utterance PCC, in h's z units;
the native condition is also broken down by utterance duration.

Usage:
    python -m sparc.compression.length_eval run --fold usc_M1 [--layer 12]
    python -m sparc.compression.length_eval summary
"""

import argparse
import json
from argparse import Namespace

import numpy as np
import torch

from . import datasets
from .encoders import load_full, normalize_wav
from .extract import lowpass
from .head_screen import train_one
from .metrics import ema_metrics
from .probe import ALPHAS, Ridge, aligned, load_targets
from .train import build, forward, load_trainable

OUT = datasets.XLSR_EMA / "analysis_length"
MODEL = "wavlm-large"
HEADS = ("conv_causal9", "conv9", "gru", "gruconv9", "mamba")
HOP, RF = 320, 400  # WavLM frame hop and receptive field (samples)
BINS = ((0, 2), (2, 3), (3, 4), (4, 6), (6, 99))


def utt_metrics(P, Y):
    """Per-utterance squared-error sums (for pooled rmse_z) and PCC."""
    return [{"sse": ((p - y) ** 2).sum(0).tolist(), "n": len(y), "pcc": ema_metrics([p], [y], n_boot=0)["pcc"]}
            for p, y in zip(P, Y)]


def pooled(rows):
    sse = np.sum([r["sse"] for r in rows], 0)
    n = np.sum([r["n"] for r in rows])
    return {"rmse_z": float(np.sqrt(sse / n).mean()), "pcc": float(np.nanmean([r["pcc"] for r in rows])), "n_utts": len(rows)}


class Encoder:
    def __init__(self, layer, device):
        self.model = load_full(MODEL).to(device).eval()
        self.layer, self.device = layer, device

    @torch.no_grad()
    def __call__(self, wav):
        x = torch.from_numpy(normalize_wav(wav)).float().unsqueeze(0).to(self.device)
        h = self.model(x, output_hidden_states=True).hidden_states[self.layer][0].float().cpu().numpy()
        return lowpass(h).astype(np.float32)


def segments(wavs, condition):
    """Inputs to run and, per utterance, where its frames come from: list of (input_index, input_frame_offset) per
    frame block, as [(inp, src_start, n_frames, dst_start)]."""
    inputs, maps = [], []
    if condition == "native":
        for i, w in enumerate(wavs):
            inputs.append(w)
            maps.append([(i, 0, len(w) // HOP - 1, 0)])
    elif condition == "long":
        cur, cur_len, starts = [], 0, []
        groups = []
        for i, w in enumerate(wavs):
            cur.append(i)
            cur_len += len(w)
            if cur_len >= 20 * 16000:
                groups.append(cur)
                cur, cur_len = [], 0
        if cur:
            groups.append(cur)
        maps = [None] * len(wavs)
        for g in groups:
            off = 0
            for i in g:
                maps[i] = [(len(inputs), off // HOP, len(wavs[i]) // HOP - 1, 0)]
                off += len(wavs[i])
            inputs.append(np.concatenate([wavs[i] for i in g]))
    else:
        C = 16000 if condition == "chunk1" else 8000
        for w in wavs:
            m, n_tot = [], len(w) // HOP - 1
            for c0 in range(0, len(w), C):
                piece = w[c0:c0 + C + RF]
                if len(piece) < RF + HOP:
                    break
                nf = min(C // HOP, len(piece) // HOP - 1, n_tot - c0 // HOP)
                if nf <= 0:
                    break
                m.append((len(inputs), 0, nf, c0 // HOP))
                inputs.append(piece)
            maps.append(m)
    return inputs, maps


def assemble(outs, maps, n_frames):
    """Per-utterance frame-level arrays from per-input outputs."""
    res = []
    for m, n in zip(maps, n_frames):
        a = np.full((n, outs[0].shape[1]), np.nan, dtype=np.float64)
        for inp, src, nf, dst in m:
            k = min(nf, len(outs[inp]) - src, n - dst)
            a[dst:dst + k] = outs[inp][src:src + k]
        res.append(a)
    return res


def run(fold, layer=12, device=None):
    from .depth_sweep import load_all

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    out_f = OUT / f"{fold}_k{layer}.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    name = f"ema_loso_{fold}"
    ds = datasets.EMAMulti(held_out=(fold,), norm_by="norm_group")
    cache = datasets.features_root("ema_multi") / MODEL
    index = json.loads((cache / "index.json").read_text())
    feats = np.load(cache / f"layer_{layer:02d}.npy", mmap_mode="r")
    sp = {k: [s for s in v if s in index] for k, v in ds.splits().items()}
    utts = load_targets(sorted(set(index) & set(ds.rows)), ds)
    depth = load_all(MODEL)[f"{layer}/{fold}"]
    shift = depth["shift"]
    data = {}
    for split in ("train", "valid"):
        X, Y, _ = aligned(feats, index, utts, sp[split], shift)
        data[split] = ([np.asarray(x, dtype=np.float32) for x in X], [y.astype(np.float64) for y in Y])
    Yall = np.concatenate(data["train"][1])
    ym, ys = Yall.mean(0), Yall.std(0)

    # systems: frame-wise functions of a full feature sequence -> standardized-undone z predictions
    systems = {}
    rg = Ridge(data["train"][0], data["train"][1], device)
    W, b = rg.weights(depth["alpha"])
    W, b = W.cpu().numpy(), b.cpu().numpy()
    systems["probe"] = lambda F: F.astype(np.float64) @ W + b
    for hname in HEADS:
        hs = json.loads((datasets.XLSR_EMA / "analysis_heads" / MODEL / f"k{layer}_{hname}.json").read_text())[fold]
        head, _ = train_one(hname, False, hs["lr"], data, ym, ys, device, seed=0)
        head.eval()

        def f(F, head=head):
            with torch.no_grad():
                y = head(torch.from_numpy(F).unsqueeze(0).to(device), None)[0].float().cpu().numpy()
            return y * ys + ym
        systems[hname] = f

    # LoRA + causal conv (its own encoder): operates on audio, not on the frozen features
    lora_dir = datasets.adapt_root(name) / MODEL / f"prefix{layer}_independent_r8_q_proj+v_proj_conv_causal_s0"
    lora = None
    if (lora_dir / "results.json").exists():
        cfg = Namespace(**json.loads((lora_dir / "results.json").read_text())["config"])
        cfg.dataset, cfg.bf16 = name, device == "cuda"
        lmodel, lhead, _ = build(cfg, device)
        load_trainable(lmodel, lhead, torch.load(lora_dir / "best_trainable.pt"))
        lmodel.eval(), lhead.eval()
        Ytr_lora = np.concatenate([ds.load_targets(s).ema_mm for s in ds.splits()["train"]])
        lym, lys = Ytr_lora.mean(0), Ytr_lora.std(0)
        lshift = cfg.shift

        def lora(w):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                y, _ = forward(lmodel, lhead, [normalize_wav(w)], device, cfg.layer_pool, train=False)
            return y[0].float().cpu().numpy() * lys + lym

    enc = Encoder(layer, device)
    test = [ds.load_utterance(s) for s in sp["test_unseen"]]
    wavs = [u.wav[: (len(u.wav) // HOP) * HOP] for u in test]
    nfr = [len(w) // HOP - 1 for w in wavs]
    durs = [len(w) / 16000 for w in wavs]
    for cond in ("native", "long", "chunk1", "chunk05"):
        if cond in res:
            continue
        inputs, maps = segments(wavs, cond)
        feats_in = [enc(x) for x in inputs]
        rec = {}
        outs_all = {s: assemble([fn(F) for F in feats_in], maps, nfr) for s, fn in systems.items()}
        if lora is not None:
            outs_all["lora"] = assemble([lora(x) for x in inputs], maps, nfr)
        for sname, frames in outs_all.items():
            sh = lshift if sname == "lora" else shift
            P, Y, D = [], [], []
            for u, a, d in zip(test, frames, durs):
                off = u.base_offset + sh
                e = u.ema_mm.astype(np.float64)
                if off < 0:
                    e, off = e[-off:], 0
                n = min(len(a) - off, len(e))
                p = a[off:off + n]
                ok = np.isfinite(p).all(1)
                if n <= 2 or ok.sum() < 3:
                    continue
                P.append(p[ok])
                Y.append(e[:n][ok])
                D.append(d)
            rows = utt_metrics(P, Y)
            for r, d in zip(rows, D):
                r["dur"] = d
            rec[sname] = rows
        res[cond] = rec
        out_f.write_text(json.dumps(res))
        print(fold, cond, {s: round(pooled(r)["rmse_z"], 4) for s, r in rec.items()}, flush=True)
    # reproduction check: native features vs the cache for the first test utterance
    if "check" not in res:
        F = enc(wavs[0])
        a, n = index[test[0].stem]
        c = np.asarray(feats[a:a + n], dtype=np.float32)
        k = min(len(F), len(c))
        res["check"] = {"max_abs_diff_vs_cache": float(np.abs(F[:k] - c[:k]).max()), "scale": float(np.abs(c).max())}
        out_f.write_text(json.dumps(res))


def summary(layer=12):
    L = ["# Short and long inputs, held-out speakers (LOSO; WavLM Large k=%d)\n" % layer,
         "rmse_z (pooled frames) / per-utterance PCC in the held-out speaker's z units, mean over the 7 folds, same EMA "
         "frames in every condition. native = utterance alone; long = test utterances concatenated into >= 20 s "
         "recordings; chunk1 / chunk05 = independent 1 s / 0.5 s pieces.\n"]
    files = sorted(OUT.glob(f"*_k{layer}.json"))
    allres = {f.stem.rsplit("_k", 1)[0]: json.loads(f.read_text()) for f in files}
    systems = sorted({s for r in allres.values() for c in ("native",) for s in r.get(c, {})})
    L.append(f"Folds: {len(allres)}. Feature reproduction (max |live - cache| / max |cache|): " + ", ".join(
        f"{h} {r['check']['max_abs_diff_vs_cache'] / r['check']['scale']:.1e}" for h, r in allres.items() if "check" in r))
    L.append("\n| system | native | long (>=20 s) | chunk 1 s | chunk 0.5 s |")
    L.append("|---|---|---|---|---|")
    for s in systems:
        cells = []
        for cond in ("native", "long", "chunk1", "chunk05"):
            v = [pooled(r[cond][s]) for r in allres.values() if cond in r and s in r[cond]]
            cells.append(f"{np.mean([x['rmse_z'] for x in v]):.4f} / {np.mean([x['pcc'] for x in v]):.4f}" if v else "-")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("\nNative condition by utterance duration (pooled over folds; rmse_z / PCC, n utterances):\n")
    L.append("| system | " + " | ".join(f"{a}-{b} s" if b < 99 else f">{a} s" for a, b in BINS) + " |")
    L.append("|---|" + "---|" * len(BINS))
    for s in systems:
        rows = [x for r in allres.values() if "native" in r and s in r["native"] for x in r["native"][s]]
        cells = []
        for a, b in BINS:
            sub = [x for x in rows if a <= x["dur"] < b]
            if sub:
                p = pooled(sub)
                cells.append(f"{p['rmse_z']:.3f} / {p['pcc']:.3f} ({len(sub)})")
            else:
                cells.append("-")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"summary_k{layer}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary"))
    ap.add_argument("--fold")
    ap.add_argument("--layer", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.fold, args.layer)
    else:
        summary(args.layer)


if __name__ == "__main__":
    main()
