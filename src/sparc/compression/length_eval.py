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
Controls (rigor plan item 8; each for 1 s and 0.5 s pieces; separate the confounded causes of the chunk loss):
    cut1/cut05          -- encode the FULL utterance, low-pass it whole (= native features), cut the features into the
                           chunk pieces and give each piece to the head alone: the head's own context loss only
                           (frozen heads only; LoRA is audio-in and has no such cut);
    enc_fulllp1/05      -- encode each chunk alone (as chunk), keep the raw features, concatenate and low-pass over the
                           whole utterance, then the head sees the whole utterance: encoder-context loss without the
                           filter-edge effect (frozen heads only);
    uttnorm1/05         -- as chunk, but each chunk's waveform is normalized with the WHOLE utterance's mean / std.
Heads are trained with seeds 0, 1, 2 (systems "<head>", "<head>_s1", "<head>_s2"); the summary reports seed means.
Finished (condition, system) pairs are kept in the output file, so a rerun only does the missing work.
Frame bookkeeping: each utterance's audio is trimmed to a multiple of 320 samples, so its frames sit at a fixed offset
inside a concatenation and inside its own chunks. Metrics: rmse_z (pooled frames) and per-utterance PCC, in h's z units;
the native condition is also broken down by utterance duration.

Usage:
    python -m sparc.compression.length_eval run --fold usc_M1 [--layer 12]
    python -m sparc.compression.length_eval summary
"""

import argparse
import json
import re
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
SEEDS = (0, 1, 2)
BASE = ("native", "long", "chunk1", "chunk05")
CONTROLS = ("cut1", "cut05", "enc_fulllp1", "enc_fulllp05", "uttnorm1", "uttnorm05")
CONDS = BASE + CONTROLS
FROZEN_ONLY = ("cut1", "cut05", "enc_fulllp1", "enc_fulllp05")
BINS = ((0, 2), (2, 3), (3, 4), (4, 6), (6, 99))


def utt_metrics(P, Y):
    """Per-utterance squared-error sums (for pooled rmse_z) and PCC."""
    return [{"sse": ((p - y) ** 2).sum(0).tolist(), "n": len(y), "pcc": ema_metrics([p], [y], n_boot=0)["pcc"]}
            for p, y in zip(P, Y)]


def pooled(rows):
    sse = np.sum([r["sse"] for r in rows], 0)
    n = np.sum([r["n"] for r in rows])
    return {"rmse_z": float(np.sqrt(sse / n).mean()), "pcc": float(np.nanmean([r["pcc"] for r in rows])), "n_utts": len(rows)}


def lowpass_any(h):
    if len(h) > 18:
        return lowpass(h).astype(np.float32)
    # pieces shorter than the zero-phase filter's default padding (18 frames): same filter, shorter padding
    from scipy.signal import butter, filtfilt
    b, a = butter(5, 10 / (0.5 * 50), btype="low")
    return filtfilt(b, a, h, axis=0, padlen=max(len(h) - 1, 0)).astype(np.float32)


class Encoder:
    def __init__(self, layer, device):
        self.model = load_full(MODEL).to(device).eval()
        self.layer, self.device = layer, device

    @torch.no_grad()
    def raw(self, wav, norm=True):
        """Un-low-passed layer features; norm=False: the waveform is already normalized."""
        x = torch.from_numpy(normalize_wav(wav) if norm else wav).float().unsqueeze(0).to(self.device)
        return self.model(x, output_hidden_states=True).hidden_states[self.layer][0].float().cpu().numpy()

    def __call__(self, wav, norm=True):
        return lowpass_any(self.raw(wav, norm))


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
        base = condition.replace("uttnorm", "chunk").replace("enc_fulllp", "chunk").replace("cut", "chunk")
        unorm = condition.startswith("uttnorm")
        C = 16000 if base == "chunk1" else 8000
        for w in wavs:
            m, n_tot = [], len(w) // HOP - 1
            wn = normalize_wav(w) if unorm else w
            for c0 in range(0, len(w), C):
                piece = wn[c0:c0 + C + RF]
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


def run(fold, layer=12, device=None, conds=None):
    conds = conds or CONDS
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
        for seed in SEEDS:
            sname = hname if seed == 0 else f"{hname}_s{seed}"
            if all(sname in res.get(c, {}) for c in conds):
                continue  # finished in an earlier run
            head, _ = train_one(hname, False, hs["lr"], data, ym, ys, device, seed=seed)
            head.eval()

            def f(F, head=head):
                with torch.no_grad():
                    y = head(torch.from_numpy(F).unsqueeze(0).to(device), None)[0].float().cpu().numpy()
                return y * ys + ym
            systems[sname] = f

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

        def lora(w, norm=True):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                y, _ = forward(lmodel, lhead, [normalize_wav(w) if norm else w], device, cfg.layer_pool, train=False)
            return y[0].float().cpu().numpy() * lys + lym

    enc = Encoder(layer, device)
    test = [ds.load_utterance(s) for s in sp["test_unseen"]]
    wavs = [u.wav[: (len(u.wav) // HOP) * HOP] for u in test]
    nfr = [len(w) // HOP - 1 for w in wavs]
    durs = [len(w) / 16000 for w in wavs]
    for cond in conds:
        have = res.get(cond, {})
        todo = {k: v for k, v in systems.items() if k not in have}
        do_lora = lora is not None and "lora" not in have and cond not in FROZEN_ONLY
        if not todo and not do_lora:
            continue
        unorm = cond.startswith("uttnorm")
        if cond.startswith("cut"):
            # encode and low-pass the full utterance, then cut the features into the chunk pieces
            inputs, maps = segments(wavs, "chunk" + cond[3:])
            full = [enc(w) for w in wavs]
            inputs = [None] * len(inputs)
            for i, m in enumerate(maps):
                for inp, _, nf, dst in m:
                    inputs[inp] = full[i][dst:dst + nf]
            feats_in = inputs
        elif cond.startswith("enc_fulllp"):
            # chunks encoded alone (raw features), concatenated per utterance, low-passed over the whole utterance
            cin, cmaps = segments(wavs, "chunk" + cond[len("enc_fulllp"):])
            raws = [enc.raw(x) for x in cin]
            feats_in, maps = [], []
            for i, m in enumerate(cmaps):
                F = np.full((nfr[i], raws[0].shape[1]), np.nan, dtype=np.float32)
                cov = np.zeros(nfr[i], bool)
                for inp, src, nf, dst in m:
                    k = min(nf, len(raws[inp]) - src, nfr[i] - dst)
                    F[dst:dst + k] = raws[inp][src:src + k]
                    cov[dst:dst + k] = True
                if cov.any():  # hold the edge values over uncovered frames (they are not scored)
                    idx = np.maximum.accumulate(np.where(cov, np.arange(len(cov)), -1))
                    idx[idx < 0] = np.argmax(cov)
                    F = F[idx]
                feats_in.append(lowpass_any(F))
                maps.append([(i, d, n_, d) for _, _, n_, d in m])
            inputs = None
        else:
            inputs, maps = segments(wavs, cond)
            feats_in = [enc(x, norm=not unorm) for x in inputs]
        rec = dict(have)
        outs_all = {s: assemble([fn(F) for F in feats_in], maps, nfr) for s, fn in todo.items()}
        if do_lora:
            outs_all["lora"] = assemble([lora(x, norm=not unorm) for x in inputs], maps, nfr)
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


def base_name(s):
    return re.sub(r"_s\d+$", "", s)


def seed_mean(rec, system):
    """Mean over the seeds of one system's pooled (rmse_z, pcc) in one condition of one fold; None if absent."""
    v = [pooled(rows) for s, rows in rec.items() if base_name(s) == system]
    if not v:
        return None
    return float(np.mean([x["rmse_z"] for x in v])), float(np.mean([x["pcc"] for x in v])), len(v)


def cell(allres, cond, system):
    v = [seed_mean(r[cond], system) for r in allres.values() if cond in r]
    v = [x for x in v if x]
    if not v:
        return None
    return float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])), len(v)


def summary(layer=12):
    L = ["# Short and long inputs, held-out speakers (LOSO; WavLM Large k=%d)\n" % layer,
         "rmse_z (pooled frames) / per-utterance PCC in the held-out speaker's z units, mean over folds (heads: mean over "
         "3 seeds per fold; probe and LoRA are single models), same EMA frames in every condition. native = utterance "
         "alone; long = test utterances concatenated into >= 20 s recordings; chunk = independent pieces of 1 s / 0.5 s. "
         "Controls: cut = full-utterance encode + low-pass, features cut into pieces, head per piece; enc_fulllp = chunks "
         "encoded alone, raw features concatenated and low-passed over the whole utterance; uttnorm = chunks normalized "
         "with the whole utterance's mean / std (LoRA has no cut / enc_fulllp variants).\n"]
    files = sorted(OUT.glob(f"*_k{layer}.json"))
    allres = {f.stem.rsplit("_k", 1)[0]: json.loads(f.read_text()) for f in files}
    systems = sorted({base_name(s) for r in allres.values() for s in r.get("native", {})},
                     key=lambda s: (s != "probe", s == "lora", s))
    L.append(f"Folds: {len(allres)}. Feature reproduction (max |live - cache| / max |cache|): " + ", ".join(
        f"{h} {r['check']['max_abs_diff_vs_cache'] / r['check']['scale']:.1e}" for h, r in allres.items() if "check" in r))
    nfold = {c: sum(c in r for r in allres.values()) for c in CONDS}
    L.append("Folds per condition: " + ", ".join(f"{c} {n}" for c, n in nfold.items()))
    for title, conds in (("Existing conditions", BASE), ("Controls", CONTROLS)):
        L.append(f"\n{title}:\n")
        L.append("| system | " + " | ".join(conds) + " |")
        L.append("|---|" + "---|" * len(conds))
        for s in systems:
            cells = []
            for c in conds:
                v = cell(allres, c, s)
                cells.append(f"{v[0]:.4f} / {v[1]:.4f}" if v else "-")
            L.append(f"| {s} | " + " | ".join(cells) + " |")

    L.append("\nDecomposition of the chunk loss (rmse_z / PCC increments, + = worse than the reference; mean over folds and "
             "seeds). total = chunk - native; head = cut - native (the head alone sees pieces); encoder = enc_fulllp - "
             "native (encoder context only, whole-utterance low-pass); edge = chunk - enc_fulllp (low-pass filter edges "
             "re-applied per piece); norm = chunk - uttnorm (per-chunk waveform normalization).\n")
    dec = {}
    for ln, tag in (("1 s", "1"), ("0.5 s", "05")):
        L.append(f"\n{ln} pieces:\n")
        L.append("| system | total | head (cut) | encoder (enc_fulllp) | edge | norm |")
        L.append("|---|---|---|---|---|---|")
        for s in systems:
            g = {c: cell(allres, c, s) for c in ("native", "chunk" + tag, "cut" + tag, "enc_fulllp" + tag, "uttnorm" + tag)}

            def d(a, b):
                if g[a] is None or g[b] is None:
                    return None
                return g[a][0] - g[b][0], g[b][1] - g[a][1]  # rmse_z increase, PCC decrease
            parts = {"total": d("chunk" + tag, "native"), "head": d("cut" + tag, "native"),
                     "encoder": d("enc_fulllp" + tag, "native"), "edge": d("chunk" + tag, "enc_fulllp" + tag),
                     "norm": d("chunk" + tag, "uttnorm" + tag)}
            dec[(ln, s)] = parts
            L.append(f"| {s} | " + " | ".join(f"{p[0]:+.4f} / {p[1]:+.4f}" if p else "-" for p in parts.values()) + " |")
    # short paragraph from the head systems pooled (probe + heads, without LoRA)
    L.append("")
    for ln in ("1 s", "0.5 s"):
        keys = [k for k in dec if k[0] == ln and k[1] != "lora"]
        m = {}
        for part in ("total", "head", "encoder", "edge", "norm"):
            v = [dec[k][part] for k in keys if dec[k][part]]
            m[part] = (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) if v else None
        if all(m.values()):
            L.append(f"{ln}, mean over probe and the 5 heads: total loss {m['total'][0]:+.3f} rmse_z / {m['total'][1]:+.3f} PCC; "
                     f"head context alone {m['head'][0]:+.3f} / {m['head'][1]:+.3f}; encoder context with whole-utterance "
                     f"filtering {m['encoder'][0]:+.3f} / {m['encoder'][1]:+.3f}; per-piece filter edges {m['edge'][0]:+.3f} / "
                     f"{m['edge'][1]:+.3f}; per-chunk normalization {m['norm'][0]:+.3f} / {m['norm'][1]:+.3f}. "
                     "The terms are measured against different references and need not add up (they interact).\n")

    L.append("\nNative condition by utterance duration (pooled over folds; rmse_z / PCC, n utterances; seed 0 systems):\n")
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
    ap.add_argument("--conds", nargs="*", help="subset of conditions (default: all)")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.fold, args.layer, conds=args.conds)
    else:
        summary(args.layer)


if __name__ == "__main__":
    main()
