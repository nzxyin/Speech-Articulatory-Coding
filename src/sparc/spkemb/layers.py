"""Phase 1 of docs/speaker_embedding_plan.md: which WavLM Large layer should SPARC pool for its speaker embedding,
judged by what resynthesis needs (speaker timbre that carries across utterances) rather than speaker verification.

extract  -- per utterance (VCTK: 30 per speaker; LibriTTS-R dev-clean / test-clean: up to 30 per speaker): all 25
            WavLM Large hidden states pooled four ways -- periodicity-weighted over the full utterance (SPARC's
            pooling, en+ periodicity: thresholded, loudness-gated), the same over the first 1 s and first 2 s, and
            uniform over the full utterance; the shipped en+ 64-d speaker embedding; and utterance-level acoustic
            targets over voiced frames (80-bin log-mel mean and std, MFCC 1-24 mean, spectral tilt) plus median F0
            (diagnostic). Written per dataset shard to OUT/<dataset>_<shard>.npz.
analyze  -- metrics per layer and pooling (probes fitted on training speakers, scored on disjoint speakers):
            M1 cross-utterance timbre (primary): ridge from utterance A's embedding to a different utterance B's
               targets (same speaker); RMSE in target-std units and PCC, mean over target dimensions;
            M2 speaker separability: nearest-centroid identification (5 enrollment utterances) and Fisher ratio;
            M3 content leakage: utterance's own target deviation from its speaker mean, predicted from its
               embedding's deviation from the speaker mean embedding (PCC; higher = more content);
            M4 short references: M1 with 1 s / 2 s references, and cosine of short to full embeddings;
            M5 pitch confound: PCC of predicted vs true median log F0.
            Shortlist rule (pre-set in the plan): layers within one SE (over test speakers) of the best M1 RMSE, plus
            layers 0 and 6. Writes OUT/summary.md and results.json.

Usage:
    python -m sparc.spkemb.layers extract --dataset vctk --shard 0 --nshards 4
    python -m sparc.spkemb.layers analyze
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("SPKEMB_OUT", "/data/user_data/xoy/spkemb/layers"))
DATASETS = {
    "vctk": Path("/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus"),
    "libri_dev": Path("/data/user_data/xoy/LibriTTS_R/dev-clean"),
    "libri_test": Path("/data/user_data/xoy/LibriTTS_R/test-clean"),
}
PER_SPK = 30
SR, HOP = 16000, 320
N_LAYERS = 25
POOLS = ("pw_full", "pw_1s", "pw_2s", "uni_full")


def list_items(dataset):
    root = DATASETS[dataset]
    files = sorted(list(root.rglob("*.wav")) + list(root.rglob("*.flac")))
    if dataset == "vctk":
        files = [f for f in files if "mic2" not in f.name]  # VCTK 0.92 has two microphones; keep one
        spk_of = lambda f: f.parent.name  # noqa: E731
    else:
        spk_of = lambda f: f.relative_to(root).parts[0]  # noqa: E731  LibriTTS-R: <speaker>/<chapter>/<file>
    by = {}
    for f in files:
        by.setdefault(spk_of(f), []).append(f)
    rng = random.Random(0)
    items = []
    for spk in sorted(by):
        fs = sorted(by[spk])
        rng.shuffle(fs)
        items += [(f"{dataset}:{spk}", str(f)) for f in fs[:PER_SPK]]
    return items


def acoustic_targets(wav, periodicity, pitch):
    import librosa

    mel = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=1024, hop_length=HOP, n_mels=80, fmax=8000)
    logmel = np.log(mel + 1e-6).T  # (T, 80)
    mfcc = librosa.feature.mfcc(S=logmel.T, n_mfcc=25).T[:, 1:]  # (T, 24)
    n = min(len(logmel), len(periodicity))
    voiced = periodicity[:n] > 0
    if voiced.sum() < 5:
        voiced = np.ones(n, dtype=bool)
    lm, mf = logmel[:n][voiced], mfcc[:n][voiced]
    mean = lm.mean(0)
    tilt = np.polyfit(np.arange(80), mean, 1)[0]
    f0 = pitch[:n][voiced[: len(pitch)]] if len(pitch) >= n else pitch[voiced[: len(pitch)]]
    f0 = f0[f0 > 0]
    return {"logmel_mean": mean, "logmel_std": lm.std(0), "mfcc_mean": mf.mean(0), "tilt": np.array([tilt]),
            "logf0": np.array([np.log(np.median(f0)) if len(f0) else np.nan])}


def extract(dataset, shard, nshards, device="cuda"):
    import torch
    from transformers import WavLMModel

    from sparc import load_model

    out_f = OUT / f"{dataset}_{shard}.npz"
    if out_f.exists():
        print(f"{out_f} exists")
        return
    items = list_items(dataset)[shard::nshards]
    coder = load_model("en+", device=device)
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(device)
    rec = {p: [] for p in POOLS}
    rec.update({"spk": [], "path": [], "en_plus": [], "logmel_mean": [], "logmel_std": [], "mfcc_mean": [],
                "tilt": [], "logf0": [], "n_frames": []})
    for i, (spk, path) in enumerate(items):
        code = coder.encode(path, seed=0)
        per = np.asarray(code["periodicity"]).reshape(-1).astype(np.float64)
        wavs = coder.process_wavfiles(path)
        with torch.no_grad():
            hs = wavlm(wavs.input_values, output_hidden_states=True).hidden_states
        H = torch.stack([h[0] for h in hs]).float().cpu().numpy()  # (25, T, 1024)
        T = min(H.shape[1], len(per))
        H, w = H[:, :T], per[:T]
        if w.sum() <= 0:
            w = np.ones(T)

        def pool(weights):
            return (H * weights[None, :, None]).sum(1) / weights.sum()

        def first(sec):
            m = np.zeros(T)
            m[: int(sec * 50)] = 1
            ww = w * m
            return pool(ww if ww.sum() > 0 else m)

        rec["pw_full"].append(pool(w).astype(np.float16))
        rec["pw_1s"].append(first(1).astype(np.float16))
        rec["pw_2s"].append(first(2).astype(np.float16))
        rec["uni_full"].append(pool(np.ones(T)).astype(np.float16))
        wav = wavs.input_values[0, : wavs.input_lens[0]].cpu().numpy()
        tg = acoustic_targets(wav, w, np.asarray(code["pitch"]).reshape(-1))
        for k, v in tg.items():
            rec[k].append(v.astype(np.float32))
        rec["en_plus"].append(np.asarray(code["spk_emb"]).reshape(-1).astype(np.float32))
        rec["spk"].append(spk)
        rec["path"].append(path)
        rec["n_frames"].append(T)
        if i % 100 == 0:
            print(f"{dataset} shard {shard}: {i}/{len(items)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(out_f, **{k: np.array(v) for k, v in rec.items()})
    print(f"wrote {out_f} ({len(items)} utterances)", flush=True)


# ---------------------------------------------------------------- analysis

def load_all():
    parts = sorted(OUT.glob("*_*.npz"))
    data = {}
    for p in parts:
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            data.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in data.items()}


def targets_matrix(d):
    return np.concatenate([d["logmel_mean"], d["logmel_std"], d["mfcc_mean"], d["tilt"]], 1).astype(np.float64)


def ridge_fit(X, Y, alphas=(1e-1, 1, 10, 100, 1e3, 1e4)):
    """Ridge with alpha chosen by leave-one-out-like holdout on a 20% split of the training rows."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    va, tr = idx[: len(X) // 5], idx[len(X) // 5:]
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-6
    ym = Y[tr].mean(0)

    def solve(rows, a):
        Xz = (X[rows] - xm) / xs
        return np.linalg.solve(Xz.T @ Xz + a * np.eye(X.shape[1]), Xz.T @ (Y[rows] - ym))

    best = min(alphas, key=lambda a: (((X[va] - xm) / xs @ solve(tr, a) + ym - Y[va]) ** 2).mean())
    W = solve(idx, best)
    return lambda Z: (Z - xm) / xs @ W + ym


def score(P, Y, ysd):
    rmse = np.sqrt(((P - Y) ** 2).mean(0)) / ysd
    pcc = [np.corrcoef(P[:, j], Y[:, j])[0, 1] for j in range(Y.shape[1]) if Y[:, j].std() > 0]
    return float(rmse.mean()), float(np.nanmean(pcc))


def analyze(seed=0):
    d = load_all()
    spk = d["spk"]
    Y = targets_matrix(d)
    logf0 = d["logf0"].reshape(-1)
    spks = sorted(set(spk))
    rng = np.random.default_rng(seed)
    test_spk = set(rng.choice(spks, len(spks) // 3, replace=False))
    is_te = np.array([s in test_spk for s in spk])
    ysd = Y[~is_te].std(0) + 1e-9
    # cross-utterance pairs: each utterance A paired with a different utterance B of its speaker (cyclic within speaker)
    order = {}
    for i, s in enumerate(spk):
        order.setdefault(s, []).append(i)
    partner = np.zeros(len(spk), dtype=int)
    for s, ids in order.items():
        for j, i in enumerate(ids):
            partner[i] = ids[(j + 1) % len(ids)]
    spk_mean_Y = {s: Y[ids].mean(0) for s, ids in order.items()}
    res = {}
    arms = [(p, k) for p in POOLS for k in range(N_LAYERS)] + [("en_plus", None)]
    for pool, k in arms:
        X = d["en_plus"].astype(np.float64) if pool == "en_plus" else d[pool][:, k].astype(np.float64)
        name = "en_plus (shipped 64-d)" if pool == "en_plus" else f"{pool} L{k}"
        r = {}
        # M1: A's embedding -> B's targets
        f = ridge_fit(X[~is_te], Y[partner[~is_te]])
        P = f(X[is_te])
        r["m1_rmse"], r["m1_pcc"] = score(P, Y[partner[is_te]], ysd)
        te = np.where(is_te)[0]
        E = (P - Y[partner[te]]) / ysd
        per_spk = [np.sqrt((E[spk[te] == s] ** 2).mean(0)).mean() for s in sorted(test_spk)]
        r["m1_rmse_se"] = float(np.std(per_spk, ddof=1) / np.sqrt(len(per_spk)))
        if pool in ("pw_full", "uni_full", "en_plus"):
            # M2: nearest centroid, 5 enrollment utterances per test speaker
            Xt = X[is_te]
            st = spk[is_te]
            Xn = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-9)
            cents, rest = {}, []
            for s in sorted(test_spk):
                ids = np.where(st == s)[0]
                cents[s] = Xn[ids[:5]].mean(0)
                rest += [(i, s) for i in ids[5:]]
            C = np.stack([cents[s] for s in sorted(test_spk)])
            names = sorted(test_spk)
            r["m2_id_acc"] = float(np.mean([names[int(np.argmax(C @ Xn[i]))] == s for i, s in rest]))
            mu = Xt.mean(0)
            sb = np.mean([np.sum((Xt[st == s].mean(0) - mu) ** 2) for s in names])
            sw = np.mean([np.mean(np.sum((Xt[st == s] - Xt[st == s].mean(0)) ** 2, 1)) for s in names])
            r["m2_fisher"] = float(sb / sw)
            # M3: content leakage, speaker-mean-removed embedding -> speaker-mean-removed own targets
            Xd = X - np.stack([X[order[s]].mean(0) for s in spk])
            Yd = Y - np.stack([spk_mean_Y[s] for s in spk])
            g = ridge_fit(Xd[~is_te], Yd[~is_te])
            r["m3_leak_pcc"] = score(g(Xd[is_te]), Yd[is_te], Yd[~is_te].std(0) + 1e-9)[1]
            # M5: pitch
            ok = np.isfinite(logf0)
            h = ridge_fit(X[~is_te & ok], logf0[~is_te & ok, None])
            r["m5_f0_pcc"] = float(np.corrcoef(h(X[is_te & ok])[:, 0], logf0[is_te & ok])[0, 1])
        if pool == "pw_full":
            for sec in ("1s", "2s"):
                Xs = d[f"pw_{sec}"][:, k].astype(np.float64)
                r[f"m4_m1_rmse_{sec}"] = score(f(Xs[is_te]), Y[partner[is_te]], ysd)[0]
                cs = np.sum(Xs * X, 1) / (np.linalg.norm(Xs, axis=1) * np.linalg.norm(X, axis=1) + 1e-9)
                r[f"m4_cos_{sec}"] = float(cs[is_te].mean())
        res[name] = r
        print(name, {a: round(b, 4) for a, b in r.items()}, flush=True)
    best = min((v["m1_rmse"], n) for n, v in res.items() if n.startswith("pw_full"))
    se = res[best[1]]["m1_rmse_se"]
    short = sorted({int(n.split("L")[1]) for n, v in res.items() if n.startswith("pw_full") and v["m1_rmse"] <= best[0] + se}
                   | {0, 6})
    out = {"n_utts": int(len(spk)), "n_spk": len(spks), "n_test_spk": len(test_spk), "results": res,
           "best_pw_full": best[1], "shortlist": short}
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    L = [f"# Speaker-embedding layer screen (WavLM Large; {len(spk)} utterances, {len(spks)} speakers, "
         f"{len(test_spk)} held-out test speakers)\n",
         "M1 = cross-utterance timbre prediction (RMSE in target-std units, PCC; primary). M2 = nearest-centroid speaker "
         "ID / Fisher ratio. M3 = content leakage PCC (lower = less content). M4 = M1 RMSE with 1 s / 2 s references. "
         "M5 = median-F0 PCC.\n",
         "| arm | M1 RMSE (SE) | M1 PCC | M4 RMSE 1s / 2s | cos 1s / 2s | M2 ID acc / Fisher | M3 leak PCC | M5 F0 PCC |",
         "|---|---|---|---|---|---|---|---|"]
    for k in range(N_LAYERS):
        r = res[f"pw_full L{k}"]
        L.append(f"| pw_full L{k} | {r['m1_rmse']:.4f} ({r['m1_rmse_se']:.4f}) | {r['m1_pcc']:.4f} | "
                 f"{r['m4_m1_rmse_1s']:.4f} / {r['m4_m1_rmse_2s']:.4f} | {r['m4_cos_1s']:.3f} / {r['m4_cos_2s']:.3f} | "
                 f"{r['m2_id_acc']:.3f} / {r['m2_fisher']:.3f} | {r['m3_leak_pcc']:.3f} | {r['m5_f0_pcc']:.3f} |")
    for n in ("en_plus (shipped 64-d)",):
        r = res[n]
        L.append(f"| {n} | {r['m1_rmse']:.4f} ({r['m1_rmse_se']:.4f}) | {r['m1_pcc']:.4f} | - | - | "
                 f"{r['m2_id_acc']:.3f} / {r['m2_fisher']:.3f} | {r['m3_leak_pcc']:.3f} | {r['m5_f0_pcc']:.3f} |")
    L.append("\nUniform pooling (uni_full) M1 RMSE by layer: " +
             ", ".join(f"L{k} {res[f'uni_full L{k}']['m1_rmse']:.4f}" for k in range(N_LAYERS)))
    L.append(f"\nBest periodicity-pooled layer by M1: {best[1]}. Shortlist (pre-set rule: within 1 SE of the best "
             f"M1 RMSE, plus layers 0 and 6): {short}.")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("extract", "analyze", "count"))
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.dataset, a.shard, a.nshards)
    elif a.cmd == "count":
        for ds in DATASETS:
            it = list_items(ds)
            print(ds, len(it), "utterances,", len({s for s, _ in it}), "speakers", it[:2])
    else:
        analyze()


if __name__ == "__main__":
    main()
