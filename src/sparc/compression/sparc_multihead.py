"""Multi-speaker inverter heads for SPARC, judged by resynthesis with the shipped vocoder.

SPARC's shipped inverter is WavLM Large layer 9 (low-passed at 10 Hz) + a linear head fitted on MNGU0 (one speaker),
and its vocoder was trained on that head's output space. Here new linear heads are fitted on the same layer-9 states
(computed through the deployed `Inversion` pipeline) with the multi-speaker EMA (USC-TIMIT + EMA_5EMO; per-group z
targets), then mapped into the shipped head's output space and dropped into `sparc.load_model('en')` unchanged.

fit      -- inverter states for every utterance; ridge heads (alignment shift -2..3 and ridge strength chosen on
            validation): one per held-out speaker h trained on the other six ('loso_<h>') and one on all seven
            ('all'). Mapping to the shipped space: per channel y_shipped = a * z + c with a = sign(corr) * sd_S / sd_Z
            and c = m_S - a m_Z, from the shipped and new heads' outputs on the head's own training audio (no EMA
            and nothing of the held-out speaker). Also reports each head's held-out rmse_z / PCC.
resynth  -- every EMA speaker's test sentences with text: 'orig', 'shipped' (load_model('en') encode/decode),
            'loso' (head that never saw the speaker), 'all' (head trained with the speaker: seen upper bound) and
            'true_ema' (approximate reference: measured EMA mapped with the 'all' head's scale, which was fitted on
            ridge-shrunken predictions, not low-passed, spliced into the shipped prediction); and 200
            LibriTTS-R test-clean utterances with 'orig', 'shipped', 'all'. Pitch, loudness and speaker embedding
            are shared (one encode per utterance, seeded CREPE); only the EMA differs.
eval     -- WER (whisper-large-v3, Whisper English normalizer), speaker similarity to the original (ECAPA
            VoxCeleb cosine), wide-band PESQ and MCD (MFCC c1-c24, frame-aligned, no DTW) vs the original.

Usage: python -m sparc.compression.sparc_multihead {fit,resynth,eval,summary}
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import datasets
from .mngu0 import align
from .probe import ALPHAS, Ridge

OUT = datasets.XLSR_EMA / "sparc_multihead"
SHIFTS = (-2, -1, 0, 1, 2, 3)
LIBRI = Path("/data/group_data/UTD-NAS/Databases/LibriTTS-R/LibriTTS_R/test-clean")
SEED = 1234


def inverter_states(inv, wav):
    """Low-passed layer-9 states exactly as Inversion._extract_ema computes them (one utterance)."""
    from sparc.inversion import butter_bandpass_filter_padded

    w = inv.process_wavfiles(wav)
    mask = torch.ones_like(w.input_values)
    with torch.no_grad():
        hs = inv.speech_model(w.input_values, attention_mask=mask, output_hidden_states=True).hidden_states
    st = hs[inv.tgt_layer].cpu().numpy()
    valid = inv.speech_model._get_feat_extract_output_lengths(torch.as_tensor(np.asarray(w.input_lens))).cpu().numpy()
    st = butter_bandpass_filter_padded(st, valid, inv.freqcut, inv.ft_sr)
    return st[0].astype(np.float32)


def shipped_head(inv):
    sd = inv.linear_model.state_dict()
    return sd["weight"].cpu().numpy().T.astype(np.float64), sd["bias"].cpu().numpy().astype(np.float64)


def fit():
    from sparc import load_model

    device = "cuda"
    coder = load_model("en", device=device)
    inv = coder.inverter
    Ws, bs = shipped_head(inv)
    ds = datasets.EMAMulti(held_out=(), norm_by="norm_group")
    rows = ds.rows
    OUT.mkdir(parents=True, exist_ok=True)
    st_f = OUT / "states.npz"
    if st_f.exists():
        z = np.load(st_f, allow_pickle=True)
        states = dict(zip(z["stems"], z["states"]))
    else:
        states = {}
        for i, (stem, r) in enumerate(sorted(rows.items())):
            states[stem] = inverter_states(inv, str(ds.root / r["audio"]))
            if i % 200 == 0:
                print("states", i, len(rows), flush=True)
        np.savez(st_f, stems=np.array(list(states)), states=np.array(list(states.values()), dtype=object))
    utts = {s: ds.load_targets(s) for s in rows}
    by = {}
    for s, r in rows.items():
        by.setdefault((r["speaker"], r["_split"]), []).append(s)
    heads = {}
    for name in [f"loso_{h}" for h in datasets.ALL_SPEAKERS] + ["all"]:
        train_spk = [s for s in datasets.ALL_SPEAKERS if name != f"loso_{s}"]
        tr = [s for sp in train_spk for s in by.get((sp, "train"), [])]
        va = [s for sp in train_spk for s in by.get((sp, "valid"), [])]

        def pairs(stems, shift):
            X, Y = [], []
            for s in stems:
                x, y = align(states[s], utts[s].ema_mm, utts[s].base_offset + shift)
                if len(x) >= 2:
                    X.append(x)
                    Y.append(y)
            return X, Y

        best = None
        for shift in SHIFTS:
            Xtr, Ytr = pairs(tr, shift)
            Xva, Yva = pairs(va, shift)
            rg = Ridge(Xtr, Ytr, device)
            for a in ALPHAS:
                W, b = rg.weights(a)
                P = np.concatenate(rg.predict(W, b, Xva))
                e = float(np.sqrt(((P - np.concatenate(Yva)) ** 2).mean(0)).mean())
                if best is None or e < best[0]:
                    best = (e, shift, a, W.cpu().numpy(), b.cpu().numpy())
        _, shift, alpha, W, b = best
        # map z -> shipped output space with the head's own training audio
        Xall = np.concatenate([states[s] for s in tr]).astype(np.float64)
        S = Xall @ Ws + bs
        Z = Xall @ W + b
        sgn = np.sign([np.corrcoef(S[:, c], Z[:, c])[0, 1] for c in range(12)])
        a_ = sgn * S.std(0) / Z.std(0)
        c_ = S.mean(0) - a_ * Z.mean(0)
        Wsh, bsh = W * a_, b * a_ + c_
        res = {"shift": int(shift), "alpha": float(alpha), "sign": sgn.tolist(), "scale": a_.tolist(), "offset": c_.tolist(),
               "corr_shipped": [float(np.corrcoef(S[:, c], Z[:, c])[0, 1]) for c in range(12)]}
        if name.startswith("loso_"):
            h = name[5:]
            X, Y = pairs(by.get((h, "test"), []), shift)
            from .metrics import ema_metrics
            P = [x.astype(np.float64) @ W + b for x in X]
            Pc, Yc = np.concatenate(P), np.concatenate(Y)
            res["heldout"] = {"rmse_z": float(np.sqrt(((Pc - Yc) ** 2).mean(0)).mean()), "pcc": ema_metrics(P, Y, n_boot=0)["pcc"]}
        heads[name] = res
        np.savez(OUT / f"head_{name}.npz", W=W, b=b, Wsh=Wsh, bsh=bsh)
        print(name, json.dumps({k: v for k, v in res.items() if k in ("shift", "alpha", "heldout")}),
              "min |corr| with shipped %.2f" % np.min(np.abs(res["corr_shipped"])), flush=True)
    (OUT / "heads.json").write_text(json.dumps(heads, indent=1))


def _set_head(inv, Wsh, bsh):
    inv.linear_model.load_state_dict({"weight": torch.from_numpy(Wsh.T.copy()).float(),
                                      "bias": torch.from_numpy(bsh.copy()).float()})
    inv.linear_model.to(inv.device)


def resynth(n_libri=200):
    from sparc import load_model

    coder = load_model("en", device="cuda")
    inv = coder.inverter
    Ws, bs = shipped_head(inv)
    ship_sd = {k: v.clone() for k, v in inv.linear_model.state_dict().items()}
    heads = json.loads((OUT / "heads.json").read_text())
    hw = {n: np.load(OUT / f"head_{n}.npz") for n in heads}
    ds = datasets.EMAMulti(held_out=(), norm_by="norm_group")
    items = []
    for stem, r in sorted(ds.rows.items()):
        if r["_split"] == "test" and (r.get("text") or "").strip():
            items.append(("ema", r["speaker"], stem, str(ds.root / r["audio"]), r["text"].strip()))
    rng = np.random.default_rng(0)
    libri = sorted(LIBRI.glob("*/*/*.wav"))
    for p in [libri[i] for i in rng.choice(len(libri), n_libri, replace=False)]:
        txt = p.with_suffix(".normalized.txt")
        items.append(("libri", "libri", p.stem, str(p), txt.read_text().strip() if txt.exists() else ""))
    man_f = OUT / "resynth_manifest.json"
    manifest = json.loads(man_f.read_text()) if man_f.exists() else []
    done = {(m["system"], m["stem"]) for m in manifest}
    a_all = heads["all"]
    for i, (src, spk, stem, path, text) in enumerate(items):
        systems = ["orig", "shipped"] + (["loso", "all", "true_ema"] if src == "ema" else ["all"])
        if all((s, stem) in done for s in systems):
            continue
        inv.linear_model.load_state_dict(ship_sd)
        code = coder.encode(path, seed=SEED)
        st = inverter_states(inv, path).astype(np.float64)
        out_paths = {}
        for s in systems:
            o = OUT / "wav" / s / spk / f"{stem}.wav"
            o.parent.mkdir(parents=True, exist_ok=True)
            if s == "orig":
                ref, sr = sf.read(path)
                sf.write(o, ref, sr)
            else:
                c = dict(code)
                if s in ("loso", "all"):
                    h = hw[f"loso_{spk}" if s == "loso" else "all"]
                    c["ema"] = (st @ h["Wsh"] + h["bsh"]).astype(np.float32)[: len(code["ema"])]
                elif s == "true_ema":
                    e = (st @ Ws + bs)
                    u = ds.load_targets(stem)
                    t0 = u.base_offset + a_all["shift"]
                    z = u.ema_mm.astype(np.float64)
                    mapped = z * np.array(a_all["scale"]) + np.array(a_all["offset"])  # z -> shipped ('all' mapping)
                    j0 = max(0, -t0)
                    t_from, t_to = max(0, t0), min(len(e), t0 + len(z))
                    if t_to > t_from:
                        e[t_from:t_to] = mapped[j0:j0 + (t_to - t_from)]
                    c["ema"] = e.astype(np.float32)[: len(code["ema"])]
                wav = coder.decode(**c)
                sf.write(o, wav, coder.sr)
            out_paths[s] = str(o)
        for s in systems:
            manifest.append({"src": src, "speaker": spk, "stem": stem, "system": s, "path": out_paths[s],
                             "ref": out_paths["orig"], "text": text})
        if i % 25 == 0:
            man_f.write_text(json.dumps(manifest))
            print("resynth", i, len(items), flush=True)
    man_f.write_text(json.dumps(manifest))


def _mcd(ref, deg, sr=16000):
    import librosa
    a = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=25, n_mels=80, hop_length=160, n_fft=1024)[1:]
    b = librosa.feature.mfcc(y=deg, sr=sr, n_mfcc=25, n_mels=80, hop_length=160, n_fft=1024)[1:]
    n = min(a.shape[1], b.shape[1])
    d = a[:, :n] - b[:, :n]
    return float(np.mean(10.0 / np.log(10) * np.sqrt(2 * (d ** 2).sum(0))))


def evaluate():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import jiwer
    import librosa
    from huggingface_hub import snapshot_download
    from pesq import pesq
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    cache = "/data/user_data/xoy/.cache/huggingface/hub"  # the shared /data/hf_cache lost these snapshot dirs
    wdir = snapshot_download("openai/whisper-large-v3", cache_dir=cache, local_files_only=True)
    proc = WhisperProcessor.from_pretrained(wdir)
    asr = WhisperForConditionalGeneration.from_pretrained(wdir, torch_dtype=torch.float16).to("cuda").eval()
    from speechbrain.inference.speaker import EncoderClassifier
    edir = snapshot_download("speechbrain/spkrec-ecapa-voxceleb", cache_dir=cache, local_files_only=True)
    spk = EncoderClassifier.from_hparams(source=edir, savedir=str(OUT / "ecapa"), run_opts={"device": "cuda"})
    manifest = json.loads((OUT / "resynth_manifest.json").read_text())
    res_f = OUT / "eval.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}

    def load16(p):
        y, sr = sf.read(p, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        return librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y

    emb_cache = {}

    def emb(p):
        if p not in emb_cache:
            with torch.no_grad():
                emb_cache[p] = spk.encode_batch(torch.from_numpy(load16(p)).unsqueeze(0).to("cuda")).squeeze().cpu()
        return emb_cache[p]

    for i, m in enumerate(manifest):
        key = f"{m['system']}|{m['stem']}"
        if key in res:
            continue
        y = load16(m["path"])
        feats = proc(y, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            ids = asr.generate(feats, language="en", task="transcribe")
        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0]
        norm = proc.tokenizer.normalize
        r_ = {"src": m["src"], "speaker": m["speaker"], "system": m["system"], "hyp": hyp,
              "wer": float(jiwer.wer(norm(m["text"]), norm(hyp))) if m["text"] and norm(m["text"]).strip() else None,
              "n_words": len(norm(m["text"]).split()) if m["text"] else 0}
        if m["system"] != "orig":
            ref = load16(m["ref"])
            n = min(len(ref), len(y))
            r_["sim"] = float(torch.nn.functional.cosine_similarity(emb(m["ref"]), emb(m["path"]), dim=0))
            try:
                r_["pesq"] = float(pesq(16000, ref[:n], y[:n], "wb"))
            except Exception:
                r_["pesq"] = None
            r_["mcd"] = _mcd(ref[:n], y[:n])
        res[key] = r_
        if i % 50 == 0:
            res_f.write_text(json.dumps(res))
            print("eval", i, len(manifest), flush=True)
    res_f.write_text(json.dumps(res))


def summary():
    heads = json.loads((OUT / "heads.json").read_text())
    res = json.loads((OUT / "eval.json").read_text())
    L = ["# Multi-speaker inverter heads in SPARC: resynthesis with the shipped vocoder\n",
         "Heads: ridge on WavLM layer 9 (deployed inverter pipeline). 'loso' never saw the speaker; 'all' did; "
         "'true_ema' uses measured EMA. WER from whisper-large-v3 (corpus-level over words), SIM = ECAPA cosine to the "
         "original, PESQ wide-band, MCD dB (no DTW). Mean over utterances.\n",
         "\nHead held-out articulation (rmse_z / PCC, ridge on layer 9, speaker never seen): " +
         ", ".join(f"{k[5:]} {v['heldout']['rmse_z']:.3f}/{v['heldout']['pcc']:.3f}" for k, v in heads.items() if "heldout" in v)]
    for src in ("ema", "libri"):
        rows = [r for r in res.values() if r["src"] == src]
        if not rows:
            continue
        L.append(f"\n## {'EMA speakers (test sentences)' if src == 'ema' else 'LibriTTS-R test-clean (200 utterances)'}\n")
        L.append("| system | n | WER % | SIM | PESQ (n failed) | MCD dB |")
        L.append("|---|---|---|---|---|---|")
        for s in ("orig", "shipped", "loso", "all", "true_ema"):
            rs = [r for r in rows if r["system"] == s]
            if not rs:
                continue
            w = [r for r in rs if r["wer"] is not None]
            wer = 100 * sum(r["wer"] * r["n_words"] for r in w) / max(1, sum(r["n_words"] for r in w))
            f = lambda k: np.mean([r[k] for r in rs if r.get(k) is not None]) if any(r.get(k) is not None for r in rs) else float("nan")  # noqa: E731
            nfail = sum(1 for r in rs if s != "orig" and r.get("pesq") is None)
            L.append(f"| {s} | {len(rs)} | {wer:.2f} | {f('sim'):.3f} | {f('pesq'):.3f} ({nfail}) | {f('mcd'):.2f} |")
        if src == "ema":
            L.append("\nPer speaker WER % (shipped / loso / all / true_ema):")
            for h in datasets.ALL_SPEAKERS:
                cells = []
                for s in ("shipped", "loso", "all", "true_ema"):
                    w = [r for r in rows if r["system"] == s and r["speaker"] == h and r["wer"] is not None]
                    nw = sum(r['n_words'] for r in w)
                    cells.append(f"{100 * sum(r['wer'] * r['n_words'] for r in w) / nw:.1f}" if nw else "n/a")
                L.append(f"- {h}: " + " / ".join(cells))
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "resynth", "eval", "summary"))
    args = ap.parse_args()
    {"fit": fit, "resynth": resynth, "eval": evaluate, "summary": summary}[args.cmd]()


if __name__ == "__main__":
    main()
