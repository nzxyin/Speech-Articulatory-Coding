"""Phase 2 of docs/speaker_embedding_plan.md: does a WavLM layer, pooled as SPARC does, drive a FROZEN vocoder well?

Only the speaker encoding FFN (`SpeakerEncodingLayer`, 1024 -> 1024 -> GELU -> 64) is refit per layer; the generator of
the base model (`en` or `en+`) is frozen. The speaker reference always comes from a DIFFERENT utterance of the same
speaker than the segment that is reconstructed (the shipped training used the same utterance), so only speaker-level
information can help. Caveat from the plan: the frozen generator was trained with its own layer's embeddings (layer 0 for
`en`, 6 for `en+`), which biases the comparison toward that layer; Phase 3 removes this.

prep    -- per utterance cache (OUT/cache/<base>/<dataset>__<speaker>__<stem>.npz): 16 kHz int16 wav, the generator input
           `art` (T, 14 = EMA 12 + pitch + loudness, from the base model's `encode`, seeded CREPE), `pooled` (25, 1024)
           periodicity-weighted mean of every WavLM Large hidden state (the pooling of spkemb/layers.py `extract`: en+
           periodicity, i.e. thresholded and loudness-gated, over the full utterance) and `spk_emb_own` (64-d, the base
           model's own shipped embedding, for the control arm). Shardable and resumable (existing files are skipped).
refit   -- train a fresh FFN on layer k (`--layer k`) with mel L1 + multi-resolution STFT between generator output and
           the target audio, 32-frame (640 ms) crops, reference = another utterance of the speaker. Training data: VCTK
           speakers except 10 held out + LibriTTS-R dev-clean. Checkpoints every `--ckpt-every` steps (`ckpt.pt`, resumable),
           `final.pt` when done (finished runs are skipped). `--layer en_ffn` is the control (no training, the base model's own
           FFN and layer) and only logs the validation loss. Output: OUT/refit/<base>_L<k>/{ckpt.pt,final.pt,log.jsonl}.
eval    -- unseen speakers: LibriTTS-R test-clean (all speakers, `--per-spk` utterances each) and the held-out VCTK speakers.
           Each utterance is resynthesized with the embedding of ANOTHER utterance of the same speaker; metrics against
           the original: speaker similarity with ECAPA and with a second, non-ECAPA SV model (microsoft/wavlm-base-plus-sv,
           fetched online on a compute node if it is not in the cache; reported as missing otherwise), MCD (no DTW,
           comparable between arms only), WER (whisper-large-v3; transcripts next to the wavs). Voice-conversion swap:
           EMA / pitch / loudness of an utterance of speaker Y, embedding from an utterance of the next speaker X in the
           same set (no pitch rescaling, so cross-gender pairs are penalized equally in every arm): similarity to a
           different real utterance of X (target), to the original Y utterance (source), and WER against Y's text.
           Writes OUT/refit/eval_<base>_<arm>.json (resumable per utterance), `summary` writes OUT/refit/summary_<base>.md.

Usage (OUT = /data/user_data/xoy/spkemb, override with SPKEMB_REFIT_OUT):
    python -m sparc.spkemb.ffn_refit prep --base en --datasets vctk,libri_dev,libri_test --shard 0 --nshards 4
    python -m sparc.spkemb.ffn_refit refit --base en --layer 9 --steps 20000
    python -m sparc.spkemb.ffn_refit eval --base en --arms en_ffn,9
    python -m sparc.spkemb.ffn_refit summary --base en --arms en_ffn,9
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .layers import DATASETS, list_items

ROOT = Path(os.environ.get("SPKEMB_REFIT_OUT", "/data/user_data/xoy/spkemb")) / "refit"
# HF cache that has `en` and `en+` (the Speech-Articulatory-Coding repo is split over two caches), WavLM, whisper and
# ECAPA: a symlink merge of /data/user_data/xoy/{.cache/huggingface/hub,hf_hub_jobs}; wavlm-base-plus-sv is downloaded
# into it. Set HF_HUB_CACHE to override.
HF_CACHE = os.environ.get("HF_HUB_CACHE", "/data/user_data/xoy/spkemb/hf_cache")
SR, HOP = 16000, 320
# The shipped generators emit a fixed output level whatever the input (waveform std 0.044-0.048 on LibriTTS-R and VCTK,
# measured on `en`), i.e. they were trained on level-normalized targets. Training targets are therefore rescaled per
# utterance to this std, otherwise the mel loss mostly measures the level mismatch.
TARGET_STD = 0.047
N_HELD_VCTK = 10
EVAL_SETS = ("libri_test", "vctk_heldout")


# ---------------------------------------------------------------- pure helpers

def bname(base):
    return base.replace("+", "plus")


def heldout_speakers(speakers, n=N_HELD_VCTK, seed=0):
    """Deterministic held-out VCTK speaker set (names as in `list_items`, i.e. 'vctk:p225')."""
    sp = sorted(speakers)
    rng = np.random.default_rng(seed)
    return set(sp[i] for i in rng.choice(len(sp), min(n, len(sp)), replace=False))


def pick_ref(n, j, rng=None):
    """Index of the reference utterance for utterance j of a speaker with n utterances (never j itself).
    Deterministic (next utterance, cyclic) when rng is None, random otherwise."""
    assert n >= 2
    if rng is None:
        return (j + 1) % n
    k = int(rng.integers(n - 1))
    return k + 1 if k >= j else k


def vc_target(speakers, si):
    """Index of the target (embedding) speaker for source speaker si: the next speaker, cyclic."""
    assert len(speakers) >= 2
    return (si + 1) % len(speakers)


def wer_corpus(rows):
    """Word-weighted corpus WER (%) over rows with keys 'wer' and 'n_words'; None-WER rows are skipped."""
    rows = [r for r in rows if r.get("wer") is not None and r.get("n_words", 0) > 0]
    n = sum(r["n_words"] for r in rows)
    return 100.0 * sum(r["wer"] * r["n_words"] for r in rows) / n if n else float("nan")


def mrstft_loss(x, y, fft_sizes=(512, 1024, 2048), hops=(50, 120, 240), wins=(240, 600, 1200)):
    """Multi-resolution STFT loss (spectral convergence + log-magnitude L1, Parallel WaveGAN style); x, y: (B, T)."""
    import torch

    total = 0.0
    for n_fft, hop, win in zip(fft_sizes, hops, wins):
        w = torch.hann_window(win, device=x.device)
        sx = torch.stft(x, n_fft, hop, win, w, return_complex=True).abs().clamp(min=1e-7)
        sy = torch.stft(y, n_fft, hop, win, w, return_complex=True).abs().clamp(min=1e-7)
        sc = torch.linalg.norm(sy - sx, dim=(1, 2)) / torch.linalg.norm(sy, dim=(1, 2))
        mag = torch.nn.functional.l1_loss(torch.log(sx), torch.log(sy))
        total = total + sc.mean() + mag
    return total / len(fft_sizes)


# ---------------------------------------------------------------- items and cache

def spk_of(item):
    return item[0]


def cache_file(base, item):
    spk, path = item
    ds, s = spk.split(":", 1)
    return ROOT / "cache" / bname(base) / f"{ds}__{s}__{Path(path).stem}.npz"


def all_items(datasets):
    out = []
    for ds in datasets:
        out += list_items(ds)
    return out


def train_items():
    v = list_items("vctk")
    held = heldout_speakers({s for s, _ in v})
    return [it for it in v if it[0] not in held] + list_items("libri_dev")


def eval_items(per_spk=10):
    """{set name: {speaker: [items]}} for the unseen-speaker evaluation."""
    out = {"libri_test": {}, "vctk_heldout": {}}
    held = heldout_speakers({s for s, _ in list_items("vctk")})
    for name, items in (("libri_test", list_items("libri_test")), ("vctk_heldout", list_items("vctk"))):
        for it in items:
            if name == "vctk_heldout" and it[0] not in held:
                continue
            out[name].setdefault(it[0], []).append(it)
    return {k: {s: v[:per_spk] for s, v in d.items() if len(v) >= 2} for k, d in out.items()}


def load_record(base, item):
    f = cache_file(base, item)
    if not f.exists():
        return None
    z = np.load(f)
    wav = z["wav"]
    return {"item": item, "wav": wav, "art": z["art"], "pooled": z["pooled"], "spk_emb_own": z["spk_emb_own"],
            "gain": TARGET_STD / max(float(wav.astype(np.float32).std() / 32768.0), 1e-6)}


def read_text(path):
    """Transcript next to the wav (LibriTTS-R .normalized.txt, VCTK txt/<spk>/<utt>.txt) or None."""
    p = Path(path)
    cand = [p.with_suffix(".normalized.txt")]
    if "VCTK" in str(p):
        root = DATASETS["vctk"]
        cand.append(root / "txt" / p.parent.name / (p.stem.replace("_mic1", "") + ".txt"))
    for c in cand:
        if c.exists():
            return c.read_text().strip()
    return None


# ---------------------------------------------------------------- prep

def smoke_items(n_train=150, per_spk=2, n_libri_spk=6):
    """Tiny item set for the smoke test: the first `n_train` training items (what `refit --max-train` uses) plus a
    few eval speakers (`eval --per-spk 2`)."""
    ev = eval_items(per_spk)
    libri = [it for s in sorted(ev["libri_test"])[:n_libri_spk] for it in ev["libri_test"][s]]
    vctk = [it for v in ev["vctk_heldout"].values() for it in v]
    return train_items()[:n_train] + libri + vctk


def prep(base, datasets, shard, nshards, max_items=None, smoke=False, device="cuda"):
    import librosa
    import soundfile as sf
    import torch
    from transformers import WavLMModel

    from sparc import load_model

    items = []
    if smoke:
        items = smoke_items()
    else:
        for ds in datasets:
            items += list_items(ds)
    items = items[shard::nshards]
    if max_items:
        items = items[:max_items]
    todo = [it for it in items if not cache_file(base, it).exists()]
    print(f"prep {base}: {len(todo)}/{len(items)} utterances to do", flush=True)
    if not todo:
        return
    coder = load_model(base, device=device)
    plus = coder if base == "en+" else load_model("en+", device=device)
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(device)
    for i, item in enumerate(todo):
        try:
            code = coder.encode(item[1], seed=0)
            codep = code if plus is coder else plus.encode(item[1], seed=0)
            per = np.asarray(codep["periodicity"]).reshape(-1).astype(np.float64)
            wavs = plus.process_wavfiles(item[1])
            with torch.no_grad():
                hs = wavlm(wavs.input_values, output_hidden_states=True).hidden_states
            H = torch.stack([h[0] for h in hs]).float().cpu().numpy()
            T = min(H.shape[1], len(per))
            w = per[:T] if per[:T].sum() > 0 else np.ones(T)
            pooled = ((H[:, :T] * w[None, :, None]).sum(1) / w.sum()).astype(np.float16)
            art = coder._match_and_cat([np.asarray(code["ema"]), np.asarray(code["pitch"]), np.asarray(code["loudness"])])
            y, sr = sf.read(item[1], dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            wav16 = np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16)
            f = cache_file(base, item)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(f) + ".tmp.npz", wav=wav16, art=art.astype(np.float32), pooled=pooled,
                     spk_emb_own=np.asarray(code["spk_emb"]).reshape(-1).astype(np.float32))
            os.replace(str(f) + ".tmp.npz", f)
        except Exception as e:  # keep going; report at the end
            print(f"FAILED {item[1]}: {type(e).__name__}: {e}", flush=True)
        if i % 100 == 0:
            print(f"prep {i}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- refit

def load_generator(base, device="cuda"):
    from sparc import load_model

    coder = load_model(base, device=device)
    gen = coder.generator.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    return coder, gen


def _synth(gen, art, emb):
    """art: (B, T, 14) tensor, emb: (B, 64) -> wav (B, T*320)."""
    return gen(art.transpose(1, 2), emb)[:, 0]


def _crop(rec, ref, frames, rng, wav_full=False):
    import torch

    T = min(len(rec["art"]), len(rec["wav"]) // HOP)
    s = int(rng.integers(0, T - frames + 1)) if T > frames else 0
    art = rec["art"][s:s + frames]
    wav = rec["wav"][s * HOP:(s + frames) * HOP].astype(np.float32) / 32768.0 * rec["gain"]
    return torch.from_numpy(art), torch.from_numpy(wav)


def refit(base, layer, steps, batch, frames, lr, ckpt_every, seed, max_train=None, device="cuda"):
    import torch

    from sparc.spk_encoder import SpeakerEncodingLayer
    from sparc.training.mel import MelSpectrogram

    out = ROOT / (f"{bname(base)}_en_ffn" if layer == "en_ffn" else f"{bname(base)}_L{layer}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.pt").exists():
        print(f"{out}/final.pt exists; skipping")
        return
    items = train_items()
    if max_train:
        items = items[:max_train]
    recs = [r for r in (load_record(base, it) for it in items) if r is not None and len(r["art"]) >= frames
            and len(r["wav"]) >= frames * HOP]
    by = {}
    for r in recs:
        by.setdefault(r["item"][0], []).append(r)
    by = {s: v for s, v in by.items() if len(v) >= 2}
    recs = [r for v in by.values() for r in v]
    print(f"{len(recs)} training utterances, {len(by)} speakers", flush=True)
    assert recs, "empty training set: run prep first"
    held = eval_items(per_spk=3)["vctk_heldout"]
    vrecs = {s: [r for r in (load_record(base, it) for it in v) if r is not None and len(r["art"]) >= frames]
             for s, v in held.items()}
    vrecs = {s: v for s, v in vrecs.items() if len(v) >= 2}

    coder, gen = load_generator(base, device)
    mel = MelSpectrogram().to(device)
    ffn = SpeakerEncodingLayer(1024, 64).to(device)
    ffn.spk_fc[2] = torch.nn.Dropout(0.2)
    opt = torch.optim.Adam(ffn.parameters(), lr=lr, betas=(0.5, 0.9))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    step = 0
    if (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location=device, weights_only=False)
        ffn.load_state_dict(ck["ffn"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    rng = np.random.default_rng(seed + step)
    spk_list = sorted(by)

    def losses(arts, wavs, embs):
        wh = _synth(gen, arts, embs)
        n = min(wh.shape[-1], wavs.shape[-1])
        wh, wt = wh[..., :n], wavs[..., :n]
        lm = torch.nn.functional.l1_loss(mel(wh), mel(wt))
        return lm, mrstft_loss(wh, wt)

    def validate():
        ffn.eval()
        vr = np.random.default_rng(123)
        res = {"mel": [], "mrstft": [], "ctrl_mel": [], "ctrl_mrstft": []}
        with torch.no_grad():
            for s, v in vrecs.items():
                for j, r in enumerate(v):
                    ref = v[pick_ref(len(v), j)]
                    a, w = _crop(r, ref, frames, vr)
                    a, w = a[None].to(device), w[None].to(device)
                    if layer != "en_ffn":
                        e = ffn(torch.from_numpy(ref["pooled"][layer].astype(np.float32))[None].to(device))
                        lm, ls = losses(a, w, e)
                        res["mel"].append(lm.item())
                        res["mrstft"].append(ls.item())
                    e = torch.from_numpy(ref["spk_emb_own"])[None].to(device)
                    lm, ls = losses(a, w, e)
                    res["ctrl_mel"].append(lm.item())
                    res["ctrl_mrstft"].append(ls.item())
        ffn.train()
        return {f"val_{k}": float(np.mean(v)) for k, v in res.items() if v}

    def log(d):
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(d) + "\n")
        print(d, flush=True)

    if layer == "en_ffn":
        log({"step": 0, **validate()})
        return
    layer = int(layer)
    ffn.train()
    t0 = time.time()
    acc = []
    while step < steps:
        arts, wavs, refs = [], [], []
        for _ in range(batch):
            s = spk_list[int(rng.integers(len(spk_list)))]
            v = by[s]
            j = int(rng.integers(len(v)))
            r, ref = v[j], v[pick_ref(len(v), j, rng)]
            a, w = _crop(r, ref, frames, rng)
            arts.append(a)
            wavs.append(w)
            refs.append(torch.from_numpy(ref["pooled"][layer].astype(np.float32)))
        arts, wavs, refs = (torch.stack(x).to(device) for x in (arts, wavs, refs))
        lm, ls = losses(arts, wavs, ffn(refs))
        loss = lm + ls
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        acc.append((lm.item(), ls.item()))
        if step % 100 == 0:
            m = np.mean(acc, 0)
            acc = []
            log({"step": step, "mel": float(m[0]), "mrstft": float(m[1]), "lr": sched.get_last_lr()[0],
                 "sec": round(time.time() - t0)})
        if step % ckpt_every == 0 or step == steps:
            v = validate()
            log({"step": step, **v})
            torch.save({"ffn": ffn.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step},
                       out / "ckpt.pt.tmp")
            os.replace(out / "ckpt.pt.tmp", out / "ckpt.pt")
    sd = {k: v.cpu() for k, v in ffn.state_dict().items()}
    torch.save({"state_dict": sd, "layer": layer, "base": base, "steps": steps}, out / "final.pt.tmp")
    os.replace(out / "final.pt.tmp", out / "final.pt")
    print("done", flush=True)


# ---------------------------------------------------------------- eval

def _fetch_sv():
    """Local snapshot dir of microsoft/wavlm-base-plus-sv, downloading online (compute node) if absent; None if missing."""
    from huggingface_hub import snapshot_download

    rid = "microsoft/wavlm-base-plus-sv"
    try:
        return snapshot_download(rid, cache_dir=HF_CACHE, local_files_only=True)
    except Exception:
        pass
    code = ("from huggingface_hub import snapshot_download;"
            f"print(snapshot_download('{rid}', cache_dir='{HF_CACHE}'))")
    env = {**os.environ, "HF_HUB_OFFLINE": "0"}
    try:
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            return r.stdout.strip().splitlines()[-1]
        print("wavlm-base-plus-sv download failed:", r.stderr[-300:], flush=True)
    except Exception as e:
        print("wavlm-base-plus-sv download failed:", e, flush=True)
    return None


def evaluate(base, arms, per_spk, max_utt=None, device="cuda"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import jiwer  # noqa: F401
    import torch
    from huggingface_hub import snapshot_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from sparc.compression.sparc_multihead import _mcd
    from sparc.spk_encoder import SpeakerEncodingLayer

    wdir = snapshot_download("openai/whisper-large-v3", cache_dir=HF_CACHE, local_files_only=True)
    proc = WhisperProcessor.from_pretrained(wdir)
    asr = WhisperForConditionalGeneration.from_pretrained(wdir, torch_dtype=torch.float16).to(device).eval()
    from speechbrain.inference.speaker import EncoderClassifier
    edir = snapshot_download("speechbrain/spkrec-ecapa-voxceleb", cache_dir=HF_CACHE, local_files_only=True)
    ecapa = EncoderClassifier.from_hparams(source=edir, savedir=str(ROOT / "ecapa"), run_opts={"device": device},
                                           overrides={"pretrained_path": edir})
    sv2dir = _fetch_sv()
    sv2 = None
    if sv2dir:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        sv2_fe = AutoFeatureExtractor.from_pretrained(sv2dir)
        sv2 = WavLMForXVector.from_pretrained(sv2dir).to(device).eval()
    else:
        print("SECOND SV MODEL MISSING: SIM-2nd will be null", flush=True)

    def emb_ecapa(y):
        with torch.no_grad():
            return ecapa.encode_batch(torch.from_numpy(y)[None].to(device)).reshape(-1).cpu()

    def emb_sv2(y):
        if sv2 is None:
            return None
        inp = sv2_fe(y, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            return sv2(input_values=inp.input_values.to(device)).embeddings.reshape(-1).cpu()

    cos = lambda a, b: None if a is None or b is None else float(torch.nn.functional.cosine_similarity(a, b, dim=0))  # noqa: E731
    norm = proc.tokenizer.normalize

    def transcribe(y):
        feats = proc(y, sampling_rate=SR, return_tensors="pt").input_features.to(device, torch.float16)
        with torch.no_grad():
            ids = asr.generate(feats, language="en", task="transcribe")
        return proc.batch_decode(ids, skip_special_tokens=True)[0]

    def wer_of(text, hyp):
        if not text or not norm(text).strip():
            return None, 0
        return float(jiwer.wer(norm(text), norm(hyp))), len(norm(text).split())

    sets = eval_items(per_spk)
    spk_emb_cache = {}

    def ref_embs(y_key, y):
        if y_key not in spk_emb_cache:
            spk_emb_cache[y_key] = (emb_ecapa(y), emb_sv2(y))
        return spk_emb_cache[y_key]

    # records
    recs = {}
    for name, d in sets.items():
        recs[name] = {s: [r for r in (load_record(base, it) for it in v) if r is not None] for s, v in d.items()}
        recs[name] = {s: v for s, v in recs[name].items() if len(v) >= 2}
    texts = {}
    gen = None
    ROOT.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        if arm == "orig":
            fname = ROOT / "eval_orig.json"
        else:
            fname = ROOT / f"eval_{bname(base)}_{arm}.json"
        res = json.loads(fname.read_text()) if fname.exists() else {}
        ffn = None
        if arm not in ("orig", "en_ffn"):
            ck = torch.load(ROOT / f"{bname(base)}_L{arm}" / "final.pt", map_location="cpu", weights_only=False)
            ffn = SpeakerEncodingLayer(1024, 64)
            ffn.load_state_dict(ck["state_dict"])
            ffn = ffn.eval().to(device)
            layer = int(arm)
        if arm != "orig" and gen is None:
            _, gen = load_generator(base, device)

        def embed(ref):
            if arm == "en_ffn":
                return torch.from_numpy(ref["spk_emb_own"])[None].to(device)
            with torch.no_grad():
                return ffn(torch.from_numpy(ref["pooled"][layer].astype(np.float32))[None].to(device))

        def synth(r, e):
            art = torch.from_numpy(r["art"])[None].to(device)
            with torch.no_grad():
                return _synth(gen, art, e)[0].cpu().numpy().astype(np.float32)

        n_done = 0
        for name, d in recs.items():
            spks = sorted(d)
            for si, s in enumerate(spks):
                v = d[s]
                for j, r in enumerate(v):
                    key = f"{name}|{s}|{Path(r['item'][1]).stem}"
                    if key in res:
                        continue
                    if max_utt and n_done >= max_utt:
                        break
                    n_done += 1
                    y = r["wav"].astype(np.float32) / 32768.0
                    p = r["item"][1]
                    if p not in texts:
                        texts[p] = read_text(p)
                    text = texts[p]
                    row = {"set": name, "speaker": s}
                    if arm == "orig":
                        hyp = transcribe(y)
                        row["wer"], row["n_words"] = wer_of(text, hyp)
                        res[key] = row
                        continue
                    ref = v[pick_ref(len(v), j)]
                    yh = synth(r, embed(ref))
                    hyp = transcribe(yh)
                    row["wer"], row["n_words"] = wer_of(text, hyp)
                    e_o = ref_embs(("o", p), y)
                    e_h = (emb_ecapa(yh), emb_sv2(yh))
                    row["sim_ecapa"], row["sim_2nd"] = cos(e_o[0], e_h[0]), cos(e_o[1], e_h[1])
                    n = min(len(y), len(yh))  # both rescaled to unit std (generator output level is fixed)
                    row["mcd"] = _mcd(y[:n] / (y[:n].std() + 1e-8), yh[:n] / (yh[:n].std() + 1e-8))
                    # voice-conversion swap: Y's EMA/pitch/loudness, X's embedding
                    xs = spks[vc_target(spks, si)]
                    xv = d[xs]
                    xref, xother = xv[j % len(xv)], xv[(j + 1) % len(xv)]
                    yv = synth(r, embed(xref))
                    e_v = (emb_ecapa(yv), emb_sv2(yv))
                    xo = ref_embs(("o", xother["item"][1]), xother["wav"].astype(np.float32) / 32768.0)
                    row["vc_target"] = xs
                    row["vc_sim_tgt_ecapa"], row["vc_sim_tgt_2nd"] = cos(xo[0], e_v[0]), cos(xo[1], e_v[1])
                    row["vc_sim_src_ecapa"], row["vc_sim_src_2nd"] = cos(e_o[0], e_v[0]), cos(e_o[1], e_v[1])
                    row["vc_wer"], row["vc_n_words"] = wer_of(text, transcribe(yv))
                    res[key] = row
                    if len(res) % 50 == 0:
                        fname.write_text(json.dumps(res))
                        print(arm, len(res), flush=True)
        fname.write_text(json.dumps(res))
        print(f"{arm}: {len(res)} utterances -> {fname}", flush=True)
    return sv2 is not None


def summary(base, arms):
    L = [f"# Phase 2 speaker FFN refit, frozen `{base}` generator, unseen speakers\n",
         "Embedding always from ANOTHER utterance of the same speaker. SIM = cosine of the speaker embedding of the "
         "resynthesis to that of the original (ECAPA / WavLM-base-plus-SV). MCD: MFCC c1-c24, no DTW (compare arms "
         "only). WER %: whisper-large-v3, word-weighted. VC: Y's EMA/pitch/loudness with the embedding of the next "
         "speaker X; target SIM vs a different real X utterance, source SIM vs the original Y utterance (ECAPA / "
         "2nd model); VC WER vs Y's text. Means over utterances; SE over speakers for ECAPA SIM.\n"]
    orig_f = ROOT / "eval_orig.json"
    orig = json.loads(orig_f.read_text()) if orig_f.exists() else {}
    for name in EVAL_SETS:
        L.append(f"\n## {name}\n")
        L.append("| arm | n | SIM-ECAPA (SE) | SIM-2nd | MCD | WER | VC target SIM | VC source SIM | VC WER |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        rows0 = [r for k, r in orig.items() if r["set"] == name]
        if rows0:
            L.append(f"| original audio | {len(rows0)} | - | - | - | {wer_corpus(rows0):.2f} | - | - | - |")
        for arm in arms:
            f = ROOT / f"eval_{bname(base)}_{arm}.json"
            if not f.exists():
                continue
            rows = [r for r in json.loads(f.read_text()).values() if r["set"] == name]
            if not rows:
                continue

            def m(k):
                v = [r[k] for r in rows if r.get(k) is not None]
                return float(np.mean(v)) if v else float("nan")

            by = {}
            for r in rows:
                if r.get("sim_ecapa") is not None:
                    by.setdefault(r["speaker"], []).append(r["sim_ecapa"])
            sp = [np.mean(v) for v in by.values()]
            se = float(np.std(sp, ddof=1) / np.sqrt(len(sp))) if len(sp) > 1 else float("nan")
            vcw = wer_corpus([{"wer": r["vc_wer"], "n_words": r["vc_n_words"]} for r in rows if r.get("vc_wer") is not None])
            label = "shipped FFN (control)" if arm == "en_ffn" else f"refit layer {arm}"
            L.append(f"| {label} | {len(rows)} | {m('sim_ecapa'):.3f} ({se:.3f}) | {m('sim_2nd'):.3f} | {m('mcd'):.2f} | "
                     f"{wer_corpus(rows):.2f} | {m('vc_sim_tgt_ecapa'):.3f} / {m('vc_sim_tgt_2nd'):.3f} | "
                     f"{m('vc_sim_src_ecapa'):.3f} / {m('vc_sim_src_2nd'):.3f} | {vcw:.2f} |")
    (ROOT / f"summary_{bname(base)}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "refit", "eval", "summary"))
    ap.add_argument("--base", choices=("en", "en+"), default="en")
    ap.add_argument("--datasets", default="vctk,libri_dev,libri_test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="prep: only the smoke-test item set")
    ap.add_argument("--max-items", type=int, default=None, help="prep: only the first N items of the shard")
    ap.add_argument("--layer", default="9", help="refit: WavLM layer 0-24, or en_ffn for the control (no training)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None, help="refit: only the first N training items (smoke test)")
    ap.add_argument("--arms", default="en_ffn", help="eval/summary: comma list of en_ffn, layer numbers, orig (eval only)")
    ap.add_argument("--per-spk", type=int, default=10)
    ap.add_argument("--max-utt", type=int, default=None, help="eval: stop after N new utterances per arm (smoke test)")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.base, a.datasets.split(","), a.shard, a.nshards, a.max_items, a.smoke)
    elif a.cmd == "refit":
        layer = a.layer if a.layer == "en_ffn" else int(a.layer)
        refit(a.base, layer, a.steps, a.batch, a.frames, a.lr, a.ckpt_every, a.seed, a.max_train)
    elif a.cmd == "eval":
        evaluate(a.base, a.arms.split(","), a.per_spk, a.max_utt)
    else:
        summary(a.base, [x for x in a.arms.split(",") if x != "orig"])


if __name__ == "__main__":
    main()
