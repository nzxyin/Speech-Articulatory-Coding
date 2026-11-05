"""Cache every layer's hidden states of a Whisper encoder, same layout as `extract.py` (docs/whisper_probe_plan.md).

Layout under <out_root>/<model>/: layer_XX.npy (N_frames, D), index.json {stem: [start, n_w]}, meta.json. Layer 0 is the
encoder input after the convolutions and positional embedding, layer k the residual stream after layer k, the last layer
includes the final LayerNorm. `probe.py` and `depth_sweep.py` read this cache unchanged.

Differences from `extract.py`:
  * each utterance is zero-padded to the 30 s window; only the first n_w = ceil(n / 320) frames (centre inside the audio)
    are kept, and the 10 Hz low-pass runs over those frames only, so the filter never sees the padding frames;
  * the waveform is not z-scored (Whisper's own log-mel front end is used);
  * layers are stored float16; a layer that overflows float16 is restarted as float32 (listed in meta.json "fp32_layers");
  * the run resumes after an interruption (progress.json), and a finished cache is not redone.

Usage: python -m sparc.compression.whisper_extract <model> [--dataset mngu0|ema_multi] [--batch 8] [--limit N]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import datasets
from .extract import FREQCUT, lowpass
from .whisper_encoders import (
    WHISPER, WINDOW_SAMPLES, hidden_states, load_bundle, log_mel, n_real_frames,
)

FP16_MAX = 65504.0
MAX_SHIFT = 0  # shortfall is measured at shift 0 (see Amendment 1 of the plan): even WavLM is short for 14% of ema_multi at shift 0
CHECK_EVERY = 10  # batches between progress saves


class Fp16Overflow(Exception):
    def __init__(self, layer):
        super().__init__(f"layer {layer} exceeds the float16 range")
        self.layer = layer


def slice_and_filter(h, n_w, raw=False):
    """(1500, D) padded-window hidden state -> (n_w, D) float32: real frames first, then the zero-phase low-pass."""
    x = np.asarray(h[:n_w], dtype=np.float32)
    return x if raw else lowpass(x).astype(np.float32)


def frames_needed(utt, max_shift=MAX_SHIFT):
    """Frames the probe may read for this utterance: base offset + largest shift + target length."""
    return int(utt.base_offset) + max_shift + len(utt.ema_mm)


def _state(out):
    p = out / "fp32_layers.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def _open(out, keep, fp32, total, D, resume):
    mm = {}
    for k in keep:
        dt = np.float32 if k in fp32 else np.float16
        f = out / f"layer_{k:02d}.npy"
        if resume and f.exists():
            m = np.load(f, mmap_mode="r+")
            if m.shape == (total, D) and m.dtype == dt:
                mm[k] = m
                continue
        mm[k] = np.lib.format.open_memmap(f, mode="w+", dtype=dt, shape=(total, D))
    return mm


def _pass(ds, stems, index, total, D, keep, enc, fe, out, fp32, batch, device, raw, fp16_limit, resume, pool, check):
    progress_f = out / "progress.json"
    done = 0
    if resume and progress_f.exists():
        done = json.loads(progress_f.read_text()).get("done", 0)
    mm = _open(out, keep, fp32, total, D, resume and done > 0)
    if not (resume and done > 0):
        done = 0
    short = worst = 0
    for b0 in range(done, len(stems), batch):
        chunk = stems[b0 : b0 + batch]
        utts = [ds.load_utterance(s) for s in chunk]
        for s, u in zip(chunk, utts):
            assert len(u.wav) == ds.n_samples(s) <= WINDOW_SAMPLES, (s, len(u.wav))
            assert n_real_frames(len(u.wav)) == index[s][1], s
        mel = log_mel(fe, [u.wav for u in utts], device=device)
        with torch.no_grad():
            hs = hidden_states(enc, mel.to(device=device, dtype=next(enc.parameters()).dtype))
        for j, (s, u) in enumerate(zip(chunk, utts)):
            a, n = index[s]
            if check and n < frames_needed(u):
                worst = max(worst, frames_needed(u) - n)
                short += 1
            rows = [hs[k][j, :n].float().cpu().numpy() for k in keep]
            filt = list(pool.map(lambda x: slice_and_filter(x, n, raw), rows))
            for k, f in zip(keep, filt):
                if k in fp32:
                    mm[k][a : a + n] = f
                    continue
                if not np.isfinite(f).all() or np.abs(f).max() > fp16_limit:
                    raise Fp16Overflow(k)
                mm[k][a : a + n] = f.astype(np.float16)
        if ((b0 // batch) + 1) % CHECK_EVERY == 0:
            for m in mm.values():
                m.flush()
            progress_f.write_text(json.dumps({"done": b0 + len(chunk)}))
            print(f"...{b0 + len(chunk)}/{len(stems)}", flush=True)
    for m in mm.values():
        m.flush()
    return short, worst


def extract_whisper(name, out_root=None, limit=None, device="cuda", dataset="mngu0", layers=None, raw=False,
                    batch=8, bundle=None, ds=None, fp16_limit=FP16_MAX, threads=8):
    ds = ds or datasets.get(dataset)
    root = Path(out_root) if out_root else datasets.features_root(dataset)
    if raw and not out_root:
        root = root.with_name(root.name + "_raw")
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("complete") and limit is None:
        print(f"{out} already complete")
        return out
    stems = ds.available_stems()
    if limit:
        sp = ds.splits()
        stems = sorted(set(sp["train"][:limit]) | {s for k, v in sp.items() if k != "train" for s in v[:limit]})
    enc, fe = bundle if bundle is not None else load_bundle(name)
    enc = enc.to(device)
    L, D = len(enc.layers), enc.config.d_model
    index, start = {}, 0
    for s in stems:
        n = n_real_frames(ds.n_samples(s))
        index[s] = [start, n]
        start += n
    total = start
    keep = list(range(L + 1)) if layers is None else list(layers)
    fp32 = _state(out)
    fp32_path = out / "fp32_layers.json"
    # a limited run or a changed layer set is a different cache: never resume it
    resume = limit is None and layers is None
    idx_path = out / "index.json"
    if resume and idx_path.exists() and json.loads(idx_path.read_text()) != index:
        resume = False
    if not resume:
        (out / "progress.json").unlink(missing_ok=True)
    idx_path.write_text(json.dumps(index))
    print(f"{name}: {len(stems)} utterances, {total} frames, layers {keep} x {D} dims, raw={raw}, fp32 layers {sorted(fp32)}")
    with ThreadPoolExecutor(threads) as pool:
        while True:
            try:
                short, worst = _pass(ds, stems, index, total, D, keep, enc, fe, out, fp32, batch, device, raw, fp16_limit,
                              resume, pool, check=True)
                break
            except Fp16Overflow as e:
                print(f"layer {e.layer} overflows float16; restarting it as float32", flush=True)
                fp32.add(e.layer)
                fp32_path.write_text(json.dumps(sorted(fp32)))
                (out / "progress.json").unlink(missing_ok=True)
                resume = False
    (out / "progress.json").unlink(missing_ok=True)
    meta_path.write_text(json.dumps({
        "model": WHISPER[name], "num_layers": L, "dim": D, "freqcut": None if raw else FREQCUT, "n_frames": total,
        "n_utts": len(stems), "layers": keep, "complete": limit is None and layers is None, "window_seconds": 30,
        "padded": True, "n_mels": int(fe.feature_size), "fp32_layers": sorted(fp32),
        "short_at_shift0_last_run": short, "max_shortfall_shift0_last_run": worst,
    }))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(WHISPER))
    ap.add_argument("out_root", nargs="?", default=None)
    ap.add_argument("--dataset", default="mngu0", choices=datasets.DATASET_NAMES)
    ap.add_argument("--limit", type=int, default=None, help="debug: N train + N val/test utterances")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layers", default=None, help="comma-separated layer indices to write (default: all)")
    ap.add_argument("--raw", action="store_true", help="no low-pass; default root <features root>_raw")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    extract_whisper(args.model, args.out_root, args.limit, args.device, args.dataset, layers, args.raw, args.batch,
                    threads=args.threads)


if __name__ == "__main__":
    main()
