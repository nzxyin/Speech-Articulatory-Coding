"""Cache every layer's hidden states of an SSL model over MNGU0, for per-layer probing.

Layout under <out_root>/<model>/:
    layer_XX.npy  -- float16 (N_frames, D) memmap; XX = 0 is the transformer input (after the
                     positional convolution), XX = k the output of transformer layer k
                     (hidden_states[k], so the last layer includes the final LayerNorm)
    index.json    -- {stem: [start, n_frames]} rows of each untrimmed utterance in every layer file
    meta.json     -- model id, layer count, dim, low-pass setting, completion flag

Features are low-pass filtered along time (zero-phase Butterworth, FREQCUT Hz) over the whole
untrimmed utterance, as in SPARC's linear inversion model, before alignment/trimming.

Usage: python -m sparc.compression.extract <model> [out_root] [--limit N]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sparc.inversion import butter_bandpass_filter

from . import mngu0
from .encoders import MODELS, frame_lengths, load_full, normalize_wav

OUT_ROOT = Path("/data/user_data/xoy/xlsr_ema/features")
FREQCUT = 10  # Hz


def lowpass(x):
    return butter_bandpass_filter(x[None], FREQCUT, mngu0.FT_SR, axis=1)[0]


@torch.no_grad()
def extract(name, out_root=OUT_ROOT, limit=None, device="cuda"):
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("complete") and limit is None:
        print(f"{out} already complete")
        return out

    stems = mngu0.available_stems()
    if limit:
        stems = [s for s in stems if mngu0.split_of(s) != "train"][:limit] + [
            s for s in stems if mngu0.split_of(s) == "train"
        ][:limit]
    model = load_full(name).to(device)
    L, D = model.config.num_hidden_layers, model.config.hidden_size

    lengths = {s: int(frame_lengths(model, sf.info(mngu0.WAV_DIR / f"{s}.wav").frames)) for s in stems}
    index, start = {}, 0
    for s in stems:
        index[s] = [start, lengths[s]]
        start += lengths[s]
    total = start
    mmaps = [
        np.lib.format.open_memmap(out / f"layer_{k:02d}.npy", mode="w+", dtype=np.float16, shape=(total, D))
        for k in range(L + 1)
    ]
    print(f"{name}: {len(stems)} utterances, {total} frames, {L} layers x {D} dims")

    for i, s in enumerate(stems):
        utt = mngu0.load_utterance(s)
        wav = torch.from_numpy(normalize_wav(utt.wav)).unsqueeze(0).to(device)
        hs = model(wav, output_hidden_states=True).hidden_states
        a, n = index[s]
        for k in range(L + 1):
            h = hs[k][0].float().cpu().numpy()
            assert len(h) == n, (s, len(h), n)
            h16 = lowpass(h).astype(np.float16)
            if not np.isfinite(h16).all():
                raise OverflowError(f"{s} layer {k}: hidden state exceeds float16 range")
            mmaps[k][a : a + n] = h16
        if (i + 1) % 100 == 0:
            print(f"...{i + 1}/{len(stems)}", flush=True)

    for m in mmaps:
        m.flush()
    (out / "index.json").write_text(json.dumps(index))
    meta_path.write_text(
        json.dumps(
            {"model": MODELS[name], "num_layers": L, "dim": D, "freqcut": FREQCUT, "n_frames": total,
             "n_utts": len(stems), "complete": limit is None}
        )
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("out_root", nargs="?", default=str(OUT_ROOT))
    ap.add_argument("--limit", type=int, default=None, help="debug: N train + N val/test utterances")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    extract(args.model, args.out_root, args.limit, args.device)


if __name__ == "__main__":
    main()
