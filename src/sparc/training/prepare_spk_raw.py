# Precomputes the raw (pre-FFN) 1024-dim pooled WavLM speaker feature for
# every utterance in an already-SPARC-encoded dataset directory.
#
# The existing `spk_emb/*.npy` caches produced by `sparc-encode` (see
# src/sparc/cli/encode.py) already have the pretrained speaker FFN applied
# (64-dim). Training a *fresh* speaker-encoder FFN from scratch (per the
# paper's training methodology) needs the FFN's raw input instead, which
# this script computes via load_model("feature_extraction") -- the same
# frozen WavLM-inversion pipeline, just with no speaker FFN or generator
# attached, so SpeakerEncoder returns the periodicity-weighted pooled
# WavLM feature untouched (see spk_encoder.py:_get_spk_emb).
#
# ema/pitch/loudness/periodicity are NOT recomputed here -- they are
# identical regardless of which trained speaker-FFN/generator checkpoint
# was used (those don't affect the frozen Inversion/SourceExtractor path),
# so the existing emasrc/*.npy caches remain valid training targets.

import sys
from pathlib import Path

import numpy as np
import tqdm

from sparc import load_model


def main(wav_dir, sparc_dir, device="cuda:0", limit=None):
    wav_dir = Path(wav_dir)
    sparc_dir = Path(sparc_dir)
    ft_dir = sparc_dir / "emasrc"
    out_dir = sparc_dir / "spk_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = load_model("feature_extraction", device=device)

    stems = [p.stem for p in ft_dir.glob("*.npy")]
    if limit is not None:
        stems = stems[:limit]
    wav_index = {}
    for ext in ("*.wav", "*.flac"):
        for p in wav_dir.glob(f"**/{ext}"):
            wav_index[p.stem] = p

    for stem in tqdm.tqdm(stems):
        out_path = out_dir / f"{stem}.npy"
        if out_path.exists():
            continue
        wav_path = wav_index.get(stem)
        if wav_path is None:
            print(f"no wav found for {stem}, skipping")
            continue
        try:
            outputs = extractor.encode(wav_path, split_batch=True, reduce=True, concat=False)
            np.save(out_path, outputs["spk_emb"].astype(np.float32))
        except Exception as e:
            print(f"Error processing {stem}: {e}")


if __name__ == "__main__":
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    main(sys.argv[1], sys.argv[2], device=sys.argv[3] if len(sys.argv) > 3 else "cuda:0", limit=limit)
