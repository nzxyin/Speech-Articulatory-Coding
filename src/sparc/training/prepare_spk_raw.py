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
#
# WARNING: load_model("feature_extraction") is NOT the en+ feature definition. It pools WavLM
# layer 0 with different periodicity settings (pitch_q 4, no threshold, no loudness gate), whereas
# en+ pools hidden_states[6] with thresholded periodicity. Features from that default path cannot
# be fed to the pretrained en+ speaker FFN. Pass --en-plus-compatible to compute the en+ pooled
# feature instead (written to spk_raw_enplus/ so it never mixes with an existing spk_raw/).

import argparse
import warnings
from pathlib import Path

import numpy as np
import tqdm

from sparc import load_model


def main(wav_dir, sparc_dir, device="cuda:0", limit=None, en_plus_compatible=False):
    wav_dir = Path(wav_dir)
    sparc_dir = Path(sparc_dir)
    ft_dir = sparc_dir / "emasrc"
    out_dir = sparc_dir / ("spk_raw_enplus" if en_plus_compatible else "spk_raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    if en_plus_compatible:
        extractor = load_model("en+", device=device)
        # Drop the pretrained speaker FFN so the pooled 1024-dim layer-6 feature is returned raw.
        extractor.speaker_encoder.spk_enc = None
    else:
        warnings.warn("Computing the feature_extraction speaker feature (WavLM layer 0), which is "
                      "not compatible with the pretrained en+ speaker FFN; "
                      "use --en-plus-compatible for the en+ feature.")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_dir")
    parser.add_argument("sparc_dir")
    parser.add_argument("device", nargs="?", default="cuda:0")
    parser.add_argument("limit", nargs="?", type=int, default=None)
    parser.add_argument("--en-plus-compatible", action="store_true",
                        help="pool WavLM layer 6 with en+'s thresholded periodicity (the feature the "
                             "pretrained en+ speaker FFN expects); output goes to spk_raw_enplus/")
    args = parser.parse_args()
    main(args.wav_dir, args.sparc_dir, device=args.device, limit=args.limit,
         en_plus_compatible=args.en_plus_compatible)
