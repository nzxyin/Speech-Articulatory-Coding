# Synthesizes standalone wav files from a trained vocoder checkpoint, for
# qualitative listening. Uses the generator + speaker-encoder FFN weights
# from a Lightning checkpoint produced by `sparc-train`; everything else
# (ema/pitch/loudness targets, raw speaker feature) comes from an
# already-SPARC-encoded dataset's cached features, same as training.

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sparc.spk_encoder import SpeakerEncodingLayer
from sparc.generator import HiFiGANGenerator
from sparc.training.lightning_module import DEFAULT_GENERATOR_CONFIG


def load_trained(ckpt_path, device="cuda:0"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"]

    generator = HiFiGANGenerator(**DEFAULT_GENERATOR_CONFIG)
    gen_sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    generator.load_state_dict(gen_sd)
    generator.remove_weight_norm()
    generator = generator.eval().to(device)

    speaker_ffn = SpeakerEncodingLayer(spk_ft_size=1024, spk_emb_size=64)
    ffn_sd = {k[len("speaker_ffn."):]: v for k, v in sd.items() if k.startswith("speaker_ffn.")}
    speaker_ffn.load_state_dict(ffn_sd)
    speaker_ffn = speaker_ffn.eval().to(device)

    return generator, speaker_ffn


def synthesize(generator, speaker_ffn, ft_path, spk_raw_path, device="cuda:0"):
    feat = np.load(ft_path)  # (T, 15)
    spk_raw = np.load(spk_raw_path).astype(np.float32)  # (1024,)
    with torch.no_grad():
        art = torch.from_numpy(feat[:, :14]).float().unsqueeze(0).transpose(1, 2).to(device)  # (1,14,T)
        spk_raw_t = torch.from_numpy(spk_raw).float().unsqueeze(0).to(device)
        spk_emb = speaker_ffn(spk_raw_t)
        wav = generator(art, spk_emb)[0, 0].cpu().numpy()
    return wav


def main(ckpt_path, sparc_dir, out_dir, n=8, device="cuda:0"):
    sparc_dir = Path(sparc_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator, speaker_ffn = load_trained(ckpt_path, device=device)

    ft_dir = sparc_dir / "emasrc"
    spk_raw_dir = sparc_dir / "spk_raw"
    stems = sorted(p.stem for p in ft_dir.glob("*.npy") if (spk_raw_dir / f"{p.stem}.npy").exists())
    rng = np.random.default_rng(0)
    sample_stems = rng.choice(stems, size=min(n, len(stems)), replace=False)

    for stem in sample_stems:
        wav = synthesize(generator, speaker_ffn, ft_dir / f"{stem}.npy", spk_raw_dir / f"{stem}.npy", device=device)
        sf.write(out_dir / f"{stem}.wav", wav, 16000)
        print(f"wrote {out_dir / f'{stem}.wav'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], n=int(sys.argv[4]) if len(sys.argv) > 4 else 8,
         device=sys.argv[5] if len(sys.argv) > 5 else "cuda:0")
