# Training-time data pipeline for the HiFi-GAN vocoder + speaker-encoder
# reproduction. Per the paper (Appendix B.7): "a random 320 ms window is
# sampled from each clip in a batch." Reuses the already-precomputed
# emasrc/*.npy (ema+pitch+loudness+periodicity, 50Hz) caches produced by
# `sparc-encode` as training targets/conditioning -- those are valid
# regardless of which trained speaker-FFN/generator checkpoint originally
# produced them, since the frozen Inversion/SourceExtractor path they come
# from doesn't depend on the (separately trained) vocoder. The raw
# (pre-FFN) speaker feature comes from spk_raw/*.npy, produced by
# prepare_spk_raw.py, and is a single pooled vector per utterance (not
# per-frame), so it's shared across all crops of that utterance.

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class VocoderDataset(Dataset):
    def __init__(self, wav_dir, sparc_dir, segment_frames=16, sr=16000, ft_sr=50, min_frames=32):
        self.wav_dir = Path(wav_dir)
        self.ft_dir = Path(sparc_dir) / "emasrc"
        self.spk_raw_dir = Path(sparc_dir) / "spk_raw"
        self.hop = sr // ft_sr
        self.segment_frames = segment_frames
        self.segment_samples = segment_frames * self.hop
        self.sr = sr

        wav_index = {}
        for ext in ("*.wav", "*.flac"):
            for p in self.wav_dir.glob(f"**/{ext}"):
                wav_index[p.stem] = p

        self.items = []
        for ft_path in sorted(self.ft_dir.glob("*.npy")):
            stem = ft_path.stem
            spk_raw_path = self.spk_raw_dir / f"{stem}.npy"
            wav_path = wav_index.get(stem)
            if wav_path is None or not spk_raw_path.exists():
                continue
            n_frames = np.load(ft_path, mmap_mode="r").shape[0]
            if n_frames < min_frames:
                continue
            self.items.append((stem, wav_path, ft_path, spk_raw_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem, wav_path, ft_path, spk_raw_path = self.items[idx]
        feat = np.load(ft_path)  # (T, 15): ema(12), pitch(1), loudness(1), periodicity(1)
        spk_raw = np.load(spk_raw_path).astype(np.float32)  # (1024,)

        wav, wav_sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(-1)
        if wav_sr != self.sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=wav_sr, target_sr=self.sr)

        n_frames = feat.shape[0]
        max_start = max(n_frames - self.segment_frames, 0)
        start = random.randint(0, max_start) if max_start > 0 else 0
        feat_crop = feat[start:start + self.segment_frames]
        if feat_crop.shape[0] < self.segment_frames:
            pad = self.segment_frames - feat_crop.shape[0]
            feat_crop = np.pad(feat_crop, ((0, pad), (0, 0)), mode="edge")

        audio_start = start * self.hop
        audio_crop = wav[audio_start:audio_start + self.segment_samples]
        if len(audio_crop) < self.segment_samples:
            audio_crop = np.pad(audio_crop, (0, self.segment_samples - len(audio_crop)))

        return {
            "art": feat_crop[:, :14].astype(np.float32),  # ema+pitch+loudness -> generator input
            "spk_raw": spk_raw,
            "audio": audio_crop.astype(np.float32),
            "stem": stem,
        }


def collate(batch):
    return {
        "art": torch.from_numpy(np.stack([b["art"] for b in batch])),  # (B, T, 14)
        "spk_raw": torch.from_numpy(np.stack([b["spk_raw"] for b in batch])),  # (B, 1024)
        "audio": torch.from_numpy(np.stack([b["audio"] for b in batch])),  # (B, T*hop)
        "stem": [b["stem"] for b in batch],
    }
