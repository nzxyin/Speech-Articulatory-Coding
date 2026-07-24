import io
from pathlib import Path

import hydra
import librosa
import numpy as np
import soundfile as sf
import tqdm
from omegaconf import DictConfig

from sparc import load_model


def _wavdir_items(wav_dir):
    wav_dir = Path(wav_dir)
    wav_files = list(wav_dir.glob("**/*.flac")) + list(wav_dir.glob("**/*.wav"))
    for wav_file in wav_files:
        name = Path(str(wav_file).replace(str(wav_dir), "")).stem
        yield name, wav_file


def _parquet_items(data_dir, glob, target_sr):
    from datasets import Audio, load_dataset

    for shard_path in sorted(Path(data_dir).glob(glob)):
        ds = load_dataset("parquet", data_files=str(shard_path), split="train")
        ds = ds.cast_column("audio", Audio(decode=False))
        for row_idx, row in enumerate(ds):
            wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            if wav.ndim > 1:
                wav = wav.mean(-1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            yield f"{shard_path.stem}-{row_idx:06d}", wav


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    save_dir = Path(cfg.dataset.save_dir)
    spk_emb_save_dir = save_dir / "spk_emb"
    spk_emb_save_dir.mkdir(parents=True, exist_ok=True)
    ft_save_dir = save_dir / "emasrc"
    ft_save_dir.mkdir(parents=True, exist_ok=True)

    coder = load_model(cfg.model.model_name, config=cfg.model.config_path, device=cfg.device)

    if cfg.dataset.get("format", "wavdir") == "parquet":
        items = _parquet_items(cfg.dataset.data_dir, cfg.dataset.get("glob", "**/*.parquet"), coder.sr)
    else:
        items = _wavdir_items(cfg.dataset.wav_dir)

    for name, audio in tqdm.tqdm(items):
        ft_save_path = ft_save_dir / f"{name}.npy"
        spk_emb_save_path = spk_emb_save_dir / f"{name}.npy"

        if spk_emb_save_path.exists():
            continue

        ft_save_path.parent.mkdir(parents=True, exist_ok=True)
        spk_emb_save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            outputs = coder.encode(audio, concat=True)
            np.save(ft_save_path, outputs["features"])
            np.save(spk_emb_save_path, outputs["spk_emb"])
        except Exception as e:
            print(f"Error processing {name}: {e}")


if __name__ == "__main__":
    main()
