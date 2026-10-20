import hashlib
import io
import json
import os
import sys
import zlib
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


def _atomic_save(path, array):
    # The temp name must not end in ".npy": consumers glob "*.npy" in these directories.
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, array)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


META_NAME = "encode_meta.json"
_INFO_ONLY_KEYS = {"linear_model_path"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_meta(cfg):
    """The settings that determine the cached outputs."""
    linear_model_path = cfg.model.get("linear_model_path")
    if linear_model_path is not None and not Path(linear_model_path).is_file():
        sys.exit(f"Error: model.linear_model_path={linear_model_path} does not exist")
    return {
        "model_name": str(cfg.model.model_name),
        "config_path": None if cfg.model.get("config_path") is None else str(cfg.model.config_path),
        "linear_model_path": None if linear_model_path is None else str(linear_model_path),
        "linear_model_sha256": None if linear_model_path is None else _sha256(linear_model_path),
        "deterministic_pitch": bool(cfg.get("deterministic_pitch", True)),
    }


def _check_meta(save_dir, meta, allow_mismatch=False):
    """Compare `meta` with save_dir/encode_meta.json; write it only for a directory without outputs."""
    meta_path = save_dir / META_NAME
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        # the head is identified by its sha256; the same file under another path is not a mismatch
        keys = sorted((set(old) | set(meta)) - _INFO_ONLY_KEYS)
        diff = {k: (old.get(k), meta.get(k)) for k in keys if old.get(k) != meta.get(k)}
        if diff:
            msg = (f"{meta_path} does not match the current settings (key: (stored, current)): {diff}. "
                   "Use a different dataset.save_dir, or set allow_meta_mismatch=true to mix outputs.")
            if not allow_mismatch:
                sys.exit(f"Error: {msg}")
            print(f"Warning: {msg}", file=sys.stderr)
        return
    has_outputs = any(save_dir.glob("emasrc/**/*.npy")) or any(save_dir.glob("spk_emb/**/*.npy"))
    if has_outputs:
        print(f"Warning: {save_dir} already has outputs but no {META_NAME}; cannot check that they were "
              "produced with the current settings.", file=sys.stderr)
        return
    tmp_path = meta_path.with_name(f".{META_NAME}.{os.getpid()}.tmp")  # concurrent array tasks
    tmp_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, meta_path)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


def run(cfg: DictConfig) -> None:
    save_dir = Path(cfg.dataset.save_dir)
    spk_emb_save_dir = save_dir / "spk_emb"
    spk_emb_save_dir.mkdir(parents=True, exist_ok=True)
    ft_save_dir = save_dir / "emasrc"
    ft_save_dir.mkdir(parents=True, exist_ok=True)
    _check_meta(save_dir, _encode_meta(cfg), allow_mismatch=bool(cfg.get("allow_meta_mismatch", False)))

    linear_model_path = cfg.model.get("linear_model_path")
    print(f"save_dir={save_dir} model={cfg.model.model_name} config_path={cfg.model.config_path} "
          f"linear_model_path={linear_model_path} (null = head embedded in the checkpoint) "
          f"deterministic_pitch={cfg.get('deterministic_pitch', True)}", file=sys.stderr)
    print("Note: utterances whose outputs already exist in save_dir are skipped, regardless of "
          "the model/head used to produce them.", file=sys.stderr)

    load_kwargs = {}
    if linear_model_path is not None:
        load_kwargs["linear_model_path"] = linear_model_path
    coder = load_model(cfg.model.model_name, config=cfg.model.config_path, device=cfg.device,
                       **load_kwargs)

    if cfg.dataset.get("format", "wavdir") == "parquet":
        items = _parquet_items(cfg.dataset.data_dir, cfg.dataset.get("glob", "**/*.parquet"), coder.sr)
    else:
        items = _wavdir_items(cfg.dataset.wav_dir)

    n_failed = 0
    for name, audio in tqdm.tqdm(items):
        ft_save_path = ft_save_dir / f"{name}.npy"
        spk_emb_save_path = spk_emb_save_dir / f"{name}.npy"

        if ft_save_path.exists() and spk_emb_save_path.exists():
            continue

        ft_save_path.parent.mkdir(parents=True, exist_ok=True)
        spk_emb_save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            seed = zlib.crc32(name.encode()) & 0x7FFFFFFF if cfg.get("deterministic_pitch", True) else None
            outputs = coder.encode(audio, concat=True, seed=seed)
            _atomic_save(ft_save_path, outputs["features"])
            _atomic_save(spk_emb_save_path, outputs["spk_emb"])
        except Exception as e:
            n_failed += 1
            print(f"Error processing {name}: {type(e).__name__}: {e}", file=sys.stderr)

    if n_failed:
        print(f"{n_failed} utterance(s) failed to encode.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
