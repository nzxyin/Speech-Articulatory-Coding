import sys
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
import tqdm
from omegaconf import DictConfig

from sparc import load_model


def _pair_by_stem(ft_dir, spk_emb_dir):
    """Pair feature and speaker-embedding files by file stem; warn about unpaired files."""
    feats = {p.stem: p for p in ft_dir.glob("*.npy")}
    spk_embs = {p.stem: p for p in spk_emb_dir.glob("*.npy")}
    for stem in sorted(feats.keys() ^ spk_embs.keys()):
        where = ft_dir if stem in feats else spk_emb_dir
        print(f"Warning: skipping {stem}: present only in {where}", file=sys.stderr)
    return [(feats[s], spk_embs[s]) for s in sorted(feats.keys() & spk_embs.keys())]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    sparc_dir = Path(cfg.dataset.save_dir)
    save_dir = Path(cfg.dataset.decode_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    spk_emb_dir = sparc_dir / "spk_emb"
    ft_dir = sparc_dir / "emasrc"

    coder = load_model(cfg.model.model_name, config=cfg.model.config_path, device=cfg.device)

    for feat_path, spk_emb_path in tqdm.tqdm(_pair_by_stem(ft_dir, spk_emb_dir)):
        save_name = save_dir / (Path(str(feat_path).replace(str(ft_dir), "")).stem + ".wav")

        if save_name.exists():
            continue

        save_name.parent.mkdir(parents=True, exist_ok=True)

        try:
            feat = np.load(feat_path)
            spk_emb = np.load(spk_emb_path)
            wav = coder.decode(feat[:, :12], feat[:, 12], feat[:, 13], spk_emb)
            sf.write(save_name, wav, 16000)
        except Exception as e:
            print(f"Error processing {feat_path}, {spk_emb_path}: {e}")


if __name__ == "__main__":
    main()
