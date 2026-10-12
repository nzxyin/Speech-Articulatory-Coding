"""UTMOS22 (strong learner) mean opinion score predictor (EVALUATION.md section 4.2).

The model is the ``utmos22_strong`` entry of ``tarepan/SpeechMOS`` loaded through ``torch.hub`` (the release tag and
the sha256 of the downloaded weight file are pinned in ``cfg.eval_metrics.utmos``). It takes 16 kHz mono audio and runs
in float32. Utterances are scored one at a time: the network normalizes over the whole waveform in the first
convolution group and averages the frame scores over time, so zero padding in a batch changes the score.
"""

import hashlib
import logging
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

MIN_SAMPLES = 400  # receptive field of the wav2vec 2.0 convolutional front end


def file_sha256(path: str | Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            digest.update(chunk)
    return digest.hexdigest()


class UTMOS:
    """UTMOS22 strong predictor; ``cfg`` is ``cfg.eval_metrics.utmos``."""

    def __init__(self, cfg: DictConfig, device: str | torch.device):
        self.cfg = cfg
        self.device = torch.device(device)
        if cfg.hub_dir is not None:
            torch.hub.set_dir(str(cfg.hub_dir))
        self.model = torch.hub.load(str(cfg.hub_repo), str(cfg.entry), trust_repo=bool(cfg.trust_repo))
        self.model = self.model.float().to(self.device).eval()
        self.checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / str(cfg.checkpoint_file)
        self.checkpoint_sha256 = file_sha256(self.checkpoint) if self.checkpoint.is_file() else None
        expected = cfg.checkpoint_sha256
        if expected is not None and self.checkpoint_sha256 != str(expected):
            raise RuntimeError(
                f"UTMOS weights {self.checkpoint} have sha256 {self.checkpoint_sha256}, expected {expected}"
            )

    def provenance(self) -> dict[str, str | None]:
        """Model identity for the stage's ``meta.json``."""
        return {
            "hub_repo": str(self.cfg.hub_repo),
            "entry": str(self.cfg.entry),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @torch.inference_mode()
    def score_one(self, wav16: np.ndarray) -> float:
        """UTMOS of one 16 kHz mono waveform (float32, any length of at least ``MIN_SAMPLES``)."""
        wav = np.asarray(wav16, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError(f"expected a mono waveform, got shape {wav.shape}")
        if len(wav) < MIN_SAMPLES:
            raise ValueError(f"waveform of {len(wav)} samples is shorter than the {MIN_SAMPLES}-sample receptive field")
        if not np.isfinite(wav).all():
            raise ValueError("waveform contains non-finite samples")
        x = torch.from_numpy(np.ascontiguousarray(wav))[None].to(self.device)
        return float(self.model(x, int(self.cfg.sample_rate)).item())

    def score(self, wavs16: list[np.ndarray], errors: list[str] | None = None) -> list[float]:
        """UTMOS of each waveform, one at a time.

        Without ``errors`` a failing utterance raises. With a list, a failing utterance scores NaN and ``errors`` gets
        one entry per input (``""`` on success, otherwise the exception text), so the caller can record it.
        """
        scores = []
        for wav in wavs16:
            try:
                scores.append(self.score_one(wav))
                message = ""
            except (ValueError, RuntimeError) as err:
                if errors is None:
                    raise
                logger.warning("UTMOS failed: %s", err)
                scores.append(math.nan)
                message = f"{type(err).__name__}: {err}"
            if errors is not None:
                errors.append(message)
        return scores
