"""Speaker embeddings for the similarity metrics (EVALUATION.md section 4.4).

Two models, both on 16 kHz mono audio and in float32: ECAPA-TDNN (``speechbrain/spkrec-ecapa-voxceleb``, 192-d, the
primary model) and WavLM-SV (``microsoft/wavlm-base-plus-sv``, ``WavLMForXVector``, 512-d). Revisions are pinned in
``cfg.eval_metrics.speaker``. Utterances are embedded one at a time: both networks normalize over the whole input
(sentence-level mean/variance normalization, group norm), so padding in a batch would change the embedding.
"""

import logging
import math
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

NAMES = ("ecapa", "wavlm_sv")
MIN_SAMPLES = 4800  # 0.3 s; the x-vector TDNN stack needs about 15 frames of 20 ms


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rows (or the vector) of ``x`` divided by their Euclidean norm."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity along the last axis, broadcasting ``a`` against ``b``.

    ``(n, d)`` with ``(n, d)`` gives ``(n,)`` (row by row), ``(n, d)`` with ``(d,)`` gives ``(n,)`` and two vectors give
    a scalar array. The inputs need not be normalized.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.sum(l2_normalize(a) * l2_normalize(b), axis=-1)


class SpeakerEmbedder:
    """Per-utterance speaker embedder; ``name`` is ``"ecapa"`` or ``"wavlm_sv"``, ``cfg`` is ``cfg.eval_metrics.speaker``."""

    def __init__(self, name: str, cfg: DictConfig, device: str | torch.device):
        if name not in NAMES:
            raise ValueError(f"speaker model must be one of {NAMES}, got {name!r}")
        self.name = name
        self.cfg = cfg
        self.model_cfg = cfg[name]
        self.device = torch.device(device)
        self.sample_rate = int(cfg.sample_rate)
        self.cache_dir = None if cfg.cache_dir is None else str(cfg.cache_dir)
        if name == "ecapa":
            self._load_ecapa()
        else:
            self._load_wavlm_sv()
        if self.dim != int(self.model_cfg.dim):
            raise RuntimeError(f"{name} embeds into {self.dim} dimensions, config says {self.model_cfg.dim}")

    def _load_ecapa(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        m = self.model_cfg
        self.snapshot = Path(
            snapshot_download(
                str(m.repo_id),
                revision=str(m.revision),
                cache_dir=self.cache_dir,
                allow_patterns=list(m.allow_patterns),
            )
        )
        # the pinned snapshot is its own save directory: speechbrain finds every file there and writes nothing
        self.model = EncoderClassifier.from_hparams(
            source=str(self.snapshot), savedir=str(self.snapshot), run_opts={"device": str(self.device)}
        )
        self.model.eval()
        probe = np.random.default_rng(0).standard_normal(self.sample_rate).astype(np.float32) * 0.01
        self.dim = int(self.embed_one(probe).shape[-1])

    def _load_wavlm_sv(self) -> None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        m = self.model_cfg
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(m.repo_id), revision=str(m.revision), cache_dir=self.cache_dir
        )
        self.model = WavLMForXVector.from_pretrained(str(m.repo_id), revision=str(m.revision), cache_dir=self.cache_dir)
        self.model = self.model.float().to(self.device).eval()
        self.dim = int(self.model.config.xvector_output_dim)

    def provenance(self) -> dict[str, str]:
        """Model identity for the stage's ``meta.json``."""
        return {"name": self.name, "repo_id": str(self.model_cfg.repo_id), "revision": str(self.model_cfg.revision)}

    @torch.inference_mode()
    def embed_one(self, wav16: np.ndarray) -> np.ndarray:
        """Raw (not yet normalized) embedding ``[dim]`` of one 16 kHz mono waveform."""
        wav = np.asarray(wav16, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError(f"expected a mono waveform, got shape {wav.shape}")
        if len(wav) < MIN_SAMPLES:
            raise ValueError(f"waveform of {len(wav)} samples is shorter than {MIN_SAMPLES} samples")
        if not np.isfinite(wav).all():
            raise ValueError("waveform contains non-finite samples")
        if self.name == "ecapa":
            x = torch.from_numpy(wav)[None].to(self.device)
            return self.model.encode_batch(x, normalize=False)[0, 0].float().cpu().numpy()
        inputs = self.feature_extractor(wav, sampling_rate=self.sample_rate, return_tensors="pt")
        out = self.model(input_values=inputs["input_values"].to(self.device))
        return out.embeddings[0].float().cpu().numpy()

    def embed(self, wavs16: list[np.ndarray], errors: list[str] | None = None) -> np.ndarray:
        """L2-normalized embeddings ``[n, dim]`` (float32), one waveform at a time.

        Without ``errors`` a failing utterance raises. With a list, its row is NaN and ``errors`` gets one entry per
        input (``""`` on success, otherwise the exception text).
        """
        rows = np.full((len(wavs16), self.dim), np.nan, dtype=np.float32)
        for i, wav in enumerate(wavs16):
            message = ""
            try:
                rows[i] = l2_normalize(self.embed_one(wav).astype(np.float64))
            except (ValueError, RuntimeError) as err:
                if errors is None:
                    raise
                logger.warning("%s embedding failed: %s", self.name, err)
                message = f"{type(err).__name__}: {err}"
            if errors is not None:
                errors.append(message)
        return rows


def mean_embedding(embeddings: np.ndarray) -> np.ndarray:
    """Mean of L2-normalized embeddings (rows); NaN rows are ignored. NaN vector if no row is finite."""
    rows = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    rows = rows[np.isfinite(rows).all(axis=1)]
    return rows.mean(axis=0) if len(rows) else np.full(embeddings.shape[-1], math.nan)
