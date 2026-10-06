"""Whisper large-v3 transcription and word/character error counts (EVALUATION.md section 4.3).

``WhisperASR`` decodes 16 kHz audio greedily (English, transcribe, no prompt) in length-sorted batches and uses
sequential long-form decoding for inputs over 30 s. Reference and hypothesis go through Whisper's
``EnglishTextNormalizer`` with the English spelling map of the model repository before the Levenshtein counts of
``edit_counts``. Word error rate and character error rate are corpus-level quantities: the tables divide the sum of
the errors by the sum of the reference lengths. Characters are those of the normalized strings, spaces included.
"""

import functools
import importlib.resources
import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import jiwer
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

logger = logging.getLogger(__name__)

_WORDS = jiwer.Compose([jiwer.ReduceToListOfListOfWords()])
_CHARS = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])
_DEFAULT_NORMALIZER: dict[str, EnglishTextNormalizer] = {}


@functools.lru_cache(maxsize=4)
def _load_normalizer(model_id: str, revision: str, cache_dir: str | None, filename: str) -> EnglishTextNormalizer:
    path = hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)
    with open(path, encoding="utf-8") as f:
        return EnglishTextNormalizer(json.load(f))


def load_normalizer(cfg: DictConfig) -> EnglishTextNormalizer:
    """``EnglishTextNormalizer`` with the English spelling map of the pinned Whisper repository (``cfg.eval_metrics.asr``)."""
    cache_dir = None if cfg.cache_dir is None else str(cfg.cache_dir)
    return _load_normalizer(str(cfg.model_id), str(cfg.revision), cache_dir, str(cfg.normalizer_file))


def set_default_normalizer(cfg: DictConfig) -> None:
    """Makes ``normalize_text`` use the normalizer of ``cfg`` (``WhisperASR`` does this at construction)."""
    _DEFAULT_NORMALIZER["normalizer"] = load_normalizer(cfg)


def normalize_text(text: str) -> str:
    """Whisper English normalization (lower case, spelling, numbers, abbreviations), whitespace collapsed.

    Uses the normalizer set by ``set_default_normalizer``; without one, it is built from the packaged default
    config (``conf/eval_metrics/default.yaml``), downloading ``normalizer.json`` of the pinned revision if needed.
    """
    if "normalizer" not in _DEFAULT_NORMALIZER:
        default = importlib.resources.files("sparc.conf").joinpath("eval_metrics", "default.yaml")
        with importlib.resources.as_file(default) as path:
            set_default_normalizer(OmegaConf.load(path).asr)
    return " ".join(_DEFAULT_NORMALIZER["normalizer"](text).split())


def edit_counts(ref_norm: str, hyp_norm: str) -> dict[str, int]:
    """Levenshtein errors of a normalized hypothesis against its reference, at word and at character level.

    Returns ``word_errors`` (substitutions + deletions + insertions), ``word_ref_len``, ``char_errors`` and
    ``char_ref_len``. Characters include spaces. An empty reference counts every hypothesis token as an insertion.
    """
    ref, hyp = " ".join(ref_norm.split()), " ".join(hyp_norm.split())
    words = jiwer.process_words([ref], [hyp], reference_transform=_WORDS, hypothesis_transform=_WORDS)
    chars = jiwer.process_characters([ref], [hyp], reference_transform=_CHARS, hypothesis_transform=_CHARS)
    return {
        "word_errors": int(words.substitutions + words.deletions + words.insertions),
        "word_ref_len": int(words.hits + words.substitutions + words.deletions),
        "char_errors": int(chars.substitutions + chars.deletions + chars.insertions),
        "char_ref_len": int(chars.hits + chars.substitutions + chars.deletions),
    }


def read_reference(wav_path: str | Path) -> str:
    """Text of the LibriTTS-R ``<id>.normalized.txt`` that sits next to ``<id>.wav``."""
    path = Path(wav_path)
    return path.with_name(path.stem + ".normalized.txt").read_text(encoding="utf-8").strip()


def _chunks(items: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


class WhisperASR:
    """Whisper large-v3 (pinned revision) transcriber; ``cfg`` is ``cfg.eval_metrics.asr``."""

    def __init__(self, cfg: DictConfig, device: str | torch.device):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" and cfg.fp16_on_cuda else torch.float32
        cache_dir = None if cfg.cache_dir is None else str(cfg.cache_dir)
        self.processor = WhisperProcessor.from_pretrained(str(cfg.model_id), revision=str(cfg.revision), cache_dir=cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(cfg.model_id), revision=str(cfg.revision), cache_dir=cache_dir, dtype=self.dtype, use_safetensors=True
        )
        self.model = self.model.to(self.device).eval()
        self.sample_rate = int(cfg.sample_rate)
        self.short_limit = int(round(float(cfg.short_form_max_s) * self.sample_rate))
        weights = hf_hub_download(str(cfg.model_id), "model.safetensors", revision=str(cfg.revision), cache_dir=cache_dir)
        self.weights_sha256 = Path(weights).resolve().name  # blobs of LFS files are named by their sha256
        set_default_normalizer(cfg)

    def provenance(self) -> dict[str, str]:
        """Model identity for the stage's ``meta.json``."""
        return {
            "model_id": str(self.cfg.model_id),
            "revision": str(self.cfg.revision),
            "weights_sha256": self.weights_sha256,
            "dtype": str(self.dtype),
        }

    def _generate_kwargs(self) -> dict:
        return {
            "language": str(self.cfg.language),
            "task": str(self.cfg.task),
            "num_beams": int(self.cfg.num_beams),
            "do_sample": False,
            "max_new_tokens": int(self.cfg.max_new_tokens),
        }

    @torch.inference_mode()
    def _decode_short(self, wavs: list[np.ndarray]) -> list[str]:
        """Utterances up to 30 s: the feature extractor pads every one to a 30 s window."""
        features = self.processor.feature_extractor(
            [np.asarray(w, dtype=np.float32) for w in wavs], sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features
        ids = self.model.generate(
            features.to(self.device, self.dtype), return_timestamps=False, **self._generate_kwargs()
        )
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

    @torch.inference_mode()
    def _decode_long(self, wav: np.ndarray) -> str:
        """One utterance over 30 s: Whisper's sequential long-form algorithm (windows of 30 s, greedy, no conditioning)."""
        inputs = self.processor.feature_extractor(
            [np.asarray(wav, dtype=np.float32)],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        ids = self.model.generate(
            inputs.input_features.to(self.device, self.dtype),
            attention_mask=inputs.attention_mask.to(self.device),
            return_timestamps=True,
            condition_on_prev_tokens=False,
            temperature=0.0,
            **self._generate_kwargs(),
        )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def transcribe(self, wavs16: list[np.ndarray], errors: list[str] | None = None) -> list[str]:
        """Transcripts of 16 kHz mono float32 waveforms, in input order.

        Waveforms up to ``short_form_max_s`` are sorted by length and decoded in batches of ``batch_size``; longer ones
        are decoded one at a time. Without ``errors`` a failure raises. With a list, a failing batch is retried
        utterance by utterance, a failing utterance yields ``""``, and ``errors`` gets one entry per input (``""`` on
        success, otherwise the exception text).
        """
        n = len(wavs16)
        texts = [""] * n
        messages = [""] * n
        order = sorted(range(n), key=lambda i: len(wavs16[i]))
        short = [i for i in order if len(wavs16[i]) <= self.short_limit]
        long = [i for i in order if len(wavs16[i]) > self.short_limit]

        def run(indices: list[int]) -> None:
            try:
                if indices and len(wavs16[indices[0]]) > self.short_limit:
                    decoded = [self._decode_long(wavs16[indices[0]])]
                else:
                    decoded = self._decode_short([wavs16[i] for i in indices])
            except (ValueError, RuntimeError) as err:
                if errors is None:
                    raise
                if len(indices) > 1:
                    for i in indices:
                        run([i])
                    return
                logger.warning("Whisper failed on utterance %d: %s", indices[0], err)
                messages[indices[0]] = f"{type(err).__name__}: {err}"
                return
            for i, text in zip(indices, decoded):
                texts[i] = text

        for batch in _chunks(short, int(self.cfg.batch_size)):
            run(batch)
        for i in long:
            run([i])
        if errors is not None:
            errors.extend(messages)
        return texts
