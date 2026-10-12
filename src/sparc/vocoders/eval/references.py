"""Reference systems of the evaluation: Vocos mel copy synthesis and the shipped SPARC en+ (EVALUATION.md section 3).

``VocosMelReference``: ``charactr/vocos-mel-24khz`` at a pinned revision, copy synthesis of the 24 kHz gt audio (mel
analysis then decoder, level unchanged). The output has exactly as many samples as the input (the gt audio has
``480 T``): the iSTFT with centre padding returns ``256 floor(n / 256)`` samples, so the tail is zero-padded.

``EnPlusReference``: ``sparc.load_model("en+")`` at the pinned snapshot with its own shipped linear head, encoding the
gt audio resampled to 16 kHz and decoding with a given 64-d speaker embedding. The shipped encoder drops one frame
for ``320 T`` input samples (WavLM, ``floor((n - 80) / 320) = T - 1``), so the decoder returns ``320 (T - 1)``
samples; the waveform is zero-padded at the end to the requested ``n16 = 320 T`` (or trimmed if longer) and then
peak-normalized to -3 dBFS, because SPARC z-scores its input and its output level is arbitrary.

Config keys read (``cfg.eval_features.references``): ``hub_cache``, ``vocos_mel.{repo, revision, weights_file,
weights_sha256, config_file}`` and ``enplus.{repo, snapshot, ckpt, ckpt_sha256, yaml, wavlm_repo, wavlm_snapshot,
input_sample_rate, hop, peak_dbfs, seed, dither, allow_tf32}``. Models are loaded from the private pinned Hugging Face
cache only (no network).
"""

import os
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml
from omegaconf import DictConfig

from sparc.vocoders.constants import SAMPLE_RATE
from sparc.vocoders.features.extractor import crepe_dither, set_fp32_numerics
from sparc.vocoders.features.cache import file_sha256

RESAMPLE_TYPE = "soxr_hq"


def fit_length(wav: np.ndarray, n: int) -> np.ndarray:
    """Trims or zero-pads a 1-D waveform at the end to exactly ``n`` samples."""
    if len(wav) >= n:
        return wav[:n]
    return np.concatenate([wav, np.zeros(n - len(wav), dtype=wav.dtype)])


def peak_normalize(wav: np.ndarray, dbfs: float) -> np.ndarray:
    """Scales a waveform so that its peak is ``dbfs`` dBFS; a silent waveform is returned unchanged."""
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    if peak == 0.0:
        return wav
    return (wav * (10.0 ** (dbfs / 20.0) / peak)).astype(wav.dtype)


def snapshot_dir(hub: str | Path, repo: str, revision: str) -> Path:
    """Directory of a pinned snapshot inside a Hugging Face hub cache."""
    path = Path(hub) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is missing from the pinned Hugging Face cache {hub}")
    return path


def _sha256(path: Path) -> str:
    """SHA-256 of a snapshot file (the blob file name for LFS files, otherwise computed)."""
    name = path.resolve().name
    if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
        return name
    return file_sha256(path)


class VocosMelReference:
    """Copy synthesis with Vocos (mel, 24 kHz): ``synth(wav24)`` returns an array of the same length."""

    def __init__(self, cfg: DictConfig, device: torch.device | str = "cpu"):
        from vocos import Vocos

        os.environ["HF_HUB_OFFLINE"] = "1"
        ref = cfg.eval_features.references
        spec = ref.vocos_mel
        self.device = torch.device(device)
        self.repo, self.revision = spec.repo, spec.revision
        directory = snapshot_dir(ref.hub_cache, spec.repo, spec.revision)
        weights = directory / spec.weights_file
        config = directory / spec.config_file
        digest = _sha256(weights)
        if digest != spec.weights_sha256:
            raise ValueError(f"{weights} has sha256 {digest}, expected {spec.weights_sha256}")
        model = Vocos.from_hparams(str(config))
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = model.eval().to(self.device)
        self.config = yaml.safe_load(config.read_text())
        self.hop = int(self.config["feature_extractor"]["init_args"]["hop_length"])
        self.sample_rate = int(self.config["feature_extractor"]["init_args"]["sample_rate"])
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"expected a {SAMPLE_RATE} Hz Vocos model, got {self.sample_rate}")
        self.weights_sha256 = digest

    def describe(self) -> dict:
        return {"repo": self.repo, "revision": self.revision, "weights_sha256": self.weights_sha256}

    @torch.inference_mode()
    def mel(self, wav24: np.ndarray) -> torch.Tensor:
        """Log-mel features ``[1, 100, frames]`` of a 24 kHz waveform (the model input)."""
        audio = torch.from_numpy(np.ascontiguousarray(wav24, dtype=np.float32))[None].to(self.device)
        return self.model.feature_extractor(audio)

    @torch.inference_mode()
    def decode(self, mel: torch.Tensor) -> np.ndarray:
        """Waveform (float32 numpy) from mel features ``[1, 100, frames]``."""
        return self.model.decode(mel)[0].float().cpu().numpy()

    def synth(self, wav24: np.ndarray) -> np.ndarray:
        """Copy synthesis of a mono 24 kHz waveform; ``len(output) == len(wav24)`` (trimmed or zero-padded)."""
        wav24 = np.asarray(wav24)
        if wav24.ndim != 1:
            raise ValueError(f"expected a mono waveform, got shape {wav24.shape}")
        out = self.decode(self.mel(wav24))
        return fit_length(out, len(wav24))


class EnPlusReference:
    """SPARC en+ with its shipped inversion head: ``encode`` the gt audio, ``decode`` with a chosen speaker embedding."""

    def __init__(self, cfg: DictConfig, device: torch.device | str = "cpu"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sparc.sparc import load_model

        ref = cfg.eval_features.references
        spec = ref.enplus
        self.device = torch.device(device)
        self.cfg = spec
        set_fp32_numerics(bool(spec.allow_tf32))
        sparc_dir = snapshot_dir(ref.hub_cache, spec.repo, spec.snapshot)
        wavlm_dir = snapshot_dir(ref.hub_cache, spec.wavlm_repo, spec.wavlm_snapshot)
        if "wavlm" not in str(wavlm_dir):
            raise ValueError("the WavLM snapshot path must contain 'wavlm' so that SPARC selects WavLMModel")
        ckpt = sparc_dir / spec.ckpt
        digest = _sha256(ckpt)
        if digest != spec.ckpt_sha256:
            raise ValueError(f"{ckpt} has sha256 {digest}, expected {spec.ckpt_sha256}")
        # same call as sparc.load_model("en+"), with the pinned files; no head is passed in, so the checkpoint's own is used
        self.coder = load_model(
            config=str(sparc_dir / spec.yaml), ckpt=str(ckpt), device=str(self.device), speech_model=str(wavlm_dir)
        )
        shipped = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]["linear_model"]
        linear = self.coder.inverter.linear_model
        if not (
            torch.equal(linear.weight.cpu(), shipped["weight"].float()) and torch.equal(linear.bias.cpu(), shipped["bias"].float())
        ):
            raise ValueError("the loaded inversion head differs from the linear_model entry of the en+ checkpoint")
        self.sample_rate = int(spec.input_sample_rate)
        self.hop = int(spec.hop)
        self.peak_dbfs = float(spec.peak_dbfs)
        self.seed = int(spec.seed)
        self.dither = bool(spec.dither)
        self.ckpt_sha256 = digest
        self.snapshot = spec.snapshot

    def describe(self) -> dict:
        return {
            "repo": self.cfg.repo,
            "snapshot": self.snapshot,
            "ckpt_sha256": self.ckpt_sha256,
            "wavlm_snapshot": self.cfg.wavlm_snapshot,
            "head": "shipped",
            "peak_dbfs": self.peak_dbfs,
            "seed": self.seed,
            "dither": self.dither,
        }

    def encode(self, wav24: np.ndarray) -> dict:
        """Shipped en+ encoding of a mono 24 kHz waveform (resampled to 16 kHz, z-scored by SPARC).

        Returns the dict of ``SPARC.encode`` (``ema [T', 12]``, ``pitch [T', 1]``, ``loudness [T', 1]``,
        ``periodicity [T', 1]``, ``pitch_stats``, ``spk_emb [64]``, ``ft_len``). ``T' = floor((n16 - 80) / 320)``.
        The numpy seed is set immediately before the call, so the shipped CREPE dither (kept unless
        ``enplus.dither`` is false) is reproducible.
        """
        wav24 = np.asarray(wav24)
        if wav24.ndim != 1:
            raise ValueError(f"expected a mono waveform, got shape {wav24.shape}")
        raw16 = librosa.resample(wav24, orig_sr=SAMPLE_RATE, target_sr=self.sample_rate, res_type=RESAMPLE_TYPE)
        np.random.seed(self.seed)
        with crepe_dither(self.dither):
            return self.coder.encode(raw16)

    @staticmethod
    def spk_emb(code: dict) -> np.ndarray:
        """The 64-d speaker embedding of an encoding (float32)."""
        return np.asarray(code["spk_emb"], dtype=np.float32)

    def decode_raw(self, code: dict, spk_emb: np.ndarray) -> np.ndarray:
        """16 kHz waveform of the shipped generator, ``320 T'`` samples at its own level (not normalized)."""
        return self.coder.decode(code["ema"], code["pitch"], code["loudness"], np.asarray(spk_emb, dtype=np.float32))

    def decode(self, code: dict, spk_emb: np.ndarray, n16: int) -> np.ndarray:
        """16 kHz waveform of exactly ``n16`` samples (zero-padded or trimmed at the end), peak at ``peak_dbfs``."""
        wav = self.decode_raw(code, spk_emb).astype(np.float32)
        return peak_normalize(fit_length(wav, int(n16)), self.peak_dbfs)
