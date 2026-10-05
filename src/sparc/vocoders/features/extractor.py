"""SPARC feature extractor for the vocoder cache: one utterance at a time, with the refit inversion head.

Runs SPARC's own en+ path (z-scored 16 kHz waveform, WavLM layer 9, low-pass, linear head, CREPE, loudness,
periodicity gating) and adds what the vocoders need on top: the un-normalized loudness ``loud_raw``, WavLM layers 0
and 6 pooled over voiced frames, and the frozen en+ 64-d speaker embedding. WavLM is run once; layers 0, 6 and 9 are
captured from that pass by a forward hook.
"""

import importlib.metadata
import os
import subprocess
from pathlib import Path

import librosa
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sparc.vocoders.constants import (
    EXTRACTOR_HOP,
    EXTRACTOR_SAMPLE_RATE,
    SAMPLE_RATE,
    feature_length,
)
from sparc.vocoders.features.cache import ENPLUS_DIM, file_sha256

PACKAGES = ("torch", "torchaudio", "transformers", "torchcrepe", "librosa", "soxr", "numpy", "scipy", "soundfile")
SPEAKER_LAYERS = (0, 6)


def package_versions() -> dict:
    """Installed versions of the packages that influence the features."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def fork_commit(root: str | Path | None = None) -> str:
    """``git rev-parse HEAD`` of the repository containing this package, or ``"unknown"``.

    A suffix ``+dirty`` marks uncommitted or untracked changes under ``src/``, so the recorded commit never claims
    code that differs from it.
    """
    root = Path(__file__).resolve().parents[4] if root is None else Path(root)
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "src"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.stdout.strip() + ("+dirty" if status.stdout.strip() else "")


def set_fp32_numerics(allow_tf32: bool) -> None:
    """Allows or forbids TF32 in cuDNN convolutions and matmuls.

    cuDNN convolutions use TF32 by default on Ampere and newer GPUs, so the same utterance would give EMA values
    that differ by about 1e-3 between GPU models; with TF32 off the cache does not depend on which node ran a shard.
    """
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)


def pool_speaker(hidden: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Periodicity-weighted mean of ``hidden [frames, 1024]``, as in ``SpeakerEncoder._get_spk_emb``.

    Both arrays are truncated to the shorter length. If the weights sum to 0 the plain mean over frames is returned
    and the third value is ``True``. Returns ``(vector float32 [1024], weight sum, used_fallback)``.
    """
    frames = min(hidden.shape[0], weights.shape[0])
    if frames == 0:
        raise ValueError("cannot pool zero frames")
    acoustics = hidden[None, :frames].astype(np.float32, copy=False)
    weight = weights[None, :frames, None].astype(np.float32, copy=False)
    total = weight.sum(1)
    wsum = float(total[0, 0])
    if wsum > 0.0:
        return ((acoustics * weight).sum(1) / total)[0].astype(np.float32), wsum, False
    return acoustics[0].mean(0).astype(np.float32), wsum, True


def raw_loudness(raw16: np.ndarray, histogram: torch.nn.Module, device: torch.device | str) -> np.ndarray:
    """Mean absolute amplitude per 320-sample frame of the un-normalized 16 kHz waveform, float32 ``[frames]``."""
    x = torch.from_numpy(np.ascontiguousarray(raw16)).float().to(device)[None]
    with torch.no_grad():
        return histogram(x)[0].cpu().numpy().astype(np.float32)


class SparcFeatureExtractor:
    """Loads SPARC en+ with the refit inversion head and extracts the cache arrays of single utterances."""

    def __init__(self, cfg: DictConfig, device: torch.device | str = "cpu"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sparc.sparc import load_model
        from sparc.src_extractor import AmplitudeHistogram

        self.cfg = cfg
        self.device = torch.device(device)
        ext = cfg.extractor
        set_fp32_numerics(ext.allow_tf32)
        hub = Path(ext.hf_hub_cache)
        sparc_dir = hub / f"models--{ext.sparc_repo.replace('/', '--')}" / "snapshots" / ext.sparc_snapshot
        wavlm_dir = hub / f"models--{ext.wavlm_repo.replace('/', '--')}" / "snapshots" / ext.wavlm_snapshot
        ckpt, yaml_path = sparc_dir / ext.sparc_ckpt, sparc_dir / ext.sparc_yaml
        for path in (ckpt, yaml_path, wavlm_dir / "config.json"):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing from the pinned Hugging Face cache {hub}")
        if "wavlm" not in str(wavlm_dir):
            raise ValueError("the WavLM snapshot path must contain 'wavlm' so that SPARC selects WavLMModel")

        npz_path = Path(cfg.inversion.linear_npz)
        npz_sha = file_sha256(npz_path)
        if npz_sha != cfg.inversion.linear_sha256:
            raise ValueError(f"{npz_path} has sha256 {npz_sha}, expected {cfg.inversion.linear_sha256}")
        with np.load(npz_path) as npz:
            head = {"weight": torch.from_numpy(npz["weight"]).float(), "bias": torch.from_numpy(npz["bias"]).float()}
        ckpt_sha = self._blob_sha256(ckpt)
        if ckpt_sha != ext.sparc_ckpt_sha256:
            raise ValueError(f"{ckpt} has sha256 {ckpt_sha}, expected {ext.sparc_ckpt_sha256}")

        self.coder = load_model(
            config=str(yaml_path),
            ckpt=str(ckpt),
            device=str(self.device),
            linear_model_state_dict=head,
            speech_model=str(wavlm_dir),
        )
        linear = self.coder.inverter.linear_model
        if not (torch.equal(linear.weight.cpu(), head["weight"]) and torch.equal(linear.bias.cpu(), head["bias"])):
            raise ValueError("the loaded inversion head differs from the refit npz")
        source = self.coder.source_extractor
        if source.q != int(cfg.inversion.pitch_q) or source.pitch_hop_length != int(cfg.inversion.pitch_hop_length):
            raise ValueError(f"expected q=2 and pitch hop 160, got q={source.q}, hop={source.pitch_hop_length}")
        if self.coder.inverter.spk_target_layer != SPEAKER_LAYERS[1]:
            raise ValueError("the en+ speaker FFN is expected to read WavLM layer 6")

        self.histogram = AmplitudeHistogram(EXTRACTOR_HOP).eval().to(self.device)
        self.wavlm = self.coder.inverter.speech_model
        self.layers = (*SPEAKER_LAYERS, self.coder.inverter.tgt_layer)
        self._hidden: dict[int, np.ndarray] = {}
        self.wavlm.register_forward_hook(self._capture)
        self.checkpoint_info = {
            "sparc_snapshot": ext.sparc_snapshot,
            "sparc_ckpt_sha256": ckpt_sha,
            "wavlm_snapshot": ext.wavlm_snapshot,
            "refit_npz_sha256": npz_sha,
        }

    @staticmethod
    def _blob_sha256(path: Path) -> str:
        """SHA-256 of a Hugging Face snapshot file; the blob name is the hash for LFS files, else it is computed."""
        name = path.resolve().name
        if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
            return name
        return file_sha256(path)

    def _capture(self, module: torch.nn.Module, args: tuple, output) -> None:
        states = output.hidden_states
        self._hidden = {i: states[i][0].detach().float().cpu().numpy() for i in self.layers}

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"

    def describe(self) -> dict:
        """Settings and versions recorded in ``meta.json``."""
        source = self.coder.source_extractor
        extractor_cfg = OmegaConf.to_container(self.cfg.extractor, resolve=True)
        return {
            "versions": package_versions(),
            "checkpoints": self.checkpoint_info,
            "extractor": {
                **extractor_cfg,
                "inversion": OmegaConf.to_container(self.cfg.inversion, resolve=True),
                "crepe_model": source.crepe_model,
                "fmin": source.fmin,
                "fmax": source.fmax,
                "periodicity_threshold": source.periodicity_threshold,
                "min_points": source.min_points,
                "reflect_loudness": source.reflect_loudness,
                "loudness_threshold": source.loudness_threshold,
                "target_layer": self.coder.inverter.tgt_layer,
                "freqcut": self.coder.inverter.freqcut,
                "normalize": self.coder.normalize,
            },
            "seed_rule": self.cfg.extractor.seed_rule,
        }

    def encode16(self, raw16: np.ndarray, seed: int) -> dict:
        """Features of one un-normalized 16 kHz waveform (the 24 kHz path calls this after resampling).

        Returns the arrays of :meth:`extract` that do not depend on the 24 kHz file.
        """
        coder = self.coder
        wavs = coder.process_wavfiles(raw16)
        outputs = coder.inverter(wavs, {})
        hidden = self._hidden
        np.random.seed(seed)
        outputs = coder.source_extractor(wavs, outputs)

        ema = outputs["ema"][0]
        frames = ema.shape[0]
        pitch = outputs["pitch"][0, :frames, 0]
        zloud = outputs["loudness"][0, :frames, 0]
        periodicity = outputs["periodicity"][0, :frames, 0]
        if min(len(pitch), len(zloud), len(periodicity)) != frames:
            raise ValueError("a feature stream is shorter than the EMA stream")
        feats = np.concatenate([ema, pitch[:, None], zloud[:, None], periodicity[:, None]], axis=1).astype(np.float32)

        loud = raw_loudness(raw16, self.histogram, self.device)[:frames]
        if len(loud) != frames:
            raise ValueError("raw loudness is shorter than the EMA stream")

        weights = outputs["periodicity"][0, :, 0]
        pooled = {}
        for layer in SPEAKER_LAYERS:
            pooled[layer], wsum, fallback = pool_speaker(hidden[layer], weights)
        ffn = self.coder.speaker_encoder.spk_enc
        with torch.no_grad():
            enplus = ffn(torch.from_numpy(pooled[6])[None].to(self.device))[0].cpu().numpy().astype(np.float32)
        if enplus.shape != (ENPLUS_DIM,):
            raise ValueError(f"unexpected en+ embedding shape {enplus.shape}")
        return {
            "feats": feats,
            "loud_raw": loud,
            "spk_l0": pooled[0],
            "spk_l6": pooled[6],
            "spk_enplus64": enplus,
            "spk_wsum": wsum,
            "spk_fallback": fallback,
            "T": frames,
        }

    def extract(self, wav24: np.ndarray, seed: int) -> dict:
        """Cache arrays of one 24 kHz mono utterance; ``seed`` fixes the pitch dither of CREPE."""
        if wav24.ndim != 1:
            raise ValueError(f"expected a mono waveform, got shape {wav24.shape}")
        raw16 = librosa.resample(
            wav24, orig_sr=SAMPLE_RATE, target_sr=EXTRACTOR_SAMPLE_RATE, res_type=self.cfg.extractor.resample_type
        )
        out = self.encode16(raw16, seed)
        expected = feature_length(len(wav24))
        if out["T"] != expected:
            raise ValueError(f"extracted {out['T']} frames, expected {expected}")
        out.update(n24=len(wav24), peak24=float(np.abs(wav24).max()), seed=seed)
        return out
