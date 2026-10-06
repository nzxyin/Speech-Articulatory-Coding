"""Feature re-extraction of system audio for the evaluation (EVALUATION.md sections 4.5 and 5).

``ReExtractor`` wraps :class:`~sparc.vocoders.features.extractor.SparcFeatureExtractor` with CREPE dither switched off,
for one of two inversion heads:

- ``"refit"``: the extractor exactly as the feature cache uses it (refit head from the ``.npz``, with its hash,
  checkpoint and weight checks);
- ``"shipped"``: the linear head stored inside the en+ checkpoint (used for the ``enplus16`` system, whose own encoder
  produces EMA in that head's units). The refit checks are skipped, and the loaded weights are instead compared with the
  ``linear_model`` entry that is read from the checkpoint file again.

Config keys read from the root config ``cfg`` (the full composed config of the caller):

- ``cfg.extractor.*`` (``hf_hub_cache, sparc_repo, sparc_snapshot, sparc_ckpt, sparc_ckpt_sha256, sparc_yaml,
  wavlm_repo, wavlm_snapshot, resample_type, allow_tf32``) and ``cfg.inversion.*`` (``linear_npz, linear_sha256,
  pitch_q, pitch_hop_length``): the blocks of ``cache_config.yaml``. If the root config has no ``extractor`` or
  ``inversion`` key they are taken from ``cache_config.yaml`` itself (and ``paths/default.yaml`` for the interpolated
  locations), so a caller does not have to include them;
- ``cfg.eval_features.reextract.{seed, sample_rate, resample_type}`` (all optional, defaults 0, 24000, soxr_hq).

``extractor.dither`` is forced to false here. Nothing else of the cache configuration is modified.
"""

import hashlib
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import sparc
from sparc.vocoders.constants import EXTRACTOR_HOP, SAMPLE_RATE
from sparc.vocoders.features.extractor import SPEAKER_LAYERS, SparcFeatureExtractor, set_fp32_numerics
from sparc.vocoders.features.cache import file_sha256

HEADS = ("refit", "shipped")
CONF_DIR = Path(sparc.__file__).parent / "conf"


def extractor_config(cfg: DictConfig) -> DictConfig:
    """Resolved copy of the ``extractor`` and ``inversion`` blocks with ``extractor.dither = false``.

    Uses the blocks of ``cfg`` when it has both; otherwise those of ``cache_config.yaml`` (locations from the
    environment through ``paths/default.yaml``). Plain (non-struct) config, safe to hand to the extractor.
    """
    if "extractor" in cfg and "inversion" in cfg:
        blocks = {key: OmegaConf.to_container(cfg[key], resolve=True) for key in ("extractor", "inversion")}
    else:
        cache = OmegaConf.load(CONF_DIR / "cache_config.yaml")
        root = OmegaConf.create(
            {
                "paths": OmegaConf.load(CONF_DIR / "paths" / "default.yaml"),
                "extractor": cache.extractor,
                "inversion": cache.inversion,
            }
        )
        blocks = {key: OmegaConf.to_container(root[key], resolve=True) for key in ("extractor", "inversion")}
    blocks["extractor"]["dither"] = False
    return OmegaConf.create(blocks)


def head_sha256(weight: torch.Tensor, bias: torch.Tensor) -> str:
    """SHA-256 over the float32 bytes of a linear head's weight and bias, for provenance."""
    digest = hashlib.sha256()
    for tensor in (weight, bias):
        digest.update(tensor.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pinned_paths(ext: DictConfig) -> dict[str, Path]:
    """Checkpoint, yaml and WavLM snapshot directory inside the private pinned Hugging Face cache."""
    hub = Path(ext.hf_hub_cache)
    sparc_dir = hub / f"models--{ext.sparc_repo.replace('/', '--')}" / "snapshots" / ext.sparc_snapshot
    wavlm_dir = hub / f"models--{ext.wavlm_repo.replace('/', '--')}" / "snapshots" / ext.wavlm_snapshot
    paths = {"ckpt": sparc_dir / ext.sparc_ckpt, "yaml": sparc_dir / ext.sparc_yaml, "wavlm": wavlm_dir}
    for path in (paths["ckpt"], paths["yaml"], wavlm_dir / "config.json"):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing from the pinned Hugging Face cache {hub}")
    if "wavlm" not in str(wavlm_dir):
        raise ValueError("the WavLM snapshot path must contain 'wavlm' so that SPARC selects WavLMModel")
    return paths


class ShippedHeadExtractor(SparcFeatureExtractor):
    """The cache extractor with the linear inversion head of the en+ checkpoint instead of the refit head.

    Only the constructor differs: ``encode16`` and ``extract`` are inherited, so the processing (resampling, z-scoring,
    WavLM layers, CREPE, loudness, speaker pooling) is identical to the refit extractor's. The refit ``.npz`` is not
    read. The head is whatever ``sparc.load_model`` takes from the checkpoint when no head is passed in.
    """

    def __init__(self, cfg: DictConfig, device: torch.device | str = "cpu"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sparc.sparc import load_model
        from sparc.src_extractor import AmplitudeHistogram

        self.cfg = cfg
        self.device = torch.device(device)
        ext = cfg.extractor
        self.dither = bool(ext.get("dither", True))
        set_fp32_numerics(ext.allow_tf32)
        paths = pinned_paths(ext)
        ckpt_sha = self._blob_sha256(paths["ckpt"])
        if ckpt_sha != ext.sparc_ckpt_sha256:
            raise ValueError(f"{paths['ckpt']} has sha256 {ckpt_sha}, expected {ext.sparc_ckpt_sha256}")
        self.coder = load_model(
            config=str(paths["yaml"]),
            ckpt=str(paths["ckpt"]),
            device=str(self.device),
            speech_model=str(paths["wavlm"]),
        )
        shipped = torch.load(paths["ckpt"], map_location="cpu", weights_only=True)["state_dict"]["linear_model"]
        linear = self.coder.inverter.linear_model
        if not (
            torch.equal(linear.weight.cpu(), shipped["weight"].float()) and torch.equal(linear.bias.cpu(), shipped["bias"].float())
        ):
            raise ValueError("the loaded inversion head differs from the linear_model entry of the en+ checkpoint")
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
            "head": "shipped (linear_model entry of the en+ checkpoint)",
        }


class ReExtractor:
    """Re-extraction of ``(T, 15)`` features and the raw loudness of system audio, without CREPE dither.

    ``head`` is ``"refit"`` or ``"shipped"`` (see the module docstring). ``extract`` takes one mono waveform at any
    sample rate; it is resampled to 24 kHz with ``soxr_hq`` (a 16 kHz system is therefore upsampled and the
    extractor downsamples it again, as for every 24 kHz input) and passed to ``SparcFeatureExtractor.extract``.
    """

    def __init__(self, cfg: DictConfig, head: str, device: torch.device | str = "cpu"):
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        self.head = head
        options = OmegaConf.select(cfg, "eval_features.reextract", default=None) or {}
        self.seed = int(options.get("seed", 0))
        self.sample_rate = int(options.get("sample_rate", SAMPLE_RATE))
        self.cfg = extractor_config(cfg)
        self.resample_type = str(options.get("resample_type", self.cfg.extractor.resample_type))
        self.device = torch.device(device)
        factory = SparcFeatureExtractor if head == "refit" else ShippedHeadExtractor
        self.extractor = factory(self.cfg, self.device)
        if self.extractor.dither:
            raise RuntimeError("the evaluation re-extraction must run without CREPE dither")
        linear = self.extractor.coder.inverter.linear_model
        self.head_weight = linear.weight.detach().cpu().clone()
        self.head_bias = linear.bias.detach().cpu().clone()
        self.head_hash = head_sha256(self.head_weight, self.head_bias)

    @property
    def coder(self):
        """The underlying SPARC model (for example for its speaker encoder)."""
        return self.extractor.coder

    def head_matches_npz(self, npz_path: str | Path | None = None) -> bool:
        """True if the active head equals the weights of the refit ``.npz`` (``cfg.inversion.linear_npz`` by default)."""
        path = Path(self.cfg.inversion.linear_npz if npz_path is None else npz_path)
        with np.load(path) as npz:
            return bool(
                torch.equal(self.head_weight, torch.from_numpy(npz["weight"]).float())
                and torch.equal(self.head_bias, torch.from_numpy(npz["bias"]).float())
            )

    def describe(self) -> dict:
        """Provenance: active head, its hash, and the extractor's own description."""
        refit = Path(self.cfg.inversion.linear_npz)
        return {
            "head": self.head,
            "head_sha256": self.head_hash,
            "refit_npz_sha256": file_sha256(refit) if refit.exists() else None,
            "dither": False,
            **self.extractor.describe(),
        }

    def extract(self, wav: np.ndarray, sr: int) -> dict:
        """``{"feats": (T, 15) float32, "loud_raw": (T,) float32, "T": int}`` of one mono waveform at ``sr`` Hz."""
        wav = np.asarray(wav)
        if wav.ndim != 1:
            raise ValueError(f"expected a mono waveform, got shape {wav.shape}")
        if int(sr) != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=int(sr), target_sr=self.sample_rate, res_type=self.resample_type)
        out = self.extractor.extract(np.ascontiguousarray(wav), self.seed)
        return {"feats": out["feats"], "loud_raw": out["loud_raw"], "T": int(out["T"])}
