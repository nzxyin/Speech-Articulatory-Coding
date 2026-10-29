"""WavLM Large k12 + BiGRU -> conv 9 articulatory inverters as a source of the vocoder's 12 articulatory channels.

The two checkpoints come from the speech-to-EMA work (``final_models.py`` on branch ``multispeaker-eval``):

* ``mngu0`` -- trained on MNGU0 day 1; the output is de-normalized with the stored target mean / std (MNGU0 mm).
* ``all7``  -- trained on the seven multi-speaker EMA speakers; its targets were z-scored per speaker (and per
  recording-session group), so the stored mean / std are those of z units and the output stays in that normalized space.

Both apply the model exactly as it was trained:

1. 16 kHz audio, zero-mean / unit-variance over the utterance (``encoders.normalize_wav``).
2. WavLM Large truncated to its first ``layer`` (= 12) transformer layers, fp32. The output equals
   ``hidden_states[12]`` of the full model (the stable-layer-norm final LayerNorm is only applied after layer 24).
3. Zero-phase 10 Hz Butterworth low-pass over the whole utterance along time (``extract.lowpass``), stored in float16
   as the training cache was, then read back as float32.
4. The BiGRU -> conv head (``make_head('gruconv9', 1024, smooth=False)``), then ``y * ys + ym``.

Alignment. In training, EMA frame ``j`` of an utterance (trimmed to its speech) was paired with SSL frame
``base_offset + shift + j``, so the head output at SSL frame ``i`` estimates the EMA at audio frame ``i - shift``.
Channel ``t`` of the vocoder grid is therefore ``out[t + shift]`` (the last frame is repeated at the end). The head
runs over the whole utterance, including the leading silence that training had cut off.

Frame grid. The vocoder grid has ``feature_length(n24)`` frames at 50 Hz, which is also the number of WavLM frames
of ``ceil(2 * n24 / 3)`` samples at 16 kHz, so no resampling in time is needed.

``apply_trained(raw16, offset)`` reproduces the training-time call (head on the trimmed features) for the checks.
"""

import hashlib
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.vocoders.constants import EXTRACTOR_SAMPLE_RATE, N_EMA, SAMPLE_RATE, feature_length

FINAL_MODELS = Path("/data/user_data/xoy/xlsr_ema/final_models")
CHECKPOINTS = {
    "mngu0": FINAL_MODELS / "mngu0" / "frozen_k12_gruconv9_s0.pt",
    "all7": FINAL_MODELS / "all7" / "frozen_k12_gruconv9_s0.pt",
}
INVERTERS = tuple(CHECKPOINTS)
WAVLM_DIR = Path(
    "/data/user_data/xoy/sparc-vocoders/hf_hub/models--microsoft--wavlm-large/snapshots/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
)
FREQCUT = 10  # Hz, the low-pass of the training features
FT_SR = 50
RESAMPLE_TYPE = "soxr_hq"  # as extractor.resample_type of cache_config.yaml


class _Front(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.inp = nn.Linear(dim, hidden)

    def forward(self, h):
        return F.gelu(self.inp(self.norm(h)))


class GruConvHead(nn.Module):
    """``make_head('gruconv<k>', dim, smooth=False)`` of ``sparc.compression``: front end, BiGRU, centered conv, linear."""

    def __init__(self, dim: int = 1024, kernel: int = 9, hidden: int = 256, n_out: int = N_EMA):
        super().__init__()
        self.kernel = kernel
        self.front = _Front(dim, hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel)
        self.rnn = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``h [B, T, dim]`` (one utterance per row, no padding) -> ``[B, T, n_out]``."""
        x = self.rnn(self.front(h))[0]
        x = F.pad(x.transpose(1, 2), ((self.kernel - 1) // 2, self.kernel // 2))
        return self.out(F.gelu(self.conv(x)).transpose(1, 2))


def lowpass(x: np.ndarray) -> np.ndarray:
    """Zero-phase 10 Hz low-pass of ``x [T, D]`` along time (``sparc.compression.extract.lowpass``)."""
    from sparc.inversion import butter_bandpass_filter

    return butter_bandpass_filter(x[None], FREQCUT, FT_SR, axis=1)[0]


def normalize_wav(wav: np.ndarray) -> np.ndarray:
    """Per-utterance zero mean / unit variance (``sparc.compression.encoders.normalize_wav``)."""
    return (wav - wav.mean()) / (wav.std() + 1e-7)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truncated_wavlm(layers: int, device) -> nn.Module:
    """WavLM Large with its first ``layers`` transformer layers; its output is ``hidden_states[layers]`` of the full model."""
    from transformers import WavLMModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = WavLMModel.from_pretrained(str(WAVLM_DIR), layerdrop=0.0, mask_time_prob=0.0, mask_feature_prob=0.0).eval()
    total = len(model.encoder.layers)
    if not 1 <= layers <= total:
        raise ValueError(f"layer must be in 1..{total}, got {layers}")
    model.encoder.layers = nn.ModuleList(list(model.encoder.layers)[:layers])
    if layers != total:
        model.encoder.layer_norm = nn.Identity()
    model.config.num_hidden_layers = layers
    return model.to(device)


def to_grid(out: np.ndarray, shift: int) -> np.ndarray:
    """Head output at SSL frame ``i`` estimates the EMA at audio frame ``i - shift``: channel ``t`` is ``out[t + shift]``
    (index clipped to the first / last frame)."""
    idx = np.clip(np.arange(len(out)) + shift, 0, len(out) - 1)
    return out[idx]


class K12Inverter:
    """One inverter checkpoint: audio to the 12 articulatory channels on the vocoder's 50 Hz grid.

    ``mode`` is ``"mngu0"`` or ``"all7"`` (selects the default checkpoint and is checked against the checkpoint's
    ``targets`` string); ``ckpt`` overrides the path. All arithmetic is fp32 with TF32 off.
    """

    def __init__(self, mode: str, device="auto", ckpt: str | Path | None = None):
        if mode not in INVERTERS:
            raise ValueError(f"mode must be one of {INVERTERS}, got {mode!r}")
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.ckpt_path = Path(ckpt) if ckpt else CHECKPOINTS[mode]
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if state.get("head") != "gruconv9" or state.get("model") != "wavlm-large" or int(state.get("dim", 0)) != 1024:
            raise ValueError(f"{self.ckpt_path} is not a WavLM Large gruconv9 checkpoint: {state.get('head')}")
        want = "mm (MNGU0)" if mode == "mngu0" else "per-normalization-group z (multi-speaker)"
        if state.get("targets") != want:
            raise ValueError(f"{self.ckpt_path} has targets {state.get('targets')!r}, mode {mode} expects {want!r}")
        self.layer, self.shift = int(state["layer"]), int(state["shift"])
        self.ym = np.asarray(state["ym"], dtype=np.float64)
        self.ys = np.asarray(state["ys"], dtype=np.float64)
        self.head = GruConvHead(1024, kernel=9)
        self.head.load_state_dict(state["state_dict"], strict=True)
        self.head = self.head.eval().to(self.device)
        self.wavlm = truncated_wavlm(self.layer, self.device)
        self.sha256 = file_sha256(self.ckpt_path)

    def describe(self) -> dict:
        return {"mode": self.mode, "ckpt": str(self.ckpt_path), "sha256": self.sha256, "layer": self.layer,
                "shift": self.shift, "target_mean": self.ym.tolist(), "target_std": self.ys.tolist()}

    @torch.no_grad()
    def features(self, raw16: np.ndarray) -> np.ndarray:
        """Low-passed layer-``layer`` features ``[T, 1024]`` float32 of one un-normalized 16 kHz waveform, with the
        float16 rounding of the training cache."""
        wav = torch.from_numpy(np.ascontiguousarray(normalize_wav(np.asarray(raw16, dtype=np.float32)))).float()[None]
        h = self.wavlm(wav.to(self.device)).last_hidden_state[0].float().cpu().numpy()
        out = lowpass(h).astype(np.float16)
        if not np.isfinite(out).all():
            raise OverflowError("hidden state exceeds the float16 range")
        return out.astype(np.float32)

    @torch.no_grad()
    def head_output(self, x: np.ndarray) -> np.ndarray:
        """Head on ``x [T, 1024]`` features, de-standardized with the stored target mean / std: ``[T, 12]`` float64."""
        y = self.head(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(self.device))[0]
        return y.float().cpu().numpy() * self.ys + self.ym

    def apply_trained(self, raw16: np.ndarray, offset: int, n: int) -> np.ndarray:
        """Training-time call: the head on features ``[offset:]`` (``offset = base_offset + shift``), the first ``n``
        output frames (``n`` = EMA frames)."""
        x = self.features(raw16)
        if offset >= 0:
            x = x[offset:]
        return self.head_output(x)[:n]

    def channels16(self, raw16: np.ndarray, shift: int | None = None) -> np.ndarray:
        """The 12 channels ``[T, 12]`` float32 of a 16 kHz waveform on its WavLM frame grid (see the module docstring).

        ``shift`` overrides the stored alignment shift (the default, as trained) for the alignment checks only."""
        out = self.head_output(self.features(raw16))
        return to_grid(out, self.shift if shift is None else shift).astype(np.float32)

    def channels24(self, wav24: np.ndarray, shift: int | None = None) -> np.ndarray:
        """The 12 channels ``[feature_length(n24), 12]`` float32 of a 24 kHz mono waveform."""
        if wav24.ndim != 1:
            raise ValueError(f"expected a mono waveform, got shape {wav24.shape}")
        raw16 = librosa.resample(
            np.asarray(wav24, dtype=np.float32), orig_sr=SAMPLE_RATE, target_sr=EXTRACTOR_SAMPLE_RATE, res_type=RESAMPLE_TYPE
        )
        out = self.channels16(raw16, shift)
        expected = feature_length(len(wav24))
        if len(out) != expected:
            raise ValueError(f"inverter produced {len(out)} frames, the vocoder grid has {expected}")
        return out
