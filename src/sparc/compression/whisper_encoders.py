"""Whisper encoders for layer probing (docs/whisper_probe_plan.md).

Whisper's encoder takes a 30 s log-mel window (3000 frames at 100 Hz), halves the rate with its second convolution
(stride 2) and adds fixed sinusoidal positions for 1500 frames, so output frame j is centred at sample 320 j (20 ms
grid, 50 Hz, the EMA rate). The checkpoint only runs on the full window: faithful use pads the utterance with zeros to
30 s, runs the whole encoder and keeps the first `n_real_frames(n)` outputs, the frames whose centre lies inside the audio.

Layer convention (same as the wav2vec2 / WavLM caches): hidden_states[0] = convolutions + positional embedding,
hidden_states[k] = residual stream after layer k, hidden_states[L] = after the final LayerNorm (applied only at k = L).

`PrefixEncoder(enc, k)` runs layers 1..k and equals hidden_states[k]; `forward_ideal` is a cost-accounting forward on the
real frames only (conv and attention over n_w frames, positions sliced). It is never used to produce probe features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WHISPER = {
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-medium": "openai/whisper-medium",
    "whisper-medium-en": "openai/whisper-medium.en",
}

SR = 16000
WINDOW_SAMPLES = 30 * SR  # 480000
MEL_FRAMES = 3000
OUT_FRAMES = 1500
FRAME_SAMPLES = 320  # output frame spacing (20 ms)


def n_real_frames(n_samples):
    """Encoder output frames whose centre (320 j) lies inside an utterance of n_samples: ceil(n / 320)."""
    return -(-int(n_samples) // FRAME_SAMPLES)


def load_encoder(name, dtype=torch.float32):
    """The checkpoint's WhisperEncoder (eval, no dropout), without the decoder."""
    from transformers import WhisperModel

    model = WhisperModel.from_pretrained(WHISPER[name], dtype=dtype)
    enc = model.get_encoder().eval()
    del model
    return enc


def load_frontend(name):
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(WHISPER[name])


def load_bundle(name, dtype=torch.float32):
    return load_encoder(name, dtype), load_frontend(name)


def log_mel(fe, wavs, device="cpu"):
    """(B, n_mels, 3000) log-mel of 16 kHz waveforms zero-padded to 30 s, un-normalized audio, the checkpoint's own front end."""
    wavs = [np.asarray(w, dtype=np.float32) for w in wavs]
    for w in wavs:
        if w.ndim != 1:
            raise ValueError("waveforms must be 1-D")
        if len(w) > WINDOW_SAMPLES:
            raise ValueError(f"utterance of {len(w) / SR:.1f} s exceeds the 30 s Whisper window")
    out = fe(wavs, sampling_rate=SR, return_tensors="pt", padding="max_length", device=device)
    x = out.input_features
    assert x.shape[-1] == MEL_FRAMES, x.shape
    return x


def _layer(layer, h):
    out = layer(h, None, layer_head_mask=None)
    return out[0] if isinstance(out, (tuple, list)) else out


def embed(enc, mel, n_frames=None):
    """Convolutions + positional embedding. n_frames: only for the idealized forward (mel is then 2 * n_frames long)."""
    h = F.gelu(enc.conv1(mel))
    h = F.gelu(enc.conv2(h)).permute(0, 2, 1)
    n = h.shape[1]
    if n_frames is not None:
        assert n == n_frames, (n, n_frames)
    return h + enc.embed_positions.weight[:n]


class PrefixEncoder(nn.Module):
    """Whisper encoder truncated after layer k (final LayerNorm only when k is the last layer). Weights are shared."""

    def __init__(self, enc, k):
        super().__init__()
        L = len(enc.layers)
        if not 0 <= k <= L:
            raise ValueError(f"k must be within 0..{L}")
        self.enc, self.k, self.last = enc, k, L

    def _run(self, h):
        for layer in self.enc.layers[: self.k]:
            h = _layer(layer, h)
        return self.enc.layer_norm(h) if self.k == self.last else h

    def forward(self, mel):
        """Stock padded forward: mel must be the full 3000-frame window."""
        if mel.shape[-1] != MEL_FRAMES:
            raise ValueError(f"Whisper expects {MEL_FRAMES} mel frames, got {mel.shape[-1]}")
        return self._run(embed(self.enc, mel))

    def forward_ideal(self, mel, n_w):
        """Cost lower bound: only the first 2 * n_w mel frames, positions sliced, attention over n_w frames."""
        return self._run(embed(self.enc, mel[..., : 2 * n_w], n_w))


@torch.no_grad()
def hidden_states(enc, mel):
    """Tuple of L + 1 tensors (B, 1500, D) for the padded window (stock forward)."""
    return enc(mel, output_hidden_states=True).hidden_states
