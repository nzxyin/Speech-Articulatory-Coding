"""SSL encoder registry and layer-subset encoders.

A layer-subset encoder keeps the CNN feature encoder, feature projection and positional
convolution of a pretrained model and runs only the chosen transformer layers, in order.
A prefix [1..k] is plain truncation; any other increasing subset is non-contiguous pruning.

The output is the hidden state after the last kept layer, matching hidden_states[k] of the full
model with output_hidden_states=True: the stable-layer-norm encoders used here (XLS-R, WavLM Large)
apply their final LayerNorm only after the last layer, so it is kept only when the last kept layer
is the model's last layer and is otherwise replaced by an identity.

WavLM computes its relative position bias in layer 1 and passes it on, so WavLM subsets must keep
layer 1. XLS-R has no such constraint.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

MODELS = {
    "wavlm-large": "microsoft/wavlm-large",
    "xlsr-300m": "facebook/wav2vec2-xls-r-300m",
    "xlsr-1b": "facebook/wav2vec2-xls-r-1b",
    "xlsr-2b": "facebook/wav2vec2-xls-r-2b",
    "w2v2-large": "facebook/wav2vec2-large-lv60",  # English LV-60k, same architecture as XLS-R 300M
}


def num_layers(name):
    return AutoConfig.from_pretrained(MODELS[name]).num_hidden_layers


def load_full(name, dtype=torch.float32):
    """Pretrained model in eval mode, with layerdrop and time masking disabled."""
    return AutoModel.from_pretrained(
        MODELS[name], layerdrop=0.0, mask_time_prob=0.0, mask_feature_prob=0.0, dtype=dtype
    ).eval()


def subset_model(model, layers):
    """Restrict a loaded model to transformer layers `layers` (1-indexed, strictly increasing), in place."""
    layers = list(layers)
    total = len(model.encoder.layers)
    if not layers or any(b <= a for a, b in zip(layers, layers[1:])) or layers[0] < 1 or layers[-1] > total:
        raise ValueError(f"layers must be strictly increasing within 1..{total}: {layers}")
    if model.config.model_type == "wavlm" and layers[0] != 1:
        raise ValueError("WavLM subsets must keep layer 1 (it computes the relative position bias)")
    model.encoder.layers = nn.ModuleList([model.encoder.layers[i - 1] for i in layers])
    if layers[-1] != total:
        model.encoder.layer_norm = nn.Identity()
    model.config.num_hidden_layers = len(layers)
    model.kept_layers = layers
    return model


def load_subset(name, layers, dtype=torch.float32):
    return subset_model(load_full(name, dtype), layers)


def normalize_wav(wav):
    """Per-utterance zero-mean / unit-variance, as the XLS-R and WavLM feature extractors do."""
    return (wav - wav.mean()) / (wav.std() + 1e-7)


def encode(model, wav, attention_mask=None):
    """(B, n_samples) normalized audio -> (B, T, D) hidden state after the last kept layer."""
    return model(wav, attention_mask=attention_mask).last_hidden_state


def frame_lengths(model, n_samples):
    return model._get_feat_extract_output_lengths(torch.as_tensor(n_samples)).long()
