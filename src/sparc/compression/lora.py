"""LoRA variants for the retained transformer layers of a layer-subset encoder.

For a target projection W (e.g. q_proj) in retained layer l, with input h:
    independent : W h + s * B_l A_l h                 (A_l: r x d_in, B_l: d_out x r per layer)
    shared_a    : W h + s * B_l A h                   (one A per projection type, shared by all layers)
    shared_gated: W h + s * B diag(g_l) A h           (one A and B per projection type; per-layer gate
                                                       g_l, a scalar or a length-r vector)
with s = alpha / r. B (or, for shared_gated, B as well) starts at zero, so training starts from the
frozen pretrained encoder. LoRA changes the trainable parameter count, not inference compute: the
updates can be merged into W after training (merge_lora), leaving the subset encoder's FLOPs unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("independent", "shared_a", "shared_gated")


class LoRALinear(nn.Module):
    """Linear with a low-rank update. Exposes the merged .weight/.bias because some attention
    implementations (WavLM's F.multi_head_attention_forward path) read projection weights directly."""

    def __init__(self, base, A, B, gate, scale):
        super().__init__()
        self.base, self.A, self.B, self.gate, self.scale = base, A, B, gate, scale
        self.in_features, self.out_features = base.in_features, base.out_features

    def delta_weight(self):
        A = self.A if self.gate is None else self.gate[:, None] * self.A
        return self.scale * (self.B @ A)

    @property
    def weight(self):
        return self.base.weight + self.delta_weight().to(self.base.weight.dtype)

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class LoRABank(nn.Module):
    """Owns every LoRA parameter, so variants share parameters by holding the same tensors."""

    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict()

    def get(self, key, shape, init):
        if key not in self.params:
            p = torch.empty(shape)
            init(p)
            self.params[key] = nn.Parameter(p)
        return self.params[key]


def _kaiming(p):
    nn.init.kaiming_uniform_(p, a=math.sqrt(5))


def apply_lora(model, variant="independent", rank=8, alpha=16, targets=("q_proj", "v_proj"), gate="scalar"):
    """Freeze `model` and wrap the `targets` projections of every encoder layer. Returns the LoRABank."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    for p in model.parameters():
        p.requires_grad_(False)
    bank = LoRABank()
    scale = alpha / rank
    for li, layer in enumerate(model.encoder.layers):
        for name, mod in list(layer.named_modules()):
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in targets or not isinstance(mod, nn.Linear):
                continue
            d_in, d_out = mod.in_features, mod.out_features
            a_key = f"{leaf}.A" if variant != "independent" else f"{leaf}.A.{li}"
            b_key = f"{leaf}.B" if variant == "shared_gated" else f"{leaf}.B.{li}"
            A = bank.get(a_key.replace(".", "__"), (rank, d_in), _kaiming)
            B = bank.get(b_key.replace(".", "__"), (d_out, rank), nn.init.zeros_)
            g = None
            if variant == "shared_gated":
                gshape = (rank,) if gate == "rank" else (1,)
                g = bank.get(f"{leaf}__gate__{li}", gshape, nn.init.ones_)
            parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
            setattr(parent, leaf, LoRALinear(mod, A, B, g, scale))
    model.lora_bank = bank
    return bank


@torch.no_grad()
def merge_lora(model):
    """Fold every LoRALinear into its base Linear (in place) and drop the bank: inference cost of the plain encoder."""
    for layer in model.encoder.layers:
        for name, mod in list(layer.named_modules()):
            if isinstance(mod, LoRALinear):
                mod.base.weight += mod.delta_weight().to(mod.base.weight.dtype)
                parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
                setattr(parent, name.rsplit(".", 1)[-1], mod.base)
    if hasattr(model, "lora_bank"):
        del model.lora_bank
    return model


def lora_param_count(bank):
    return sum(p.numel() for p in bank.parameters())
