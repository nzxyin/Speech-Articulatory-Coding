"""Speaker encoders trained jointly with each vocoder: the FFN of the comparison and the Phase 3 layer-pool encoder."""

import math

import torch
from torch import nn

from sparc.vocoders.constants import SPEAKER_DIM, SPEAKER_RAW_DIM


class SpeakerFFN(nn.Module):
    """Maps a pooled 1024-d WavLM vector to a 64-d speaker embedding.

    Same layers as ``sparc.spk_encoder.SpeakerEncodingLayer``, preceded by a fixed per-dimension z-score whose
    statistics come from the training cache and are stored as buffers.
    """

    def __init__(
        self,
        in_dim: int = SPEAKER_RAW_DIM,
        hidden_dim: int = SPEAKER_RAW_DIM,
        out_dim: int = SPEAKER_DIM,
        dropout: float = 0.2,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ):
        super().__init__()
        mean = torch.zeros(in_dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        std = torch.ones(in_dim) if std is None else torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean", mean.reshape(in_dim).clone())
        self.register_buffer("std", std.reshape(in_dim).clamp_min(1e-6).clone())
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, spk_raw: torch.Tensor) -> torch.Tensor:
        return self.net((spk_raw - self.mean) / self.std)

    def embed(self, spk_raw: torch.Tensor, mask: torch.Tensor | None = None, sample: bool = False):
        """``(z, info)`` as :meth:`LayerPoolSpeakerEncoder.embed`; ``spk_raw`` is ``[B, 1024]`` or ``[B, K, 1024]``.

        With ``K`` references the embedding is the mean of the per-reference outputs. Deterministic: ``info`` is empty.
        """
        if spk_raw.dim() == 2:
            return self(spk_raw), {}
        return masked_mean(self(spk_raw), mask), {}


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of ``x [B, K, D]`` over the references marked in ``mask [B, K]`` (all of them when ``mask`` is None)."""
    if mask is None:
        return x.mean(1)
    w = mask.to(x.dtype)[..., None]
    return (x * w).sum(1) / w.sum(1).clamp_min(1.0)


class LayerPoolSpeakerEncoder(nn.Module):
    """Speaker encoder over the pooled statistics of all WavLM Large layers (speaker-embedding Phase 3).

    Input per reference: periodicity-weighted mean and std of every hidden state, ``[2, L, 1024]`` (cache of
    ``sparc.vocoders.features.spk_layers``). Each statistic is z-scored per layer and dimension with training-set
    statistics (buffers), mixed over layers with learned softmax weights (one set for the mean, one for the std;
    uniform at initialization), and passed through Linear 2048 -> hidden, GELU, dropout, Linear -> out (``use_std``
    False: the mean alone, 1024 inputs).

    Deterministic (``vae`` False): the embedding is the mean over the references of the per-reference outputs.
    VAE: each reference gives a diagonal Gaussian ``(mu_k, log var_k)``; the posterior is their product with the
    N(0, I) prior, so precision ``1 + sum_k exp(-log var_k)`` and mean ``sum_k mu_k exp(-log var_k) / precision``.
    ``embed(..., sample=True)`` draws from it (training); otherwise it returns the posterior mean.
    """

    def __init__(
        self,
        n_layers: int = 25,
        dim: int = SPEAKER_RAW_DIM,
        hidden_dim: int = 1024,
        out_dim: int = SPEAKER_DIM,
        dropout: float = 0.2,
        use_std: bool = True,
        vae: bool = False,
        logvar_init: float = -2.0,
        stat_mean: torch.Tensor | None = None,
        stat_std: torch.Tensor | None = None,
    ):
        super().__init__()
        n_stats = 2 if use_std else 1
        shape = (2, n_layers, dim)
        mean = torch.zeros(shape) if stat_mean is None else torch.as_tensor(stat_mean, dtype=torch.float32)
        std = torch.ones(shape) if stat_std is None else torch.as_tensor(stat_std, dtype=torch.float32)
        self.register_buffer("stat_mean", mean.reshape(shape).clone())
        self.register_buffer("stat_std", std.reshape(shape).clamp_min(1e-6).clone())
        self.use_std = bool(use_std)
        self.vae = bool(vae)
        self.out_dim = int(out_dim)
        self.layer_logits = nn.Parameter(torch.zeros(n_stats, n_layers))
        self.net = nn.Sequential(
            nn.Linear(n_stats * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim * (2 if vae else 1)),
        )
        if vae:
            # references start moderately confident (variance e^-2), so the posterior is not the prior at step 0
            nn.init.constant_(self.net[-1].bias[out_dim:], logvar_init)

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=-1)

    def per_reference(self, spk_raw: torch.Tensor) -> torch.Tensor:
        """Network output ``[..., out]`` (``[..., 2 out]`` for the VAE) of statistics ``[..., 2, L, 1024]``."""
        n = 2 if self.use_std else 1
        x = (spk_raw[..., :n, :, :].float() - self.stat_mean[:n]) / self.stat_std[:n]
        mixed = torch.einsum("sl,...sld->...sd", self.layer_weights(), x)
        return self.net(mixed.flatten(-2))

    def forward(self, spk_raw: torch.Tensor) -> torch.Tensor:
        """Per-reference code ``[..., out]`` (the VAE's per-reference mean, without the prior)."""
        return self.per_reference(spk_raw)[..., : self.out_dim]

    def posterior(self, spk_raw: torch.Tensor, mask: torch.Tensor | None = None):
        """``(mu, log var)`` ``[B, out]`` of the product of the references' Gaussians and the prior."""
        if spk_raw.dim() == 4:
            spk_raw = spk_raw[:, None]
        out = self.per_reference(spk_raw)
        mu_k, logvar_k = out[..., : self.out_dim], out[..., self.out_dim :].clamp(-12.0, 8.0)
        prec_k = torch.exp(-logvar_k)
        if mask is not None:
            prec_k = prec_k * mask.to(prec_k.dtype)[..., None]
        prec = 1.0 + prec_k.sum(1)
        mu = (mu_k * prec_k).sum(1) / prec
        return mu, -torch.log(prec)

    def embed(self, spk_raw: torch.Tensor, mask: torch.Tensor | None = None, sample: bool = False):
        """``(z [B, out], info)``; ``spk_raw`` is ``[B, 2, L, D]`` or ``[B, K, 2, L, D]``; VAE ``info``: mu, logvar."""
        if not self.vae:
            if spk_raw.dim() == 4:
                return self.per_reference(spk_raw), {}
            return masked_mean(self.per_reference(spk_raw), mask), {}
        mu, logvar = self.posterior(spk_raw, mask)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return z, {"mu": mu, "logvar": logvar}


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, diag exp(logvar)) || N(0, I) ) per row, in nats."""
    return 0.5 * (logvar.exp() + mu**2 - 1.0 - logvar).sum(-1)


def total_correlation(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, dataset_size: int) -> torch.Tensor:
    """Minibatch-weighted-sampling estimate of the total correlation of q(z) (Chen et al. 2018, beta-TC-VAE).

    ``z, mu, logvar`` are ``[M, D]``: one sample per posterior of a minibatch of ``M`` references drawn from a set
    of ``dataset_size``.
    """
    m = z.shape[0]
    log_qzi_xj = -0.5 * ((z[:, None] - mu[None]) ** 2 / logvar.exp()[None] + logvar[None] + math.log(2 * math.pi))
    log_norm = math.log(m * dataset_size)
    log_qz = torch.logsumexp(log_qzi_xj.sum(-1), dim=1) - log_norm
    log_prod_qzd = (torch.logsumexp(log_qzi_xj, dim=1) - log_norm).sum(-1)
    return (log_qz - log_prod_qzd).mean()


class PitchCode(nn.Module):
    """Phase 4 (R-F0): additive speaker-code term from the reference utterance's median voiced F0.

    The generator sees only pitch ratios, so the pitch level has to come from the speaker code: this maps
    ``(ln f0_ref - logf0_mean) / logf0_std`` (training-set statistics of the frontend, so the scalar is ~N(0, 1)
    across utterances) through a linear layer to ``out_dim``, and the training module adds it to the encoder's code.
    With several references the caller passes the geometric mean of their medians. Weight and bias start at zero (the
    arm starts as B0 without the pitch input); a NaN or non-positive median (no voiced frames) maps to 0, the
    training mean.
    """

    def __init__(self, logf0_mean: float, logf0_std: float, out_dim: int = SPEAKER_DIM):
        super().__init__()
        self.register_buffer("logf0_mean", torch.tensor(float(logf0_mean)))
        self.register_buffer("logf0_std", torch.tensor(max(float(logf0_std), 1e-6)))
        self.proj = nn.Linear(1, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def normalize(self, ref_f0_hz: torch.Tensor) -> torch.Tensor:
        """``[B]`` median F0 in Hz -> ``[B, 1]`` normalized log median (NaN or non-positive -> 0)."""
        z = (torch.log(ref_f0_hz.float().clamp_min(1.0)) - self.logf0_mean) / self.logf0_std
        return torch.where(ref_f0_hz > 0, z, torch.zeros_like(z))[:, None]

    def forward(self, ref_f0_hz: torch.Tensor) -> torch.Tensor:
        return self.proj(self.normalize(ref_f0_hz))
