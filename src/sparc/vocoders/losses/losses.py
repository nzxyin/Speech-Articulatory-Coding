"""Adversarial, feature-matching, mel and multi-scale spectral losses for the 24 kHz vocoder comparison.

The hinge losses and the feature-matching loss follow Vocos (https://github.com/gemelo-ai/vocos, commit
eb39abfc42c1dee4854d9b10d44dd7d4fd3b0e53, MIT License, Copyright (c) 2023 Charactr Inc.); the LSGAN losses follow
HiFi-GAN (Kong et al., 2020) as in the fork's ``sparc.training.losses``; ``MultiScaleSpectralLoss`` is ported from
DDSP-Articulatory-Vocoder (https://github.com/Drake-Lin/DDSP-Articulatory-Vocoder, commit
dc6df4ca0ed7bdd50630b7b26882291c5cee8a57, MIT License, Copyright (c) 2024 Louis Liu, Drake Lin).

Every adversarial helper averages over the sub-discriminators of one discriminator set and returns
``(total, per_disc)``; weights between sets (for example MPD 1.0 and MRD 0.1) are applied by the caller. Waveform
losses take ``(y_hat, y)`` as ``[B, 1, L]`` or ``[B, L]`` tensors and compute the spectra in float32 with autocast
disabled. Padding and centring use ``sparc.vocoders.losses.ops``, whose backward is deterministic on CUDA.
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from sparc.vocoders.losses.ops import stft


def _mean(per_disc: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(per_disc).mean()


def _float(x: torch.Tensor) -> torch.Tensor:
    """Logits and feature maps under autocast are bf16/fp16; the losses reduce them in float32."""
    return x.float()


def hinge_d_loss(
    real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Hinge discriminator loss ``relu(1 - D(y)) + relu(1 + D(y_hat))``, averaged over sub-discriminators."""
    per_disc = [F.relu(1 - _float(r)).mean() + F.relu(1 + _float(g)).mean() for r, g in zip(real_logits, fake_logits)]
    return _mean(per_disc), per_disc


def hinge_g_loss(fake_logits: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Hinge generator loss ``relu(1 - D(y_hat))``, averaged over sub-discriminators."""
    per_disc = [F.relu(1 - _float(g)).mean() for g in fake_logits]
    return _mean(per_disc), per_disc


def lsgan_d_loss(
    real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Least-squares discriminator loss ``(1 - D(y))^2 + D(y_hat)^2``, averaged over sub-discriminators."""
    per_disc = [((1 - _float(r)) ** 2).mean() + (_float(g) ** 2).mean() for r, g in zip(real_logits, fake_logits)]
    return _mean(per_disc), per_disc


def lsgan_g_loss(fake_logits: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Least-squares generator loss ``(1 - D(y_hat))^2``, averaged over sub-discriminators."""
    per_disc = [((1 - _float(g)) ** 2).mean() for g in fake_logits]
    return _mean(per_disc), per_disc


ADVERSARIAL_LOSSES: dict[str, tuple[Callable, Callable]] = {
    "hinge": (hinge_d_loss, hinge_g_loss),
    "lsgan": (lsgan_d_loss, lsgan_g_loss),
}


def feature_matching_loss(real_fmaps: list[list[torch.Tensor]], fake_fmaps: list[list[torch.Tensor]]) -> torch.Tensor:
    """L1 feature matching: sum over layers of ``mean |real - fake|``, divided by the number of sub-discriminators."""
    total = 0.0
    for fmap_r, fmap_g in zip(real_fmaps, fake_fmaps):
        for r, g in zip(fmap_r, fmap_g):
            total = total + (_float(r) - _float(g)).abs().mean()
    return total / len(real_fmaps)


def _as_2d(wav: torch.Tensor) -> torch.Tensor:
    return wav.squeeze(1) if wav.dim() == 3 else wav


def _check_aligned(y_hat: torch.Tensor, y: torch.Tensor) -> None:
    """Rejects waveform pairs of different shape, which could otherwise give equally many frames and misalign."""
    if _as_2d(y_hat).shape != _as_2d(y).shape:
        raise ValueError(f"generated {tuple(y_hat.shape)} and reference {tuple(y.shape)} waveforms differ in shape")


class MelSpectrogramLoss(nn.Module):
    """L1 distance between natural-log mel magnitude spectrograms.

    The defaults are the shared 24 kHz definition: 1024-point Hann window, hop 256, 100 HTK mel bins from 0 to
    12 kHz, magnitude (``power=1``), reflect-padded centred frames, ``log(clamp(mel, 1e-5))``. The spectra are
    numerically identical to ``torchaudio.transforms.MelSpectrogram`` with the same arguments.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        f_min: float = 0.0,
        f_max: float = 12000.0,
        power: float = 1.0,
        center: bool = True,
        mel_scale: str = "htk",
        clamp: float = 1e-5,
    ):
        super().__init__()
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.power, self.center, self.clamp = power, center, clamp
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        fb = torchaudio.functional.melscale_fbanks(
            n_fft // 2 + 1, f_min, f_max, n_mels, sample_rate, norm=None, mel_scale=mel_scale
        )
        self.register_buffer("mel_fb", fb, persistent=False)

    def log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Log-mel spectrogram ``[B, n_mels, frames]`` of ``wav`` given as ``[B, 1, L]`` or ``[B, L]``."""
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = _as_2d(wav).float()
            spec = stft(x, self.n_fft, self.hop_length, self.win_length, self.window, self.center).abs()
            if self.power != 1.0:
                spec = spec.pow(self.power)
            mel = torch.matmul(spec.transpose(-1, -2), self.mel_fb).transpose(-1, -2)
            return torch.log(mel.clamp(min=self.clamp))

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _check_aligned(y_hat, y)
        return F.l1_loss(self.log_mel(y_hat), self.log_mel(y))


class _MagnitudeSpectra(nn.Module):
    """Hann-window STFT magnitudes at several FFT sizes with ``hop = int(n_fft * hop_ratio)``."""

    def __init__(self, fft_sizes: Sequence[int], hop_ratio: float):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.hop_ratio = hop_ratio
        for n in self.fft_sizes:
            self.register_buffer(f"window_{n}", torch.hann_window(n), persistent=False)

    def magnitude(self, wav: torch.Tensor, n_fft: int) -> torch.Tensor:
        window = getattr(self, f"window_{n_fft}")
        return stft(_as_2d(wav).float(), n_fft, int(n_fft * self.hop_ratio), n_fft, window).abs()


class MultiScaleSpectralLoss(_MagnitudeSpectra):
    """DDSP-AV multi-scale spectral loss, summed over FFT sizes.

    Per scale: ``L1(S_hat, S) + alpha * L1(log(S_hat + eps), log(S + eps))`` on STFT magnitudes with a window of
    ``n_fft`` samples and a hop of ``n_fft * hop_ratio``.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (2048, 1024, 512, 256, 128, 64),
        hop_ratio: float = 0.25,
        alpha: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__(fft_sizes, hop_ratio)
        self.alpha, self.eps = alpha, eps

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _check_aligned(y_hat, y)
        with torch.autocast(device_type=y.device.type, enabled=False):
            total = 0.0
            for n_fft in self.fft_sizes:
                s_hat, s = self.magnitude(y_hat, n_fft), self.magnitude(y, n_fft)
                total = total + F.l1_loss(s_hat, s) + self.alpha * F.l1_loss(
                    torch.log(s_hat + self.eps), torch.log(s + self.eps)
                )
            return total


@torch.no_grad()
def mr_stft_distance(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    fft_sizes: Sequence[int] = (2048, 1024, 512, 256, 128, 64),
    hop_ratio: float = 0.25,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Multi-resolution STFT distance for validation, averaged over FFT sizes.

    Per scale: spectral convergence ``||S - S_hat||_F / ||S||_F`` plus the mean L1 distance of the natural-log
    magnitudes. Lower is better; the value is a metric, not a training loss.
    """
    _check_aligned(y_hat, y)
    spectra = _MagnitudeSpectra(fft_sizes, hop_ratio).to(y.device)
    with torch.autocast(device_type=y.device.type, enabled=False):
        scores = []
        for n_fft in fft_sizes:
            s_hat, s = spectra.magnitude(y_hat, n_fft), spectra.magnitude(y, n_fft)
            convergence = torch.linalg.norm(s - s_hat) / (torch.linalg.norm(s) + eps)
            log_distance = F.l1_loss(torch.log(s_hat + eps), torch.log(s + eps))
            scores.append(convergence + log_distance)
        return torch.stack(scores).mean()
