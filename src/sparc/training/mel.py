# Mel-spectrogram extraction matching the parameters reported in the SPARC
# paper (arXiv:2406.12998), Appendix B.6:
#   {fs: 16000, fft_size: 1024, hop_size: 160, win_length: null,
#    window: "hann", num_mels: 80, fmin: 0, fmax: 8000}
# These parameters are not defined anywhere else in this codebase (the
# inference pipeline has no mel-spectrogram loss), so this module exists
# solely for the vocoder training loss below.

import torch
import torchaudio


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_size=160,
        win_length=None,
        num_mels=80,
        fmin=0,
        fmax=8000,
    ):
        super().__init__()
        self.mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_length or n_fft,
            window_fn=torch.hann_window,
            n_mels=num_mels,
            f_min=fmin,
            f_max=fmax,
            power=1.0,
            center=True,
        )

    def forward(self, wav):
        """wav: (B, T) -> log-mel (B, num_mels, T // hop_size + 1)"""
        mel = self.mel_fn(wav)
        return torch.log(torch.clamp(mel, min=1e-5))
