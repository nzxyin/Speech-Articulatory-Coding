"""Frozen WeSpeaker ResNet34 speaker-verification embedding (VoxCeleb2, 256-d) for the out-of-domain screen.

Not ECAPA, WavLM-SV or CAM++ (the three models of the cloning evaluation), so it can be used as an encoder input
without touching the held-out judges. Architecture ported from wenet-e2e/wespeaker ``wespeaker/models/resnet.py``
(Apache-2.0; ResNet34, 32 base channels, temporal statistics pooling, one embedding layer ``seg_1``); inference frontend
as in ``eval/metrics/campplus.py`` (16 kHz audio scaled to int16 range, Kaldi fbank 80 bins, per-utterance mean
normalization). Checkpoint ``Wespeaker/wespeaker-voxceleb-resnet34`` (``avg_model.pt``) in ``CHECKPOINT``.

``prep`` writes the 16 kHz waveforms of a cache set (``<root>/ood_sv/wave16_<set>.npy`` + offsets) so the encoder can
embed the exact crop a training batch uses::

    python -m sparc.vocoders.spkscreen.ood_sv prep --sets train heldout globe globe_spk
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.vocoders.spkscreen import cache as C

CHECKPOINT = Path(__import__("os").environ.get("OOD_SV_CKPT", "/data/user_data/xoy/spkemb/ood_sv/wespeaker_resnet34"))
EMBED_DIM = 256
FRAME_SAMPLES = 320  # one cached frame = 20 ms at 16 kHz
WIN = 400


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet34(nn.Module):
    def __init__(self, feat_dim=80, embed_dim=EMBED_DIM, m=32, blocks=(3, 4, 6, 3)):
        super().__init__()
        self.in_planes = m
        self.conv1 = nn.Conv2d(1, m, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(m)
        self.layer1 = self._make(m, blocks[0], 1)
        self.layer2 = self._make(m * 2, blocks[1], 2)
        self.layer3 = self._make(m * 4, blocks[2], 2)
        self.layer4 = self._make(m * 8, blocks[3], 2)
        self.stats_dim = (feat_dim // 8) * m * 8
        self.seg_1 = nn.Linear(self.stats_dim * 2, embed_dim)

    def _make(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        """``x [B, T, F]`` fbank; ``lengths [B]`` valid input frames (statistics pooled over the valid frames only)."""
        x = x.permute(0, 2, 1).unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = out.reshape(out.shape[0], -1, out.shape[-1])  # [B, C*F, T/8]
        if lengths is None:
            mean, std = out.mean(-1), out.var(-1, unbiased=True).clamp_min(1e-7).sqrt()
        else:
            n = ((lengths + 7) // 8).clamp(min=2, max=out.shape[-1])
            m = (torch.arange(out.shape[-1], device=out.device)[None] < n[:, None]).to(out.dtype)[:, None]
            mean = (out * m).sum(-1) / n[:, None]
            var = (((out - mean[..., None]) ** 2) * m).sum(-1) / (n[:, None] - 1)
            std = var.clamp_min(1e-7).sqrt()
        return self.seg_1(torch.cat([mean, std], 1))


def load_resnet(location: str | Path = CHECKPOINT) -> ResNet34:
    model = ResNet34()
    state = torch.load(Path(location) / "avg_model.pt", map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("projection.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"ResNet34 checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.eval()


def fbank(wav16: torch.Tensor) -> torch.Tensor:
    """Kaldi fbank of one waveform ``[n]`` in [-1, 1] on its device, mean-normalized: ``[frames, 80]``."""
    import torchaudio.compliance.kaldi as kaldi

    f = kaldi.fbank(wav16[None] * (1 << 15), num_mel_bins=80, frame_length=25, frame_shift=10, sample_frequency=16000,
                    window_type="hamming", dither=0.0)  # fmt: skip
    return f - f.mean(0)


class WaveStore:
    """16 kHz waveforms of one cache set (fp16, concatenated) as a torch tensor on a device."""

    def __init__(self, set_name: str, root: Path = C.ROOT, device="cpu"):
        d = root / "ood_sv"
        self.wave = torch.from_numpy(np.load(d / f"wave16_{set_name}.npy")).to(device)
        self.off = np.load(d / f"wave16_{set_name}_off.npy")  # [N + 1]

    def crop(self, utt: int, start_frame: int, n_frames: int) -> torch.Tensor:
        a, b = int(self.off[utt]), int(self.off[utt + 1])
        s = a + start_frame * FRAME_SAMPLES
        e = min(b, a + (start_frame + n_frames - 1) * FRAME_SAMPLES + WIN)
        return self.wave[s : max(e, s + WIN)].float() / 32768.0


class SVEmbedder(nn.Module):
    """Frozen ResNet34 embedding of waveform crops (the crops of a ``FrameBatch``)."""

    def __init__(self, root: Path = C.ROOT):
        super().__init__()
        self.net = load_resnet()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.root = root
        self._stores: dict[str, WaveStore] = {}

    def store(self, name: str, device) -> WaveStore:
        if name not in self._stores:
            self._stores[name] = WaveStore(name, self.root, device)
        return self._stores[name]

    def train(self, mode: bool = True):
        return super().train(False)  # BatchNorm statistics stay frozen

    @torch.no_grad()
    def forward(self, batch) -> torch.Tensor:
        st = self.store(batch.set_name, batch.mask.device)
        feats = [fbank(st.crop(int(u), int(s), int(n))) for u, s, n in zip(batch.utt, batch.start, batch.length)]
        lens = torch.as_tensor([len(f) for f in feats], device=batch.mask.device)
        x = torch.zeros(len(feats), int(lens.max()), 80, device=batch.mask.device)
        for i, f in enumerate(feats):
            x[i, : len(f)] = f
        return self.net(x, lens)


def prep(set_name: str, root: Path = C.ROOT) -> None:
    import librosa

    man = C.load_manifest(set_name, root)
    out = root / "ood_sv"
    out.mkdir(parents=True, exist_ok=True)
    waves, off = [], [0]
    for p in man["wav_path"]:
        w24 = C.read_wav24(p)
        w = librosa.resample(w24, orig_sr=24000, target_sr=16000, res_type="soxr_hq")
        waves.append(np.clip(w * 32768.0, -32768, 32767).astype(np.int16))
        off.append(off[-1] + len(w))
    np.save(out / f"wave16_{set_name}.npy", np.concatenate(waves))
    np.save(out / f"wave16_{set_name}_off.npy", np.array(off, np.int64))
    print(f"{set_name}: {len(man)} utterances, {off[-1] / 16000 / 3600:.2f} h", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("prep",))
    p.add_argument("--sets", nargs="+", required=True)
    a = p.parse_args()
    for s in a.sets:
        prep(s)


if __name__ == "__main__":
    main()
