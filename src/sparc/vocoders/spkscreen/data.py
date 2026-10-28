"""Loading the frame cache and building padded frame batches (``FrameBatch``) for the speaker-encoder screen."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sparc.vocoders.spkscreen import cache as C


@dataclass
class FrameBatch:
    """Padded frames of ``B`` utterance crops.

    ``hidden [B, L, T, D]`` float32 holds the cached WavLM layers requested by the module, in the order of its
    ``layers`` tuple; ``periodicity``, ``zloud`` (SPARC's z-scored loudness), ``loud`` (raw loudness) and ``f0`` (Hz,
    continuous through unvoiced frames) are ``[B, T]``; ``mel [B, T, 80]`` is the 24 kHz log-mel (None unless
    requested); ``mask [B, T]`` is True on real frames. Padded frames hold zeros.
    """

    hidden: torch.Tensor
    periodicity: torch.Tensor
    zloud: torch.Tensor
    loud: torch.Tensor
    f0: torch.Tensor
    mel: torch.Tensor | None
    mask: torch.Tensor
    # provenance of the crops (set by ``FrameSet.batch``): lets a module read the waveform of the same crop
    set_name: str = ""
    utt: np.ndarray | None = None
    start: np.ndarray | None = None
    length: np.ndarray | None = None

    @property
    def lengths(self) -> torch.Tensor:
        return self.mask.sum(1)


class FrameSet:
    """Concatenated frame arrays of one cache set (train or heldout) with per-utterance offsets."""

    def __init__(self, hidden: dict[int, torch.Tensor], aux: torch.Tensor, mel: torch.Tensor | None, lens: np.ndarray,
                 speakers: np.ndarray, targets: np.ndarray, logf0: np.ndarray, ids: list[str] | None = None):
        self.hidden, self.aux, self.mel = hidden, aux, mel
        self.lens = np.asarray(lens, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lens)[:-1]]).astype(np.int64)
        self.speakers = np.asarray(speakers).astype(str)
        self.targets, self.logf0 = np.asarray(targets, np.float32), np.asarray(logf0, np.float64)
        self.ids = ids if ids is not None else [str(i) for i in range(len(lens))]
        self.name = ""
        self.device = aux.device
        self.n = len(self.lens)
        total = int(self.lens.sum())
        for k, h in hidden.items():
            if h.shape[0] != total:
                raise ValueError(f"layer {k}: {h.shape[0]} frames, expected {total}")
        if aux.shape[0] != total:
            raise ValueError("aux frames do not match lengths")
        self._offsets_t = torch.as_tensor(self.offsets, device=self.device)
        self._lens_t = torch.as_tensor(self.lens, device=self.device)

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.hidden))

    def batch(self, utt: np.ndarray, start: np.ndarray, length: np.ndarray, layers: tuple[int, ...], need_mel: bool = False,
              frames_cap: int | None = None) -> FrameBatch:
        """Crops ``[start, start + length)`` frames of utterances ``utt`` (all ``[B]`` int arrays), padded."""
        dev = self.device
        utt_t = torch.as_tensor(utt, device=dev)
        length_t = torch.as_tensor(length, device=dev).clamp(min=1)
        tmax = int(length_t.max()) if frames_cap is None else min(frames_cap, int(length_t.max()))
        ar = torch.arange(tmax, device=dev)
        mask = ar[None] < length_t[:, None]
        idx = self._offsets_t[utt_t][:, None] + torch.as_tensor(start, device=dev)[:, None] + ar[None]
        idx = torch.where(mask, idx, torch.zeros_like(idx))
        for k in layers:
            if k not in self.hidden:
                raise KeyError(f"layer {k} is not loaded (have {self.layers})")
        hid = torch.stack([self.hidden[k][idx] for k in layers], 1).float()
        hid = hid * mask[:, None, :, None]
        aux = self.aux[idx] * mask[..., None]
        mel = None
        if need_mel:
            if self.mel is None:
                raise ValueError("mel frames were not loaded")
            mel = self.mel[idx].float() * mask[..., None]
        return FrameBatch(hid, aux[..., 0], aux[..., 1], aux[..., 2], aux[..., 3], mel, mask, self.name, np.asarray(utt),
                          np.asarray(start), np.asarray(length))

    def full(self, utt: np.ndarray, layers: tuple[int, ...], need_mel: bool = False, max_frames: int | None = None) -> FrameBatch:
        """Whole utterances (optionally only the first ``max_frames`` frames)."""
        utt = np.asarray(utt)
        length = self.lens[utt] if max_frames is None else np.minimum(self.lens[utt], max_frames)
        return self.batch(utt, np.zeros(len(utt), np.int64), length, layers, need_mel)


def load_set(set_name: str, layers: tuple[int, ...], mel: bool = False, device: str | torch.device = "cpu",
             root: Path = C.ROOT) -> FrameSet:
    """Loads the chunks of a cache set; only the requested layers are read."""
    man = C.load_manifest(set_name, root)
    chunks = [C.chunk_dir(root, set_name, c) for c in range(C.n_chunks(len(man)))]
    missing = [str(d) for d in chunks if not (d / "done").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} chunks of {set_name} are not done, e.g. {missing[0]}")

    def cat(name):
        return np.concatenate([np.load(d / name) for d in chunks])

    hidden = {k: torch.from_numpy(cat(f"hid_L{k}.npy")).to(device) for k in layers}
    aux = torch.from_numpy(cat("aux.npy")).to(device)
    melt = torch.from_numpy(cat("mel.npy")).to(device) if mel else None
    lens = cat("lens.npy")
    if set_name == "heldout":
        p1 = np.load(root / "heldout" / "phase1.npz")
        targets, logf0 = p1["targets"], p1["logf0"]  # Phase 1 definition, read from its files
    elif set_name == "train_aug":  # augmented copies of the training utterances keep the clean utterances' targets
        clean = [C.chunk_dir(root, "train", c) for c in range(C.n_chunks(len(man)))]
        targets = np.concatenate([np.load(d / "targets.npy") for d in clean])
        logf0 = np.concatenate([np.load(d / "logf0.npy") for d in clean])
    else:
        targets, logf0 = cat("targets.npy"), cat("logf0.npy")
    fs = FrameSet(hidden, aux, melt, lens, man["speaker"].values, targets, logf0, list(man["id"]))
    fs.name = set_name
    if set_name == "heldout":
        fs.is_test = man["is_test"].values.astype(bool)
        fs.en_plus = np.load(root / "heldout" / "phase1.npz")["en_plus"]
        fs.own_targets = cat("targets.npy")
    return fs


def concat_sets(a: FrameSet, b: FrameSet) -> FrameSet:
    """Utterances of ``a`` followed by those of ``b`` (same layers, same device); speaker names are shared."""
    if a.layers != b.layers or (a.mel is None) != (b.mel is None):
        raise ValueError("sets differ in loaded layers or mel")
    hidden = {k: torch.cat([a.hidden[k], b.hidden[k]]) for k in a.layers}
    mel = None if a.mel is None else torch.cat([a.mel, b.mel])
    fs = FrameSet(hidden, torch.cat([a.aux, b.aux]), mel, np.concatenate([a.lens, b.lens]),
                  np.concatenate([a.speakers, b.speakers]), np.concatenate([a.targets, b.targets]),
                  np.concatenate([a.logf0, b.logf0]), a.ids + b.ids)
    fs.name = f"{a.name}+{b.name}"
    return fs
