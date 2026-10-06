"""MNGU0 day-1 EMA corpus: standard split, EMA in mm, and label-derived EMA/audio alignment.

Split: the corpus's own standard file sets (mngu0_s1_ema_filesets README): validation = IDs ending
in an odd digit followed by 0, test = IDs ending in an even digit followed by 0, train = the rest
(1137/63/63 of the 1263 EMA files). These are the sets used in prior MNGU0 inversion work.

Units: ema_norm stores (x - mean) / (4 * std) per channel, with mean/std in norm_parms/. The raw
positions are in cm (tongue-dorsum y has mean 5.44 and std 0.23), so mm = 10 * (4 * std * x + mean).

Alignment: ema_norm was silence-trimmed using the forced-alignment labels, then a few more frames
were trimmed to match an acoustic context window. The wav files are untrimmed. So EMA frame j
pairs with SSL frame round(sil_end * 50) + shift + j, where sil_end is the end of the leading
silence in the .lab file. Over the 1046 utterances aligned earlier by cross-correlation against the
shipped WavLM model (mngu0_peralign_refit.py), the cross-correlation offset minus sil_end * 50 has
mean 1.39 and std 0.97 frames (r = 0.95), so the label rule recovers the alignment to about a frame
without depending on any SSL model. The remaining global shift is selected per model on the
validation set (see SHIFT_CANDIDATES), since encoders can differ in their effective latency.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import decimate

MNGU0_ROOT = Path("/data/group_data/UTD-NAS/Databases/mngu0/extracted")
EMA_NORM_DIR = MNGU0_ROOT / "ema_norm" / "mngu0_s1_ema_norm_1.0.1"
WAV_DIR = MNGU0_ROOT / "wav_16kHz" / "mngu0_s1_wav_16kHz_1.1.0"
LAB_DIR = MNGU0_ROOT / "lab" / "mngu0_s1_lab_1.1.1"

FT_SR = 50  # Hz, SSL frame rate and EMA rate after decimation
EMA_NATIVE_SR = 200
SR = 16000
MIN_EMA_SECONDS = 0.2
SHIFT_CANDIDATES = (0, 1, 2, 3)  # frames added to round(sil_end * 50); chosen on validation
CHANNELS = ("TD_x", "TD_y", "TB_x", "TB_y", "TT_x", "TT_y", "LI_x", "LI_y", "UL_x", "UL_y", "LL_x", "LL_y")
ARTICULATORS = ("TD", "TB", "TT", "LI", "UL", "LL")  # MNGU0 T3, T2, T1, jaw, upper lip, lower lip
CM_TO_MM = 10.0
SILENCE = "#"


def split_of(stem):
    """Standard MNGU0 split of an utterance ID like 'mngu0_s1_0130'."""
    num = stem.rsplit("_", 1)[1]
    if num.endswith("0"):
        return "valid" if int(num[-2]) % 2 == 1 else "test"
    return "train"


@dataclass
class Utterance:
    stem: str
    wav: np.ndarray  # (n_samples,) float32, 16 kHz, untrimmed
    ema_mm: np.ndarray  # (T, 12) float32 at 50 Hz, trimmed to speech
    sil_end: float  # seconds, end of leading silence in the wav
    phones: list = field(default_factory=list)  # [(start_s, end_s, label)] in wav time

    @property
    def base_offset(self):
        return int(round(self.sil_end * FT_SR))


def read_est_track(path):
    """Minimal EST_Track binary reader; returns (times, data) with data = all channels."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"EST_File Track":
            raise ValueError(f"not an EST Track file: {path}")
        order, nrows, ncols = "<", 0, 0
        while True:
            line = f.readline().strip()
            if line == b"EST_Header_End":
                break
            if line == b"ByteOrder 10":
                order = ">"
            elif m := re.match(rb"NumFrames\s+(\d+)", line):
                nrows = int(m.group(1))
            elif m := re.match(rb"NumChannels\s+(\d+)", line):
                ncols = int(m.group(1))
        data = np.fromfile(f, dtype=np.dtype(order + "f4")).reshape(nrows, ncols + 2)
    return data[:, 0], data[:, 2:]


@lru_cache(maxsize=1)
def norm_params():
    d = EMA_NORM_DIR / "norm_parms"
    mean = np.loadtxt(d / "ema_means.txt")[:12]
    std = np.loadtxt(d / "ema_stds.txt")[:12]
    return mean, std


def load_ema_mm(stem):
    _, data = read_est_track(EMA_NORM_DIR / f"{stem}.ema")
    mean, std = norm_params()
    pos = data[:, :12].astype(np.float64) * 4 * std + mean
    pos = decimate(pos, EMA_NATIVE_SR // FT_SR, axis=0, zero_phase=True)
    return (pos * CM_TO_MM).astype(np.float32)


def load_phones(stem):
    """ESPS label file -> [(start_s, end_s, label)] covering the wav from 0."""
    text = (LAB_DIR / f"{stem}.lab").read_text()
    body = text.split("#\n", 1)[1]
    phones, start = [], 0.0
    for line in body.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        end = float(parts[0])
        phones.append((start, end, parts[-1]))
        start = end
    return phones


def ema_num_frames(stem):
    with open(EMA_NORM_DIR / f"{stem}.ema", "rb") as f:
        for line in f:
            if m := re.match(rb"NumFrames\s+(\d+)", line.strip()):
                return int(m.group(1))
    raise ValueError(stem)


@lru_cache(maxsize=1)
def available_stems():
    """Utterances with EMA, audio and labels and at least MIN_EMA_SECONDS of (trimmed) EMA, sorted.

    A few EMA files are only a handful of samples long after silence trimming; they are too short
    to decimate (scipy's zero-phase filter needs > 27 samples) and carry no usable trajectory.
    """
    ema = {p.stem for p in EMA_NORM_DIR.glob("mngu0_s1_*.ema")}
    wav = {p.stem for p in WAV_DIR.glob("mngu0_s1_*.wav")}
    lab = {p.stem for p in LAB_DIR.glob("mngu0_s1_*.lab")}
    stems = sorted(ema & wav & lab)
    return [s for s in stems if ema_num_frames(s) >= MIN_EMA_SECONDS * EMA_NATIVE_SR]


def stems_by_split():
    out = {"train": [], "valid": [], "test": []}
    for s in available_stems():
        out[split_of(s)].append(s)
    return out


def load_utterance(stem):
    wav, sr = sf.read(WAV_DIR / f"{stem}.wav", dtype="float32")
    assert sr == SR, (stem, sr)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    phones = load_phones(stem)
    sil_end = phones[0][1] if phones and phones[0][2] == SILENCE else 0.0
    return Utterance(stem=stem, wav=wav, ema_mm=load_ema_mm(stem), sil_end=sil_end, phones=phones)


def align(feats, ema, offset):
    """Pair SSL frames feats[offset + j] with ema[j]; returns equal-length (feats, ema) slices."""
    if offset >= 0:
        f, e = feats[offset:], ema
    else:
        f, e = feats, ema[-offset:]
    n = min(len(f), len(e))
    return f[:n], e[:n]


def frame_phones(utt, offset, n):
    """Phone label for each of the n aligned EMA frames (EMA frame j sits at SSL frame offset + j)."""
    times = (offset + np.arange(n) + 0.5) / FT_SR
    ends = np.array([p[1] for p in utt.phones])
    idx = np.minimum(np.searchsorted(ends, times, side="right"), len(utt.phones) - 1)
    return [utt.phones[i][2] for i in idx]
