"""Datasets for the compression experiments: a common interface over MNGU0 and the multi-speaker EMA corpora.

Interface (what probe / train / extract / prune / components need):
    name, shift_candidates, splits() -> {split: [stem]}, available_stems(), n_samples(stem) (16 kHz),
    load_utterance(stem) -> Utterance (stem, wav, ema_mm = fitting targets, base_offset, phones, speaker),
    frame_phones(utt, offset, n), to_mm(utt, arr) -> mm, evaluate(utts, preds, trues, phones) -> metrics.

mngu0      -- the single-speaker MNGU0 setup used so far (targets in mm; splits: train / valid / test).
ema_multi  -- USC-TIMIT EMA + USC EMA_5EMO (sparc.ema_corpora outputs). Targets are z-scored per speaker and
              channel with that speaker's training-split statistics (standard per-articulator normalization,
              no cross-speaker transform). Metrics are reported in mm (z-errors scaled back by each speaker's
              channel std, comparable with MNGU0), in z units (rmse_z), and per speaker.
              Speakers usc_F1 and 5emo_kf are held out entirely: their test sentences form test_unseen (and
              their validation sentences valid_unseen); the other five speakers' train / valid / test splits
              are text-disjoint (USC-TIMIT by sentence id, 5EMO by prompt). Utterances are trimmed to their
              valid EMA frames (edge frames lost to the latency correction) and the 38 utterances with internal
              dropout gaps are excluded.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from . import mngu0
from .metrics import ema_metrics


@dataclass
class Utterance:
    stem: str
    wav: np.ndarray
    ema_mm: np.ndarray  # (T, 12) fitting targets (mm for mngu0, per-speaker z for ema_multi)
    base_offset: int  # SSL frame of target frame 0 before the alignment shift
    phones: list = field(default_factory=list)
    speaker: str = ""
    sil_end: float = 0.0
    norm_key: str = ""  # normalization group (speaker, or a session group such as usc_F5_s1)


class MNGU0:
    name = "mngu0"
    shift_candidates = mngu0.SHIFT_CANDIDATES

    def splits(self):
        return mngu0.stems_by_split()

    def available_stems(self):
        return mngu0.available_stems()

    def n_samples(self, stem):
        return sf.info(mngu0.WAV_DIR / f"{stem}.wav").frames

    def load_utterance(self, stem):
        u = mngu0.load_utterance(stem)
        return Utterance(stem=stem, wav=u.wav, ema_mm=u.ema_mm, base_offset=u.base_offset, phones=u.phones,
                         speaker="mngu0", sil_end=u.sil_end)

    def load_targets(self, stem):
        """Like load_utterance but without audio (cheap)."""
        phones = mngu0.load_phones(stem)
        sil_end = phones[0][1] if phones and phones[0][2] == mngu0.SILENCE else 0.0
        u = mngu0.Utterance(stem=stem, wav=None, ema_mm=mngu0.load_ema_mm(stem), sil_end=sil_end, phones=phones)
        return Utterance(stem=stem, wav=None, ema_mm=u.ema_mm, base_offset=u.base_offset, phones=phones,
                         speaker="mngu0", sil_end=sil_end)

    def frame_phones(self, u, offset, n):
        return mngu0.frame_phones(mngu0.Utterance(stem=u.stem, wav=None, ema_mm=None, sil_end=u.sil_end,
                                                  phones=u.phones), offset, n)

    def to_mm(self, u, arr):
        return arr

    def evaluate(self, utts, preds, trues, phones=None, n_boot=1000):
        return ema_metrics(preds, trues, phones=phones, n_boot=n_boot)


EMA_ROOT = Path("/data/user_data/xoy/ema_corpora")
HELD_OUT = ("usc_F1", "5emo_kf")
EMO_TEST_PROMPTS = {"sent3", "sent7", "pssg1_phrase03", "pssg1_phrase08", "pssg2_phrase03"}
EMO_VALID_PROMPTS = {"sent5", "pssg1_phrase05", "pssg2_phrase01"}


def emo_split(prompt):
    if prompt in EMO_TEST_PROMPTS:
        return "test"
    if prompt in EMO_VALID_PROMPTS:
        return "valid"
    return "train"


class EMAMulti:
    name = "ema_multi"
    shift_candidates = (-1, 0, 1)

    def __init__(self, root=EMA_ROOT, held_out=HELD_OUT, norm_by="speaker"):
        from sparc.ema_corpora.dataset import load_manifest

        self.root = Path(root)
        self.held_out = tuple(held_out)
        if norm_by not in ("speaker", "norm_group"):
            raise ValueError(norm_by)
        self.norm_by = norm_by  # 'norm_group' gives usc_F5's two recording sessions separate statistics
        rows = load_manifest(self.root)
        self.rows = {}
        for r in rows:
            v = np.load(self.root / r["ema"])["valid"].all(1)
            idx = np.flatnonzero(v)
            if len(idx) < 25 or not v[idx[0]:idx[-1] + 1].all():
                continue  # no usable EMA, or an internal dropout gap
            r = dict(r)
            r["_first"], r["_last"] = int(idx[0]), int(idx[-1]) + 1
            split = r["split"] if r["corpus"] == "usc_timit" else emo_split(r["prompt"])
            r["_split"] = split
            self.rows[r["utt_id"]] = r
        self._stats = self._speaker_stats()

    def _speaker_stats(self):
        """Per-speaker channel mean/std from each speaker's training-split utterances (valid frames)."""
        acc = {}
        for r in self.rows.values():
            if r["_split"] != "train":
                continue
            e = np.load(self.root / r["ema"])["ema"][r["_first"]:r["_last"]].astype(np.float64)
            a = acc.setdefault(r[self.norm_by], [np.zeros(12), np.zeros(12), 0])
            a[0] += e.sum(0)
            a[1] += (e * e).sum(0)
            a[2] += len(e)
        return {s: (m := a[0] / a[2], np.sqrt(a[1] / a[2] - m**2)) for s, a in acc.items()}

    def splits(self):
        out = {"train": [], "valid": [], "test": [], "valid_unseen": [], "test_unseen": []}
        for stem, r in sorted(self.rows.items()):
            if r["speaker"] in self.held_out:
                if r["_split"] in ("valid", "test"):
                    out[f"{r['_split']}_unseen"].append(stem)
            else:
                out[r["_split"]].append(stem)
        return out

    def available_stems(self):
        return sorted(self.rows)

    def n_samples(self, stem):
        return sf.info(self.root / self.rows[stem]["audio"]).frames

    def speaker(self, stem):
        return self.rows[stem]["speaker"]

    def load_targets(self, stem, with_audio=False):
        r = self.rows[stem]
        mean, std = self._stats[r[self.norm_by]]
        e = np.load(self.root / r["ema"])["ema"][r["_first"]:r["_last"]].astype(np.float64)
        z = ((e - mean) / std).astype(np.float32)
        wav = sf.read(self.root / r["audio"], dtype="float32")[0] if with_audio else None
        return Utterance(stem=stem, wav=wav, ema_mm=z, base_offset=r["_first"], speaker=r["speaker"],
                         norm_key=r[self.norm_by])

    def load_utterance(self, stem):
        return self.load_targets(stem, with_audio=True)

    def frame_phones(self, u, offset, n):
        return []

    def to_mm(self, u, arr):
        mean, std = self._stats[u.norm_key or u.speaker]
        return arr * std + mean

    def evaluate(self, utts, preds, trues, phones=None, n_boot=1000):
        """Metrics in mm (per-speaker de-normalization), z-unit RMSE, and per-speaker breakdown."""
        pm = [self.to_mm(u, p) for u, p in zip(utts, preds)]
        tm = [self.to_mm(u, t) for u, t in zip(utts, trues)]
        out = ema_metrics(pm, tm, n_boot=n_boot)
        P, Y = np.concatenate(preds), np.concatenate(trues)
        out["rmse_z"] = float(np.sqrt(((P - Y) ** 2).mean(0)).mean())
        out["per_speaker"] = {}
        for spk in sorted({u.speaker for u in utts}):
            idx = [i for i, u in enumerate(utts) if u.speaker == spk]
            m = ema_metrics([pm[i] for i in idx], [tm[i] for i in idx], n_boot=0)
            Pz = np.concatenate([preds[i] for i in idx])
            Yz = np.concatenate([trues[i] for i in idx])
            out["per_speaker"][spk] = {"n_utts": len(idx), "rmse": m["rmse"], "pcc": m["pcc"],
                                       "rmse_z": float(np.sqrt(((Pz - Yz) ** 2).mean(0)).mean()),
                                       "vel_rmse": m["vel_rmse"]}
        return out


@lru_cache(maxsize=4)
def get(name):
    if name == "mngu0":
        return MNGU0()
    if name == "ema_multi":
        return EMAMulti()
    if name.startswith("ema_loso_"):  # leave-one-speaker-out fold: train on the other six speakers
        spk = name[len("ema_loso_"):]
        if spk not in ALL_SPEAKERS:
            raise ValueError(f"unknown speaker {spk!r}; one of {ALL_SPEAKERS}")
        return EMAMulti(held_out=(spk,), norm_by="norm_group")
    raise ValueError(name)


ALL_SPEAKERS = ("usc_M1", "usc_F1", "usc_M3", "usc_F5", "5emo_jn", "5emo_jr", "5emo_kf")
DATASET_NAMES = ("mngu0", "ema_multi") + tuple(f"ema_loso_{s}" for s in ALL_SPEAKERS)


XLSR_EMA = Path("/data/user_data/xoy/xlsr_ema")


def features_root(name):
    """Feature caches / probe results: features/ for mngu0 (unchanged), features_<name>/ otherwise."""
    return XLSR_EMA / ("features" if name == "mngu0" else f"features_{name}")


def adapt_root(name):
    return XLSR_EMA / ("adapt" if name == "mngu0" else f"adapt_{name}")
