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
ema_loso_<speaker>  -- leave-one-speaker-out fold: that speaker held out, normalization per norm group.
<name>_<target>     -- the same split with a PCA-based target reparameterization (TARGETS), fitted per normalization
              group on its training-split frames and inverted for evaluation, so metrics stay in the original
              per-channel z space (rmse_target is the error in the fitting space):
              zca   -- per-articulator whitening of the (x, y) mm coordinates that keeps their orientation
                       (C^-1/2: removes the within-articulator x/y correlation and equalizes the two variances);
              pca   -- per-articulator principal axes, each scaled to unit variance, with each group's axes ordered
                       and signed to match a reference (the mean training-group covariance), paired by direction
                       rather than by variance rank, so no group is rotated by more than 45 deg. Equal to zca followed
                       by a per-group rotation onto shared axes, up to one rotation common to all groups;
              zca12 -- whitening of all 12 channels jointly (also removes cross-articulator correlation).
              Like the z-score, a new speaker's transform needs that speaker's own EMA statistics.
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


TARGETS = ("z", "zca", "pca", "zca12")
N_ART = 6


def _inv_sqrt(C):
    w, V = np.linalg.eigh(C)
    return (V / np.sqrt(w)) @ V.T


def _sorted_eig(C):
    """Eigenvectors as columns, descending variance, sign fixed so the largest-magnitude entry is positive."""
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    return w, V * np.sign(V[np.abs(V).argmax(0), range(V.shape[1])])


def _aligned_eig(C, R):
    """Eigen-decomposition of C with its axes permuted and signed to best match the reference axes R (columns)."""
    w, V = _sorted_eig(C)
    perms = [(0, 1), (1, 0)]
    p = max(perms, key=lambda p: np.abs(np.diag(V[:, p].T @ R)).sum())
    w, V = w[list(p)], V[:, list(p)]
    return w, V * np.where(np.diag(V.T @ R) >= 0, 1.0, -1.0)


def target_matrices(kind, cov, std, ref=None):
    """A (12 x 12) mapping a per-channel z-scored frame to the fitting target y = A z, and its inverse.

    cov: (12, 12) mm covariance of the group; std: (12,) channel std; ref: per-articulator reference axes for 'pca'.
    Channels are ordered (art0_x, art0_y, art1_x, ...)."""
    S = np.diag(std)
    if kind == "z":
        A = np.eye(12)
    elif kind == "zca12":
        A = _inv_sqrt(cov) @ S
    elif kind in ("zca", "pca"):
        A = np.zeros((12, 12))
        for a in range(N_ART):
            i = slice(2 * a, 2 * a + 2)
            C = cov[i, i]
            if kind == "zca":
                A[i, i] = _inv_sqrt(C) @ S[i, i]
            else:
                w, V = _aligned_eig(C, ref[a])
                A[i, i] = (V / np.sqrt(w)).T @ S[i, i]
    else:
        raise ValueError(kind)
    return A, np.linalg.inv(A)


def emo_split(prompt):
    if prompt in EMO_TEST_PROMPTS:
        return "test"
    if prompt in EMO_VALID_PROMPTS:
        return "valid"
    return "train"


class EMAMulti:
    name = "ema_multi"
    shift_candidates = (-1, 0, 1)

    def __init__(self, root=EMA_ROOT, held_out=HELD_OUT, norm_by="speaker", target="z"):
        from sparc.ema_corpora.dataset import load_manifest

        self.root = Path(root)
        self.held_out = tuple(held_out)
        if norm_by not in ("speaker", "norm_group"):
            raise ValueError(norm_by)
        self.norm_by = norm_by  # 'norm_group' gives usc_F5's two recording sessions separate statistics
        if target not in TARGETS:
            raise ValueError(target)
        self.target = target
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
        self._stats, self._cov = self._speaker_stats()
        self._tf = self._target_transforms()

    def _speaker_stats(self):
        """Per-group channel mean/std and mm covariance from the group's training-split utterances (valid frames)."""
        acc = {}
        for r in self.rows.values():
            if r["_split"] != "train":
                continue
            e = np.load(self.root / r["ema"])["ema"][r["_first"]:r["_last"]].astype(np.float64)
            a = acc.setdefault(r[self.norm_by], [np.zeros(12), np.zeros((12, 12)), 0])
            a[0] += e.sum(0)
            a[1] += e.T @ e
            a[2] += len(e)
        stats, cov = {}, {}
        for g, (s1, s2, n) in acc.items():
            m = s1 / n
            cov[g] = s2 / n - np.outer(m, m)
            stats[g] = (m, np.sqrt(np.diag(cov[g])))
        return stats, cov

    def _target_transforms(self):
        """{group: (A, A^-1)} with y = A z. The 'pca' reference axes come from the training groups only."""
        ref = None
        if self.target == "pca":
            train_groups = {r[self.norm_by] for r in self.rows.values() if r["speaker"] not in self.held_out}
            C = np.mean([self._cov[g] for g in sorted(train_groups)], 0)
            ref = [_sorted_eig(C[2 * a:2 * a + 2, 2 * a:2 * a + 2])[1] for a in range(N_ART)]
        return {g: target_matrices(self.target, self._cov[g], self._stats[g][1], ref) for g in self._stats}

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
        z = (e - mean) / std
        if self.target != "z":
            z = z @ self._tf[r[self.norm_by]][0].T
        z = z.astype(np.float32)
        wav = sf.read(self.root / r["audio"], dtype="float32")[0] if with_audio else None
        return Utterance(stem=stem, wav=wav, ema_mm=z, base_offset=r["_first"], speaker=r["speaker"],
                         norm_key=r[self.norm_by])

    def load_utterance(self, stem):
        return self.load_targets(stem, with_audio=True)

    def frame_phones(self, u, offset, n):
        return []

    def to_z(self, u, arr):
        """Fitting targets -> per-channel z units (identity unless a reparameterized target is used)."""
        if self.target == "z":
            return arr
        return arr @ self._tf[u.norm_key or u.speaker][1].T

    def to_mm(self, u, arr):
        mean, std = self._stats[u.norm_key or u.speaker]
        return self.to_z(u, arr) * std + mean

    def evaluate(self, utts, preds, trues, phones=None, n_boot=1000):
        """Metrics in mm (per-speaker de-normalization), z-unit RMSE, and per-speaker breakdown.

        With a reparameterized target, predictions are mapped back to per-channel z first, so every metric is
        comparable with target 'z'; rmse_target is the error in the fitting space."""
        rmse_target = None
        if self.target != "z":
            P, Y = np.concatenate(preds), np.concatenate(trues)
            rmse_target = float(np.sqrt(((P - Y) ** 2).mean(0)).mean())
            preds = [self.to_z(u, p) for u, p in zip(utts, preds)]
            trues = [self.to_z(u, t) for u, t in zip(utts, trues)]
        pm = [self._z_to_mm(u, p) for u, p in zip(utts, preds)]
        tm = [self._z_to_mm(u, t) for u, t in zip(utts, trues)]
        out = ema_metrics(pm, tm, n_boot=n_boot)
        P, Y = np.concatenate(preds), np.concatenate(trues)
        out["rmse_z"] = float(np.sqrt(((P - Y) ** 2).mean(0)).mean())
        if rmse_target is not None:
            out["rmse_target"] = rmse_target
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

    def _z_to_mm(self, u, z):
        mean, std = self._stats[u.norm_key or u.speaker]
        return z * std + mean


def split_target(name):
    """'ema_loso_usc_M1_pca' -> ('ema_loso_usc_M1', 'pca'); names without a target suffix use 'z'."""
    for t in TARGETS[1:]:
        if name.endswith("_" + t):
            return name[:-len(t) - 1], t
    return name, "z"


@lru_cache(maxsize=4)
def get(name):
    if name == "mngu0":
        return MNGU0()
    base, target = split_target(name)
    if base == "ema_multi":
        return EMAMulti(target=target)
    if base.startswith("ema_loso_"):  # leave-one-speaker-out fold: train on the other six speakers
        spk = base[len("ema_loso_"):]
        if spk not in ALL_SPEAKERS:
            raise ValueError(f"unknown speaker {spk!r}; one of {ALL_SPEAKERS}")
        return EMAMulti(held_out=(spk,), norm_by="norm_group", target=target)
    raise ValueError(name)


ALL_SPEAKERS = ("usc_M1", "usc_F1", "usc_M3", "usc_F5", "5emo_jn", "5emo_jr", "5emo_kf")
_BASES = ("ema_multi",) + tuple(f"ema_loso_{s}" for s in ALL_SPEAKERS)
DATASET_NAMES = ("mngu0",) + _BASES + tuple(f"{b}_{t}" for b in _BASES for t in TARGETS[1:])


XLSR_EMA = Path("/data/user_data/xoy/xlsr_ema")


def features_root(name):
    """Feature caches / probe results: features/ for mngu0 (unchanged), features_<name>/ otherwise."""
    return XLSR_EMA / ("features" if name == "mngu0" else f"features_{name}")


def adapt_root(name):
    return XLSR_EMA / ("adapt" if name == "mngu0" else f"adapt_{name}")
