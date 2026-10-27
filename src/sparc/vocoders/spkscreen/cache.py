"""Frame-level cache of the speaker-encoder screen (docs/spkemb_encoder_screen.md).

Two sets, each a manifest plus chunks of ``CHUNK`` utterances (resumable; a chunk is complete when ``done`` exists):

- ``train``: LibriTTS-R train.clean.100, up to 30 utterances per speaker from the packed cache index;
- ``eval``: the 5470 utterances of the Phase 1 layer screen (``/data/user_data/xoy/spkemb/layers/*.npz``), in Phase 1's
  row order, with Phase 1's targets and its 62 held-out test speakers.

Per chunk (``<root>/<set>/chunks/chunk_<c>/``): ``hid_L<k>.npy`` fp16 ``[frames, 1024]`` for the layers in ``LAYERS``,
``aux.npy`` float32 ``[frames, 4]`` (periodicity, z-scored loudness, raw loudness, F0 Hz), ``mel.npy`` fp16
``[frames, 80]``, ``lens.npy`` int32, ``targets.npy`` float32 ``[n, TARGET_DIM]``, ``logf0.npy``, ``check.json``.

Usage::

    python -m sparc.vocoders.spkscreen.cache manifest
    python -m sparc.vocoders.spkscreen.cache extract --set train --worker 0 --nworkers 4
    python -m sparc.vocoders.spkscreen.cache check
"""

import argparse
import glob
import json
import os
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SPKSCREEN_ROOT", "/data/user_data/xoy/spkemb/encoder_screen"))
SV_ROOT = Path(os.environ.get("SV_ROOT", "/data/user_data/xoy/sparc-vocoders"))
PACKED = SV_ROOT / "cache" / "packed" / "train.clean.100"
PHASE1 = Path("/data/user_data/xoy/spkemb/layers")
LAYERS = (0, 1, 3, 6, 12, 24)
CHUNK = 250
PER_SPK = 30
T_MIN, T_MAX = 100, 800
MIN_UTT_PER_SPK = 20
MIN_SAMPLES = 9600  # 0.4 s at 24 kHz; the extractor fails below 9239 samples
N_MEL = 80
TARGET_DIM = 80 + 80 + 24 + 1  # logmel mean, logmel std, MFCC 1-24 mean, tilt (Phase 1 targets_matrix order)
AUX_COLS = ("periodicity", "zloud", "loud", "f0")


def utterance_seed(utt_id: str) -> int:
    return zlib.crc32(utt_id.encode()) & 0x7FFFFFFF


# ------------------------------------------------------------------------------------------------ manifests


def phase1_rows() -> dict:
    """Phase 1 data concatenated in its analysis order (sorted npz files)."""
    keys = ("spk", "path", "logmel_mean", "logmel_std", "mfcc_mean", "tilt", "logf0", "en_plus")
    parts = sorted(glob.glob(str(PHASE1 / "*_*.npz")))
    if not parts:
        raise FileNotFoundError(f"no Phase 1 npz files in {PHASE1}")
    out = {k: [] for k in keys}
    for p in parts:
        z = np.load(p, allow_pickle=True)
        for k in keys:
            out[k].append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def phase1_test_speakers(speakers: np.ndarray) -> list[str]:
    """Phase 1's held-out speakers (``layers.analyze``: ``default_rng(0).choice(sorted(speakers), n // 3)``)."""
    spks = sorted(set(speakers))
    rng = np.random.default_rng(0)
    return sorted(rng.choice(spks, len(spks) // 3, replace=False))


def build_manifests(root: Path = ROOT) -> None:
    import pandas as pd

    # eval
    p1 = phase1_rows()
    test = set(phase1_test_speakers(p1["spk"]))
    ids = [f"{s}:{Path(str(p)).stem}" for s, p in zip(p1["spk"], p1["path"])]
    ev = pd.DataFrame(
        {"id": ids, "speaker": [str(s) for s in p1["spk"]], "wav_path": [str(p) for p in p1["path"]],
         "is_test": [s in test for s in p1["spk"]]}
    )
    ev["seed"] = [utterance_seed(i) for i in ev["id"]]
    if ev["id"].duplicated().any():
        raise ValueError("duplicate evaluation ids")
    targets = np.concatenate([p1["logmel_mean"], p1["logmel_std"], p1["mfcc_mean"], p1["tilt"]], 1).astype(np.float32)
    if targets.shape[1] != TARGET_DIM:
        raise ValueError(f"unexpected target width {targets.shape}")
    (root / "heldout").mkdir(parents=True, exist_ok=True)
    ev.to_parquet(root / "heldout" / "manifest.parquet")
    np.savez(root / "heldout" / "phase1.npz", targets=targets, logf0=p1["logf0"].reshape(-1).astype(np.float64),
             en_plus=p1["en_plus"].astype(np.float32))
    # train
    idx = pd.read_parquet(PACKED / "index.parquet")
    idx["row"] = np.arange(len(idx))
    ok = idx[(idx["T"] >= T_MIN) & (idx["T"] <= T_MAX) & (~idx["spk_fallback"])]
    rng = np.random.default_rng(0)
    rows, dropped = [], []
    for spk in sorted(ok["speaker"].unique(), key=str):
        sub = ok[ok["speaker"] == spk]
        if len(sub) < MIN_UTT_PER_SPK:
            dropped.append(str(spk))
            continue
        pick = rng.permutation(len(sub))[:PER_SPK]
        rows.append(sub.iloc[np.sort(pick)])
    tr = pd.concat(rows)
    tr = pd.DataFrame({"id": tr["id"].astype(str).values, "speaker": tr["speaker"].astype(str).values,
                       "wav_path": tr["wav_path"].astype(str).values, "seed": tr["seed"].astype(np.int64).values,
                       "packed_row": tr["row"].values, "packed_offset": tr["offset"].values,
                       "packed_T": tr["T"].values})
    ev_numeric = {s.split(":")[1] for s in ev["speaker"] if s.startswith("libri")}
    overlap = ev_numeric & set(tr["speaker"])
    if overlap:
        raise ValueError(f"evaluation speakers in training: {sorted(overlap)}")
    (root / "train").mkdir(parents=True, exist_ok=True)
    tr.to_parquet(root / "train" / "manifest.parquet")
    summary = {"train_utts": len(tr), "train_speakers": int(tr["speaker"].nunique()), "train_dropped_speakers": dropped,
               "train_frames_packed": int(tr["packed_T"].sum()), "eval_utts": len(ev),
               "eval_speakers": int(ev["speaker"].nunique()), "eval_test_speakers": len(test),
               "layers": list(LAYERS)}
    (root / "manifest_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


GLOBE_TABLE = Path("/data/user_data/xoy/spkemb/phase4/pools/globe/table.parquet")


def build_globe_manifest(root: Path = ROOT) -> None:
    """Out-of-domain set of the precondition checks: the Phase 4 GLOBE pool (400 speakers, 3 utterances each)."""
    import pandas as pd

    t = pd.read_parquet(GLOBE_TABLE)
    t = t[t["n24"] >= MIN_SAMPLES].copy()
    g = pd.DataFrame({"id": "globe:" + t["id"].astype(str), "speaker": "globe:" + t["speaker"].astype(str),
                      "wav_path": t["wav_path"].astype(str)})
    g = g[g.groupby("speaker")["id"].transform("size") >= 2].reset_index(drop=True)
    g["seed"] = [utterance_seed(i) for i in g["id"]]
    (root / "globe").mkdir(parents=True, exist_ok=True)
    g.to_parquet(root / "globe" / "manifest.parquet")
    print(f"globe: {len(g)} utterances, {g['speaker'].nunique()} speakers")


def load_manifest(set_name: str, root: Path = ROOT):
    import pandas as pd

    return pd.read_parquet(root / set_name / "manifest.parquet")


# ------------------------------------------------------------------------------------------------ extraction


def acoustic_targets(raw16: np.ndarray, periodicity: np.ndarray, f0_hz: np.ndarray) -> tuple[np.ndarray, float]:
    """Phase 1 targets of one utterance (``sparc.spkemb.layers.acoustic_targets``) on the z-scored 16 kHz waveform.

    Returns ``(targets [TARGET_DIM], ln median F0 over voiced frames or NaN)``.
    """
    import librosa

    wav = (raw16 - raw16.mean()) / raw16.std()
    mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=1024, hop_length=320, n_mels=80, fmax=8000)
    logmel = np.log(mel + 1e-6).T
    mfcc = librosa.feature.mfcc(S=logmel.T, n_mfcc=25).T[:, 1:]
    n = min(len(logmel), len(periodicity))
    voiced = periodicity[:n] > 0
    if voiced.sum() < 5:
        voiced = np.ones(n, dtype=bool)
    lm, mf = logmel[:n][voiced], mfcc[:n][voiced]
    mean = lm.mean(0)
    tilt = np.polyfit(np.arange(80), mean, 1)[0]
    f0 = f0_hz[:n][voiced[: len(f0_hz)]]
    f0 = f0[f0 > 0]
    tg = np.concatenate([mean, lm.std(0), mf.mean(0), [tilt]]).astype(np.float32)
    return tg, float(np.log(np.median(f0))) if len(f0) else float("nan")


def logmel24(wav24: np.ndarray, frames: int) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(y=wav24, sr=24000, n_fft=1024, hop_length=480, n_mels=N_MEL, fmax=12000)
    out = np.log(mel + 1e-6).T.astype(np.float32)
    if len(out) < frames:
        out = np.concatenate([out, np.repeat(out[-1:], frames - len(out), 0)])
    return out[:frames]


class FrameExtractor:
    """SPARC features and the selected WavLM layers of one 24 kHz waveform."""

    def __init__(self, device: str = "auto"):
        import torch
        from omegaconf import OmegaConf

        from sparc.vocoders.eval.reextract import extractor_config
        from sparc.vocoders.features.extractor import SparcFeatureExtractor
        from sparc.vocoders.features.spk_layers import load_wavlm

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        cfg = extractor_config(OmegaConf.create({}))
        cfg.extractor.dither = True
        self.extractor = SparcFeatureExtractor(cfg, self.device)
        self.wavlm = load_wavlm(self.device)

    def encode(self, wav24: np.ndarray, seed: int) -> dict:
        """Features of one utterance. Utterances under ``MIN_SAMPLES`` (cannot be encoded, issue #12) are padded with
        digital silence at the end and their frames cut back to the utterance's own length."""
        if len(wav24) >= MIN_SAMPLES:
            return self._encode(wav24, seed)
        keep = max(1, len(wav24) // 480 - 1)
        out = self._encode(np.concatenate([wav24, np.zeros(MIN_SAMPLES - len(wav24), np.float32)]), seed)
        out["hidden"], out["aux"], out["mel"] = out["hidden"][:, :keep], out["aux"][:keep], out["mel"][:keep]
        out["padded"] = True
        return out

    def _encode(self, wav24: np.ndarray, seed: int) -> dict:
        import librosa

        from sparc.vocoders.constants import F0_CHANNEL, LOUDNESS_CHANNEL, PERIODICITY_CHANNEL, SAMPLE_RATE
        from sparc.vocoders.features.spk_layers import hidden_states

        out = self.extractor.extract(np.ascontiguousarray(wav24, dtype=np.float32), seed)
        feats = out["feats"]
        hid = hidden_states(self.wavlm, wav24, self.device)
        frames = min(len(feats), hid.shape[1])
        if abs(len(feats) - hid.shape[1]) > 1:
            raise ValueError(f"feature frames {len(feats)} and WavLM frames {hid.shape[1]} differ")
        aux = np.stack(
            [feats[:frames, PERIODICITY_CHANNEL], feats[:frames, LOUDNESS_CHANNEL], out["loud_raw"][:frames],
             feats[:frames, F0_CHANNEL]], 1).astype(np.float32)
        raw16 = librosa.resample(wav24, orig_sr=SAMPLE_RATE, target_sr=16000, res_type="soxr_hq")
        tg, logf0 = acoustic_targets(raw16, aux[:, 0], aux[:, 3])
        return {"hidden": hid[list(LAYERS), :frames].astype(np.float16), "aux": aux,
                "mel": logmel24(wav24, frames).astype(np.float16), "targets": tg, "logf0": logf0,
                "spk_l6": out["spk_l6"]}


def read_wav24(path: str) -> np.ndarray:
    import librosa
    import soundfile as sf

    wav, rate = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim != 1:
        wav = wav.mean(1)
    if rate != 24000:
        wav = librosa.resample(wav, orig_sr=rate, target_sr=24000, res_type="soxr_hq")
    return np.ascontiguousarray(wav, dtype=np.float32)


def n_chunks(n: int) -> int:
    return -(-n // CHUNK)


def chunk_dir(root: Path, set_name: str, c: int) -> Path:
    return root / set_name / "chunks" / f"chunk_{c:05d}"


def extract(set_name: str, worker: int, nworkers: int, root: Path = ROOT, limit_chunks: int | None = None) -> None:
    man = load_manifest(set_name, root)
    todo = [c for c in range(n_chunks(len(man))) if c % nworkers == worker and not (chunk_dir(root, set_name, c) / "done").exists()]
    if limit_chunks is not None:
        todo = todo[:limit_chunks]
    if not todo:
        print(f"{set_name} worker {worker}: nothing to do")
        return
    fx = FrameExtractor()
    packed_feats = np.load(PACKED / "feats.npy", mmap_mode="r") if set_name == "train" else None
    for c in todo:
        rows = range(c * CHUNK, min(len(man), (c + 1) * CHUNK))
        hid = {k: [] for k in LAYERS}
        aux, mel, lens, tgs, f0s = [], [], [], [], []
        worst = {"periodicity_maxdiff": 0.0, "l6_cosine_min": 1.0, "padded_utts": 0}
        for r in rows:
            row = man.iloc[r]
            enc = fx.encode(read_wav24(row["wav_path"]), int(row["seed"]))
            for j, k in enumerate(LAYERS):
                hid[k].append(enc["hidden"][j])
            aux.append(enc["aux"])
            mel.append(enc["mel"])
            lens.append(len(enc["aux"]))
            worst["padded_utts"] += int(enc.get("padded", False))
            tgs.append(enc["targets"])
            f0s.append(enc["logf0"])
            if packed_feats is not None:
                o, t = int(row["packed_offset"]), int(row["packed_T"])
                n = min(t, len(enc["aux"]))
                ref = np.asarray(packed_feats[o : o + n, 14])
                worst["periodicity_maxdiff"] = max(worst["periodicity_maxdiff"], float(np.abs(ref - enc["aux"][:n, 0]).max()))
                ref6 = np.load(PACKED / "spk_l6.npy", mmap_mode="r")[int(row["packed_row"])]
                a = enc["spk_l6"].astype(np.float64)
                worst["l6_cosine_min"] = min(worst["l6_cosine_min"], float(a @ ref6 / (np.linalg.norm(a) * np.linalg.norm(ref6))))
        d = chunk_dir(root, set_name, c)
        d.mkdir(parents=True, exist_ok=True)
        for k in LAYERS:
            np.save(d / f"hid_L{k}.npy", np.concatenate(hid[k]))
        np.save(d / "aux.npy", np.concatenate(aux))
        np.save(d / "mel.npy", np.concatenate(mel))
        np.save(d / "lens.npy", np.array(lens, np.int32))
        np.save(d / "targets.npy", np.stack(tgs))
        np.save(d / "logf0.npy", np.array(f0s, np.float64))
        (d / "check.json").write_text(json.dumps({**worst, "utts": len(lens), "frames": int(sum(lens))}))
        (d / "done").touch()
        print(f"{set_name} chunk {c}: {len(lens)} utts, {sum(lens)} frames, {worst}", flush=True)


# ------------------------------------------------------------------------------------------------ check


def check(root: Path = ROOT) -> dict:
    report = {}
    for s in ("train", "heldout"):
        man = load_manifest(s, root)
        missing = [c for c in range(n_chunks(len(man))) if not (chunk_dir(root, s, c) / "done").exists()]
        frames, pd_max, cos_min = 0, 0.0, 1.0
        for c in range(n_chunks(len(man))):
            f = chunk_dir(root, s, c) / "check.json"
            if f.exists():
                j = json.loads(f.read_text())
                frames += j["frames"]
                pd_max = max(pd_max, j.get("periodicity_maxdiff", 0.0))
                cos_min = min(cos_min, j.get("l6_cosine_min", 1.0))
        report[s] = {"utts": len(man), "missing_chunks": missing, "frames": frames, "periodicity_maxdiff_vs_packed": pd_max,
                     "l6_cosine_min_vs_packed": cos_min}
    ev = np.load(root / "heldout" / "phase1.npz")
    tg = np.concatenate([np.load(chunk_dir(root, "heldout", c) / "targets.npy") for c in range(n_chunks(len(load_manifest("heldout", root))))
                         if (chunk_dir(root, "heldout", c) / "done").exists()])
    n = len(tg)
    if n:
        cc = [np.corrcoef(tg[:, j], ev["targets"][:n, j])[0, 1] for j in range(TARGET_DIM)]
        report["heldout"]["target_corr_vs_phase1_mean"] = float(np.nanmean(cc))
        report["heldout"]["target_corr_vs_phase1_min"] = float(np.nanmin(cc))
    total = sum(f.stat().st_size for f in root.rglob("*.npy"))
    report["size_gb"] = total / 1e9
    print(json.dumps(report, indent=1))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    sub.add_parser("globe-manifest")
    e = sub.add_parser("extract")
    e.add_argument("--set", dest="set_name", choices=("train", "heldout", "globe"), required=True)
    e.add_argument("--worker", type=int, default=0)
    e.add_argument("--nworkers", type=int, default=1)
    e.add_argument("--limit-chunks", type=int, default=None)
    sub.add_parser("check")
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args()
    if a.cmd == "manifest":
        build_manifests(a.root)
    elif a.cmd == "globe-manifest":
        build_globe_manifest(a.root)
    elif a.cmd == "extract":
        extract(a.set_name, a.worker, a.nworkers, a.root, a.limit_chunks)
    else:
        check(a.root)


if __name__ == "__main__":
    main()
