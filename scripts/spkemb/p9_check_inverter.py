"""Phase 9 checks of ``sparc.vocoders.features.k12_inverter`` (run on a GPU node through ``p5_job.sh``).

``sanity``    reproduces the stored test scores of ``frozen_k12_gruconv9`` (mngu0: all 61 test utterances; all7: all seen
              speakers' test sentences, overall and per speaker) from audio, with the training-time alignment, and
              compares the live features with the cached ones of the speech-to-EMA work.
``align``     N utterances of a packed split: frame count equals the packed ``T``, and the lag of the peak of the
              cross-correlation of every new channel with the packed layer-9 channel of the same index.
``timing``    seconds per audio hour of ``channels24`` on the first N utterances of a packed split (after warm-up).

The speech-to-EMA code (``sparc.compression``) lives on branch ``multispeaker-eval``; it is only read here, by adding
that checkout's ``sparc`` directory to the package path of this process.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REF_SRC = "/home/xoy/Speech-Articulatory-Coding/.claude/worktrees/fix-issue-16/src/sparc"


def reference_modules():
    import sparc

    if REF_SRC not in sparc.__path__:
        sparc.__path__.append(REF_SRC)
    from sparc.compression import datasets, probe
    from sparc.compression.metrics import ema_metrics

    return datasets, probe, ema_metrics


def scores(P, Y, ys, ema_metrics):
    Pc, Yc = np.concatenate(P), np.concatenate(Y)
    rmse_c = np.sqrt(((Pc - Yc) ** 2).mean(0))
    return {"rmse": float(rmse_c.mean()), "rmse_z": float((rmse_c / ys).mean()), "pcc": ema_metrics(P, Y, n_boot=0)["pcc"]}


def sanity(mode: str, limit: int | None, device: str) -> dict:
    from sparc.vocoders.features.k12_inverter import K12Inverter

    datasets, probe, ema_metrics = reference_modules()
    inv = K12Inverter(mode, device)
    if mode == "mngu0":
        ds, root = datasets.get("mngu0"), datasets.features_root("mngu0") / "wavlm-large"
    else:
        ds, root = datasets.EMAMulti(held_out=(), norm_by="norm_group"), datasets.features_root("ema_multi") / "wavlm-large"
    index = json.loads((root / "index.json").read_text())
    cache = np.load(root / f"layer_{inv.layer:02d}.npy", mmap_mode="r")
    stems = [s for s in ds.splits()["test"] if s in index][:limit]
    P, Y, kept, feat_err = [], [], [], []
    for s in stems:
        u = ds.load_utterance(s)
        x = inv.features(u.wav)
        a, n = index[s]
        if len(x) != n:
            raise ValueError(f"{s}: live {len(x)} frames, cache {n}")
        c = np.asarray(cache[a : a + n], dtype=np.float32)
        feat_err.append(float(np.abs(x - c).max() / (np.abs(c).max() + 1e-9)))
        offset = u.base_offset + inv.shift
        f, y = x[offset:], u.ema_mm
        m = min(len(f), len(y))
        if m < 2:
            continue
        P.append(inv.head_output(f[:m]))
        Y.append(y[:m].astype(np.float64))
        kept.append(s)
    res = {"mode": mode, "n_utterances": len(kept), "feature_max_rel_err_vs_cache": float(max(feat_err)),
           "feature_mean_rel_err_vs_cache": float(np.mean(feat_err)), "test": scores(P, Y, inv.ys, ema_metrics)}
    if mode == "all7":
        per = {}
        for spk in sorted({ds.rows[s]["speaker"] for s in kept}):
            idx = [i for i, s in enumerate(kept) if ds.rows[s]["speaker"] == spk]
            per[spk] = scores([P[i] for i in idx], [Y[i] for i in idx], inv.ys, ema_metrics)
        res["test_per_speaker"] = per
    stored = json.loads((Path(inv.ckpt_path).parent / "frozen_k12_gruconv9.json").read_text())["seeds"]["0"]
    res["stored_seed0"] = stored
    return res


def packed_rows(packed: str, n: int, spread: bool = True):
    import pandas as pd

    index = pd.read_parquet(Path(packed) / "index.parquet")
    pick = np.linspace(0, len(index) - 1, n).astype(int) if spread else np.arange(n)
    return index, index.iloc[np.unique(pick)]


def align(mode: str, packed: str, n: int, device: str) -> dict:
    """Per grid shift (``stored`` = the checkpoint's own shift, the default of the packing tool) the lag of the peak of the
    cross-correlation of every new channel with the packed layer-9 channel of the same index (lag < 0: the new channel
    leads the packed one) and the correlation at lag 0."""
    import soundfile as sf

    from sparc.vocoders.features.k12_inverter import K12Inverter

    inv = K12Inverter(mode, device)
    index, rows = packed_rows(packed, n)
    feats = np.load(Path(packed) / "feats.npy", mmap_mode="r")
    shifts = sorted({inv.shift, -1, 0, 1, 2, 3})
    lags = {k: [] for k in shifts}
    corr0 = {k: [] for k in shifts}
    peakc = {k: [] for k in shifts}
    same_t, names = [], []
    for _, r in rows.iterrows():
        wav, rate = sf.read(r["wav_path"], dtype="float32")
        raw16 = librosa_resample(wav)
        old = np.asarray(feats[int(r["offset"]) : int(r["offset"]) + int(r["T"]), :12])
        for k in shifts:
            out = inv.channels16(raw16, k)
            same_t.append(len(out) == len(old) == int(r["T"]))
            lag_row, c0_row, pk_row = [], [], []
            for c in range(12):
                a, b = out[:, c] - out[:, c].mean(), old[:, c] - old[:, c].mean()
                full = np.correlate(a, b, mode="full") / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                mid = len(b) - 1
                w = full[mid - 10 : mid + 11]
                lag_row.append(int(np.argmax(w) - 10))
                c0_row.append(float(w[10]))
                pk_row.append(float(w.max()))
            lags[k].append(lag_row)
            corr0[k].append(c0_row)
            peakc[k].append(pk_row)
        names.append(r["id"])
    res = {"mode": mode, "packed": packed, "n_utterances": len(names), "stored_shift": inv.shift,
           "all_T_equal": bool(all(same_t)), "by_grid_shift": {}}
    for k in shifts:
        L = np.array(lags[k])
        res["by_grid_shift"][str(k)] = {
            "lag_peak_hist": {int(v): int((L == v).sum()) for v in np.unique(L)},
            "fraction_lag0": float((L == 0).mean()), "median_lag_per_channel": np.median(L, 0).tolist(),
            "mean_corr_at_lag0": float(np.mean(corr0[k])), "mean_peak_corr": float(np.mean(peakc[k]))}
    return res


def librosa_resample(wav24):
    import librosa

    return librosa.resample(wav24, orig_sr=24000, target_sr=16000, res_type="soxr_hq")


def timing(mode: str, packed: str, n: int, device: str) -> dict:
    import soundfile as sf
    import torch

    from sparc.vocoders.features.k12_inverter import K12Inverter

    inv = K12Inverter(mode, device)
    index, rows = packed_rows(packed, n, spread=False)
    wavs = [sf.read(p, dtype="float32")[0] for p in rows["wav_path"]]
    inv.channels24(wavs[0])
    if inv.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for w in wavs:
        inv.channels24(w)
    if inv.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    hours = sum(len(w) for w in wavs) / 24000 / 3600
    return {"mode": mode, "device": torch.cuda.get_device_name(inv.device) if inv.device.type == "cuda" else "cpu",
            "n_utterances": len(wavs), "audio_hours": hours, "seconds": dt, "seconds_per_audio_hour": dt / hours}


def live(mode: str, packed: str, n: int, device: str, smoke: str) -> dict:
    """Packing tool vs the live encoder: extracts chunk 0 of ``packed`` to ``smoke`` (timed, includes audio reading), then
    encodes ``n`` of its utterances live through ``FeatureEncoder(inverter=mode)`` with the packed utterance's seed and
    compares channels 0-11 with the chunk and channels 12-14 / loud_raw with the packed source arrays."""
    import soundfile as sf

    from sparc.vocoders.eval.clone_pools import FeatureEncoder
    from sparc.vocoders.features import inverter_pack as ip

    src, out = Path(packed), Path(smoke)
    t0 = time.time()
    ip.run_extract(src, out, mode, 0, 1, device, 1)
    chunk_seconds = time.time() - t0
    import pandas as pd

    index = pd.read_parquet(src / "index.parquet")
    part = np.load(ip.chunk_path(out, 0))
    offsets = np.concatenate([[0], np.cumsum(index["T"].to_numpy(dtype=np.int64))])
    rows = np.unique(np.linspace(0, min(len(index), ip.CHUNK) - 1, n).astype(int))
    src_feats = np.load(src / "feats.npy", mmap_mode="r")
    src_loud = np.load(src / "loud_raw.npy", mmap_mode="r")
    enc = FeatureEncoder(device, dither=True, inverter=mode)
    res = {"mode": mode, "chunk0_rows": int(min(len(index), ip.CHUNK)), "chunk0_seconds_incl_load_and_io": chunk_seconds,
           "chunk0_audio_hours": float(index["duration"].iloc[: ip.CHUNK].sum() / 3600), "utterances": []}
    for i in rows:
        r = index.iloc[i]
        wav = sf.read(r["wav_path"], dtype="float32")[0]
        e = enc.encode(wav, int(r["seed"]))
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        packed_new = part[lo:hi]
        res["utterances"].append({
            "id": r["id"], "T": int(r["T"]), "T_live": int(e["T"]),
            "max_abs_diff_ema_live_vs_chunk": float(np.abs(e["feats"][:, :12] - packed_new).max()),
            "max_abs_diff_other_channels_vs_source": float(np.abs(e["feats"][:, 12:] - np.asarray(src_feats[lo:hi, 12:])).max()),
            "max_abs_diff_loud_raw_vs_source": float(np.abs(e["loud_raw"] - np.asarray(src_loud[lo:hi])).max())})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sanity", "align", "timing", "live"))
    ap.add_argument("--mode", required=True, choices=("mngu0", "all7"))
    ap.add_argument("--packed", default="/data/user_data/xoy/spkemb/phase5/packed/libritts_f/train.clean.100")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", default="/data/user_data/xoy/spkemb/phase9/smoke/split")
    a = ap.parse_args()
    if a.cmd == "sanity":
        res = sanity(a.mode, a.limit, a.device)
    elif a.cmd == "align":
        res = align(a.mode, a.packed, a.n, a.device)
    elif a.cmd == "timing":
        res = timing(a.mode, a.packed, a.n, a.device)
    else:
        res = live(a.mode, a.packed, a.n, a.device, a.smoke)
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)


if __name__ == "__main__":
    sys.exit(main())
