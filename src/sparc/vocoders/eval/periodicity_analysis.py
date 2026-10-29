"""What does the periodicity channel bring to the HiFi-GAN articulatory vocoder? (``docs/vocoder_periodicity_analysis.md``)

Systems (all HiFi-GAN, 100k generator steps, trained on LibriTTS-R only, one seed):

* ``ref``   the vocoder comparison's reference run ``ablation_100k`` (fork input set: 12 EMA, pitch, loudness, periodicity);
* ``none``  the same recipe without the periodicity channel (original SPARC input set: 12 EMA, pitch, loudness), run
  ``periodicity_none_100k``;
* ``probe_const`` / ``probe_binary`` / ``probe_smooth``  the reference model with its periodicity input replaced at inference
  (no training): the training mean, a voiced / unvoiced mask, a 9-frame moving average. The speaker vector and every other
  input stay those of the recording.

Items: own-utterance resynthesis (condition T1: the speaker code comes from the utterance itself, features from its own cache)
of a fixed subset of LibriTTS-R dev-clean (validation) and test-clean (test), and of the validation / test speaker halves of
VCTK (``docs/spkemb_validation_splits.json``). Stages (resumable, CPU unless noted)::

    python -m sparc.vocoders.eval.periodicity_analysis prepare                       # subset, MFA corpus
    python -m sparc.vocoders.eval.periodicity_analysis synth --systems ref,probe_const,... [--check]   # GPU
    python -m sparc.vocoders.eval.periodicity_analysis metrics --systems gt,ref,...  # GPU: UTMOS, Whisper, speaker, PESQ ...
    python -m sparc.vocoders.eval.periodicity_analysis summarize                      # classes, bootstrap, tables

``metrics`` also writes the per-frame arrays (50 Hz grid) that ``summarize`` turns into per-sound-class numbers using the
forced alignments (Montreal Forced Aligner, ``english_mfa``), so the class definitions can change without a GPU rerun.
"""

import argparse
import json
import logging
import os
import re
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sparc.vocoders.constants import F0_CHANNEL, HOP, PERIODICITY_CHANNEL, SAMPLE_RATE

logger = logging.getLogger(__name__)

ROOT = Path("/data/user_data/xoy/sparc-vocoders-periodicity")
REF_RUN = "/data/user_data/xoy/sparc-vocoders/runs/ablation_100k/hifigan"
ABL_RUN = "/data/user_data/xoy/spkemb/phase3/runs/periodicity_none_100k/hifigan"
PHASE4 = "/data/user_data/xoy/spkemb/phase4"
SPLITS_FILE = Path(__file__).resolve().parents[4] / "docs" / "spkemb_validation_splits.json"
GAIN_DB = -3.0
MFA_BIN = "/data/user_data/xoy/envs/mfa/bin/mfa"

# split name -> (corpus, label, role)
SPLITS = {
    "dev": ("libritts", "LibriTTS-R dev-clean", "validation"),
    "test": ("libritts", "LibriTTS-R test-clean", "test"),
    "vctk_val": ("vctk", "VCTK validation half", "validation (out of domain)"),
    "vctk_test": ("vctk", "VCTK test half", "test (out of domain)"),
}
LIBRI_SPLIT = {"dev": "dev.clean", "test": "test.clean"}
SYSTEM_RUN = {"ref": REF_RUN, "none": ABL_RUN, "probe_const": REF_RUN, "probe_binary": REF_RUN, "probe_smooth": REF_RUN}
SYSTEM_LABEL = {
    "gt": "recording",
    "ref": "fork input set (+ periodicity)",
    "none": "original SPARC input set (EMA + pitch + loudness)",
    "probe_const": "reference, periodicity = training mean",
    "probe_binary": "reference, periodicity = voiced/unvoiced mask",
    "probe_smooth": "reference, periodicity smoothed (9 frames)",
}
SMOOTH_FRAMES = 9
PER_UTT = 15  # utterances per speaker
DUR_RANGE = (2.0, 12.0)
CLASSES = ("vowels", "voiced_fricatives", "voiceless_fricatives", "stop_closure", "stop_burst", "affricates", "nasals",
           "approximants", "silence", "voicing_onset_offset", "voicing_onset", "voicing_offset")
CLASS_LABEL = {
    "vowels": "vowels", "voiced_fricatives": "voiced fricatives", "voiceless_fricatives": "voiceless fricatives",
    "stop_closure": "stops, closure", "stop_burst": "stops, burst", "affricates": "affricates", "nasals": "nasals",
    "approximants": "approximants / liquids", "silence": "silence", "voicing_onset_offset": "within 30 ms of a voicing change",
    "voicing_onset": "within 30 ms of a voicing onset", "voicing_offset": "within 30 ms of a voicing offset",
    "all": "all frames",
}
BOOT = 10000


# ------------------------------------------------------------------------------------------------ small helpers


def stable_rng(*parts) -> np.random.Generator:
    import hashlib

    h = int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


def setup_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sv = os.environ.get("SV_ROOT", "/data/user_data/xoy/sparc-vocoders")
    os.environ.setdefault("SV_ROOT", sv)
    os.environ.setdefault("SPARC_VOC_CACHE", f"{sv}/cache")
    os.environ.setdefault("SPARC_VOC_RUNS", f"{sv}/runs")


def cache_root() -> Path:
    return Path(os.environ.get("SPARC_VOC_CACHE", "/data/user_data/xoy/sparc-vocoders/cache"))


def open_store(split: str):
    """``(store, table)`` of a split; the table has ``id, speaker, n24, wav_path`` (and ``text`` for VCTK)."""
    corpus = SPLITS[split][0]
    if corpus == "libritts":
        from sparc.vocoders.data.dataset import PackedStore
        from sparc.vocoders.eval.io import packed_dir

        d = packed_dir(cache_root(), LIBRI_SPLIT[split])
        store = PackedStore([d], "l6")
        table = pd.read_parquet(d / "index.parquet")[["id", "speaker", "n24", "wav_path"]]
    else:
        from sparc.vocoders.eval.clone_pools import Pool

        pool = Pool("vctk", PHASE4)
        store = pool.store("l6")
        table = pool.table[["id", "speaker", "n24", "wav_path"]]
    table = table.copy()
    table["id"], table["speaker"] = table["id"].astype(str), table["speaker"].astype(str)
    return store, table


def subset_path() -> Path:
    return ROOT / "subset.json"


def load_subset() -> dict:
    return json.loads(subset_path().read_text())


def audio_path(system: str, split: str, uid: str) -> Path:
    return ROOT / "audio" / system / split / f"{uid}.wav"


# ------------------------------------------------------------------------------------------------ prepare


def choose(table: pd.DataFrame, speakers: Sequence[str] | None, split: str) -> pd.DataFrame:
    """Up to ``PER_UTT`` utterances of 2-12 s per speaker, seeded by the speaker (sorted ids, no replacement)."""
    dur = table["n24"] / SAMPLE_RATE
    t = table[(dur >= DUR_RANGE[0]) & (dur <= DUR_RANGE[1])]
    if speakers is not None:
        t = t[t["speaker"].isin(set(speakers))]
    rows = []
    for spk, g in t.groupby("speaker"):
        g = g.sort_values("id")
        pick = stable_rng("periodicity", split, spk).permutation(len(g))[:PER_UTT]
        rows.append(g.iloc[sorted(pick)])
    return pd.concat(rows).sort_values(["speaker", "id"]).reset_index(drop=True)


def run_prepare(args: argparse.Namespace) -> None:
    setup_env()
    from sparc.vocoders.eval.io import read_transcript

    splits_json = json.loads(SPLITS_FILE.read_text())
    out: dict = {}
    corpus = ROOT / "mfa" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for split, (kind, label, role) in SPLITS.items():
        store, table = open_store(split)
        speakers = None
        if kind == "vctk":
            speakers = splits_json["cloning"]["val" if split == "vctk_val" else "test"]["vctk"]
            from sparc.vocoders.eval.clone_pools import Pool

            texts = pd.read_parquet(Pool("vctk", PHASE4).directory / "table.parquet", columns=["id", "text"]).set_index("id")["text"]
        sel = choose(table, speakers, split)
        items = []
        for r in sel.itertuples():
            text = str(texts.loc[r.id]) if kind == "vctk" else read_transcript(r.wav_path)
            items.append({"id": r.id, "speaker": r.speaker, "n24": int(r.n24), "wav_path": r.wav_path, "text": text})
            d = corpus / r.speaker
            d.mkdir(exist_ok=True)
            link = d / f"{r.id}.wav"
            if not link.exists():
                link.symlink_to(r.wav_path)
            (d / f"{r.id}.lab").write_text(text + "\n")
        out[split] = {"label": label, "role": role, "n_speakers": int(sel["speaker"].nunique()), "items": items}
        print(split, label, len(items), "utterances,", out[split]["n_speakers"], "speakers")
    ROOT.mkdir(parents=True, exist_ok=True)
    subset_path().write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------ synthesis


def probe_periodicity(system: str, per: np.ndarray, const: float, voiced_value: float) -> np.ndarray:
    """The reference model's periodicity input under a reliance probe (``per`` is the cached raw channel, 0 when unvoiced)."""
    if system == "probe_const":
        return np.full_like(per, const)
    if system == "probe_binary":
        return np.where(per > 0, voiced_value, 0.0).astype(per.dtype)
    if system == "probe_smooth":
        from scipy.ndimage import uniform_filter1d

        return uniform_filter1d(per.astype(np.float64), size=SMOOTH_FRAMES, mode="nearest").astype(per.dtype)
    raise KeyError(system)


def voiced_value_from_dev() -> tuple[float, dict]:
    """Mean periodicity of the voiced frames (periodicity > 0) of LibriTTS-R dev-clean, from the packed cache."""
    d = cache_root() / "packed" / "dev.clean"
    feats = np.load(d / "feats.npy", mmap_mode="r")
    per = np.asarray(feats[:, PERIODICITY_CHANNEL], dtype=np.float64)
    f0 = np.asarray(feats[:, F0_CHANNEL], dtype=np.float64)
    voiced = per > 0
    return float(per[voiced].mean()), {"frames": int(len(per)), "voiced_fraction": float(voiced.mean()), "f0_gt_0_fraction": float((f0 > 0).mean())}


def run_synth(args: argparse.Namespace) -> None:
    setup_env()
    import soundfile as sf
    import torch

    from sparc.vocoders.eval.interp import load_run, utterance_features
    from sparc.vocoders.eval.io import gt_audio, write_json
    from sparc.vocoders.models.frontend import load_stats
    from sparc.vocoders.training.callbacks import EVAL_SEED, fixed_torch_rng

    subset = load_subset()
    stats = load_stats(os.environ["SPARC_VOC_CACHE"] + "/stats/train_stats.json")
    const = float(stats["per_mean"])
    voiced_value, vinfo = voiced_value_from_dev()
    systems = args.systems.split(",")
    splits = args.splits.split(",")
    stores = {s: open_store(s)[0] for s in splits}
    loaded: dict[str, tuple] = {}
    meta = {"per_mean_training": const, "voiced_mean_dev": voiced_value, "dev_frame_stats": vinfo, "smooth_frames": SMOOTH_FRAMES}
    for system in systems:
        if system == "gt":
            run = None
        else:
            run = SYSTEM_RUN[system]
            if run not in loaded:
                module, cfg, info = load_run(run, None, "auto")
                if info["g_step"] != 100000:
                    raise RuntimeError(f"{run}: checkpoint at step {info['g_step']}, expected 100000")
                for p in module.parameters():
                    p.requires_grad_(False)
                loaded[run] = (module, info)
            module, info = loaded[run]
            meta[system] = info
        device = next(module.parameters()).device if run else None
        for split in splits:
            store = stores[split]
            done = 0
            for item in subset[split]["items"]:
                uid = item["id"]
                path = audio_path(system, split, uid)
                if path.exists() and not args.overwrite:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                u = store.index_of(uid)
                T = int(store.T[u])
                if system == "gt":
                    wav = gt_audio(store, u, GAIN_DB)[: HOP * T]
                else:
                    feats = utterance_features(store, uid, GAIN_DB)
                    if system.startswith("probe"):
                        feats = feats.copy()
                        feats[PERIODICITY_CHANNEL] = probe_periodicity(system, feats[PERIODICITY_CHANNEL], const, voiced_value)
                    batch = {
                        "features": torch.from_numpy(np.ascontiguousarray(feats))[None].to(device),
                        "spk_raw": torch.from_numpy(np.asarray(store.speaker_vector(u), dtype=np.float32))[None].to(device),
                    }
                    with torch.inference_mode(), fixed_torch_rng(device, EVAL_SEED):
                        wav = module.synthesize(batch)[0, 0].float().cpu().numpy()
                sf.write(str(path) + ".tmp.wav", np.asarray(wav, dtype=np.float32), SAMPLE_RATE, subtype="FLOAT")
                os.replace(str(path) + ".tmp.wav", path)
                done += 1
            print(f"synth {system} {split}: wrote {done}", flush=True)
    if args.check:
        run_check(subset, stores)
    write_json(ROOT / f"synth_meta_{'_'.join(systems)}.json", meta)


def run_check(subset: dict, stores: dict) -> None:
    """Compares the reference resynthesis written here with the comparison's stored test-clean T1 predictions."""
    import soundfile as sf

    if "test" not in stores:
        return
    pred = Path(REF_RUN) / "predictions" / "test.clean" / "T1"
    diffs = []
    for item in subset["test"]["items"][:20]:
        mine, theirs = audio_path("ref", "test", item["id"]), pred / f"{item['id']}.wav"
        if mine.exists() and theirs.exists():
            a, _ = sf.read(mine, dtype="float32")
            b, _ = sf.read(theirs, dtype="float32")
            n = min(len(a), len(b))
            diffs.append(float(np.abs(a[:n] - b[:n]).max()))
    print("check vs stored predictions: n =", len(diffs), "max |diff| =", max(diffs) if diffs else None, flush=True)


# ------------------------------------------------------------------------------------------------ frame analysis


def frame_centres(n_frames: int) -> np.ndarray:
    return HOP * np.arange(n_frames) + HOP // 2


def windows(x: np.ndarray, centres: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    pad = np.pad(x, (half, half))
    return pad[centres[:, None] + np.arange(width)[None, :]]  # padded index = centre - half + half


def hf_band_db(x: np.ndarray, centres: np.ndarray, lo_hz: float = 4000.0, hi_hz: float = 12000.0, n: int = 1024) -> np.ndarray:
    """Log energy (dB) of the 4-12 kHz band of Hann frames of ``n`` samples centred on ``centres``."""
    fr = windows(np.asarray(x, dtype=np.float64), centres, n) * np.hanning(n)[None]
    p = np.abs(np.fft.rfft(fr, n, axis=1)) ** 2
    k0, k1 = int(np.ceil(lo_hz * n / SAMPLE_RATE)), int(hi_hz * n / SAMPLE_RATE)
    return 10.0 * np.log10(p[:, k0:k1].sum(1) + 1e-12)


def hnr_db(x: np.ndarray, centres: np.ndarray, win: int = 1200, fmin: float = 70.0, fmax: float = 545.0) -> np.ndarray:
    """Harmonic-to-noise ratio (dB) per frame by the normalized-autocorrelation method (Praat-style, window-corrected).

    ``win`` samples (50 ms) Hann window; the peak of the normalized autocorrelation over lags of ``fmax``..``fmin`` gives
    ``r``; ``HNR = 10 log10(r / (1 - r))`` with ``r`` clipped to [1e-6, 1 - 1e-6] (so the value lies in -60..60 dB). NaN for
    frames of digital silence.
    """
    x = np.asarray(x, dtype=np.float64)
    fr = windows(x, centres, win)
    fr = fr - fr.mean(1, keepdims=True)
    w = np.hanning(win)
    nfft = 4096
    lag_hi = int(SAMPLE_RATE / fmin)
    f = np.fft.rfft(fr * w[None], nfft, axis=1)
    acf = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, : lag_hi + 1]
    fw = np.fft.rfft(w, nfft)
    acw = np.fft.irfft(fw * np.conj(fw), nfft)[: lag_hi + 1]
    r = acf / np.maximum(acf[:, :1], 1e-30) / (acw / acw[0])[None]
    lag_lo = int(SAMPLE_RATE / fmax)
    peak = np.clip(r[:, lag_lo : lag_hi + 1].max(1), 1e-6, 1 - 1e-6)
    out = 10.0 * np.log10(peak / (1.0 - peak))
    out[acf[:, 0] < 1e-12 * win] = np.nan
    return out


def mel_frame_l1(ref: np.ndarray, deg: np.ndarray, cfg, n_frames: int) -> np.ndarray:
    """Per 50 Hz frame mean over mel bins of |log-mel(deg) - log-mel(ref)| (training mel, hop 256, frames pooled by start time)."""
    import torch

    from sparc.vocoders.eval.metrics.signal import mel_module

    module = mel_module(cfg)
    with torch.inference_mode():
        a = module.log_mel(torch.from_numpy(np.asarray(ref, dtype=np.float32))[None])
        b = module.log_mel(torch.from_numpy(np.asarray(deg, dtype=np.float32))[None])
        d = (b - a).abs()[0].mean(0).double().numpy()  # [mel frames]
    which = np.minimum((256 * np.arange(len(d))) // HOP, n_frames - 1)
    num = np.bincount(which, weights=d, minlength=n_frames)
    den = np.bincount(which, minlength=n_frames)
    return num / np.maximum(den, 1)


def pad_to(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)[:n]
    return np.concatenate([x, np.full(n - len(x), np.nan)]) if len(x) < n else x


def run_metrics(args: argparse.Namespace) -> None:
    setup_env()
    import soundfile as sf
    import torch

    from sparc.vocoders.eval.io import to_16k, write_parquet
    from sparc.vocoders.eval.metrics.asr import edit_counts, normalize_text
    from sparc.vocoders.eval.metrics.campplus import CAMPPlusEmbedder
    from sparc.vocoders.eval.metrics.signal import signal_metrics
    from sparc.vocoders.eval.stages import make_model
    from sparc.vocoders.eval.systems import compose_config

    cfg = compose_config("eval_config", [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = SimpleNamespace(cfg=cfg, device=device)
    utmos, asr = make_model(ctx, "utmos"), make_model(ctx, "asr")
    ecapa, rex = make_model(ctx, "spk:ecapa"), make_model(ctx, "reextract:refit")
    camp = CAMPPlusEmbedder(device=device)
    mel_cfg = cfg.eval_metrics.signal  # the block signal_metrics and mel_module read
    subset = load_subset()
    systems = sorted(args.systems.split(","), key=lambda s: s != "gt")  # the recording first: its embeddings are the reference
    for split in args.splits.split(","):
        items = subset[split]["items"]
        ids = [it["id"] for it in items]
        speakers = [it["speaker"] for it in items]
        emb_path = ROOT / "metrics" / split / "gt_embeddings.npz"
        for system in systems:
            out_path = ROOT / "metrics" / split / f"{system}.parquet"
            if out_path.exists() and not args.overwrite:
                print(f"metrics {system} {split}: exists", flush=True)
                continue
            t0 = time.time()
            rows, arrays, emb_ecapa, emb_camp = [], {}, [], []
            for start in range(0, len(items), 32):
                chunk = items[start : start + 32]
                wavs, gts = [], []
                for it in chunk:
                    w, rate = sf.read(audio_path(system, split, it["id"]), dtype="float32")
                    g, _ = sf.read(audio_path("gt", split, it["id"]), dtype="float32")
                    assert rate == SAMPLE_RATE and len(w) == len(g), (it["id"], len(w), len(g))
                    wavs.append(w)
                    gts.append(g)
                w16 = [to_16k(w, SAMPLE_RATE) for w in wavs]
                errs: list[str] = []
                mos = utmos.score(w16, errs)
                hyps = asr.transcribe(w16, [])
                ecapa_rows = ecapa.embed(w16, [])
                for k, it in enumerate(chunk):
                    uid, w, g = it["id"], wavs[k], gts[k]
                    row = {"id": uid, "speaker": it["speaker"], "dur_s": len(w) / SAMPLE_RATE, "utmos": mos[k], "hyp": hyps[k]}
                    ref_norm, hyp_norm = normalize_text(it["text"]), normalize_text(hyps[k])
                    row.update(edit_counts(ref_norm, hyp_norm))
                    cm = camp(w16[k])
                    emb_ecapa.append(ecapa_rows[k])
                    emb_camp.append(cm / np.linalg.norm(cm))
                    T = len(w) // HOP
                    centres = frame_centres(T)
                    ext = rex.extract(w, SAMPLE_RATE)["feats"]
                    arrays[f"{uid}/hb"] = hf_band_db(w, centres).astype(np.float32)
                    arrays[f"{uid}/hnr"] = hnr_db(w, centres).astype(np.float32)
                    arrays[f"{uid}/per"] = pad_to(ext[:, PERIODICITY_CHANNEL], T).astype(np.float32)
                    arrays[f"{uid}/f0"] = pad_to(ext[:, F0_CHANNEL], T).astype(np.float32)
                    if system != "gt":
                        try:
                            row.update(signal_metrics(g, w, SAMPLE_RATE, mel_cfg))
                        except Exception as error:  # a failed metric must not stop the run
                            logger.warning("signal metrics %s: %s", uid, error)
                        arrays[f"{uid}/mel"] = mel_frame_l1(g, w, mel_cfg, T).astype(np.float32)
                    rows.append(row)
                print(f"metrics {system} {split}: {len(rows)}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
            ecapa_arr, camp_arr = np.stack(emb_ecapa), np.stack(emb_camp)
            if system == "gt":
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(emb_path, ids=np.array(ids), ecapa=ecapa_arr, campplus=camp_arr)
            else:
                ref = np.load(emb_path)
                assert list(ref["ids"]) == ids
                for name, arr in (("ecapa", ecapa_arr), ("campplus", camp_arr)):
                    sim = (arr * ref[name]).sum(1) / (np.linalg.norm(arr, axis=1) * np.linalg.norm(ref[name], axis=1))
                    for row, s in zip(rows, sim):
                        row[f"sim_{name}"] = float(s)
            frame_file = ROOT / "frames" / split / f"{system}.npz"
            frame_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(frame_file, **arrays)
            write_parquet(out_path, pd.DataFrame(rows))
            print(f"metrics {system} {split}: done in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ alignments and classes

VOWEL_CHARS = set("aeiouyæɑɒɔəɐɚɛɜɝɪʊʌʉøœɯɨɵɤɞɘɛ")
NASAL_CHARS = set("mnŋɲɱɳɴ")
STOP_CHARS = set("pbtdkɡgcɟʈɖqɢʔ")
VFRIC_CHARS = set("zʒvðɣʐʑβʝ")
UFRIC_CHARS = set("sʃfθhçxɕʂɸχħ")
APPROX_CHARS = set("lɫɹɻjwʋɾɭʎɰɥʁr")
AFFRICATES = {"tʃ", "dʒ", "ts", "dz", "tɕ", "dʑ"}
SILENCE_LABELS = {"", "sil", "sp"}
UNKNOWN_LABELS = {"spn", "oov", "unk", "<unk>"}
STOP_BURST_S = 0.04
STOP_MIN_CLOSURE_S = 0.02


def phone_class(p: str) -> str | None:
    """Sound class of an ``english_mfa`` (IPA) phone label; ``None`` for spoken-noise / unknown labels.

    Diacritics and length / aspiration / palatalization marks are ignored (syllabic nasals and liquids count as nasals and
    approximants). Diphthongs (``aj, aw, ej, ɔj, əw``) are vowels; ``tʃ, dʒ`` are affricates; the flap ``ɾ`` is an
    approximant / liquid.
    """
    if p in SILENCE_LABELS:
        return "silence"
    if p in UNKNOWN_LABELS:
        return None
    s = "".join(c for c in unicodedata.normalize("NFD", p) if not unicodedata.combining(c) and c not in "ʰʲʷˠˤːˑ")
    if not s:
        return None
    if s in AFFRICATES:
        return "affricates"
    c0 = s[0]
    for chars, name in ((VOWEL_CHARS, "vowels"), (NASAL_CHARS, "nasals"), (STOP_CHARS, "stops"), (VFRIC_CHARS, "voiced_fricatives"),
                        (UFRIC_CHARS, "voiceless_fricatives"), (APPROX_CHARS, "approximants")):
        if c0 in chars:
            return name
    return None


def parse_textgrid_phones(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8")
    for tier in re.split(r"item \[\d+\]:", text)[1:]:
        m = re.search(r'name = "([^"]*)"', tier)
        if m and m.group(1) == "phones":
            return [(float(a), float(b), s) for a, b, s in re.findall(r'xmin = (\S+)\s*\n\s*xmax = (\S+)\s*\n\s*text = "(.*?)"', tier)]
    raise ValueError(f"no phones tier in {path}")


MIN_RUN_FRAMES = 3


def smooth_runs(v: np.ndarray, min_len: int) -> np.ndarray:
    """Voicing sequence (1 voiced, 0 unvoiced, NaN unavailable) with every run shorter than ``min_len`` frames given the value of
    the run before it (the first run keeps its value), so that tracker flicker does not create voicing transitions."""
    out = v.copy()
    n = len(out)

    def same(a: float, b: float) -> bool:
        return a == b or (np.isnan(a) and np.isnan(b))

    while True:
        runs, s = [], 0
        for t in range(1, n + 1):
            if t == n or not same(out[t], out[s]):
                runs.append((s, t))
                s = t
        short = [(a, b) for k, (a, b) in enumerate(runs) if k > 0 and b - a < min_len and np.isfinite(out[a])]
        if not short:
            return out
        a, b = short[0]
        out[a:b] = out[a - 1]


def class_masks(phones: Sequence[tuple[float, float, str]], n: int, voiced_g: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Boolean frame masks per class on the 50 Hz grid (frame ``t`` is at ``(t + 0.5) * 20 ms``), and label bookkeeping.

    Stops: the final 40 ms of a stop interval are ``stop_burst``, the rest ``stop_closure`` (a stop shorter than 60 ms
    is all burst). Voicing transitions come from the voicing of the recording's re-extraction: frames whose centre is within
    30 ms of a change between voiced and unvoiced (4 frames around it); ``voiced_g`` is NaN where unavailable.
    """
    centre = (np.arange(n) + 0.5) * (HOP / SAMPLE_RATE)
    masks = {c: np.zeros(n, dtype=bool) for c in CLASSES}
    info = {"labelled": 0, "unknown": 0, "unmapped": 0}
    for s, e, p in phones:
        cls = phone_class(p)
        inside = (centre >= s) & (centre < e)
        if not inside.any():
            continue
        if cls is None:
            info["unknown" if p in UNKNOWN_LABELS else "unmapped"] += int(inside.sum())
            continue
        info["labelled"] += int(inside.sum())
        if cls == "stops":
            burst = inside & ((centre >= e - STOP_BURST_S) | ((e - s) < STOP_BURST_S + STOP_MIN_CLOSURE_S))
            masks["stop_burst"] |= burst
            masks["stop_closure"] |= inside & ~burst
        else:
            masks[cls] |= inside
    v = smooth_runs(np.asarray(voiced_g, dtype=float), MIN_RUN_FRAMES)
    for t in range(1, n):
        if np.isfinite(v[t]) and np.isfinite(v[t - 1]) and v[t] != v[t - 1]:
            lo, hi = max(0, t - 2), min(n, t + 2)
            masks["voicing_onset_offset"][lo:hi] = True
            masks["voicing_onset" if v[t] > 0 else "voicing_offset"][lo:hi] = True
    return masks, info


def frame_stats(mask: np.ndarray, a: dict[str, np.ndarray], g: dict[str, np.ndarray] | None) -> dict[str, float]:
    """Per-utterance statistics of one frame mask. ``a`` are the system's frame arrays, ``g`` the recording's."""
    nan = float("nan")

    def mean(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(x.mean()) if len(x) else nan

    out = {"n": float(mask.sum())}
    out["mel_l1"] = mean(a["mel"][mask]) if "mel" in a else nan
    if g is not None:
        hb = a["hb"] - g["hb"]
        hn = a["hnr"] - g["hnr"]
        out["hb_abs"], out["hb_bias"] = mean(np.abs(hb[mask])), mean(hb[mask])
        out["hnr_abs"], out["hnr_bias"] = mean(np.abs(hn[mask])), mean(hn[mask])
        out["hnr_gt"] = mean(g["hnr"][mask])
        ok = mask & np.isfinite(a["per"]) & np.isfinite(g["per"])
        vs, vg = a["per"] > 0, g["per"] > 0
        out["vde"] = float((vs[ok] != vg[ok]).mean()) if ok.any() else nan
        out["per_mae"] = mean(np.abs(a["per"] - g["per"])[mask])
        out["voiced_gt"] = float(vg[ok].mean()) if ok.any() else nan
        both = ok & vs & vg & (a["f0"] > 0) & (g["f0"] > 0)
        cents = 1200.0 * np.log2(a["f0"][both] / g["f0"][both])
        out["f0_abs"] = float(np.abs(cents).mean()) if both.any() else nan
        out["f0_sq"] = float((cents**2).mean()) if both.any() else nan
        out["f0_n"] = float(both.sum())
    return out


def load_frames(split: str, system: str, ids: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(ROOT / "frames" / split / f"{system}.npz")
    keys = ("hb", "hnr", "per", "f0", "mel")
    return {uid: {k: z[f"{uid}/{k}"].astype(np.float64) for k in keys if f"{uid}/{k}" in z.files} for uid in ids}


def frame_table(split: str, systems: Sequence[str]) -> tuple[pd.DataFrame, dict]:
    """Per utterance and class statistics of every system against the recording, plus alignment coverage."""
    items = load_subset()[split]["items"]
    ids = [it["id"] for it in items]
    spk = {it["id"]: it["speaker"] for it in items}
    gt = load_frames(split, "gt", ids)
    masks, cover = {}, {"utterances": len(ids), "aligned": 0, "frames": 0, "labelled": 0, "unknown": 0, "unmapped": 0, "per_class": dict.fromkeys(CLASSES, 0)}
    for uid in ids:
        n = len(gt[uid]["hb"])
        cover["frames"] += n
        tg = ROOT / "mfa" / "out" / spk[uid] / f"{uid}.TextGrid"
        if not tg.exists():
            continue
        cover["aligned"] += 1
        m, info = class_masks(parse_textgrid_phones(tg), n, np.where(np.isfinite(gt[uid]["per"]), gt[uid]["per"] > 0, np.nan))
        masks[uid] = m
        for k in ("labelled", "unknown", "unmapped"):
            cover[k] += info[k]
        for c in CLASSES:
            cover["per_class"][c] += int(m[c].sum())
    rows = []
    for system in systems:
        fr = load_frames(split, system, ids)
        for uid in ids:
            if uid not in masks:
                continue
            n = len(gt[uid]["hb"])
            groups = {"all": np.ones(n, dtype=bool), **masks[uid]}
            for c, m in groups.items():
                if c != "all" and not m.any():
                    continue
                rows.append({"split": split, "system": system, "id": uid, "speaker": spk[uid], "class": c, **frame_stats(m, fr[uid], gt[uid] if system != "gt" else None)})
    return pd.DataFrame(rows), cover


# ------------------------------------------------------------------------------------------------ bootstrap and tables


def speaker_means(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("speaker")[col].apply(lambda x: x.dropna().mean() if x.notna().any() else np.nan)


def boot_mean(values: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Mean over speakers and its percentile bootstrap CI95 (speakers resampled with replacement)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(v), size=(BOOT, len(v)))
    means = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(df_a: pd.DataFrame, df_b: pd.DataFrame, col: str) -> dict[str, float]:
    """``a - b`` on the utterances present and finite in both, averaged per speaker, with a speaker-level bootstrap CI."""
    m = df_a[["id", "speaker", col]].merge(df_b[["id", col]], on="id", suffixes=("_a", "_b"))
    m = m[np.isfinite(m[f"{col}_a"]) & np.isfinite(m[f"{col}_b"])]
    if m.empty:
        return {"a": np.nan, "b": np.nan, "diff": np.nan, "lo": np.nan, "hi": np.nan, "n_speakers": 0}
    g = m.groupby("speaker")[[f"{col}_a", f"{col}_b"]].mean()
    d, lo, hi = boot_mean((g[f"{col}_a"] - g[f"{col}_b"]).to_numpy())
    return {"a": float(g[f"{col}_a"].mean()), "b": float(g[f"{col}_b"].mean()), "diff": d, "lo": lo, "hi": hi, "n_speakers": int(len(g))}


def paired_ratio(df_a: pd.DataFrame, df_b: pd.DataFrame, num: str, den: str) -> dict[str, float]:
    """Corpus ratio (sum of ``num`` over sum of ``den``) of ``a`` minus ``b``, resampling speakers."""
    sa = df_a.groupby("speaker")[[num, den]].sum()
    sb = df_b.groupby("speaker")[[num, den]].sum().reindex(sa.index)
    idx = np.random.default_rng(0).integers(0, len(sa), size=(BOOT, len(sa)))
    ra = sa[num].to_numpy()[idx].sum(1) / sa[den].to_numpy()[idx].sum(1)
    rb = sb[num].to_numpy()[idx].sum(1) / sb[den].to_numpy()[idx].sum(1)
    pa, pb = sa[num].sum() / sa[den].sum(), sb[num].sum() / sb[den].sum()
    d = ra - rb
    return {"a": float(pa), "b": float(pb), "diff": float(pa - pb), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "n_speakers": int(len(sa))}


GLOBAL_METRICS = [  # column, label, digits, kind
    ("utmos", "UTMOS", 3, "mean"), ("wer", "WER (%)", 2, "wer"), ("cer", "CER (%)", 2, "cer"), ("mel_l1", "mel L1", 4, "mean"),
    ("mel_l1_4_8k", "mel L1, 4-8 kHz", 4, "mean"), ("mel_l1_8_12k", "mel L1, 8-12 kHz", 4, "mean"), ("mrstft", "multi-res. STFT distance", 4, "mean"),
    ("pesq_wb", "PESQ (WB)", 3, "mean"), ("mcd", "MCD (dB)", 3, "mean"), ("sim_ecapa", "ECAPA similarity to recording", 4, "mean"),
    ("sim_campplus", "CAM++ similarity to recording", 4, "mean"),
]
FRAME_GLOBAL = [  # from the 'all' class of the frame table
    ("vde", "voiced/unvoiced decision error (fraction of frames)", 4), ("f0_rmse", "F0 RMSE on frames voiced in both (cents)", 2),
    ("per_mae", "periodicity MAE", 4), ("hnr_abs", "HNR error, mean absolute dB", 3), ("hnr_bias", "HNR error, mean dB (output - recording)", 3),
    ("hb_abs", "4-12 kHz log-energy error, mean absolute dB", 3), ("hb_bias", "4-12 kHz log-energy error, mean dB", 3),
]
CLASS_METRICS = [("mel_l1", "mel L1", 4), ("hb_abs", "4-12 kHz log-energy error, mean absolute dB", 3), ("hb_bias", "4-12 kHz log-energy error, mean dB (output - recording)", 3),
                 ("hnr_abs", "HNR error, mean absolute dB", 3), ("hnr_bias", "HNR error, mean dB (output - recording)", 3), ("vde", "voicing decision error", 4),
                 ("f0_abs", "F0 error on both-voiced frames, mean absolute cents", 2)]
COMPARE = ("none", "probe_const", "probe_binary", "probe_smooth")


def load_global(split: str, system: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "metrics" / split / f"{system}.parquet")
    df["wer"] = df["word_errors"] / df["word_ref_len"]
    df["cer"] = df["char_errors"] / df["char_ref_len"]
    return df


def fmt(x: float, digits: int, pct: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * (100 if pct else 1):.{digits}f}"


def fmt_ci(r: dict, digits: int, pct: bool = False) -> str:
    if not np.isfinite(r["diff"]):
        return "n/a"
    s = 100 if pct else 1
    sig = "*" if (r["lo"] > 0 or r["hi"] < 0) else ""
    return f"{r['diff'] * s:+.{digits}f} [{r['lo'] * s:+.{digits}f}, {r['hi'] * s:+.{digits}f}]{sig}"


def run_summarize(args: argparse.Namespace) -> None:
    setup_env()
    subset = load_subset()
    available = {sp: [s for s in ("gt", "ref", "none", "probe_const", "probe_binary", "probe_smooth") if (ROOT / "metrics" / sp / f"{s}.parquet").exists()] for sp in SPLITS}
    lines = ["# Periodicity analysis: all tables", "", "Speaker-level paired bootstrap, 10,000 resamples, 95% intervals; `*` marks an interval that excludes 0.",
             "Differences are system minus the reference (fork input set, with periodicity).", ""]
    results: dict = {}
    for split, (kind, label, role) in SPLITS.items():
        systems = available[split]
        if "ref" not in systems:
            continue
        n_utt, n_spk = len(subset[split]["items"]), subset[split]["n_speakers"]
        lines += [f"## {label} ({role}): {n_utt} utterances, {n_spk} speakers", ""]
        glob = {s: load_global(split, s) for s in systems}
        frame_df, cover = frame_table(split, systems)
        (ROOT / "summary").mkdir(exist_ok=True)
        frame_df.to_parquet(ROOT / "summary" / f"frames_{split}.parquet")
        comps = [s for s in COMPARE if s in systems]
        header = "| metric | recording | " + " | ".join([SYSTEM_LABEL["ref"]] + [SYSTEM_LABEL[s] for s in comps]) + " |"
        lines += ["### Global metrics", "", "Reference column: mean; other systems: difference from the reference with CI.", "", header, "|" + "---|" * (3 + len(comps))]
        rec = {}
        for col, name, digits, kind_ in GLOBAL_METRICS:
            row = [name]
            if kind_ in ("wer", "cer"):
                num, den = ("word_errors", "word_ref_len") if kind_ == "wer" else ("char_errors", "char_ref_len")
                gsum = glob["gt"][num].sum() / glob["gt"][den].sum() if "gt" in glob else np.nan
                row.append(fmt(gsum, digits, True))
                ref = glob["ref"][num].sum() / glob["ref"][den].sum()
                row.append(fmt(ref, digits, True))
                for s in comps:
                    r = paired_ratio(glob[s], glob["ref"], num, den)
                    rec[(col, s)] = r
                    row.append(fmt_ci(r, digits, True))
            else:
                row.append(fmt(glob["gt"][col].mean(), digits) if "gt" in glob and col in glob["gt"] else "-")
                row.append(fmt(glob["ref"][col].mean(), digits))
                for s in comps:
                    r = paired(glob[s], glob["ref"], col)
                    rec[(col, s)] = r
                    row.append(fmt_ci(r, digits))
            lines.append("| " + " | ".join(row) + " |")
        allf = frame_df[frame_df["class"] == "all"].copy()
        allf["f0_rmse"] = np.sqrt(allf["f0_sq"])
        for col, name, digits in FRAME_GLOBAL:
            row = [name, "-"]
            ref = allf[allf["system"] == "ref"]
            row.append(fmt(speaker_means(ref, col).mean(), digits))
            for s in comps:
                r = paired(allf[allf["system"] == s], ref, col)
                rec[(col, s)] = r
                row.append(fmt_ci(r, digits))
            lines.append("| " + " | ".join(row) + " |")
        lines += [""]
        lines += [f"Alignment coverage ({label}): {cover['aligned']}/{cover['utterances']} utterances aligned; {cover['labelled']}/{cover['frames']} frames "
                  f"({100 * cover['labelled'] / max(cover['frames'], 1):.1f} %) in a sound class (spoken-noise labels {cover['unknown']}, unmapped {cover['unmapped']}); "
                  "frames per class: " + ", ".join(f"{CLASS_LABEL[c]} {n}" for c, n in cover["per_class"].items()), ""]
        for col, name, digits in CLASS_METRICS:
            lines += [f"### {name}, per sound class", "", "| class | frames | reference | " + " | ".join(f"{SYSTEM_LABEL[s]} minus reference" for s in comps) + " |", "|" + "---|" * (3 + len(comps))]
            for c in ("all", *CLASSES):
                sub = {s: frame_df[(frame_df["system"] == s) & (frame_df["class"] == c)] for s in systems}
                if sub["ref"].empty:
                    continue
                row = [CLASS_LABEL[c], str(int(cover["per_class"].get(c, sub["ref"]["n"].sum())))]
                row.append(fmt(speaker_means(sub["ref"], col).mean(), digits))
                for s in comps:
                    r = paired(sub[s], sub["ref"], col)
                    rec[(f"{c}/{col}", s)] = r
                    row.append(fmt_ci(r, digits))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        results[split] = {f"{k[0]}|{k[1]}": v for k, v in rec.items()}
        results[split]["coverage"] = cover
    (ROOT / "summary").mkdir(exist_ok=True)
    (ROOT / "summary" / "summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "summary" / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("wrote", ROOT / "summary" / "summary.md")


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def run_cfgdiff(args: argparse.Namespace) -> None:
    """Prints every training-config key that differs between the reference run and the ablation (stored in the checkpoints)."""
    import torch

    from sparc.vocoders.training.callbacks import list_step_checkpoints

    cfgs = {}
    for name, run in (("ref", REF_RUN), ("none", ABL_RUN)):
        step, path = list_step_checkpoints(Path(run) / "ckpt")[-1]
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfgs[name] = flatten(state["hyper_parameters"]["cfg"])
        print(name, path, "g_step", int(state["vocoder_counters"]["g_step"]))
    keys = sorted(set(cfgs["ref"]) | set(cfgs["none"]))
    diff = [(k, cfgs["ref"].get(k, "<absent>"), cfgs["none"].get(k, "<absent>")) for k in keys if cfgs["ref"].get(k, "<absent>") != cfgs["none"].get(k, "<absent>")]
    print(f"{len(keys)} config keys, {len(diff)} differ:")
    for k, a, b in diff:
        print(f"  {k}: ref={a!r} none={b!r}")


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    s = sub.add_parser("synth")
    s.add_argument("--systems", default="gt,ref,probe_const,probe_binary,probe_smooth")
    s.add_argument("--splits", default=",".join(SPLITS))
    s.add_argument("--check", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    m = sub.add_parser("metrics")
    m.add_argument("--systems", default="gt,ref,probe_const,probe_binary,probe_smooth")
    m.add_argument("--splits", default=",".join(SPLITS))
    m.add_argument("--overwrite", action="store_true")
    sub.add_parser("summarize")
    sub.add_parser("cfgdiff")
    a = ap.parse_args(argv)
    {"prepare": run_prepare, "synth": run_synth, "metrics": run_metrics, "summarize": run_summarize, "cfgdiff": run_cfgdiff}[a.cmd](a)


if __name__ == "__main__":
    main()
