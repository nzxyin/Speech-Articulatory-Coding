# Reproduces the SPARC paper's Section III-A1 / Appendix A.1 methodology:
# fits the SSL-linear acoustic-to-articulatory inversion (AAI) model from
# scratch on MNGU0 -- WavLM Large layer 9 features -> 12-dim EMA (X/Y of
# tongue dorsum/blade/tip, jaw, upper lip, lower lip) via ordinary least
# squares. This was previously reused from the shipped checkpoint (frozen)
# since MNGU0 requires manual dataset-owner approval; now that the data is
# available, this validates that frozen component from scratch rather than
# just trusting it.
#
# MNGU0 data layout used here (the "day1"/"s1" release):
#   ema_norm/*.ema  -- head-corrected EMA, EST Track binary format, 200Hz,
#                      36 channels: 12 position (T3,T2,T1,JAW,UL,LL x X,Y)
#                      + 12 velocity + 12 acceleration. Only the first 12
#                      (position) channels are used here.
#   wav_16kHz/*.wav -- matching audio, already 16kHz.
#
# MNGU0's own channel order (T3, T2, T1, JAW, UL, LL) already matches the
# paper's articulator order (TD, TB, TT, LI, UL, LL) exactly -- T1/T2/T3
# are tongue tip/body/dorsum respectively in the Carstens AG500 convention
# MNGU0 uses, so no channel reordering is needed, just taking columns
# 0:12 of the norm file directly.

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import decimate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from transformers import WavLMModel

from sparc.inversion import butter_bandpass_filter

MNGU0_ROOT = Path("/data/group_data/UTD-NAS/Databases/mngu0/extracted")
EMA_DIR = MNGU0_ROOT / "ema_norm" / "mngu0_s1_ema_norm_1.0.1"
WAV_DIR = MNGU0_ROOT / "wav_16kHz" / "mngu0_s1_wav_16kHz_1.1.0"
OUT_DIR = Path("/data/user_data/xoy/mngu0_linear_aai_refit")

TARGET_LAYER = 9  # paper Appendix A.1: selected via 5-fold CV; reused here, not re-swept
FREQCUT = 10  # Hz, Butterworth low-pass on WavLM features
FT_SR = 50  # Hz, WavLM/EMA common frame rate
EMA_NATIVE_SR = 200  # Hz, MNGU0's native EMA sample rate
DOWNSAMPLE_FACTOR = EMA_NATIVE_SR // FT_SR  # 4

# MNGU0's EMA capture and the audio recording are NOT triggered at the same
# instant despite both files' own internal time references nominally
# starting at t=0 -- there's a fixed hardware/software sync lag between the
# two subsystems. Measured empirically via cross-correlation against the
# shipped (pretrained) model's predictions on 7 utterances spanning the
# corpus (shift range -20 to -24 frames, mean ~-22.6, correlation 0.86-0.94
# once aligned vs. ~0.17 unaligned) -- NOT used as a training signal, only
# as an independent diagnostic to discover this fixed timing constant once.
# Applied as: drop the first N frames of the WavLM feature sequence before
# pairing frame-for-frame with the EMA sequence (which starts unshifted).
EMA_AUDIO_OFFSET_FRAMES = 23


class ESTTrack:
    """Minimal Python-3 EST_Track binary-format reader (port of the
    corpus-provided estfile.py, which is Python 2-only)."""

    def __init__(self, filename):
        with open(filename, "rb") as f:
            if f.readline().strip() != b"EST_File Track":
                raise ValueError(f"not an EST Track file: {filename}")
            byte_order = "<"
            nrows = ncols = 0
            names = {}
            while True:
                line = f.readline().strip()
                if line == b"EST_Header_End":
                    break
                if line == b"ByteOrder 01":
                    byte_order = "<"
                elif line == b"ByteOrder 10":
                    byte_order = ">"
                else:
                    text = line.decode("ascii", errors="ignore")
                    mo = re.search(
                        r"name\s+(?P<name>\S+)|NumFrames\s+(?P<nf>\d+)|"
                        r"NumChannels\s+(?P<nc>\d+)|Channel_(?P<ch>\d+)\s+(?P<n>\S+)",
                        text,
                    )
                    if mo is None:
                        continue
                    if mo.group("nf"):
                        nrows = int(mo.group("nf"))
                    elif mo.group("nc"):
                        ncols = int(mo.group("nc"))
                    elif mo.group("ch"):
                        names[int(mo.group("ch"))] = mo.group("n")
            data = np.fromfile(f, dtype=np.dtype(byte_order + "f4")).reshape(nrows, ncols + 2)
        self.names = names
        self.T = data[:, 0]
        self.D = data[:, 2 : ncols + 2]  # first 2 cols are time + a break/frame marker


def load_ema_50hz_zscored(path):
    track = ESTTrack(path)
    pos = track.D[:, :12].astype(np.float64)  # position channels only, drop velocity/accel
    pos = decimate(pos, DOWNSAMPLE_FACTOR, axis=0, zero_phase=True)
    pos = (pos - pos.mean(axis=0, keepdims=True)) / pos.std(axis=0, keepdims=True)
    return pos.astype(np.float32)


def load_wavlm_layer9(wav_path, model, device):
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000
    wav = (wav - wav.mean()) / wav.std()
    with torch.no_grad():
        inp = torch.from_numpy(wav).unsqueeze(0).to(device)
        out = model(inp, output_hidden_states=True)
    states = out.hidden_states[TARGET_LAYER].cpu().numpy()[0]  # (T, 1024)
    states = butter_bandpass_filter(states[None], FREQCUT, FT_SR, axis=1)[0]
    return states.astype(np.float32)


def main(device="cuda:0"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ema_files = {p.stem: p for p in EMA_DIR.glob("*.ema")}
    wav_files = {p.stem: p for p in WAV_DIR.glob("*.wav")}
    stems = sorted(set(ema_files) & set(wav_files))
    print(f"EMA files: {len(ema_files)}, WAV files: {len(wav_files)}, matched: {len(stems)}")

    model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    model.encoder.layers = model.encoder.layers[: TARGET_LAYER + 1]
    model = model.eval().to(device)

    per_utt_X, per_utt_Y = {}, {}
    for i, stem in enumerate(stems):
        try:
            ema = load_ema_50hz_zscored(ema_files[stem])
            wavlm = load_wavlm_layer9(wav_files[stem], model, device)
            wavlm = wavlm[EMA_AUDIO_OFFSET_FRAMES:]  # see EMA_AUDIO_OFFSET_FRAMES above
            n = min(len(ema), len(wavlm))
            if n < 5:
                continue
            per_utt_X[stem] = wavlm[:n]
            per_utt_Y[stem] = ema[:n]
        except Exception as e:
            print(f"skip {stem}: {e}")
        if (i + 1) % 200 == 0:
            print(f"...{i + 1}/{len(stems)}")

    print(f"usable utterances: {len(per_utt_X)}")
    all_stems = sorted(per_utt_X.keys())

    # --- 5-fold CV, matching paper Appendix A.1's reported evaluation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_corrs = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_stems)):
        train_stems = [all_stems[i] for i in train_idx]
        test_stems = [all_stems[i] for i in test_idx]
        Xtr = np.concatenate([per_utt_X[s] for s in train_stems])
        Ytr = np.concatenate([per_utt_Y[s] for s in train_stems])
        reg = LinearRegression().fit(Xtr, Ytr)

        corrs = []
        for s in test_stems:
            pred = reg.predict(per_utt_X[s])
            for c in range(12):
                if per_utt_Y[s][:, c].std() > 1e-6 and pred[:, c].std() > 1e-6:
                    corrs.append(np.corrcoef(pred[:, c], per_utt_Y[s][:, c])[0, 1])
        fold_mean = float(np.mean(corrs))
        fold_corrs.append(fold_mean)
        print(f"fold {fold}: n_test_utt={len(test_stems)} mean_PCC={fold_mean:.4f}")

    print(f"5-fold CV mean PCC: {np.mean(fold_corrs):.4f} +/- {np.std(fold_corrs):.4f}")

    # --- final model, fit on ALL data (matches how the shipped checkpoint's
    #     linear head was produced, per paper: "All data in MNGU0 dataset
    #     are used for training the main model after selecting the best
    #     layer") ---
    Xall = np.concatenate([per_utt_X[s] for s in all_stems])
    Yall = np.concatenate([per_utt_Y[s] for s in all_stems])
    final_reg = LinearRegression().fit(Xall, Yall)

    with open(OUT_DIR / "linear_aai_mngu0_refit.pkl", "wb") as f:
        pickle.dump(final_reg, f)
    np.savez(
        OUT_DIR / "linear_aai_mngu0_refit.npz",
        weight=final_reg.coef_.astype(np.float32),
        bias=final_reg.intercept_.astype(np.float32),
    )
    with open(OUT_DIR / "cv_results.txt", "w") as f:
        f.write(f"per-fold PCC: {fold_corrs}\n")
        f.write(f"mean +/- std: {np.mean(fold_corrs):.4f} +/- {np.std(fold_corrs):.4f}\n")
        f.write(f"n utterances used: {len(all_stems)}\n")
        f.write("paper's reported figure (Appendix A.1): 0.878 +/- 0.012\n")

    print("saved:", OUT_DIR / "linear_aai_mngu0_refit.pkl")


if __name__ == "__main__":
    main(device=sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
