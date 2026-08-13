# Reproduces the SPARC paper's Appendix A.1 / Fig. 7 layer-selection sweep:
# probes every WavLM Large transformer layer's correlation with MNGU0 EMA,
# to verify the paper's claim that layer 9 is the peak rather than just
# trusting it (as refit_mngu0_linear_aai.py otherwise does by default).
#
# Uses a single held-out utterance split per layer (not full 5-fold CV --
# this is a comparative sweep to find the peak, not the final reported
# number) on a subsample of utterances to keep memory bounded, since all
# 25 hidden-state layers must be cached at once per utterance (one WavLM
# forward pass per utterance, sliced afterward) rather than re-running
# WavLM per layer.

import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from transformers import WavLMModel

from sparc.inversion import butter_bandpass_filter
from sparc.training.refit_mngu0_linear_aai import (
    EMA_AUDIO_OFFSET_FRAMES,
    EMA_DIR,
    FREQCUT,
    FT_SR,
    WAV_DIR,
    load_ema_50hz_zscored,
)

N_UTTERANCES = 400  # subsample for memory; paper's own layer-selection used 5-fold CV on the full set
LAYERS_TO_TEST = list(range(1, 25))  # WavLM Large has 24 transformer layers


def load_all_layers(wav_path, model, device):
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000
    wav = (wav - wav.mean()) / wav.std()
    with torch.no_grad():
        inp = torch.from_numpy(wav).unsqueeze(0).to(device)
        out = model(inp, output_hidden_states=True)
    # hidden_states[i] for i=1..24 is the output after the i-th transformer layer
    layers = {}
    for L in LAYERS_TO_TEST:
        states = out.hidden_states[L].cpu().numpy()[0]
        states = butter_bandpass_filter(states[None], FREQCUT, FT_SR, axis=1)[0]
        layers[L] = states.astype(np.float32)[EMA_AUDIO_OFFSET_FRAMES:]
    return layers


def main(device="cuda:0"):
    random.seed(0)
    ema_files = {p.stem: p for p in EMA_DIR.glob("*.ema")}
    wav_files = {p.stem: p for p in WAV_DIR.glob("*.wav")}
    stems = sorted(set(ema_files) & set(wav_files))
    stems = random.sample(stems, min(N_UTTERANCES, len(stems)))
    print(f"sweeping {len(stems)} utterances x {len(LAYERS_TO_TEST)} layers")

    model = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().to(device)  # full model, no truncation

    per_layer_X = {L: [] for L in LAYERS_TO_TEST}
    per_layer_Y = {L: [] for L in LAYERS_TO_TEST}
    per_layer_utt_bounds = {L: [] for L in LAYERS_TO_TEST}  # for a per-utterance train/test split

    for i, stem in enumerate(stems):
        try:
            ema = load_ema_50hz_zscored(ema_files[stem])
            layers = load_all_layers(wav_files[stem], model, device)
            for L in LAYERS_TO_TEST:
                n = min(len(ema), len(layers[L]))
                if n < 5:
                    continue
                per_layer_X[L].append(layers[L][:n])
                per_layer_Y[L].append(ema[:n])
        except Exception as e:
            print(f"skip {stem}: {e}")
        if (i + 1) % 100 == 0:
            print(f"...{i + 1}/{len(stems)}")

    results = {}
    for L in LAYERS_TO_TEST:
        n_utt = len(per_layer_X[L])
        idx = list(range(n_utt))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=0)
        Xtr = np.concatenate([per_layer_X[L][i] for i in train_idx])
        Ytr = np.concatenate([per_layer_Y[L][i] for i in train_idx])
        reg = LinearRegression().fit(Xtr, Ytr)

        corrs = []
        for i in test_idx:
            pred = reg.predict(per_layer_X[L][i])
            true = per_layer_Y[L][i]
            for c in range(12):
                if true[:, c].std() > 1e-6 and pred[:, c].std() > 1e-6:
                    corrs.append(np.corrcoef(pred[:, c], true[:, c])[0, 1])
        mean_corr = float(np.mean(corrs))
        results[L] = mean_corr
        print(f"layer {L:2d}: mean_PCC={mean_corr:.4f}  (n_train_utt={len(train_idx)} n_test_utt={len(test_idx)})")

    best_layer = max(results, key=results.get)
    print()
    print(f"peak layer: {best_layer} (PCC={results[best_layer]:.4f})")
    print("paper's reported peak: layer 9 (PCC=0.878 +/- 0.012)")

    out_dir = Path("/data/user_data/xoy/mngu0_linear_aai_refit")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "layer_sweep_results.txt", "w") as f:
        for L, c in sorted(results.items()):
            f.write(f"{L}\t{c:.4f}\n")
        f.write(f"peak: {best_layer}\n")


if __name__ == "__main__":
    main(device=sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
