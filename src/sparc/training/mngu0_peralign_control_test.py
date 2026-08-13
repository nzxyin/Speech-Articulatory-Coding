# Control/sanity-check for mngu0_peralign_refit.py.
#
# The per-utterance alignment run scored 0.9186 CV PCC -- *above* the
# paper's own reported 0.878 -- which is suspicious rather than a clean
# win: the offset was picked by maximizing correlation against the SHIPPED
# model's own prediction, a signal produced by basically the same
# architecture (WavLM layer 9 -> linear) being fit and evaluated here.
# Searching over 101 candidate shifts and keeping whichever one best
# matches a structurally-similar, smooth, autocorrelated predictor can
# inflate held-out correlation via a selection-bias artifact, independent
# of any real physical timing fix.
#
# This script repeats the identical procedure but with MISMATCHED pairing:
# each utterance's ground-truth EMA is aligned against a different,
# randomly assigned utterance's shipped-model prediction (fixed derangement,
# no genuine temporal correspondence should exist between the two). If the
# resulting CV PCC is still inflated, that proves the effect is a
# search/leakage artifact, not a genuine alignment improvement.

import pickle
import random

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from transformers import WavLMModel

from sparc import load_model
from sparc.training.mngu0_peralign_refit import best_offset, load_wavlm_layer9
from sparc.training.refit_mngu0_linear_aai import (
    EMA_DIR,
    OUT_DIR,
    TARGET_LAYER,
    WAV_DIR,
    load_ema_50hz_zscored,
)

MIN_OVERLAP_FRAMES = 50


def derangement(n, seed=0):
    """A random permutation of range(n) with no fixed points."""
    rng = random.Random(seed)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def main(device="cuda:0"):
    ema_files = {p.stem: p for p in EMA_DIR.glob("*.ema")}
    wav_files = {p.stem: p for p in WAV_DIR.glob("*.wav")}
    stems = sorted(set(ema_files) & set(wav_files))
    print(f"matched: {len(stems)}")

    coder = load_model("feature_extraction", device=device)
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    wavlm.encoder.layers = wavlm.encoder.layers[: TARGET_LAYER + 1]
    wavlm = wavlm.eval().to(device)

    true_ema_cache, shipped_pred_cache = {}, {}
    for i, stem in enumerate(stems):
        try:
            true_ema_cache[stem] = load_ema_50hz_zscored(ema_files[stem])
            out = coder.encode(wav_files[stem], split_batch=True, reduce=True, concat=False)
            shipped_pred_cache[stem] = out["ema"]
        except Exception as e:
            print(f"skip {stem}: {e}")
        if (i + 1) % 300 == 0:
            print(f"...cached {i + 1}/{len(stems)}")

    valid_stems = sorted(set(true_ema_cache) & set(shipped_pred_cache))
    perm = derangement(len(valid_stems), seed=0)
    mismatched_ref = {valid_stems[i]: shipped_pred_cache[valid_stems[perm[i]]] for i in range(len(valid_stems))}

    per_utt_X, per_utt_Y, offsets = {}, {}, {}
    for i, stem in enumerate(valid_stems):
        try:
            true_ema = true_ema_cache[stem]
            control_ref = mismatched_ref[stem]  # deliberately WRONG utterance's shipped prediction
            s, c = best_offset(true_ema, control_ref)
            if s is None:
                print(f"skip {stem}: no valid alignment found in search range")
                continue
            offsets[stem] = (s, c)

            feats = load_wavlm_layer9(wav_files[stem], wavlm, device)
            if s >= 0:
                feats_al, ema_al = feats[s:], true_ema
            else:
                feats_al, ema_al = feats, true_ema[-s:]
            n = min(len(feats_al), len(ema_al))
            if n < MIN_OVERLAP_FRAMES:
                continue
            per_utt_X[stem] = feats_al[:n]
            per_utt_Y[stem] = ema_al[:n]
        except Exception as e:
            print(f"skip {stem}: {e}")
        if (i + 1) % 300 == 0:
            print(f"...aligned {i + 1}/{len(valid_stems)}")

    print(f"usable utterances: {len(per_utt_X)}")
    s_vals = np.array([s for s, c in offsets.values()])
    c_vals = np.array([c for s, c in offsets.values()])
    print(f"[CONTROL] mismatched-pairing offset stats: mean={s_vals.mean():.2f} std={s_vals.std():.2f}")
    print(f"[CONTROL] mismatched-pairing best-alignment corr: mean={c_vals.mean():.4f} std={c_vals.std():.4f}")

    all_stems = sorted(per_utt_X.keys())
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
        print(f"[CONTROL] fold {fold}: n_test_utt={len(test_stems)} mean_PCC={fold_mean:.4f}")

    print(f"[CONTROL] 5-fold CV mean PCC (mismatched-pairing alignment): {np.mean(fold_corrs):.4f} +/- {np.std(fold_corrs):.4f}")
    print("real per-utterance alignment (matched pairing): 0.9186 +/- 0.0010")
    print("base script (global 23-frame offset): 0.7021 +/- 0.0090")
    print("paper's reported figure (Appendix A.1): 0.878 +/- 0.012")

    with open(OUT_DIR / "cv_results_peralign_CONTROL_mismatched.txt", "w") as f:
        f.write(f"per-fold PCC: {fold_corrs}\n")
        f.write(f"mean +/- std: {np.mean(fold_corrs):.4f} +/- {np.std(fold_corrs):.4f}\n")
        f.write("This is a mismatched-pairing control: each utterance's ground truth was\n")
        f.write("aligned against a DIFFERENT utterance's shipped-model prediction (no real\n")
        f.write("temporal correspondence should exist). A high CV PCC here means the matched\n")
        f.write("per-utterance alignment result is inflated by search/leakage, not genuine.\n")


if __name__ == "__main__":
    import sys

    main(device=sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
