"""LoRA in the frozen-head pipeline (pre-registered contrasts T1/T2, docs/rigor_prereg.md addendum).

WavLM Large k12 + LoRA r8 q/v + BiGRU -> conv 9 head, trained like the frozen heads of head_screen: encoder output
low-passed in-graph with the cached features' zero-phase 10 Hz filter, no output smoother, the fold's depth-sweep
shift, encoder dropout off, fp32, head learning rate = head_screen's per-fold choice for gruconv9 at k12; LoRA learning
rate from LRS chosen on the training speakers' validation with seed 0, then seeds 1 and 2.

Usage:
    python -m sparc.compression.matched lr --fold usc_M1 --lr 2e-4    # seed 0 at one LoRA learning rate
    python -m sparc.compression.matched seeds --fold usc_M1           # pick the rate, run seeds 1-2
    python -m sparc.compression.matched choices                       # print the chosen rates
"""

import argparse
import json

from . import datasets

MODEL, K = "wavlm-large", 12
LRS = (1e-4, 2e-4, 5e-4)
OUT = datasets.XLSR_EMA / "analysis_rigor" / "matched"


def run_name(lr, s):
    return f"prefix{K}_independent_r8_q_proj+v_proj_gruconv_k9_nosm_m{lr:g}_s{s}"


def fold_settings(fold):
    from .depth_sweep import load_all
    from .head_screen import OUT as HS

    shift = load_all(MODEL)[f"{K}/{fold}"]["shift"]
    head_lr = json.loads((HS / MODEL / f"k{K}_gruconv9.json").read_text())[fold]["lr"]
    return shift, head_lr


def train_one(fold, lr, seed):
    from .train import main as train_main

    shift, head_lr = fold_settings(fold)
    train_main(["--model", MODEL, "--dataset", f"ema_loso_{fold}", "--layers", f"1-{K}", "--variant", "independent",
                "--rank", "8", "--targets", "q_proj,v_proj", "--head", "gruconv", "--kernel", "9", "--no-smooth",
                "--input-lowpass", "--no-enc-dropout", "--no-bf16", "--shift", str(shift), "--head-lr", str(head_lr),
                "--lr", str(lr), "--seed", str(seed), "--tag", f"m{lr:g}"])


def valid_score(fold, lr):
    f = datasets.adapt_root(f"ema_loso_{fold}") / MODEL / run_name(lr, 0) / "results.json"
    if not f.exists():
        return None
    v = json.loads(f.read_text())["valid"]
    return v.get("rmse_z", v.get("rmse"))


def choose(fold):
    scores = {lr: valid_score(fold, lr) for lr in LRS}
    missing = [lr for lr, v in scores.items() if v is None]
    if missing:
        raise RuntimeError(f"{fold}: seed-0 runs missing for {missing}")
    lr = min(scores, key=scores.get)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{fold}.json").write_text(json.dumps({"lr": lr, "valid": {str(k): v for k, v in scores.items()}}, indent=1))
    return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("lr", "seeds", "choices"))
    ap.add_argument("--fold")
    ap.add_argument("--lr", type=float)
    a = ap.parse_args()
    if a.cmd == "lr":
        train_one(a.fold, a.lr, 0)
    elif a.cmd == "seeds":
        lr = choose(a.fold)
        print(f"{a.fold}: LoRA lr {lr:g}", flush=True)
        for s in (1, 2):
            train_one(a.fold, lr, s)
    else:
        for h in datasets.ALL_SPEAKERS:
            f = OUT / f"{h}.json"
            print(h, json.loads(f.read_text()) if f.exists() else "-")


if __name__ == "__main__":
    main()
