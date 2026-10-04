#!/bin/bash

#SBATCH --job-name=sparc_train
#SBATCH --error=/data/user_data/xoy/slurm_logs/sparc_train_%j.err
#SBATCH --output=/data/user_data/xoy/slurm_logs/sparc_train_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xoy@andrew.cmu.edu
#
# HiFi-GAN vocoder + speaker-encoder training (reproduction of the SPARC
# paper's Section III-B / Appendix B training methodology). Requires
# per-dataset spk_raw/*.npy features -- run
# `uv run python -m sparc.training.prepare_spk_raw <wav_dir> <sparc_dir>`
# first (see src/sparc/training/prepare_spk_raw.py).
#
# Usage:
#   sbatch scripts/train_slurm.sh dataset=vctk
#   sbatch scripts/train_slurm.sh dataset=vctk max_steps=1500000
#
#   Full reproduction (paper: 1.5M steps, batch 64, ~555h LibriTTS-R) will
#   run well past the general partition's 2-day cap -- use preempt instead.
#   The job is requeue-safe: --requeue + --signal=B:USR1@120 make SLURM send
#   SIGUSR1 to this script 120s before preemption, which is forwarded to
#   python; Lightning then writes <ckpt_dir>/hpc_ckpt_N.ckpt and calls
#   `scontrol requeue`. The requeued job keeps its SLURM_JOB_ID, so
#   sparc-train finds <dataset.save_dir>/vocoder_ckpt/slurm_<jobid>/ and
#   resumes from the newest last.ckpt/hpc_ckpt_*.ckpt automatically. To
#   continue a run across separate submissions, pass the same run_name=<name>.
#   max_steps counts batches (one discriminator + one generator update).
#     sbatch --partition=preempt --gres=gpu:1 --time=20-00:00:00 \
#         scripts/train_slurm.sh dataset=librittsr_train_clean_360 \
#         max_steps=1500000

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export WANDB_CACHE_DIR=/data/user_data/xoy/.cache/wandb
cd "$SLURM_SUBMIT_DIR/.."

# Run python directly (not under `uv run`) in the background so the batch
# shell can forward SLURM's SIGUSR1 to it (--signal=B:... only signals this shell).
PY=$(uv run python -c 'import sys; print(sys.executable)')
"$PY" -m sparc.cli.train "$@" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$pid" 2>/dev/null || true' TERM
# `wait` returns early when a trapped signal arrives; keep waiting for the real exit.
while kill -0 "$pid" 2>/dev/null; do
    wait "$pid" && rc=0 || rc=$?
done
exit "${rc:-0}"
