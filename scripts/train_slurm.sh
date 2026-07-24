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
#   run well past the general/cpu partitions' 2-day cap -- use preempt
#   instead (it can kill and requeue the job from the start of the script
#   at any time, so pass --resume_from_checkpoint pointing at the last
#   Lightning checkpoint under <dataset.save_dir>/vocoder_ckpt/last.ckpt to
#   make a requeued run pick up where it left off rather than restart):
#     sbatch --partition=preempt --gres=gpu:1 --time=20-00:00:00 \
#         scripts/train_slurm.sh dataset=librittsr_train_clean_360 \
#         max_steps=1500000 resume_from_checkpoint=/data/user_data/xoy/LibriTTS_R/train-clean-360-sparc/vocoder_ckpt/last.ckpt

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export WANDB_CACHE_DIR=/data/user_data/xoy/.cache/wandb
cd "$SLURM_SUBMIT_DIR/.."

uv run sparc-train "$@"
