#!/bin/bash

#SBATCH --job-name=sparc_spk_raw
#SBATCH --error=/data/user_data/xoy/slurm_logs/sparc_spk_raw_%j.err
#SBATCH --output=/data/user_data/xoy/slurm_logs/sparc_spk_raw_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xoy@andrew.cmu.edu
#
# Precomputes spk_raw/*.npy (raw pre-FFN pooled WavLM speaker feature) for
# an already-SPARC-encoded dataset -- required by the vocoder training
# pipeline (see src/sparc/training/prepare_spk_raw.py).
#
# Usage:
#   sbatch scripts/prepare_spk_raw_slurm.sh <wav_dir> <sparc_dir> [limit]

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
cd "$SLURM_SUBMIT_DIR/.."

uv run python -m sparc.training.prepare_spk_raw "$@"
