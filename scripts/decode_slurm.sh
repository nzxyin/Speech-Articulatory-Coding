#!/bin/bash

#SBATCH --job-name=sparc_decode
#SBATCH --error=/data/user_data/xoy/slurm_logs/sparc_decode_%A_%a.err
#SBATCH --output=/data/user_data/xoy/slurm_logs/sparc_decode_%A_%a.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=xoy@andrew.cmu.edu
#
# Generic SPARC resynthesis job. Reads features from dataset.save_dir
# (the output of encode_slurm.sh) and writes wavs to dataset.decode_dir.
#
# Usage:
#   Single dataset:
#     sbatch scripts/decode_slurm.sh dataset=vctk
#
#   Array over multiple datasets (one dataset name per array index):
#     sbatch --partition=array --array=0-1 scripts/decode_slurm.sh \
#         librittsr_dev_clean librittsr_test_clean
#
#   Large datasets that may run past the general/cpu partitions' 2-day cap:
#   use the preempt partition instead (up to 31 days, but jobs there can be
#   killed and requeued from the start of the script at any time by higher-
#   priority partitions -- this is safe here because the per-file skip-if-
#   exists check below makes a requeued run resume rather than redo work):
#     sbatch --partition=preempt --gres=gpu:1 --time=6-00:00:00 \
#         scripts/decode_slurm.sh dataset=globe_v2

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Keep huggingface model-cache downloads off $HOME (its quota is tight).
export HF_HOME=/data/user_data/xoy/.cache/huggingface
cd "$SLURM_SUBMIT_DIR/.."

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    datasets=("$@")
    dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
    echo "Processing dataset: $dataset"
    uv run sparc-decode dataset="$dataset"
else
    uv run sparc-decode "$@"
fi
