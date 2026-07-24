#!/bin/bash

#SBATCH --job-name=sparc_encode
#SBATCH --error=/data/user_data/xoy/slurm_logs/sparc_encode_%A_%a.err
#SBATCH --output=/data/user_data/xoy/slurm_logs/sparc_encode_%A_%a.out
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
# Generic SPARC feature extraction job. Dataset and model are Hydra config
# groups defined in src/sparc/conf/{dataset,model}.
#
# Usage:
#   Single dataset:
#     sbatch scripts/encode_slurm.sh dataset=vctk
#     sbatch scripts/encode_slurm.sh dataset=vctk model=en
#
#   Array over multiple datasets (one dataset name per array index):
#     sbatch --partition=array --array=0-3 scripts/encode_slurm.sh \
#         librittsr_train_clean_100 librittsr_train_clean_360 \
#         librittsr_dev_clean librittsr_test_clean

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Keep huggingface model-cache downloads off $HOME (its quota is tight).
export HF_HOME=/data/user_data/xoy/.cache/huggingface
cd "$SLURM_SUBMIT_DIR/.."

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    datasets=("$@")
    dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
    echo "Processing dataset: $dataset"
    uv run sparc-encode dataset="$dataset"
else
    uv run sparc-encode "$@"
fi
