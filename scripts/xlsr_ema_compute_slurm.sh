#!/bin/bash

#SBATCH --job-name=xlsr_ema_compute
#SBATCH --error=/data/user_data/xoy/slurm_logs/xlsr_ema_compute_%j.err
#SBATCH --output=/data/user_data/xoy/slurm_logs/xlsr_ema_compute_%j.out
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=babel-n9-32
#
# Inference cost (params, FLOPs, activation memory, latency/RTF) of every prefix truncation of each
# model, all in one job so latencies come from the same GPU type (pinned to L40S).
#
# Usage: sbatch scripts/xlsr_ema_compute_slurm.sh [model ...]   # default: xlsr-300m wavlm-large xlsr-1b

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=../../../.venv/bin/python

MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(xlsr-300m wavlm-large xlsr-1b)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for m in "${MODELS[@]}"; do
    $PY -m sparc.compression.compute "$m"
done
