#!/bin/bash

#SBATCH --job-name=xlsr_ema_probe
#SBATCH --error=/data/user_data/xoy/slurm_logs/xlsr_ema_probe_%j.err
#SBATCH --output=/data/user_data/xoy/slurm_logs/xlsr_ema_probe_%j.out
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=babel-n9-32
#
# Phase 1 of the SSL-compression study: cache every layer's hidden states of one SSL model over
# MNGU0, then fit a ridge EMA probe per layer (sparc.compression.extract / .probe).
# Restart-safe: extraction is redone unless it completed; probing resumes per layer.
#
# Usage: sbatch scripts/xlsr_ema_probe_slurm.sh <model>   # xlsr-1b | xlsr-300m | wavlm-large | xlsr-2b

set -euo pipefail
MODEL=${1:?model name}
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=../../../.venv/bin/python  # worktrees under .claude/worktrees share the main venv

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -m sparc.compression.extract "$MODEL"
$PY -m sparc.compression.probe "$MODEL"
