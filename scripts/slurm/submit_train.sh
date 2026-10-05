#!/bin/bash
# Submits one preemption-safe vocoder training job.
# usage: scripts/slurm/submit_train.sh <vocoder> <experiment> [gpus] [hydra overrides...]
#   scripts/slurm/submit_train.sh vocos main
#   scripts/slurm/submit_train.sh hifigan precision_pilot 1 trainer.trainer.precision=bf16-mixed
# TIME overrides the time limit (default 2-00:00:00). The run directory is fixed by the experiment and vocoder names,
# so submitting the same pair again resumes the run (or exits at once if it is finished).
set -euo pipefail
source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
vocoder=${1:?usage: submit_train.sh <vocoder> <experiment> [gpus] [hydra overrides...]}
experiment=${2:?usage: submit_train.sh <vocoder> <experiment> [gpus] [hydra overrides...]}
gpus=${3:-1}
shift $(( $# < 3 ? $# : 3 ))
mkdir -p "$SV_ROOT/slurm_logs"
exec sbatch \
    --job-name="sv_${experiment}_${vocoder}" \
    --ntasks-per-node="$gpus" \
    --gres="gpu:$gpus" \
    --mem="$((32 * gpus))G" \
    --time="${TIME:-2-00:00:00}" \
    --output="$SV_ROOT/slurm_logs/%x_%j.out" \
    "$SV_REPO/scripts/slurm/train_vocoder.sh" "$vocoder" "$experiment" "$@"
