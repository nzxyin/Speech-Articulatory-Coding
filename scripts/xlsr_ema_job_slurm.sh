#!/bin/bash

#SBATCH --job-name=xlsr_ema
#SBATCH --error=/data/user_data/xoy/slurm_logs/xlsr_ema_%x_%j.err
#SBATCH --output=/data/user_data/xoy/slurm_logs/xlsr_ema_%x_%j.out
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB|VRAM_80GB
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=babel-n9-32
#SBATCH --requeue
#
# Generic runner for the SSL-compression study: runs `python -m sparc.compression.<module> <args>`.
# The train and prune modules resume from their saved state, so the job is safe to requeue or to
# submit to preempt instead (sbatch --partition=preempt --qos=preempt_qos ...).
#
# Usage:
#   sbatch -J lora_1b scripts/xlsr_ema_job_slurm.sh train --model xlsr-1b --layers 1-12 --variant independent
#   sbatch -J prune_1b scripts/xlsr_ema_job_slurm.sh prune xlsr-1b --start 24 --min 8
#   Several runs in one job (sequential): separate them with ';;'
#   sbatch -J sweep scripts/xlsr_ema_job_slurm.sh train --model xlsr-1b --layers 1-12 ';;' train --model ...

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=../../../.venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

status=0
cmd=()
run() {
    [ ${#cmd[@]} -gt 0 ] || return 0
    echo "=== sparc.compression.${cmd[*]}"
    $PY -m "sparc.compression.${cmd[0]}" "${cmd[@]:1}" || status=1
    cmd=()
}
for a in "$@"; do
    if [ "$a" = ";;" ]; then run; else cmd+=("$a"); fi
done
run
exit $status
