#!/bin/bash
# Runner for slurm-alloc submissions (no #SBATCH placement headers: the allocator owns partition, QoS and GPUs).
# Runs `python <args>` from the repo root with the project environment, e.g.
#   slurm-alloc submit --agent sparc-xlsr-ema --no-msp --gpus 1 --cpus 8 --mem 64G --time 04:00:00 \
#       -- "$PWD/scripts/rigor_job.sh" -m sparc.compression.rigor head-preds --model wavlm-large --layer 12 --head mamba
set -uo pipefail
# slurm-alloc sets --chdir to the caller's directory (submit from the repo root); plain sbatch: the submit dir
[ -d src/sparc ] || cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
[ -d src/sparc ] || { echo "not in the repo root: $PWD" >&2; exit 2; }
export HF_HUB_OFFLINE=1
if [ -d /data/user_data/xoy/hf_hub_jobs ]; then export HF_HUB_CACHE=/data/user_data/xoy/hf_hub_jobs; fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=/home/xoy/Speech-Articulatory-Coding/.venv/bin/python
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$PY" "$@"
