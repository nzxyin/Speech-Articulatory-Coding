#!/bin/bash
# Runner for speaker-embedding Phase 5 jobs submitted with slurm-alloc (no #SBATCH placement headers):
# `python <args>` from this checkout with the vocoder environment, offline model caches. Outputs go under
# /data/user_data/$USER/spkemb/phase5.
set -uo pipefail
source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
REPO=${P5_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" && [ -d src/sparc/vocoders ] || { echo "not a checkout with src/sparc/vocoders: $REPO" >&2; exit 2; }
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$SV_REPO/.venv/bin/python" "$@"
