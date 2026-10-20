#!/bin/bash
# Runner for speaker-embedding Phase 3 jobs submitted with slurm-alloc (no #SBATCH placement headers):
# `python <args>` from this checkout with the vocoder environment. Phase 3 runs and caches go under
# /data/user_data/$USER/spkemb/phase3, never into the vocoder comparison's $SV_ROOT/runs.
set -uo pipefail
source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
REPO=${P3_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" && [ -d src/sparc/vocoders ] || { echo "not a checkout with src/sparc/vocoders: $REPO" >&2; exit 2; }
export SPARC_VOC_RUNS=/data/user_data/$USER/spkemb/phase3/runs
export SPKEMB_P3_CACHE=/data/user_data/$USER/spkemb/phase3/cache
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$SV_REPO/.venv/bin/python" "$@"
