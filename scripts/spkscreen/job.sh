#!/bin/bash
# Runner for speaker-encoder screen jobs (docs/spkemb_encoder_screen.md), also used with slurm-alloc (no #SBATCH
# placement headers): `python <args>` from this checkout with the vocoder environment and offline model caches.
# Usage: job.sh -m sparc.vocoders.spkscreen.train --variant baseline --seed 0
set -uo pipefail
source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
REPO=${SPKSCREEN_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" && [ -d src/sparc/vocoders/spkscreen ] || { echo "not a checkout with src/sparc/vocoders/spkscreen: $REPO" >&2; exit 2; }
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export SPKSCREEN_ROOT=${SPKSCREEN_ROOT:-/data/user_data/$USER/spkemb/encoder_screen}
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec /data/user_data/xoy/venvs/sparc-fork/bin/python "$@"
