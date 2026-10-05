#!/bin/bash
# Builds and runs the sbatch command line of the feature-cache array job.
#   submit_cache.sh [--shards N] [--concurrency M] [--only 3,7,12] [--time HH:MM:SS] [--dry-run] [-- hydra overrides]
# --only resubmits just the listed shards of the same N (for failed tasks); --dry-run prints the command and exits.
set -euo pipefail

source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
SHARDS=64
CONCURRENCY=12
ONLY=""
TIME_LIMIT=""
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --shards) SHARDS=$2; shift 2 ;;
    --concurrency) CONCURRENCY=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    --time) TIME_LIMIT=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; break ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=$SV_ROOT/slurm_logs
if [ -n "$ONLY" ]; then ARRAY="$ONLY%$CONCURRENCY"; else ARRAY="0-$((SHARDS - 1))%$CONCURRENCY"; fi

CMD=(sbatch --array="$ARRAY" --export="ALL,NUM_SHARDS=$SHARDS" --output="$LOG_DIR/%x_%A_%a.out")
[ -n "$TIME_LIMIT" ] && CMD+=(--time="$TIME_LIMIT")
CMD+=("$SCRIPT_DIR/cache_features.sh" "$@")

if [ "$DRY_RUN" -eq 1 ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
if [ ! -f "$SPARC_VOC_CACHE/manifest.parquet" ]; then
  echo "manifest missing: run 'sparc-cache stage=manifest' first" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
"${CMD[@]}"
