#!/bin/bash
# Submits one training run per seed of a variant: submit_train.sh <variant> [seeds...] (default seeds 0 1 2).
# Restartable (a run is a few minutes and finished runs are skipped), one GPU each.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
VARIANT=$1
shift
AGENT=${SLURM_ALLOC_AGENT:-sparc-xlsr-ema}
for s in ${@:-0 1 2}; do
  /home/xoy/slurm-alloc/bin/slurm-alloc submit --agent "$AGENT" --no-wait --gpus 1 --cpus 4 --mem 64G --time 02:00:00 \
    --class train --restartable --job-name "ss_${VARIANT}_s${s}" -- "$REPO/scripts/spkscreen/job.sh" \
    -m sparc.vocoders.spkscreen.train --variant "$VARIANT" --seed "$s" 2>&1 | grep -E "^[0-9a-f]{12}$|^[0-9]+$"
done
