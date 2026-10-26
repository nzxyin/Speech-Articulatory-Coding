#!/bin/bash
# Submits the frame-cache extraction (3 GPU workers per set; restartable, finished chunks are skipped).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
AGENT=${SLURM_ALLOC_AGENT:-sparc-xlsr-ema}
for s in ${@:-train heldout}; do
  for w in 0 1 2; do
    /home/xoy/slurm-alloc/bin/slurm-alloc submit --agent "$AGENT" --no-wait --gpus 1 --cpus 4 --mem 48G --time 06:00:00 \
      --class train --restartable --job-name "ss_ext_${s}_${w}" -- "$REPO/scripts/spkscreen/job.sh" \
      -m sparc.vocoders.spkscreen.cache extract --set "$s" --worker "$w" --nworkers 3
  done
done
