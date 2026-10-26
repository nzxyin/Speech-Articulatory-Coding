#!/bin/bash
# Submits seeds 0 1 2 of the multi-scale family (H9, H10, H12, H14): submit_multiscale.sh [variants...].
# Multi-layer variants hold 2 to 4 cached layers on the GPU (about 8 GB each for train plus held-out), so they ask for
# a 40+ GB GPU and more host memory; the single-layer ones use the default placement of submit_train.sh.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
AGENT=${SLURM_ALLOC_AGENT:-sparc-xlsr-ema}
BIG="L40S|L40|6000Ada|RTX_PRO_6000|A100_80GB|A100_80G|A100_40GB|H100|H200"
for v in ${@:-ms_l1_l6 ms_layerw ms_seg05 ms_mfa ms_tdnn ms_xformer ms_grl_content ms_grl_f0 ms_grl_both}; do
  case $v in
    ms_layerw) MEM=128G; export SBATCH_CONSTRAINT="$BIG" ;;
    ms_mfa) MEM=112G; export SBATCH_CONSTRAINT="$BIG" ;;
    ms_l1_l6) MEM=96G; export SBATCH_CONSTRAINT="$BIG" ;;
    *) MEM=64G; unset SBATCH_CONSTRAINT ;;
  esac
  for s in 0 1 2; do
    /home/xoy/slurm-alloc/bin/slurm-alloc submit --agent "$AGENT" --no-wait --gpus 1 --cpus 4 --mem "$MEM" --time 02:00:00 \
      --class train --restartable --job-name "ss_${v}_s${s}" --output "/data/user_data/$USER/slurm_logs/ss_${v}_s${s}_%j.out" \
      -- "$REPO/scripts/spkscreen/job.sh" -m sparc.vocoders.spkscreen.train --variant "$v" --seed "$s" 2>&1 \
      | grep -E "^[0-9a-f]{12}$|^[0-9]+$"
  done
done
