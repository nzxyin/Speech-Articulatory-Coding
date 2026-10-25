#!/bin/bash
# Phase 5 quality scoring of the two anchor sets (LibriTTS-R test.clean, VCTK evaluation pool); resumable.
set -uo pipefail
HERE=${SLURM_SUBMIT_DIR:-$PWD}/scripts/spkemb  # not dirname $0: sbatch stages a copy of the script
status=0
for corpus in libritts_test_clean vctk_eval; do
  "$HERE/p5_job.sh" -m sparc.vocoders.features.quality run --corpus "$corpus" --workers 8 || status=$?
done
exit $status
