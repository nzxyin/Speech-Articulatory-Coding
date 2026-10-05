#!/bin/bash
# Array job of the feature cache: task k extracts shard k of NUM_SHARDS (one GPU per task). Submit through
# scripts/slurm/submit_cache.sh. Extra arguments are Hydra overrides for the cache driver.
# Idempotent from line 1: a requeued or resubmitted task skips cache files that are already valid and exits at once if
# its shard is marked done. Exit codes of the python driver: 0 done, 75 stopped by a signal (requeued below), else
# error. The driver runs as `python -m` so the script works before the sparc-cache console script is installed.
#SBATCH --job-name=sv_cache
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=USR1@120
#SBATCH --output=slurm_logs/%x_%A_%a.out

source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
export REQUEUE_CMD="scontrol requeue $SLURM_JOB_ID"
cd "$SV_REPO" || exit 1

SHARD=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
TAG=$(printf '%05dof%05d' "$SHARD" "$NUM_SHARDS")
DONE_DIR=$SPARC_VOC_CACHE/done
BAD_NODES=$SPARC_VOC_CACHE/bad_gpu_nodes.txt
mkdir -p "$DONE_DIR"
echo "=== attempt start $(date -Is) job=$SLURM_JOB_ID shard=$TAG restart_count=${SLURM_RESTART_COUNT:-0} node=$SLURMD_NODENAME"

if [ -s "$DONE_DIR/extract_$TAG" ]; then echo "shard $TAG already done"; exit 0; fi

# Slurm does not requeue FAILED jobs and about 4.5 % of preempt GPU starts fail within minutes, so a node whose GPU
# cannot create a context is excluded for this job and the job is requeued.
reject_node() {
  echo "$SLURMD_NODENAME" >> "$BAD_NODES"
  local excluded
  excluded=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^ExcNodeList=//p')
  [ "$excluded" = "(null)" ] && excluded=""
  scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="${excluded:+$excluded,}$SLURMD_NODENAME" \
    || echo "ExcNodeList update refused; $SLURMD_NODENAME is recorded in $BAD_NODES instead"
  $REQUEUE_CMD
  sleep 120
  exit 1
}

if [ -f "$BAD_NODES" ] && grep -qx "$SLURMD_NODENAME" "$BAD_NODES"; then
  echo "node $SLURMD_NODENAME is listed in $BAD_NODES"
  reject_node
fi
if ! timeout 300 "$SV_PY" -c 'import torch; x = torch.zeros(1).cuda(); torch.cuda.synchronize(); print(torch.cuda.get_device_name())'; then
  echo "GPU preflight failed on $SLURMD_NODENAME"
  reject_node
fi

srun --kill-on-bad-exit=1 uv run --no-sync python -m sparc.cli.cache_features stage=extract shard_index="$SHARD" num_shards="$NUM_SHARDS" "$@"
rc=$?
if [ $rc -eq 0 ]; then
  date -Is > "$DONE_DIR/extract_$TAG"
elif [ $rc -eq 75 ]; then
  echo "stopped by a signal; requeueing"
  $REQUEUE_CMD
  sleep 120
fi
echo "=== attempt end $(date -Is) rc=$rc"
exit $rc
