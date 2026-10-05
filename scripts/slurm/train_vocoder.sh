#!/bin/bash
# Preemption-safe training of one vocoder: sbatch scripts/slurm/train_vocoder.sh <vocoder> <experiment> [hydra overrides...]
# Use scripts/slurm/submit_train.sh, which sets the job name, log path, GPU count and memory. Idempotent from line 1:
# a requeued job keeps its id, re-runs this script and truncates --output unless --open-mode=append (set below).
# Cluster facts (compute.md, Part A): preempt sends USR1 at preemption (send_user_signal, GraceTime 120 s, KillWait 120 s);
# --signal without B: reaches the srun tasks (python), not this shell.
# The default --output is relative (the submit script overrides it with a path under $SV_ROOT/slurm_logs).
#SBATCH --job-name=sv_train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=USR1@120

set -u
source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
VOCODER=${1:?vocoder name}
EXPERIMENT=${2:?experiment name}
shift 2
GPUS=${SLURM_NTASKS_PER_NODE:-1}
ARGS=(vocoder="$VOCODER" experiment="$EXPERIMENT" "$@")

cd "$SV_REPO" || exit 1
CFG=$(uv run --no-sync sparc-train "${ARGS[@]}" --cfg job --resolve)
RUN=$(printf '%s\n' "$CFG" | sed -n 's/^run_dir: //p')
[ -n "$RUN" ] || { echo "could not resolve run_dir"; exit 1; }
# Bit-exact GPU resume needs deterministic cuBLAS workspaces (set before any process touches cuBLAS).
if printf '%s\n' "$CFG" | grep -qE '^    deterministic: true$'; then export CUBLAS_WORKSPACE_CONFIG=:4096:8; fi
mkdir -p "$RUN/ckpt"
echo "=== attempt start $(date -Is) job=$SLURM_JOB_ID restart_count=${SLURM_RESTART_COUNT:-0} node=$SLURMD_NODENAME run=$RUN"

if [ -s "$RUN/DONE" ]; then echo "already done: $(cat "$RUN/DONE")"; exit 0; fi

# Only the sbatch script requeues: ssh sessions adopted into a job also carry SLURM_JOB_ID (the python callback runs this
# command only when the variable is set).
export REQUEUE_CMD="scontrol requeue $SLURM_JOB_ID"

# GPU preflight: about 4.5 % of preempt GPU starts fail within minutes and Slurm does not requeue FAILED jobs.
if ! "$SV_REPO/.venv/bin/python" -c "
import sys, torch
n = torch.cuda.device_count()
for i in range(n):
    torch.zeros(1, device=f'cuda:{i}')
sys.exit(0 if n >= $GPUS else 1)
"; then
    echo "GPU preflight failed on $SLURMD_NODENAME"
    BAD="$SPARC_VOC_RUNS/bad_gpu_nodes.txt"
    echo "$SLURMD_NODENAME" >> "$BAD"
    EXCLUDE=$(sort -u "$BAD" | paste -sd,)
    scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="$EXCLUDE" || echo "ExcNodeList update refused for a running job; excluded nodes are listed in $BAD"
    eval "$REQUEUE_CMD" || exit 1
    sleep 600
    exit 1
fi

export NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NODE_RANK=0

# One task per GPU; LOCAL_RANK makes Lightning's LightningEnvironment treat the processes as externally launched.
srun --kill-on-bad-exit=1 --ntasks="$GPUS" --ntasks-per-node="$GPUS" --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" bash -c '
    export LOCAL_RANK=$SLURM_LOCALID
    exec uv run --no-sync sparc-train "$@"
' _ "${ARGS[@]}" trainer.trainer.devices="$GPUS" trainer.trainer.num_nodes=1
rc=$?

if [ "$rc" -eq 0 ] && grep -q '"finished": true' "$RUN/status.json" 2>/dev/null; then
    date -Is > "$RUN/DONE"
    echo "finished: $(cat "$RUN/DONE")"
elif [ "$rc" -eq 0 ]; then
    echo "stopped before max_g_steps: requeueing"
    eval "$REQUEUE_CMD"
    sleep 300
fi
exit $rc
