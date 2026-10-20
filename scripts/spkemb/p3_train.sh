#!/bin/bash
# Preemption-safe training of one Phase 3 arm through slurm-alloc (it owns partition, GPUs, --requeue):
#   slurm-alloc submit --agent sparc-xlsr-ema --restartable --ckpt-signal USR1 --gpus 1 --cpus 8 --mem 64G \
#     --time 14:00:00 --sbatch "--signal=USR1@120" -- "$PWD/scripts/spkemb/p3_train.sh" hifigan p3_b2 [overrides]
# Same logic as scripts/slurm/train_vocoder.sh (resume from the newest checkpoint, DONE marker, requeue when stopped
# early), but code from this checkout and runs under /data/user_data/$USER/spkemb/phase3/runs.
set -u
source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
VOCODER=${1:?vocoder name}
EXPERIMENT=${2:?experiment name}
shift 2
REPO=${P3_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" && [ -d src/sparc/vocoders ] || { echo "not a checkout with src/sparc/vocoders: $REPO" >&2; exit 2; }
export SPARC_VOC_RUNS=/data/user_data/$USER/spkemb/phase3/runs
export SPKEMB_P3_CACHE=/data/user_data/$USER/spkemb/phase3/cache
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
PY="$SV_REPO/.venv/bin/python"
ARGS=(vocoder="$VOCODER" experiment="$EXPERIMENT" "$@")

CFG=$("$PY" -m sparc.cli.train "${ARGS[@]}" --cfg job --resolve)
RUN=$(printf '%s\n' "$CFG" | sed -n 's/^run_dir: //p')
[ -n "$RUN" ] || { echo "could not resolve run_dir"; exit 1; }
case "$RUN" in "$SPARC_VOC_RUNS"/*) ;; *) echo "run_dir $RUN is outside $SPARC_VOC_RUNS"; exit 1 ;; esac
if printf '%s\n' "$CFG" | grep -qE '^    deterministic: true$'; then export CUBLAS_WORKSPACE_CONFIG=:4096:8; fi
mkdir -p "$RUN/ckpt"
echo "=== attempt start $(date -Is) job=${SLURM_JOB_ID:-} restart_count=${SLURM_RESTART_COUNT:-0} node=${SLURMD_NODENAME:-} run=$RUN"
if [ -s "$RUN/DONE" ]; then echo "already done: $(cat "$RUN/DONE")"; exit 0; fi
[ -n "${SLURM_JOB_ID:-}" ] && export REQUEUE_CMD="scontrol requeue $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -m sparc.cli.train "${ARGS[@]}" trainer.trainer.devices=1 trainer.trainer.num_nodes=1
rc=$?
if [ "$rc" -eq 0 ] && grep -q '"finished": true' "$RUN/status.json" 2>/dev/null; then
    date -Is > "$RUN/DONE"
    echo "finished: $(cat "$RUN/DONE")"
elif [ "$rc" -eq 0 ] && [ -n "${REQUEUE_CMD:-}" ]; then
    echo "stopped before max_g_steps: requeueing"
    eval "$REQUEUE_CMD"
    sleep 300
fi
exit $rc
