#!/bin/bash
# Preemption-safe training of one Phase 3 arm through slurm-alloc (it owns partition, GPUs, --requeue):
#   slurm-alloc submit --agent sparc-xlsr-ema --restartable --ckpt-signal USR1 --gpus 1 --cpus 8 --mem 64G \
#     --time 14:00:00 --sbatch "--signal=B:USR1@120" -- "$PWD/scripts/spkemb/p3_train.sh" hifigan p3_b2 [overrides]
# B: is required: the training python runs inside this batch shell, which forwards the signal (see below).
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

# python runs directly in the batch shell (no srun step), so the stop signal must be sent to this shell
# (--signal=B:USR1@120) and forwarded: a foreground child would block the trap until it exits, and an untrapped USR1
# (also sent by `scancel --signal=USR1 --full`) would kill this shell before the checkpoint is written. The training
# callback (PreemptionCheckpoint) checkpoints at the next step boundary and exits 0; after USR1 it also runs $REQUEUE_CMD.
# TERM (scancel) checkpoints and stops without any requeue, so cancelling a job ends it.
"$PY" -m sparc.cli.train "${ARGS[@]}" trainer.trainer.devices=1 trainer.trainer.num_nodes=1 &
PID=$!
GOT_USR1=0
forward() { echo "batch shell got SIG$1 at $(date -Is): forwarding to python ($PID)"; kill -s "$1" "$PID" 2>/dev/null; }
trap 'GOT_USR1=1; forward USR1' USR1
trap 'forward TERM' TERM
while :; do
    wait "$PID"
    rc=$?
    kill -0 "$PID" 2>/dev/null || break
done
trap - USR1 TERM
if [ "$rc" -eq 0 ] && grep -q '"finished": true' "$RUN/status.json" 2>/dev/null; then
    date -Is > "$RUN/DONE"
    echo "finished: $(cat "$RUN/DONE")"
elif [ "$rc" -eq 0 ] && [ "$GOT_USR1" -eq 1 ] && [ -n "${REQUEUE_CMD:-}" ]; then
    # requeue only after the time-limit / yield signal; a TERM (scancel) stop must stay stopped
    echo "stopped before max_g_steps: requeueing"
    eval "$REQUEUE_CMD"
    sleep 300
fi
exit $rc
