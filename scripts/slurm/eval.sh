#!/bin/bash
# Preemption-safe evaluation items: sbatch scripts/slurm/eval.sh [hydra overrides...] <stage:system:condition>...
# Use scripts/slurm/submit_eval.sh, which sets the job name, log path and resources. Each item is `stage:system:condition`
# (for example `signal:hifigan:all`, `utmos:all:all`, `aggregate:all:all`); an argument containing `=` is a Hydra override for
# every item. Items run in order; one whose DONE marker exists is skipped; the first failure stops the list. The python
# driver exits 75 when a stop signal interrupted it between chunks: the job requeues itself and resumes at the first
# unfinished chunk (finished chunks are never recomputed).
# Idempotent from line 1: a requeued job keeps its id, re-runs this script and appends to --output (--open-mode=append).
# Cluster facts as in train_vocoder.sh: preempt sends USR1 at preemption (GraceTime 120 s); --signal without B: reaches the
# srun task (python), not this shell. Only this batch script requeues (REQUEUE_CMD), never the python code.
# Environment: SV_EVAL_REPO (default ~/sparc-eval-wt, the worktree with the evaluation venv), SV_EVAL_SPLIT (test.clean),
# SV_EVAL_LIMIT (smoke subset size, unset for the full split), SV_EVAL_CPU=1 (no GPU preflight, eval.device=cpu).
#SBATCH --job-name=sv_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=USR1@120

set -u
source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
export REQUEUE_CMD="scontrol requeue $SLURM_JOB_ID"
REPO=${SV_EVAL_REPO:-$HOME/sparc-eval-wt}
SPLIT=${SV_EVAL_SPLIT:-test.clean}
LIMIT=${SV_EVAL_LIMIT:-}
CPU_ONLY=${SV_EVAL_CPU:-0}
PY="$REPO/.venv/bin/python"
cd "$REPO" || exit 1

ITEMS=()
OVERRIDES=()
for arg in "$@"; do
    case "$arg" in
        *=*) OVERRIDES+=("$arg") ;;
        *:*) ITEMS+=("$arg") ;;
        *) echo "bad argument: $arg (expected stage:system:condition or a Hydra override)"; exit 2 ;;
    esac
done
[ "${#ITEMS[@]}" -gt 0 ] || { echo "no items given"; exit 2; }

ROOT="$SV_ROOT/eval${LIMIT:+/smoke_$LIMIT}"
DONE_DIR="$ROOT/done/$SPLIT"
EXTRA=(eval.split="$SPLIT")
[ -n "$LIMIT" ] && EXTRA+=(eval.limit="$LIMIT")
[ "$CPU_ONLY" = 1 ] && EXTRA+=(eval.device=cpu)
echo "=== attempt start $(date -Is) job=$SLURM_JOB_ID restart_count=${SLURM_RESTART_COUNT:-0} node=$SLURMD_NODENAME repo=$REPO split=$SPLIT limit=${LIMIT:-none} items=${ITEMS[*]}"

marker() { local IFS=:; local p=($1); echo "$DONE_DIR/${p[0]}__${p[1]}__${p[2]}"; }
# aggregate and samples depend on every other stage and are cheap: the CLI never writes a DONE marker for them and always
# reruns them, so a marker left by an older version must not make this script skip them either.
finished() { case "$1" in aggregate:*|samples:*) return 1 ;; esac; [ -s "$(marker "$1")" ]; }

pending=0
for item in "${ITEMS[@]}"; do finished "$item" || pending=1; done
if [ "$pending" = 0 ]; then echo "all items already done"; exit 0; fi

# Slurm does not requeue FAILED jobs and about 4.5 % of preempt GPU starts fail within minutes: a node whose GPU cannot
# create a context is excluded for this job and the job is requeued (as in train_vocoder.sh).
if [ "$CPU_ONLY" != 1 ]; then
    if ! timeout 300 "$PY" -c "
import sys, torch
n = torch.cuda.device_count()
for i in range(n):
    torch.zeros(1, device=f'cuda:{i}')
sys.exit(0 if n >= 1 else 1)
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
fi

for item in "${ITEMS[@]}"; do
    IFS=: read -r stage system condition <<< "$item"
    if finished "$item"; then echo "skip $item (done: $(cat "$(marker "$item")"))"; continue; fi
    echo "=== item $item start $(date -Is)"
    srun --kill-on-bad-exit=1 "$PY" -m sparc.cli.eval_vocoder stage="$stage" system="${system:-all}" condition="${condition:-all}" "${EXTRA[@]}" "${OVERRIDES[@]}"
    rc=$?
    echo "=== item $item end $(date -Is) rc=$rc"
    if [ "$rc" -eq 75 ]; then
        echo "stopped by a signal: requeueing"
        eval "$REQUEUE_CMD"
        sleep 300
        exit 75
    elif [ "$rc" -ne 0 ]; then
        exit "$rc"
    fi
done
echo "=== attempt end $(date -Is) all items done"
