#!/bin/bash
# Submits one preemption-safe evaluation job.
# usage: scripts/slurm/submit_eval.sh [--cpu] [--name NAME] <items and hydra overrides...>
#   scripts/slurm/submit_eval.sh gt:gt:T1 refs:all:all signal:all:all           (GPU job, partition preempt, preempt_qos)
#   scripts/slurm/submit_eval.sh --cpu prosody:all:all aggregate:all:all samples:all:all   (no GPU: preempt_cpu_qos)
#   SV_EVAL_LIMIT=8 scripts/slurm/submit_eval.sh gt:gt:T1 ...                    (smoke subset under eval_root/smoke_8)
# Items are stage:system:condition; arguments with `=` are Hydra overrides (for example eval.require_final=false). The list
# runs in order and finished items are skipped, so submitting the same list again resumes it. Environment: TIME (default
# 12:00:00), SV_EVAL_SPLIT, SV_EVAL_LIMIT, SV_EVAL_REPO. The evaluation uses at most one GPU (training holds the others).
# CPU stages: gt, refs on CPU are slow, so use the GPU job for them; --cpu suits signal, prosody, aggregate, samples and the
# CPU timings of the efficiency stage (eval.device=cpu is set automatically).
set -euo pipefail
source "${SPARC_ENV_FILE:-$HOME/sparc-vocoders-work/env.sh}"
cpu=0
name=sv_eval
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) cpu=1; shift ;;
        --name) name=$2; shift 2 ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] || { sed -n '2,12p' "$0"; exit 2; }
mkdir -p "$SV_ROOT/slurm_logs" 2>/dev/null || true   # /data is not mounted on the login node (the directory exists)
repo=${SV_EVAL_REPO:-$HOME/sparc-eval-wt}
args=(--job-name="$name" --time="${TIME:-12:00:00}" --output="$SV_ROOT/slurm_logs/%x_%j.out"
      --export="ALL,SV_EVAL_CPU=$cpu${SV_EVAL_SPLIT:+,SV_EVAL_SPLIT=$SV_EVAL_SPLIT}${SV_EVAL_LIMIT:+,SV_EVAL_LIMIT=$SV_EVAL_LIMIT}${SV_EVAL_REPO:+,SV_EVAL_REPO=$SV_EVAL_REPO}")
if [ "$cpu" = 1 ]; then
    args+=(--qos=preempt_cpu_qos --gres=none --cpus-per-task=16 --mem=64G)
else
    args+=(--gres=gpu:1)
fi
exec sbatch "${args[@]}" "$repo/scripts/slurm/eval.sh" "$@"
