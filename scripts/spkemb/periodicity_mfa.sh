#!/bin/bash
# Forced alignment of the periodicity-analysis subset (docs/vocoder_periodicity_analysis.md) with the Montreal Forced Aligner
# (english_mfa dictionary and acoustic model, as the group's other alignments). CPU job: submit with slurm-alloc.
# Corpus: /data/user_data/xoy/sparc-vocoders-periodicity/mfa/corpus (written by `periodicity_analysis prepare`).
set -uo pipefail
ROOT=/data/user_data/xoy/sparc-vocoders-periodicity
export MFA_ROOT_DIR=$ROOT/mfa_root
export PATH=/data/user_data/xoy/envs/mfa/bin:$PATH
mkdir -p "$MFA_ROOT_DIR"
[ -d "$MFA_ROOT_DIR/pretrained_models" ] || cp -r /data/user_data/xoy/mfa_root/pretrained_models "$MFA_ROOT_DIR/pretrained_models"
/data/user_data/xoy/envs/mfa/bin/mfa align "$ROOT/mfa/corpus" english_mfa english_mfa "$ROOT/mfa/out" \
    --clean --num_jobs 8 --beam 100 --retry_beam 400 --output_format long_textgrid
rc=$?
echo "mfa exit $rc"
echo "textgrids: $(find "$ROOT/mfa/out" -name '*.TextGrid' | wc -l)"
exit $rc
