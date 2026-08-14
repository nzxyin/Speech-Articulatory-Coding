#!/bin/bash

#SBATCH --job-name=decode_librittsr
#SBATCH --error=err/decode_librittsr_%A_%a.err
#SBATCH --output=out/decode_librittsr_%A_%a.out
#SBATCH --partition=array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --array=0-1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=xoy@andrew.cmu.edu

source ../.venv/bin/activate

tasks=(
    # "train-clean-100"
    # "train-clean-360"
    "dev-clean-sparc"
    "test-clean-sparc"
)

task=${tasks[$SLURM_ARRAY_TASK_ID]}

echo "Processing task: $task"

python decode_audio.py \
    --sparc_dir=/data/user_data/xoy/LibriTTS_R/$task \
    --save_dir=/data/user_data/xoy/LibriTTS_R/${task}-resynth \
    --device=cuda:0

echo "Completed task: $task"


