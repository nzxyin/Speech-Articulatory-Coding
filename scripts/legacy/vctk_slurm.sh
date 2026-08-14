#!/bin/bash

#SBATCH --job-name=encode_vctk
#SBATCH --error=err/encode_vctk_%j.err
#SBATCH --output=out/encode_vctk_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=xoy@andrew.cmu.edu

source ../.venv/bin/activate
echo "Processing VCTK dataset"

python encode_audio.py \
    --wav_dir=/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus \
    --save_dir=/data/user_data/xoy/VCTK/VCTK-Corpus \
    --device=cuda:0

echo "Completed VCTK"
