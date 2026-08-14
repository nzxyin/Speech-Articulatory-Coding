#!/bin/bash

#SBATCH --job-name=encode_ljspeech
#SBATCH --error=err/encode_ljspeech_%j.err
#SBATCH --output=out/encode_ljspeech_%j.out
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
echo "Processing LJSpeech dataset"

python encode_audio.py \
    --wav_dir=/data/user_data/xoy/LJSpeech-1.1/wavs \
    --save_dir=/data/user_data/xoy/LJSpeech-1.1/preprocessed \
    --device=cuda:0

echo "Completed task: $task"


