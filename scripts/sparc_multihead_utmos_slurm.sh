#!/bin/bash
# UTMOSv2 naturalness of the sparc_multihead resynthesis systems (orig, shipped, loso, all, true_ema).
# Links each system's wavs into one flat directory per system and source, then runs UTMOSv2 (its own venv).
#SBATCH --job-name=smh_utmos
#SBATCH --output=/data/user_data/xoy/slurm_logs/smh_utmos_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/smh_utmos_%j.err
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=babel-n9-32
set -e
OUT=/data/user_data/xoy/xlsr_ema/sparc_multihead
UTMOS=/home/xoy/UTMOSv2
mkdir -p "$OUT/utmos_in" "$OUT/utmos"
for sysdir in "$OUT"/wav/*; do
  sys=$(basename "$sysdir")
  for spkdir in "$sysdir"/*; do
    spk=$(basename "$spkdir")
    src=ema
    [ "$spk" = libri ] && src=libri
    dst="$OUT/utmos_in/${sys}_${src}"
    mkdir -p "$dst"
    for w in "$spkdir"/*.wav; do
      ln -sf "$w" "$dst/${spk}__$(basename "$w")"
    done
  done
done
cd "$UTMOS"
source .venv/bin/activate
for d in "$OUT"/utmos_in/*; do
  name=$(basename "$d")
  [ -f "$OUT/utmos/$name.csv" ] && continue
  python inference.py --input_dir "$d" --out_path "$OUT/utmos/$name.csv"
done
