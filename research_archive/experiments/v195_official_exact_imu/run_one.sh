#!/usr/bin/env bash
set -euo pipefail

fold="$1"
seed="$2"
gpu="$3"
mkdir -p experiments/v195_official_exact_imu/logs
exec ${HOME}/miniconda3/envs/Train/bin/python \
  experiments/v195_official_exact_imu/train.py \
  --fold "$fold" --seed "$seed" --gpu "$gpu" \
  > "experiments/v195_official_exact_imu/logs/v195_official_exact_s${seed}_f${fold}.log" 2>&1
