#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v170_radar_point_expert"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
FOLD="${1:?fold required}"
GPU="${2:?GPU required}"

if [[ ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "V170 may only use GPUs 4-7" >&2
  exit 2
fi
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi

mkdir -p "$HERE/logs" "$HERE/outputs"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u -B "$HERE/train.py" \
  --fold "$FOLD" --name "v170_points_f${FOLD}" \
  --epochs 35 --batch-size 64 --workers 4 \
  --lr 0.0004 --weight-decay 0.001 --mixup-alpha 0 \
  --label-smoothing 0.03 --auxiliary-weight 0.2 --ema-decay 0.995 \
  --swanlab-mode cloud
