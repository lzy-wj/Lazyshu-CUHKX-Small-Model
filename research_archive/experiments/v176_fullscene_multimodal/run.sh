#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v176_fullscene_multimodal"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
FOLD="${1:?fold required}"
GPU="${2:?GPU required}"

if [[ ! "$FOLD" =~ ^[01]$ || ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "Expected fold 0/1 and GPU 4-7" >&2
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
  --fold "$FOLD" --name "v176_fullscene_f${FOLD}" \
  --frames 4 --image-size 160 --epochs 15 --batch-size 8 --workers 8 \
  --lr 0.0004 --backbone-lr-factor 0.1 --weight-decay 0.0005 \
  --label-smoothing 0.03 --auxiliary-weight 0.15 --ema-decay 0.995 \
  --swanlab-mode cloud
