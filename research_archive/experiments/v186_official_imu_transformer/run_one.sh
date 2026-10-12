#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 FOLD GPU" >&2
  exit 2
fi

FOLD="$1"
GPU="$2"
ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v186_official_imu_transformer"
NAME="v186_official_transformer_f${FOLD}"
mkdir -p "$HERE/logs"
CUDA_VISIBLE_DEVICES="$GPU" "$HOME/miniconda3/envs/Train/bin/python" "$ROOT/scripts/train.py" \
  --name "$NAME" --modalities imu_official --fold "$FOLD" \
  --strict-scratch --fusion concat --imu-token-encoder transformer \
  --epochs 120 --batch-size 48 --lr 0.0005 --workers 4 --frames 16 \
  --eval-every 2 --patience 120 --warmup-epochs 5 --ema-decay 0.99 \
  --auxiliary-weight 0.15 --swanlab-mode online \
  > "$HERE/logs/${NAME}.log" 2>&1
