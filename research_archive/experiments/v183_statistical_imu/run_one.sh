#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FOLD GPU VARIANT" >&2
  exit 2
fi

FOLD="$1"
GPU="$2"
VARIANT="$3"
case "$VARIANT" in
  official_stats) MODALITIES=imu_official ;;
  magnetic_stats) MODALITIES=imu_magnetic ;;
  dual_stats) MODALITIES=imu_world,imu_official ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v183_statistical_imu"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
NAME="v183_${VARIANT}_f${FOLD}"
mkdir -p "$HERE/logs"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/scripts/train.py" \
  --name "$NAME" --modalities "$MODALITIES" --fold "$FOLD" \
  --strict-scratch --fusion concat --imu-token-encoder statistics \
  --epochs 80 --batch-size 48 --lr 0.0005 --workers 4 --frames 16 \
  --eval-every 2 --patience 80 --warmup-epochs 5 --ema-decay 0.99 \
  --auxiliary-weight 0.15 --swanlab-mode online \
  > "$HERE/logs/${NAME}.log" 2>&1

