#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FOLD GPU VARIANT" >&2
  exit 2
fi

FOLD="$1"
GPU="$2"
VARIANT="$3"
ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v181_relative_imu_experts"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"

case "$VARIANT" in
  relative)
    MODALITIES=imu_relative
    FUSION=concat
    ;;
  dual)
    MODALITIES=imu_world,imu_relative
    FUSION=transformer
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

NAME="v181_${VARIANT}_f${FOLD}"
mkdir -p "$HERE/logs"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/scripts/train.py" \
  --name "$NAME" --modalities "$MODALITIES" --fold "$FOLD" \
  --strict-scratch --fusion "$FUSION" --epochs 60 --batch-size 24 \
  --lr 0.001 --width 96 --workers 4 --frames 16 --eval-every 2 \
  --patience 60 --warmup-epochs 5 --ema-decay 0.99 \
  --auxiliary-weight 0.15 --swanlab-mode online \
  > "$HERE/logs/${NAME}.log" 2>&1

