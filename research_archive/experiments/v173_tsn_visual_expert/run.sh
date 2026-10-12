#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v173_tsn_visual_expert"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
FOLD="${1:?fold required}"
GPU="${2:?GPU required}"
MODALITY="${3:-Thermal}"

case "$MODALITY" in
  Thermal) SLUG=thermal ;;
  IR) SLUG=ir ;;
  Depth_Color) SLUG=depth ;;
  *) echo "Unknown modality: $MODALITY" >&2; exit 2 ;;
esac
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
exec "$PYTHON" -u -B "$ROOT/experiments/v172_thermal_frame_resnet/train.py" \
  --fold "$FOLD" --name "v173_tsn_${SLUG}_f${FOLD}" --modality "$MODALITY" \
  --clip-training --epochs 18 --batch-size 16 --validation-batch-size 12 \
  --workers 8 --train-views 4 --validation-views 12 \
  --lr 0.0004 --backbone-lr-factor 0.1 --weight-decay 0.0005 \
  --label-smoothing 0.03 --ema-decay 0.995 --swanlab-mode cloud
