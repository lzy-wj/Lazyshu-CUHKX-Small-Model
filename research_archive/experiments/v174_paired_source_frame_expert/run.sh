#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v174_paired_source_frame_expert"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
FOLD="${1:?fold required}"
TARGET="${2:?target user required}"
GPU="${3:?GPU required}"

case "$FOLD:$TARGET" in
  0:8|0:18|1:7|1:17) ;;
  *) echo "Expected 0:8, 0:18, 1:7, or 1:17" >&2; exit 2 ;;
esac
if [[ ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "V174 may only use GPUs 4-7" >&2
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
  --fold "$FOLD" --validation-users "$TARGET" \
  --name "v174_thermal_user${TARGET}_f${FOLD}" --modality Thermal \
  --output-root "$HERE/outputs" \
  --epochs 15 --batch-size 64 --validation-batch-size 12 --workers 8 \
  --train-views 4 --validation-views 12 --lr 0.0003 \
  --weight-decay 0.0001 --label-smoothing 0.03 --ema-decay 0.995 \
  --swanlab-mode cloud
