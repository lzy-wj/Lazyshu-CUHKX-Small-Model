#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU FOLD}"
FOLD="${2:?Usage: $0 GPU FOLD}"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$FOLD" =~ ^[01]$ ]] || { echo "Fold must be 0 or 1" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi

NAME="v105_posec3d_heatmap_f${FOLD}"
mkdir -p "$HERE/logs" "$HERE/outputs"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u -B "$HERE/train.py" \
  --name "$NAME" --fold "$FOLD" \
  --cache-dir "$ROOT/data/processed/posec3d_32_48" \
  --epochs 30 --batch-size 24 --workers 4 --lr 0.0004 \
  --weight-decay 0.001 --warmup-epochs 2 --mixup-alpha 0.2 \
  --label-smoothing 0.03 --ema-decay 0.99 --seed 42 \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  2>&1 | tee "$HERE/logs/${NAME}.log"
