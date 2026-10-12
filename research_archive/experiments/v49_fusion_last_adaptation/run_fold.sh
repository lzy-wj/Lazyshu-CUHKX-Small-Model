#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/Train/bin/python}"
GPU="${1:?Usage: $0 GPU FOLD}"
FOLD="${2:?Usage: $0 GPU FOLD}"

mkdir -p "$HERE/outputs" "$HERE/logs"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
  "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${FOLD}.pt" \
  --pseudo-labels "$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${FOLD}.npz" \
  --fold "$FOLD" --scope fusion_last --epochs 2 --lr 0.0001 \
  --confidence-threshold 0.6 --retention-weight 0.25 \
  --batch-size 16 --workers 4 \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --tta-hflip --tta-crop-scale 0.75 \
  --output "$HERE/outputs/fold${FOLD}.npz" \
  2>&1 | tee "$HERE/logs/fold${FOLD}.log"

