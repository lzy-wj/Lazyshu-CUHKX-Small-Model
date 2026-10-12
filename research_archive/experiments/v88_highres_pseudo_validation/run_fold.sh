#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU FOLD}"
FOLD="${2:?Usage: $0 GPU FOLD}"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160"
CHECKPOINT="$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${FOLD}.pt"
PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${FOLD}.npz"
OUTPUT="$HERE/outputs/v23_warm60_pseudo_adapt_highres160_f${FOLD}.npz"

mkdir -p "$HERE/outputs" "$HERE/logs"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
  "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
  --checkpoint "$CHECKPOINT" \
  --pseudo-labels "$PSEUDO" \
  --fold "$FOLD" \
  --scope heads \
  --epochs 2 \
  --lr 0.0002 \
  --confidence-threshold 0.6 \
  --retention-weight 0.25 \
  --batch-size 16 \
  --workers 4 \
  --cache-dir "$CACHE" \
  --frames 32 \
  --image-size 160 \
  --tta-hflip \
  --tta-crop-scale 0.75 \
  --output "$OUTPUT" 2>&1 | tee "$HERE/logs/fold${FOLD}.log"
