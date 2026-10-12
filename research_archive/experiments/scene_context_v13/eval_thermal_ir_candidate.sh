#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
FOLD="${2:-0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA_VISIBLE_DEVICES="$GPU" ${HOME}/miniconda3/envs/Train/bin/python -B \
  "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_thermal_ir_mixup_f${FOLD}/best_int8.pt" \
  --fold "$FOLD" --tta-hflip --tta-crop-scale 0.75 --batch-size 32 --workers 4 \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_thermal_ir_16_128" --frames 16 \
  --output "$ROOT/experiments/scene_context_v13/outputs/predictions/v14_thermal_ir_mixup_f${FOLD}_tta.npz"
