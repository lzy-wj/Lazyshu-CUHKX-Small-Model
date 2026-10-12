#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/Train/bin/python}"
GPU="${1:?Usage: $0 GPU FOLD}"
FOLD="${2:?Usage: $0 GPU FOLD}"

mkdir -p "$HERE/outputs" "$HERE/logs"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$HERE/adapt_predict_envbn.py" \
  --fold "$FOLD" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${FOLD}.pt" \
  --pseudo-labels "$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${FOLD}.npz" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --output "$HERE/outputs/fold${FOLD}.npz" \
  2>&1 | tee "$HERE/logs/fold${FOLD}.log"

