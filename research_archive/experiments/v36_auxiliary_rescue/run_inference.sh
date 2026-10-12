#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
mkdir -p "$HERE/outputs" "$HERE/logs"

for fold in 0 1; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
    --checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${fold}.pt" \
    --fold "$fold" --cache-dir "$CACHE" --frames 32 --batch-size 16 --workers 4 \
    --tta-hflip --tta-crop-scale 0.75 --adabn global --adabn-passes 1 --save-auxiliary \
    --output "$HERE/outputs/v23_aux_adabn_f${fold}.npz" \
    2>&1 | tee "$HERE/logs/inference_f${fold}.log"
done
