#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: run_adabn_keep_validation.sh GPU}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="$ROOT/experiments/scene_context_v13/outputs"

export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="$HOME/miniconda3/envs/Train/bin:$PATH"
mkdir -p "$OUTPUT_ROOT/predictions/shared6_adabn_keep"

for adaptation in global_keep user_keep; do
  for fold in 0 1; do
    python -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
      --checkpoint "$ROOT/experiments/blend_aware_v12/outputs/checkpoints/v12_shared6_f${fold}/best_int8.pt" \
      --fold "$fold" \
      --tta-hflip \
      --tta-crop-scale 0.75 \
      --adabn "$adaptation" \
      --batch-size 16 \
      --workers 4 \
      --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
      --frames 32 \
      --output "$OUTPUT_ROOT/predictions/shared6_adabn_keep/${adaptation}_f${fold}.npz"
  done
done
