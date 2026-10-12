#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: run_mixup_adabn_validation.sh GPU}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="$ROOT/experiments/scene_context_v13/outputs"

export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="$HOME/miniconda3/envs/Train/bin:$PATH"
mkdir -p "$OUTPUT_ROOT/predictions/shared6_mixup_adabn"

for adaptation in global user; do
  for fold in 0 1; do
    if [[ "$fold" == 0 ]]; then
      checkpoint="$ROOT/experiments/blend_aware_v12/outputs/checkpoints/v12_shared6_mixup_f0/best_int8.pt"
    else
      checkpoint="$OUTPUT_ROOT/checkpoints/v14_shared6_mixup_f1/best_int8.pt"
    fi
    python -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
      --checkpoint "$checkpoint" \
      --fold "$fold" \
      --tta-hflip \
      --tta-crop-scale 0.75 \
      --adabn "$adaptation" \
      --batch-size 16 \
      --workers 4 \
      --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
      --frames 32 \
      --output "$OUTPUT_ROOT/predictions/shared6_mixup_adabn/${adaptation}_f${fold}.npz"
  done
done
