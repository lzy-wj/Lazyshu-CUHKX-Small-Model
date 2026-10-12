#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${HOME}/miniconda3/envs/Train/bin/python"
for FOLD in 0 1; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
    --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_fixedsoup_uniform_l3int6_l4b0_f${FOLD}.pt" \
    --pseudo-labels "$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${FOLD}.npz" \
    --fold "$FOLD" --scope heads --epochs 3 --lr 0.0002 --confidence-threshold 0.7 \
    --retention-weight 0.25 --batch-size 16 --workers 4 \
    --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" --frames 32 \
    --tta-hflip --tta-crop-scale 0.75 \
    --output "$ROOT/experiments/scene_context_v13/outputs/predictions/v14_fixedsoup_uniform_pseudo_adapt_f${FOLD}.npz"
done
