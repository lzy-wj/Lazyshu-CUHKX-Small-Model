#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${HOME}/miniconda3/envs/Train/bin/python"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
OUT="$ROOT/experiments/scene_context_v13/outputs/predictions/shared6_mixup_fixedsoup_quant"
mkdir -p "$OUT"
for VARIANT in nf4_l3int6 uniform_l3int6_l4b0; do
  for FOLD in 0 1; do
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
      --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_fixedsoup_${VARIANT}_f${FOLD}.pt" \
      --fold "$FOLD" --tta-hflip --tta-crop-scale 0.75 --batch-size 16 --workers 4 \
      --cache-dir "$CACHE" --frames 32 --output "$OUT/${VARIANT}_f${FOLD}.npz"
  done
done
