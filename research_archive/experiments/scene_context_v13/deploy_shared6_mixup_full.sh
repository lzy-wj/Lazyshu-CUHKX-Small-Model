#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${HOME}/miniconda3/envs/Train/bin/python"
SOURCE="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_full/best.pt"
COMPACT="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_full_hybrid_nf4_l3int6_compact.pt"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"

"$PYTHON" -B "$ROOT/scripts/quantize_r2plus1d_hybrid.py" \
  --input "$SOURCE" --output "$COMPACT" --deep-scheme nf4 --layer3-bits 6

for ADABN in none global; do
  NAME="v14_shared6_mixup_full_nf4_${ADABN}_tta"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/scripts/predict_r2plus1d.py" \
    --checkpoints "$COMPACT" --cache-dir "$CACHE" --frames 32 \
    --tta-hflip --tta-crop-scale 0.75 --adabn "$ADABN" --batch-size 16 --workers 4 \
    --name "$NAME"
done
