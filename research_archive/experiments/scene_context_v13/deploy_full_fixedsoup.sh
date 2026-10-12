#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${HOME}/miniconda3/envs/Train/bin/python"
RUN="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_full18"
SOUP="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_full_fixedsoup_fp32.pt"
COMPACT="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_full_fixedsoup_uniform_l3int6_l4b0.pt"

"$PYTHON" -B "$ROOT/scripts/soup_r2plus1d_checkpoints.py" \
  --checkpoints "$RUN/epochs/epoch_10.pt" "$RUN/epochs/epoch_14.pt" "$RUN/epochs/epoch_18.pt" \
  --weights 0.2 0.3 0.5 --storage fp32 --output "$SOUP"
"$PYTHON" -B "$ROOT/scripts/quantize_r2plus1d_hybrid.py" \
  --input "$SOUP" --output "$COMPACT" --deep-scheme uniform4 --layer3-bits 6 \
  --layer4-int6-blocks 0
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/scripts/predict_r2plus1d.py" \
  --checkpoints "$COMPACT" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" --frames 32 \
  --tta-hflip --tta-crop-scale 0.75 --batch-size 16 --workers 4 \
  --name v14_shared6_mixup_full_fixedsoup_uniform_tta
