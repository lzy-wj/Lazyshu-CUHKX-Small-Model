#!/usr/bin/env bash
set -euo pipefail

NAME="${1:?usage: queue_shared6_validation.sh NAME GPU FOLD}"
GPU="${2:?usage: queue_shared6_validation.sh NAME GPU FOLD}"
FOLD="${3:?usage: queue_shared6_validation.sh NAME GPU FOLD}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="$ROOT/experiments/scene_context_v13/outputs"
CHECKPOINT="$OUTPUT_ROOT/checkpoints/$NAME/best_int8.pt"
PREDICTION="$OUTPUT_ROOT/predictions/${NAME}_tta.npz"

export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="$HOME/miniconda3/envs/Train/bin:$PATH"
mkdir -p "$OUTPUT_ROOT/predictions" "$OUTPUT_ROOT/logs"

flock "$ROOT/outputs/logs/gpu-${GPU}.lock" \
  python -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
    --checkpoint "$CHECKPOINT" \
    --fold "$FOLD" \
    --tta-hflip \
    --tta-crop-scale 0.75 \
    --batch-size 16 \
    --workers 4 \
    --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
    --frames 32 \
    --output "$PREDICTION"
