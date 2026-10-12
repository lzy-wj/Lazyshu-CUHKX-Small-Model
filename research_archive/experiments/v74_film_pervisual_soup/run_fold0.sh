#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/experiments/v74_film_pervisual_soup"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
mkdir -p "$HERE/outputs/checkpoints" "$HERE/outputs/predictions" "$HERE/outputs/logs"

for TAG in 025 050 075; do
  case "$TAG" in
    025) WEIGHT=0.25 ;;
    050) WEIGHT=0.50 ;;
    075) WEIGHT=0.75 ;;
  esac
  CHECKPOINT="$HERE/outputs/checkpoints/soup_w${TAG}_f0_int8.pt"
  "$PYTHON" -B "$HERE/build_soup.py" --fold 0 --v65-weight "$WEIGHT" --output "$CHECKPOINT"
  flock "$ROOT/outputs/logs/gpu-4.lock" env CUDA_VISIBLE_DEVICES=4 \
    "$PYTHON" -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
    --checkpoint "$CHECKPOINT" --fold 0 \
    --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" --frames 32 \
    --batch-size 16 --workers 4 \
    --output "$HERE/outputs/predictions/soup_w${TAG}_f0.npz" \
    2>&1 | tee "$HERE/outputs/logs/soup_w${TAG}_f0.log"
done
