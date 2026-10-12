#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/experiments/v74_film_pervisual_soup"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
CHECKPOINT="$HERE/outputs/checkpoints/soup_w050_f1_int8.pt"
mkdir -p "$HERE/outputs/checkpoints" "$HERE/outputs/predictions" "$HERE/outputs/logs"

"$PYTHON" -B "$HERE/build_soup.py" --fold 1 --v65-weight 0.5 --output "$CHECKPOINT"
flock "$ROOT/outputs/logs/gpu-4.lock" env CUDA_VISIBLE_DEVICES=4 \
  "$PYTHON" -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
  --checkpoint "$CHECKPOINT" --fold 1 \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" --frames 32 \
  --batch-size 16 --workers 4 \
  --output "$HERE/outputs/predictions/soup_w050_f1.npz" \
  2>&1 | tee "$HERE/outputs/logs/soup_w050_f1.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$HERE/evaluate_fold1.py" \
  2>&1 | tee "$HERE/outputs/logs/fold1_evaluate.log"
