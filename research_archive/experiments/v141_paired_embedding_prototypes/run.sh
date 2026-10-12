#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
HERE="$ROOT/experiments/v141_paired_embedding_prototypes"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
mkdir -p "$HERE/outputs" "$HERE/logs"

CUDA_VISIBLE_DEVICES=4 "$PYTHON" "$HERE/extract.py" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f0/best.pt" \
  --fold 0 --cache-dir "$CACHE" --output "$HERE/outputs/features_f0.npz" \
  > "$HERE/logs/extract_f0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$HERE/extract.py" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f1/best.pt" \
  --fold 1 --cache-dir "$CACHE" --output "$HERE/outputs/features_f1.npz" \
  > "$HERE/logs/extract_f1.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
"$PYTHON" "$HERE/evaluate.py" | tee "$HERE/logs/evaluate.log"

