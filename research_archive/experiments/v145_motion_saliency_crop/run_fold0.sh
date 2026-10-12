#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
HERE="$ROOT/experiments/v145_motion_saliency_crop"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
CHECKPOINT="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f0/best.pt"
mkdir -p "$HERE/outputs" "$HERE/logs"

CUDA_VISIBLE_DEVICES=4 "$PYTHON" "$HERE/predict.py" --checkpoint "$CHECKPOINT" \
  --fold 0 --cache-dir "$CACHE" --crop-fraction 0.50 \
  --output "$HERE/outputs/motion_0.50_f0.npz" > "$HERE/logs/motion_0.50_f0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$HERE/predict.py" --checkpoint "$CHECKPOINT" \
  --fold 0 --cache-dir "$CACHE" --crop-fraction 0.65 \
  --output "$HERE/outputs/motion_0.65_f0.npz" > "$HERE/logs/motion_0.65_f0.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
"$PYTHON" "$HERE/evaluate_fold0.py" | tee "$HERE/logs/evaluate_fold0.log"

