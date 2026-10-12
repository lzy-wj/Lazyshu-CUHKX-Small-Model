#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
EXTRACT="$ROOT/experiments/v141_paired_embedding_prototypes/extract.py"
HERE="$ROOT/experiments/v142_cross_environment_metric"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
mkdir -p "$HERE/outputs" "$HERE/logs"

CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$EXTRACT" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f0/best.pt" \
  --fold 0 --scope train --cache-dir "$CACHE" --output "$HERE/outputs/features_f0.npz" \
  > "$HERE/logs/extract_f0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=7 "$PYTHON" "$EXTRACT" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f1/best.pt" \
  --fold 1 --scope train --cache-dir "$CACHE" --output "$HERE/outputs/features_f1.npz" \
  > "$HERE/logs/extract_f1.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

for fold in 0 1; do
  for representation in fused visual all; do
    CUDA_VISIBLE_DEVICES=$((6 + fold)) "$PYTHON" "$HERE/train_metric.py" \
      --features "$HERE/outputs/features_f${fold}.npz" --fold "$fold" \
      --representation "$representation" \
      --output "$HERE/outputs/metric_${representation}_f${fold}.pt" \
      > "$HERE/logs/metric_${representation}_f${fold}.log" 2>&1
  done &
done
wait

"$PYTHON" "$HERE/evaluate.py" | tee "$HERE/logs/evaluate.log"
