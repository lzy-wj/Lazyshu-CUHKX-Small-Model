#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
HERE="$ROOT/experiments/v149_multimodal_person_windows"
CACHE="$ROOT/data/processed/r2plus1d_multimodal_windows_32_128"

for modality in Depth_Color IR Thermal; do
  while [ ! -s "$HERE/outputs/${modality}_windows.json" ]; do sleep 10; done
done
"$PYTHON" "$HERE/merge_windows.py" | tee "$HERE/logs/merge.log"
"$PYTHON" "$ROOT/scripts/prepare_r2plus1d_cache.py" \
  --output "$CACHE" --windows-from "$HERE/outputs/robust_windows.json" \
  --frames 32 --image-size 128 --rgb-modalities Depth_Color,Thermal \
  --workers 20 > "$HERE/logs/cache.log" 2>&1

for fold in 0 1; do
  CUDA_VISIBLE_DEVICES=$((4 + fold)) "$PYTHON" "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
    --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f${fold}/best.pt" \
    --fold "$fold" --cache-dir "$CACHE" --frames 32 --image-size 128 \
    --batch-size 16 --workers 4 --output "$HERE/outputs/robust_v23_f${fold}.npz" \
    > "$HERE/logs/predict_f${fold}.log" 2>&1 &
done
wait
"$PYTHON" "$HERE/evaluate.py" | tee "$HERE/logs/evaluate.log"

