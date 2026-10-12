#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
HERE="$ROOT/experiments/v149_multimodal_person_windows"

for threshold in 0.1 0.2; do
  suffix=${threshold#0.}
  windows="$HERE/outputs/fallback_q${suffix}_windows.json"
  cache="$ROOT/data/processed/r2plus1d_multimodal_fallback_q${suffix}_32_128"
  "$PYTHON" "$HERE/merge_fallback.py" --ir-threshold "$threshold" --output "$windows"
  "$PYTHON" "$ROOT/scripts/prepare_r2plus1d_cache.py" \
    --output "$cache" --windows-from "$windows" --frames 32 --image-size 128 \
    --rgb-modalities Depth_Color,Thermal --workers 20 \
    > "$HERE/logs/cache_fallback_q${suffix}.log" 2>&1
done

for threshold in 0.1 0.2; do
  suffix=${threshold#0.}
  CUDA_VISIBLE_DEVICES=$((3 + suffix)) "$PYTHON" "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
    --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f0/best.pt" \
    --fold 0 --cache-dir "$ROOT/data/processed/r2plus1d_multimodal_fallback_q${suffix}_32_128" \
    --frames 32 --image-size 128 --batch-size 16 --workers 4 \
    --output "$HERE/outputs/fallback_q${suffix}_f0.npz" \
    > "$HERE/logs/predict_fallback_q${suffix}_f0.log" 2>&1 &
done
wait
"$PYTHON" "$HERE/evaluate_fallback_fold0.py" | tee "$HERE/logs/evaluate_fallback_fold0.log"
