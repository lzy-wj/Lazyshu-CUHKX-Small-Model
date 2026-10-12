#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: run_shared6_quantization_sweep.sh GPU}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="$ROOT/experiments/scene_context_v13/outputs"
CHECKPOINT_ROOT="$OUTPUT_ROOT/checkpoints"
PREDICTION_ROOT="$OUTPUT_ROOT/predictions/shared6_quantization_sweep"

export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="$HOME/miniconda3/envs/Train/bin:$PATH"
mkdir -p "$PREDICTION_ROOT"

for variant in \
  hybrid_l3int6 \
  hybrid_l3int8 \
  hybrid_l3int6_l4b0int6 \
  hybrid_l3int6_l4b1int6 \
  hybrid_l3int6_l4b2int6 \
  hybrid_nf4 \
  hybrid_nf4_l3int6 \
  hybrid_nf4_l3int6_l4b0int6; do
  python -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
    --checkpoint "$CHECKPOINT_ROOT/v12_shared6_f0_${variant}_compact.pt" \
    --fold 0 \
    --tta-hflip \
    --tta-crop-scale 0.75 \
    --batch-size 16 \
    --workers 4 \
    --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
    --frames 32 \
    --output "$PREDICTION_ROOT/${variant}_f0.npz"
done

for adaptation in global user; do
  for fold in 0 1; do
    python -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
      --checkpoint "$ROOT/experiments/blend_aware_v12/outputs/checkpoints/v12_shared6_f${fold}/best_int8.pt" \
      --fold "$fold" \
      --tta-hflip \
      --tta-crop-scale 0.75 \
      --adabn "$adaptation" \
      --batch-size 16 \
      --workers 4 \
      --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
      --frames 32 \
      --output "$PREDICTION_ROOT/adabn_${adaptation}_f${fold}.npz"
  done
done
