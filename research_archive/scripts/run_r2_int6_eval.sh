#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
for fold in 0 1; do
  conda run --no-capture-output -n Train python scripts/evaluate_r2plus1d_checkpoint.py \
    --checkpoint "outputs/checkpoints/r2_staged_soup_f${fold}_w075/best_int6.pt" \
    --fold "$fold" \
    --cache-dir data/processed/r2plus1d_yolo_32_128 \
    --frames 32 \
    --batch-size 8 \
    --workers 4 \
    --tta-hflip \
    --tta-crop-scale 0.75 \
    --output "outputs/study_r2plus1d/r2_soup_int6/fold${fold}.npz"
done

export TORCH_HOME="$PWD/.tools/torch"
for fold in 0 1; do
  for crop in 0.75 0.875; do
    conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
      --checkpoint "outputs/checkpoints/v7_x3dm_f${fold}/best.pt" \
      --fold "$fold" \
      --output "outputs/study_x3dm_spatial/crop${crop}_fold${fold}.npz" \
      --batch-size 16 \
      --workers 6 \
      --tta-crop-scale "$crop"
  done
done
