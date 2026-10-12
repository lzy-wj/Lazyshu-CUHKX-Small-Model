#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
for fold in 0 1; do
  conda run --no-capture-output -n Train python scripts/evaluate_r2plus1d_checkpoint.py \
    --checkpoint "outputs/checkpoints/r2_staged_soup_f${fold}_w075/best_int8.pt" \
    --fold "$fold" \
    --cache-dir data/processed/r2plus1d_yolo_32_128 \
    --frames 32 \
    --batch-size 8 \
    --workers 4 \
    --tta-hflip \
    --tta-crop-scale 0.75 \
    --save-auxiliary \
    --output "outputs/study_r2plus1d/r2_soup_auxiliary/fold${fold}.npz"
done
