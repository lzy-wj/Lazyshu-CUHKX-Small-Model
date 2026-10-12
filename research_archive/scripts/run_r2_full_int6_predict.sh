#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
conda run --no-capture-output -n Train python scripts/predict_r2plus1d.py \
  --checkpoints outputs/checkpoints/r2_staged_soup_full_w075/best_int6.pt \
  --cache-dir data/processed/r2plus1d_yolo_32_128 \
  --yolo .tools/checkpoints/yolo11n.pt \
  --frames 32 \
  --batch-size 8 \
  --workers 4 \
  --tta-hflip \
  --tta-crop-scale 0.75 \
  --name r2_staged_soup_full_int6_crop075
