#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export TORCH_HOME="$PWD/.tools/torch"
conda run --no-capture-output -n Train python scripts/train.py \
  --name v8_x3dm_yolo_f0_r1 \
  --modalities yolo_rgb,yolo_ir,skeleton,imu_world \
  --fold 0 \
  --seed 42 \
  --epochs 60 \
  --batch-size 16 \
  --lr 0.0005 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 2 \
  --patience 20 \
  --swanlab-mode online \
  --strong-augment \
  --canonical-pose \
  --image-size 128 \
  --backbone x3d_s \
  --kinetics \
  --kinetics-source m \
  --backbone-lr-factor 0.15 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.15
