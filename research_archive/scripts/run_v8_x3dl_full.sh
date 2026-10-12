#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export TORCH_HOME="$PWD/.tools/torch"
conda run --no-capture-output -n Train python scripts/train.py \
  --name v8_x3dl_full \
  --modalities person_ir,person_thermal,skeleton,imu_world \
  --fold -1 \
  --seed 42 \
  --epochs 30 \
  --batch-size 12 \
  --lr 0.0004 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 2 \
  --patience 20 \
  --swanlab-mode online \
  --strong-augment \
  --canonical-pose \
  --image-size 224 \
  --backbone x3d_s \
  --kinetics \
  --kinetics-source l \
  --backbone-lr-factor 0.12 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.15
