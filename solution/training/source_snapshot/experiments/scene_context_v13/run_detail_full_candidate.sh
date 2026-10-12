#!/usr/bin/env bash
set -euo pipefail

name="v14_x3dl_detail160_full"

python -B scripts/train.py \
  --name "$name" \
  --modalities person_ir_detail,person_thermal_detail,skeleton,imu_world \
  --fold -1 \
  --seed 42 \
  --epochs 16 \
  --schedule-epochs 32 \
  --warmup-epochs 2 \
  --batch-size 12 \
  --lr 0.0004 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 2 \
  --patience 20 \
  --swanlab-mode online \
  --strong-augment \
  --synchronized-temporal-augment \
  --canonical-pose \
  --image-size 160 \
  --backbone x3d_s \
  --fusion concat \
  --kinetics \
  --kinetics-source l \
  --backbone-lr-factor 0.12 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.15
