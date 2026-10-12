#!/usr/bin/env bash
set -euo pipefail

fold="${1:?fold is required}"
name="v13_x3dl_rgb_f${fold}"

python -B scripts/train.py \
  --name "$name" \
  --modalities person_ir,person_thermal,person_depth,skeleton,imu_world \
  --fold "$fold" \
  --seed 42 \
  --epochs 60 \
  --batch-size 8 \
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
  --image-size 224 \
  --backbone x3d_s \
  --fusion concat \
  --kinetics \
  --kinetics-source l \
  --backbone-lr-factor 0.12 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.15
