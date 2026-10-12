#!/usr/bin/env bash
set -euo pipefail

fold="${1:?fold is required}"
name="v13_x3dl_detail224_f${fold}"

python -B scripts/train.py \
  --name "$name" \
  --modalities person_ir_detail,person_thermal_detail,skeleton,imu_world \
  --fold "$fold" \
  --seed 42 \
  --epochs 32 \
  --schedule-epochs 32 \
  --warmup-epochs 1 \
  --batch-size 12 \
  --lr 0.00015 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 2 \
  --patience 12 \
  --swanlab-mode online \
  --strong-augment \
  --synchronized-temporal-augment \
  --canonical-pose \
  --image-size 224 \
  --backbone x3d_s \
  --fusion concat \
  --kinetics \
  --kinetics-source l \
  --backbone-lr-factor 0.08 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.15 \
  --init "outputs/checkpoints/v13_x3dl_detail_f${fold}/best.pt"
