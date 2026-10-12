#!/usr/bin/env bash
set -euo pipefail

fold="${1:?fold is required}"
name="v13_swin3d_ir_f${fold}"

export TORCH_HOME="${LAZYSHU_PROJECT_ROOT}/.tools/torch"
python -B scripts/train.py \
  --name "$name" \
  --modalities person_ir,skeleton,imu_world \
  --fold "$fold" \
  --seed 42 \
  --epochs 40 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --lr 0.0002 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 2 \
  --patience 14 \
  --swanlab-mode online \
  --strong-augment \
  --synchronized-temporal-augment \
  --canonical-pose \
  --image-size 160 \
  --backbone swin3d_t \
  --fusion concat \
  --kinetics \
  --backbone-lr-factor 0.08 \
  --mixup 0.1 \
  --ema-decay 0.98 \
  --auxiliary-weight 0.15
