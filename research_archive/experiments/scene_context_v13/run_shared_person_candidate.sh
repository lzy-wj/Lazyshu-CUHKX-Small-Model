#!/usr/bin/env bash
set -euo pipefail

fold="${1:?fold is required}"
name="v13_shared_person_f${fold}"

python -B scripts/train.py \
  --name "$name" \
  --modalities person_ir,person_thermal,person_depth,skeleton,imu_world \
  --fold "$fold" \
  --seed 42 \
  --epochs 40 \
  --schedule-epochs 40 \
  --warmup-epochs 2 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --lr 0.00025 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 2 \
  --patience 14 \
  --swanlab-mode online \
  --strong-augment \
  --synchronized-temporal-augment \
  --canonical-pose \
  --image-size 224 \
  --backbone x3d_shared_tokens \
  --fusion transformer \
  --kinetics \
  --kinetics-source l \
  --backbone-lr-factor 0.08 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.25 \
  --init "outputs/checkpoints/v13_shared_tokens_f${fold}/best.pt"
