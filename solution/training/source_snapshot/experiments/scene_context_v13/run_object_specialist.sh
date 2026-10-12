#!/usr/bin/env bash
set -euo pipefail

fold="${1:?fold is required}"
view="${2:?view is required}"
labels="6,7,8,9,10,11,14,17,18,19,20,21,22,23,24,25,26,27,37,38,39"

if [[ "$view" == "person" ]]; then
  name="v13_object_person_f${fold}"
  modalities="person_ir,person_thermal,skeleton,imu_world"
  image_size=224
  init="outputs/checkpoints/v8_x3dl_f${fold}/best.pt"
elif [[ "$view" == "detail" ]]; then
  name="v13_object_detail_f${fold}"
  modalities="person_ir_detail,person_thermal_detail,skeleton,imu_world"
  image_size=160
  init="outputs/checkpoints/v13_x3dl_detail_f${fold}/best.pt"
else
  echo "unknown view: $view" >&2
  exit 2
fi

python -B scripts/train.py \
  --name "$name" \
  --modalities "$modalities" \
  --fold "$fold" \
  --seed 42 \
  --epochs 24 \
  --schedule-epochs 24 \
  --warmup-epochs 1 \
  --batch-size 12 \
  --lr 0.00012 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 1 \
  --patience 8 \
  --swanlab-mode online \
  --strong-augment \
  --synchronized-temporal-augment \
  --canonical-pose \
  --image-size "$image_size" \
  --backbone x3d_s \
  --fusion concat \
  --kinetics \
  --kinetics-source l \
  --backbone-lr-factor 0.08 \
  --mixup 0.05 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.15 \
  --training-labels "$labels" \
  --selection-labels "$labels" \
  --init "$init"
