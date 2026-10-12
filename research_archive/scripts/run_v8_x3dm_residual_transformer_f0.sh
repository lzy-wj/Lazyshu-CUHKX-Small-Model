#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export TORCH_HOME="$PWD/.tools/torch"
conda run --no-capture-output -n Train python scripts/train.py \
  --name v8_x3dm_residual_transformer_f0 \
  --modalities person_ir,person_thermal,skeleton,imu_world \
  --fold 0 \
  --seed 42 \
  --epochs 30 \
  --batch-size 16 \
  --lr 0.0005 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 1 \
  --patience 10 \
  --swanlab-mode online \
  --init outputs/checkpoints/v7_x3dm_f0/best.pt \
  --freeze-inherited \
  --strong-augment \
  --canonical-pose \
  --image-size 224 \
  --backbone x3d_s \
  --kinetics \
  --kinetics-source m \
  --backbone-lr-factor 0.05 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.15 \
  --fusion residual_transformer
