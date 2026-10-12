#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export TORCH_HOME="$PWD/.tools/torch"
conda run --no-capture-output -n Train python scripts/adapt_x3d_simclr.py \
  --checkpoint outputs/checkpoints/v7_x3dm_f0/best.pt \
  --name v8_x3dm_simclr_f0 \
  --fold 0 \
  --epochs 8 \
  --batch-size 16 \
  --workers 6 \
  --lr 0.00005 \
  --backbone-lr-factor 0.1 \
  --contrastive-weight 0.2 \
  --temperature 0.5 \
  --steps-per-epoch 60 \
  --swanlab-mode online
