#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export TORCH_HOME="$PWD/.tools/torch"
conda run --no-capture-output -n Train python scripts/adapt_x3d_simclr.py \
  --checkpoint outputs/checkpoints/v7_x3dm_f1/best.pt \
  --name v8_x3dm_simclr_f1 \
  --fold 1 \
  --epochs 1 \
  --batch-size 16 \
  --workers 6 \
  --lr 0.00005 \
  --backbone-lr-factor 0.1 \
  --contrastive-weight 0.2 \
  --temperature 0.5 \
  --steps-per-epoch 60 \
  --swanlab-mode online

for fold in 0 1; do
  conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
    --checkpoint "outputs/checkpoints/v7_x3dm_f${fold}/best.pt" \
    --fold "$fold" \
    --output "outputs/study_x3dm_auxiliary/fold${fold}.npz" \
    --batch-size 16 \
    --workers 6 \
    --save-auxiliary \
    --save-embeddings
  for crop in 0.75 0.875; do
    conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
      --checkpoint "outputs/checkpoints/v7_x3dm_f${fold}/best.pt" \
      --fold "$fold" \
      --output "outputs/study_x3dm_temporal/crop${crop}_fold${fold}.npz" \
      --batch-size 16 \
      --workers 6 \
      --temporal-crop "$crop"
  done
done
