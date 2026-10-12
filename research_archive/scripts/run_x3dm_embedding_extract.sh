#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export TORCH_HOME="$PWD/.tools/torch"
for fold in 0 1; do
  conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
    --checkpoint "outputs/checkpoints/v7_x3dm_f${fold}/best.pt" \
    --fold "$fold" \
    --output "outputs/study_x3dm_embeddings/train_fold${fold}.npz" \
    --batch-size 16 \
    --workers 6 \
    --indices-scope train \
    --save-embeddings
done
