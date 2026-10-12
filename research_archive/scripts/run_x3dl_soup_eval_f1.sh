#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export TORCH_HOME="$PWD/.tools/torch"
for left in 20 22; do
  for weight in 025 050 075; do
    conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
      --checkpoint "outputs/checkpoints/v8_x3dl_soup${left}_28_f1_w${weight}/best_int8.pt" \
      --fold 1 \
      --output "outputs/study_x3dl_soup/${left}_28_w${weight}_fold1.npz" \
      --batch-size 12 \
      --workers 6
  done
done
conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
  --checkpoint outputs/checkpoints/v8_x3dl_f1/best_int8.pt \
  --fold 1 \
  --output outputs/study_x3dl_soup/best28_crop0875_fold1.npz \
  --batch-size 12 \
  --workers 6 \
  --tta-crop-scale 0.875
