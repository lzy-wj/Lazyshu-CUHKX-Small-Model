#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export TORCH_HOME="$PWD/.tools/torch"
for weight in 025 050 075; do
  conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
    --checkpoint "outputs/checkpoints/v8_x3dl_soup_f0_w${weight}/best_int8.pt" \
    --fold 0 \
    --output "outputs/study_x3dl_soup/w${weight}_fold0.npz" \
    --batch-size 12 \
    --workers 6
done
conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
  --checkpoint outputs/checkpoints/v8_x3dl_f0/best_int8.pt \
  --fold 0 \
  --output outputs/study_x3dl_soup/best30_crop0875_fold0.npz \
  --batch-size 12 \
  --workers 6 \
  --tta-crop-scale 0.875
