#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export TORCH_HOME="$PWD/.tools/torch"
checkpoints=(
  outputs/checkpoints/v8_x3dl_soup_f0_w075/best_int8.pt
  outputs/checkpoints/v8_x3dl_soup22_28_f1_w075/best_int8.pt
)
for fold in 0 1; do
  conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
    --checkpoint "${checkpoints[$fold]}" \
    --fold "$fold" \
    --output "outputs/study_x3dl_soup/selected_crop0875_fold${fold}.npz" \
    --batch-size 12 \
    --workers 6 \
    --tta-crop-scale 0.875
done
