#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export TORCH_HOME="$PWD/.tools/torch"
for crop in 0.75 0.875; do
  conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
    --checkpoint outputs/checkpoints/v8_x3dl_f0/snapshot_for_tta.pt \
    --fold 0 \
    --output "outputs/study_x3dl_tta/crop${crop}_fold0.npz" \
    --batch-size 12 \
    --workers 6 \
    --tta-crop-scale "$crop"
done
conda run --no-capture-output -n Train python scripts/evaluate_legacy_checkpoint.py \
  --checkpoint outputs/checkpoints/v8_x3dl_f0/snapshot_for_tta.pt \
  --fold 0 \
  --output outputs/study_x3dl_tta/hflip_fold0.npz \
  --batch-size 12 \
  --workers 6 \
  --tta-hflip
