#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

experiment=experiments/dynamic_crop
cache="$experiment/cache/ablation_dynamic_crop_32_128"
mkdir -p "$experiment/checkpoints/ablation_dynamic_crop_soup_f0_w075" "$experiment/predictions" "$experiment/reports" "$experiment/logs"

conda run --no-capture-output -n Train python scripts/interpolate_r2plus1d_checkpoints.py \
  --left outputs/checkpoints/r2_staged_multimodal_f0/best.pt \
  --right "$experiment/checkpoints/ablation_dynamic_crop_staged_f0/best.pt" \
  --right-weight 0.75 \
  --output "$experiment/checkpoints/ablation_dynamic_crop_soup_f0_w075/best_int8.pt" \
  2>&1 | tee "$experiment/logs/interpolate_f0.log"

flock outputs/logs/gpu-7.lock bash -lc "
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=7
  conda run --no-capture-output -n Train python scripts/evaluate_r2plus1d_checkpoint.py \
    --checkpoint $experiment/checkpoints/ablation_dynamic_crop_soup_f0_w075/best_int8.pt \
    --fold 0 --cache-dir $cache --frames 32 \
    --tta-hflip --tta-crop-scale 0.75 --batch-size 8 --workers 4 \
    --output $experiment/predictions/ablation_dynamic_crop_fold0.npz \
    2>&1 | tee $experiment/logs/evaluate_f0.log
"

conda run --no-capture-output -n Train python scripts/evaluate_single_ablation.py \
  --candidate "$experiment/predictions/ablation_dynamic_crop_fold0.npz" \
  --availability "$cache/available.npy" --fold 0 \
  --output "$experiment/reports/fold0_metrics.json" \
  2>&1 | tee "$experiment/logs/score_f0.log"
