#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
HERE="$ROOT/experiments/v148_environment_specific_teachers"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
mkdir -p "$HERE/outputs" "$HERE/logs"

for fold in 0 1; do
  for environment in 0 1; do
    gpu=$((4 + 2 * fold + environment))
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/scripts/train_r2plus1d.py" \
      --name "v148_env${environment}_f${fold}" --fold "$fold" \
      --mode shared_multimodal \
      --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f${fold}/best.pt" \
      --cache-dir "$CACHE" --frames 32 --image-size 128 --epochs 7 \
      --batch-size 8 --workers 4 --lr 0.00010 --backbone-lr-factor 0.05 \
      --warmup-epochs 1 --patience 7 --ema-decay 0.99 --dropout 0.3 \
      --auxiliary-weight 0.15 --label-smoothing 0.03 --mixup-alpha 0.3 \
      --masked-modality-distillation-weight 0.5 \
      --training-environment "$environment" --selection-environment "$environment" \
      --visual-strategy joint --allow-method-combination \
      --output-root "$HERE/outputs" --swanlab-mode cloud \
      > "$HERE/logs/v148_env${environment}_f${fold}.log" 2>&1 &
  done
done
wait

