#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v156_cross_environment_contrastive"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
mkdir -p "$HERE/logs" "$HERE/outputs"

CUDA_VISIBLE_DEVICES=7 "$PYTHON" "$ROOT/scripts/train_r2plus1d.py" \
  --name v156_cross_environment_f0 --fold 0 --mode shared_multimodal \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f0/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 7 --batch-size 12 --workers 4 \
  --lr 0.00010 --backbone-lr-factor 0.05 --warmup-epochs 1 --patience 7 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
  --label-smoothing 0.03 --mixup-alpha 0.3 \
  --masked-modality-distillation-weight 0.5 \
  --cross-environment-contrastive-weight 0.01 \
  --contrastive-temperature 0.1 --contrastive-queue-size 2048 \
  --contrastive-class-cap 64 --contrastive-warmup-items 1024 \
  --visual-strategy joint --allow-method-combination \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  > "$HERE/logs/v156_cross_environment_f0.log" 2>&1
