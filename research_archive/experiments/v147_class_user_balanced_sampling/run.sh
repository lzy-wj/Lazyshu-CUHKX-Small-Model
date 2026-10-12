#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
HERE="$ROOT/experiments/v147_class_user_balanced_sampling"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
mkdir -p "$HERE/outputs" "$HERE/logs"

for fold in 0 1; do
  gpu=$((6 + fold))
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/scripts/train_r2plus1d.py" \
    --name "v147_class_user_f${fold}" --fold "$fold" --mode shared_multimodal \
    --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f${fold}/best.pt" \
    --cache-dir "$CACHE" --frames 32 --image-size 128 --epochs 5 \
    --batch-size 8 --workers 4 --lr 0.00010 --backbone-lr-factor 0.05 \
    --warmup-epochs 1 --patience 5 --ema-decay 0.99 --dropout 0.3 \
    --auxiliary-weight 0.15 --label-smoothing 0.03 --mixup-alpha 0.3 \
    --masked-modality-distillation-weight 0.5 --class-user-balanced-sampling \
    --visual-strategy joint --allow-method-combination \
    --output-root "$HERE/outputs" --swanlab-mode cloud \
    > "$HERE/logs/v147_class_user_f${fold}.log" 2>&1 &
done
wait

