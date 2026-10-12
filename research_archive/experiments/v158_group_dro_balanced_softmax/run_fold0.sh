#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v158_group_dro_balanced_softmax"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
mkdir -p "$HERE/logs" "$HERE/outputs"

CUDA_VISIBLE_DEVICES=4 "$PYTHON" "$ROOT/scripts/train_r2plus1d.py" \
  --name v158_group_dro_balanced_f0 --fold 0 --mode shared_multimodal \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f0/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 7 --batch-size 12 --workers 4 \
  --lr 0.00010 --backbone-lr-factor 0.05 --warmup-epochs 1 --patience 7 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
  --label-smoothing 0.03 --mixup-alpha 0.3 \
  --masked-modality-distillation-weight 0.5 \
  --group-dro-eta 0.01 --balanced-softmax-temperature 1 \
  --visual-strategy joint --allow-method-combination \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  > "$HERE/logs/v158_group_dro_balanced_f0.log" 2>&1
