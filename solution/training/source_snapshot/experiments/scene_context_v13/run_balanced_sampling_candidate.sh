#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU index required}"
FOLD="${2:?fold required}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="$HOME/miniconda3/envs/Train/bin:$PATH"

python -B "$ROOT/scripts/train_r2plus1d.py" \
  --name "v18_shared6_joint_balanced_f${FOLD}" --fold "$FOLD" \
  --mode shared_multimodal --initialization kinetics \
  --weights "$ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 20 --batch-size 8 --workers 4 \
  --lr 0.0003 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 2 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
  --class-balance-power 0 --class-balanced-sampling --seed 42 --patience 20 \
  --eval-every 1 --use-radar --mixup-alpha 0.3 --visual-strategy joint \
  --allow-method-combination \
  --output-root "$ROOT/experiments/scene_context_v13/outputs" --swanlab-mode cloud
