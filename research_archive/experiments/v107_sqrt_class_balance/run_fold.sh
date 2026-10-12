#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU index required}"
FOLD="${2:?fold required}"
if [[ ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "V107 may only use GPUs 4-7" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -B "$ROOT/scripts/train_r2plus1d.py" \
  --name "v107_sqrt_class_balance_f${FOLD}" --fold "$FOLD" \
  --mode shared_multimodal --initialization kinetics \
  --weights "$ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth" \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v18_shared5_joint_noradar_f${FOLD}/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 10 --batch-size 8 --workers 4 \
  --lr 0.00015 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 1 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
  --class-balance-power 0.5 --seed 42 --patience 10 --eval-every 1 \
  --mixup-alpha 0.3 --visual-strategy joint \
  --masked-modality-distillation-weight 0.5 --allow-method-combination \
  --output-root "$ROOT/experiments/v107_sqrt_class_balance/outputs" --swanlab-mode cloud
