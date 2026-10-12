#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v171_full_modality_highres"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
FOLD="${1:?fold required}"
GPU="${2:?GPU required}"

if [[ ! "$FOLD" =~ ^[01]$ || ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "Expected fold 0/1 and GPU 4-7" >&2
  exit 2
fi
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi

mkdir -p "$HERE/logs" "$HERE/outputs"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" "$ROOT/scripts/train_r2plus1d.py" \
  --name "v171_full6_160_f${FOLD}" --fold "$FOLD" --mode shared_multimodal \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f${FOLD}/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --frames 32 --image-size 160 --epochs 7 --batch-size 8 --workers 4 \
  --lr 0.00010 --backbone-lr-factor 0.05 --warmup-epochs 1 --patience 7 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
  --label-smoothing 0.03 --mixup-alpha 0.3 \
  --training-minimum-modalities 6 --selection-minimum-modalities 6 \
  --masked-modality-distillation-weight 0.5 \
  --visual-strategy joint --allow-method-combination \
  --output-root "$HERE/outputs" --swanlab-mode cloud
