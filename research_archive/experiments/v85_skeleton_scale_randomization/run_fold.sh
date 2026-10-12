#!/usr/bin/env bash
set -euo pipefail

FOLD="${1:?fold required}"
GPU="${2:?gpu required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
NAME="v85_scale_randomization_isolated_f${FOLD}"

[[ "$FOLD" =~ ^[01]$ ]] || { echo "fold must be 0 or 1" >&2; exit 2; }
[[ "$GPU" =~ ^[4-7]$ ]] || { echo "use only physical GPUs 4-7" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u scripts/train_r2plus1d.py \
  --name "$NAME" --fold "$FOLD" --mode shared_multimodal --visual-strategy joint \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v18_shared5_joint_noradar_f${FOLD}/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 10 --batch-size 8 --workers 4 \
  --lr 0.00015 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 1 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
  --mixup-alpha 0.3 --masked-modality-distillation-weight 0.5 \
  --masked-modality-distillation-temperature 2.0 --allow-method-combination \
  --skeleton-scale-randomization \
  --output-root "$HERE/outputs" --swanlab-mode cloud
