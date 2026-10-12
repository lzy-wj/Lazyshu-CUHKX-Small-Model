#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-6}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
NAME="v86_highres160_spatial_masked_final_f0"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "use only physical GPUs 4-7" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u experiments/v80_spatial_masked_encoder/train.py \
  --name "$NAME" --fold 0 --select-final \
  --init-checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f0.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" --image-size 160 \
  --dimension 512 --depth 4 --heads 8 --feedforward-multiplier 3 \
  --residual-scale 0.25 --reconstruction-mask-probability 0.35 \
  --reconstruction-weight 0.2 --retention-weight 0.5 --retention-temperature 2.0 \
  --epochs 6 --batch-size 8 --workers 4 --lr 0.00015 --weight-decay 0.0005 \
  --warmup-epochs 1 --ema-decay 0.99 --label-smoothing 0.03 --mixup-alpha 0.3 \
  --seed 42 --output-root "$HERE/outputs" --swanlab-mode cloud
