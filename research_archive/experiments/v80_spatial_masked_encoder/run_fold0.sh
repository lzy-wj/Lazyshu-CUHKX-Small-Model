#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-5}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
NAME="v80_spatial_masked_encoder_f0"
LOG="$HERE/outputs/logs/${NAME}.log"

[[ "$GPU" =~ ^[5-7]$ ]] || { echo "V80 may use only currently unreserved rear GPUs 5-7" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi
mkdir -p "$HERE/outputs/logs"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u "$HERE/train.py" \
  --name "$NAME" --fold 0 \
  --init-checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f0.pt" \
  --dimension 512 --depth 4 --heads 8 --feedforward-multiplier 3 \
  --residual-scale 0.25 --reconstruction-mask-probability 0.35 \
  --reconstruction-weight 0.2 --retention-weight 0.5 --retention-temperature 2.0 \
  --epochs 12 --batch-size 8 --workers 4 --lr 0.00015 --weight-decay 0.0005 \
  --warmup-epochs 1 --ema-decay 0.99 --label-smoothing 0.03 --mixup-alpha 0.3 \
  --seed 42 --output-root "$HERE/outputs" --swanlab-mode cloud 2>&1 | tee "$LOG"

