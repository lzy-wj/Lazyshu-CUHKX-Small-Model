#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU FOLD SPLIT_STAGE}"
FOLD="${2:?Usage: $0 GPU FOLD SPLIT_STAGE}"
SPLIT_STAGE="${3:?Usage: $0 GPU FOLD SPLIT_STAGE}"
NAME="v96_modality_specialized_s${SPLIT_STAGE}_f${FOLD}"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$FOLD" =~ ^[01]$ ]] || { echo "Fold must be 0 or 1" >&2; exit 2; }
[[ "$SPLIT_STAGE" =~ ^[34]$ ]] || { echo "Split stage must be 3 or 4" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi
mkdir -p "$HERE/outputs" "$HERE/logs"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u -B "$HERE/train.py" \
  --name "$NAME" --fold "$FOLD" --split-stage "$SPLIT_STAGE" \
  --init-checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f${FOLD}.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --image-size 160 --frames 32 --epochs 6 --batch-size 8 --workers 4 \
  --lr 0.00003 --weight-decay 0.0001 --warmup-epochs 1 --ema-decay 0.99 \
  --label-smoothing 0.03 --mixup-alpha 0.3 --auxiliary-weight 0.15 \
  --retention-weight 0.5 --retention-temperature 2 --seed 42 \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  2>&1 | tee "$HERE/logs/${NAME}.log"
