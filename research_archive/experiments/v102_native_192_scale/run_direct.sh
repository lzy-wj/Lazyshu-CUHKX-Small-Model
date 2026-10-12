#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU FOLD}"
FOLD="${2:?Usage: $0 GPU FOLD}"
NAME="v102_v23_native192_f${FOLD}"
[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$FOLD" =~ ^[01]$ ]] || { echo "Fold must be 0 or 1" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi
mkdir -p "$HERE/outputs" "$HERE/logs"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u -B scripts/evaluate_r2plus1d_checkpoint.py \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f${FOLD}.pt" \
  --fold "$FOLD" --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_192" \
  --frames 32 --image-size 192 --batch-size 16 --workers 4 \
  --output "$HERE/outputs/${NAME}.npz" \
  2>&1 | tee "$HERE/logs/${NAME}.log"
