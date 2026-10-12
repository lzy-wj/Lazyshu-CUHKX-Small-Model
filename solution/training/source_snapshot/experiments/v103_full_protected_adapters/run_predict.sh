#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU LOCATION}"
LOCATION="${2:?Usage: $0 GPU LOCATION}"
[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$LOCATION" =~ ^(intermediate|dual)$ ]] || { echo "Location must be intermediate or dual" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi
mkdir -p "$HERE/predictions" "$HERE/logs"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u -B "$HERE/predict_test.py" \
  --source "$ROOT/experiments/v24_full_refit/outputs/final/v24_full_warm60_soup_int8.pt" \
  --adapter "$HERE/outputs/v103_full_protected_${LOCATION}/adapter.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --output "$HERE/predictions/v103_full_protected_${LOCATION}.npz" \
  --batch-size 16 --workers 4 \
  2>&1 | tee "$HERE/logs/predict_${LOCATION}.log"
