#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU old|refreshed}"
VARIANT="${2:?Usage: $0 GPU old|refreshed}"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$VARIANT" =~ ^(old|refreshed)$ ]] || { echo "Variant must be old or refreshed" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi

PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v14_test_pseudo_labels.npz"
if [[ "$VARIANT" == "refreshed" ]]; then
  PSEUDO="$HERE/outputs/v104_refreshed_test_pseudo.npz"
fi
NAME="v104_target_${VARIANT}_cross"
mkdir -p "$HERE/logs" "$HERE/outputs"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u -B "$HERE/adapt_test.py" \
  --name "$NAME" \
  --checkpoint "$ROOT/experiments/v24_full_refit/outputs/final/v24_full_warm60_soup_fp32.pt" \
  --pseudo-labels "$PSEUDO" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --bottleneck 128 --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
  --retention-weight 0.25 --stream-distillation-weight 0.15 --stream-temperature 2 \
  --batch-size 16 --workers 4 --seed 42 --output-root "$HERE/outputs" \
  --swanlab-mode cloud 2>&1 | tee "$HERE/logs/${NAME}.log"
