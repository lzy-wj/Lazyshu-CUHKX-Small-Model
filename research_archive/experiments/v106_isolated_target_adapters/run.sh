#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU FOLD old|refreshed}"
FOLD="${2:?Usage: $0 GPU FOLD old|refreshed}"
VARIANT="${3:?Usage: $0 GPU FOLD old|refreshed}"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$FOLD" =~ ^[01]$ ]] || { echo "Fold must be 0 or 1" >&2; exit 2; }
[[ "$VARIANT" =~ ^(old|refreshed)$ ]] || { echo "Variant must be old or refreshed" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi

PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${FOLD}.npz"
if [[ "$VARIANT" == "refreshed" ]]; then
  PSEUDO="$ROOT/experiments/v99_refreshed_sequence_teacher/outputs/v99_v89_pseudo_f${FOLD}.npz"
fi
NAME="v106_isolated_${VARIANT}_f${FOLD}"
mkdir -p "$HERE/logs" "$HERE/outputs"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u -B "$ROOT/experiments/v98_cross_modal_target_adapter/adapt.py" \
  --name "$NAME" --fold "$FOLD" --target-domain 1 \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${FOLD}.pt" \
  --pseudo-labels "$PSEUDO" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --bottleneck 128 --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
  --retention-weight 0.25 --stream-distillation-weight 0.15 --stream-temperature 2 \
  --batch-size 16 --workers 4 --frames 32 --image-size 160 \
  --tta-hflip --tta-crop-scale 0.75 --seed 42 \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  2>&1 | tee "$HERE/logs/${NAME}.log"
