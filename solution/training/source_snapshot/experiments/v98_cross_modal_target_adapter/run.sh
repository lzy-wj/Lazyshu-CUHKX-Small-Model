#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU FOLD VARIANT}"
FOLD="${2:?Usage: $0 GPU FOLD VARIANT}"
VARIANT="${3:?Usage: $0 GPU FOLD VARIANT}"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$FOLD" =~ ^[01]$ ]] || { echo "Fold must be 0 or 1" >&2; exit 2; }
[[ "$VARIANT" =~ ^(main|cross)$ ]] || { echo "Variant must be main or cross" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi
STREAM_WEIGHT=0
if [[ "$VARIANT" == "cross" ]]; then STREAM_WEIGHT=0.15; fi
NAME="v98_target_adapter_${VARIANT}_f${FOLD}"
mkdir -p "$HERE/outputs" "$HERE/logs"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u -B "$HERE/adapt.py" \
  --name "$NAME" --fold "$FOLD" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${FOLD}.pt" \
  --pseudo-labels "$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${FOLD}.npz" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --bottleneck 128 --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
  --retention-weight 0.25 --stream-distillation-weight "$STREAM_WEIGHT" --stream-temperature 2 \
  --batch-size 16 --workers 4 --frames 32 --image-size 160 \
  --tta-hflip --tta-crop-scale 0.75 --seed 42 \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  2>&1 | tee "$HERE/logs/${NAME}.log"
