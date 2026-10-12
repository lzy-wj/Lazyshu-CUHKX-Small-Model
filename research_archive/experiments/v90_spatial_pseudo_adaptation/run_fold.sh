#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU FOLD}"
FOLD="${2:?Usage: $0 GPU FOLD}"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$FOLD" =~ ^[01]$ ]] || { echo "Fold must be 0 or 1" >&2; exit 2; }
if [[ "$FOLD" == 0 ]]; then
  NAME="v86_highres160_spatial_masked_final_f0"
else
  NAME="v86_highres160_spatial_masked_f1"
fi

mkdir -p "$HERE/outputs" "$HERE/logs"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -B "$HERE/adapt.py" \
  --checkpoint "$ROOT/experiments/v86_highres_spatial_encoder/outputs/$NAME/best.pt" \
  --pseudo-labels "$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${FOLD}.npz" \
  --fold "$FOLD" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --frames 32 --image-size 160 \
  --epochs 2 --lr 0.0002 --confidence-threshold 0.6 --retention-weight 0.25 \
  --batch-size 16 --workers 4 --tta-hflip --tta-crop-scale 0.75 \
  --output "$HERE/outputs/v90_spatial_pseudo_f${FOLD}.npz" \
  --adapter-output "$HERE/outputs/v90_spatial_pseudo_head_f${FOLD}.pt" \
  2>&1 | tee "$HERE/logs/fold${FOLD}.log"
