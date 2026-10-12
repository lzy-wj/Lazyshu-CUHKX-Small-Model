#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU FOLD}"
FOLD="${2:?Usage: $0 GPU FOLD}"
PSEUDO="$HERE/outputs/consensus_pseudo_f${FOLD}.npz"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$FOLD" =~ ^[01]$ ]] || { echo "Fold must be 0 or 1" >&2; exit 2; }

mkdir -p "$HERE/outputs" "$HERE/logs"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -B \
  "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${FOLD}.pt" \
  --pseudo-labels "$PSEUDO" --fold "$FOLD" \
  --scope heads --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
  --retention-weight 0.25 --batch-size 16 --workers 4 \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --frames 32 --image-size 160 --tta-hflip --tta-crop-scale 0.75 \
  --output "$HERE/outputs/v92_consensus_highres_f${FOLD}.npz" \
  2>&1 | tee "$HERE/logs/fold${FOLD}.log"
