#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU FOLD DIMENSION HEADS}"
FOLD="${2:?Usage: $0 GPU FOLD DIMENSION HEADS}"
DIMENSION="${3:?Usage: $0 GPU FOLD DIMENSION HEADS}"
HEADS="${4:?Usage: $0 GPU FOLD DIMENSION HEADS}"
NAME="v95_phase_experts_d${DIMENSION}_f${FOLD}"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "Use only physical GPUs 4-7" >&2; exit 2; }
[[ "$FOLD" =~ ^[01]$ ]] || { echo "Fold must be 0 or 1" >&2; exit 2; }
mkdir -p "$HERE/outputs" "$HERE/logs"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -B "$HERE/train.py" \
  --name "$NAME" --fold "$FOLD" \
  --init-checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f${FOLD}.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --image-size 160 --frames 32 --dimension "$DIMENSION" --heads "$HEADS" \
  --phase-depth 1 --expert-depth 2 --feedforward-multiplier 3 \
  --residual-scale 0.25 --phases 16 --epochs 6 --batch-size 8 --workers 4 \
  --lr 0.00015 --weight-decay 0.0005 --warmup-epochs 1 --ema-decay 0.99 \
  --label-smoothing 0.03 --mixup-alpha 0.3 \
  --expert-loss-weight 0.25 --retention-weight 0.5 --retention-temperature 2 \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  2>&1 | tee "$HERE/logs/${NAME}.log"
