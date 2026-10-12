#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/Train/bin/python}"
GPU="${1:?Usage: $0 GPU FOLD}"
FOLD="${2:?Usage: $0 GPU FOLD}"

case "$FOLD" in
  0) USERS=(3 8 18 23) ;;
  1) USERS=(5 7 17 22) ;;
  *) echo "Fold must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "$HERE/outputs" "$HERE/logs"
for USER in "${USERS[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
    "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
    --checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${FOLD}.pt" \
    --pseudo-labels "$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${FOLD}.npz" \
    --fold "$FOLD" --target-user "$USER" \
    --scope heads --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
    --retention-weight 0.25 --batch-size 16 --workers 4 \
    --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
    --frames 32 --tta-hflip --tta-crop-scale 0.75 \
    --output "$HERE/outputs/fold${FOLD}_user${USER}.npz" \
    2>&1 | tee "$HERE/logs/fold${FOLD}_user${USER}.log"
done
"$PYTHON" -B "$HERE/merge_evaluate.py" --fold "$FOLD"

