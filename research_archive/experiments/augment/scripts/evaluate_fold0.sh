#!/usr/bin/env bash
set -euo pipefail

SHIFT_TOKENS="${1:?usage: evaluate_fold0.sh SHIFT_TOKENS}"
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "$EXPERIMENT_ROOT/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
NAME="ablation_async_shift_f0_p${SHIFT_TOKENS}"
SOUP_DIR="$EXPERIMENT_ROOT/outputs/checkpoints/${NAME}_soup_w075"
PREDICTION="$EXPERIMENT_ROOT/outputs/predictions/${NAME}_candidate.npz"
REPORT="$EXPERIMENT_ROOT/outputs/reports/${NAME}.json"
mkdir -p "$SOUP_DIR" "$(dirname "$PREDICTION")" "$(dirname "$REPORT")" "$EXPERIMENT_ROOT/outputs/logs"

flock "$REPOSITORY_ROOT/outputs/logs/gpu-2.lock" bash -c '
set -o pipefail
export CUDA_VISIBLE_DEVICES=2
"$1" "$2/scripts/interpolate_r2plus1d_checkpoints.py" \
  --left "$2/outputs/checkpoints/r2_staged_multimodal_f0/best.pt" \
  --right "$3/outputs/checkpoints/$4/best.pt" \
  --right-weight 0.75 \
  --output "$5/best_int8.pt"
"$1" "$2/scripts/evaluate_r2plus1d_checkpoint.py" \
  --checkpoint "$5/best_int8.pt" \
  --fold 0 \
  --cache-dir "$2/data/processed/r2plus1d_yolo_32_128" \
  --frames 32 \
  --tta-hflip \
  --tta-crop-scale 0.75 \
  --batch-size 16 \
  --workers 4 \
  --output "$6"
"$1" "$2/scripts/evaluate_single_ablation.py" \
  --candidate "$6" \
  --fold 0 \
  --output "$7"
' bash "$PYTHON" "$REPOSITORY_ROOT" "$EXPERIMENT_ROOT" "$NAME" "$SOUP_DIR" "$PREDICTION" "$REPORT" \
  2>&1 | tee "$EXPERIMENT_ROOT/outputs/logs/${NAME}_evaluation.log"
