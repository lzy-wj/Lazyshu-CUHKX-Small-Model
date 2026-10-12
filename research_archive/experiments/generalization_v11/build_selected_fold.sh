#!/usr/bin/env bash
set -euo pipefail

FOLD="${1:?usage: build_selected_fold.sh FOLD GPU}"
GPU="${2:?usage: build_selected_fold.sh FOLD GPU}"
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$EXPERIMENT_ROOT/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
CONDITIONAL_SOUP="$EXPERIMENT_ROOT/outputs/checkpoints/v11_conditional_adv_f${FOLD}_soup_w075/best_int8.pt"
SELECTED_CHECKPOINT="$EXPERIMENT_ROOT/outputs/checkpoints/v11_selected_r2_f${FOLD}/best_int8.pt"
SELECTED_PREDICTION="$EXPERIMENT_ROOT/outputs/predictions/v11_selected_r2_f${FOLD}.npz"
SELECTED_REPORT="$EXPERIMENT_ROOT/outputs/reports/v11_selected_fold${FOLD}.json"
mkdir -p "$(dirname "$SELECTED_CHECKPOINT")" "$(dirname "$SELECTED_PREDICTION")" "$(dirname "$SELECTED_REPORT")"

test -f "$CONDITIONAL_SOUP"
flock "$REPOSITORY_ROOT/outputs/logs/gpu-${GPU}.lock" bash -c '
set -euo pipefail
export CUDA_VISIBLE_DEVICES="$1"
"$2" "$3/scripts/interpolate_r2plus1d_checkpoints.py" \
  --left "$3/outputs/checkpoints/r2_staged_soup_f${4}_w075/best_int8.pt" \
  --right "$5" \
  --right-weight 0.5 \
  --output "$6"
"$2" "$3/scripts/evaluate_r2plus1d_checkpoint.py" \
  --checkpoint "$6" \
  --fold "$4" \
  --cache-dir "$3/data/processed/r2plus1d_yolo_32_128" \
  --frames 32 \
  --tta-hflip \
  --tta-crop-scale 0.75 \
  --batch-size 16 \
  --workers 4 \
  --output "$7"
' bash "$GPU" "$PYTHON" "$REPOSITORY_ROOT" "$FOLD" "$CONDITIONAL_SOUP" "$SELECTED_CHECKPOINT" "$SELECTED_PREDICTION"

"$PYTHON" "$EXPERIMENT_ROOT/evaluate_selected_blend.py" \
  --candidate "$SELECTED_PREDICTION" \
  --fold "$FOLD" \
  --config "$EXPERIMENT_ROOT/selected_config.json" \
  --output "$SELECTED_REPORT"
"$PYTHON" "$EXPERIMENT_ROOT/analyze_candidate.py" \
  --candidate "$SELECTED_PREDICTION" \
  --ensemble "${SELECTED_REPORT%.json}.npz" \
  --fold "$FOLD" \
  --output "$EXPERIMENT_ROOT/outputs/reports/v11_selected_fold${FOLD}_per_user.json"
