#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: evaluate_fold.sh METHOD GPU [FOLD]}"
GPU="${2:?usage: evaluate_fold.sh METHOD GPU [FOLD]}"
FOLD="${3:-0}"
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$EXPERIMENT_ROOT/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
NAME="v11_${METHOD}_f${FOLD}"
SOUP="$EXPERIMENT_ROOT/outputs/checkpoints/${NAME}_soup_w075/best_int8.pt"
CANDIDATE="$EXPERIMENT_ROOT/outputs/predictions/${NAME}_candidate.npz"
AUDIT="$EXPERIMENT_ROOT/outputs/reports/${NAME}_main_audit.json"
GROUPED="$EXPERIMENT_ROOT/outputs/reports/${NAME}_per_user.json"
LOG="$EXPERIMENT_ROOT/outputs/logs/${NAME}_evaluation.log"
mkdir -p "$(dirname "$SOUP")" "$(dirname "$CANDIDATE")" "$(dirname "$AUDIT")"

set +e
flock "$REPOSITORY_ROOT/outputs/logs/gpu-${GPU}.lock" bash -c '
set -euo pipefail
export CUDA_VISIBLE_DEVICES="$1"
"$2" "$3/scripts/interpolate_r2plus1d_checkpoints.py" \
  --left "$3/outputs/checkpoints/r2_staged_multimodal_f${4}/best.pt" \
  --right "$5/outputs/checkpoints/${6}/best.pt" \
  --right-weight 0.75 \
  --output "$7"
"$2" "$3/scripts/evaluate_r2plus1d_checkpoint.py" \
  --checkpoint "$7" \
  --fold "$4" \
  --cache-dir "$3/data/processed/r2plus1d_yolo_32_128" \
  --frames 32 \
  --tta-hflip \
  --tta-crop-scale 0.75 \
  --batch-size 16 \
  --workers 4 \
  --output "$8"
"$2" "$3/scripts/evaluate_single_ablation.py" \
  --candidate "$8" \
  --fold "$4" \
  --output "$9"
"$2" "$5/analyze_candidate.py" \
  --candidate "$8" \
  --ensemble "${9%.json}.npz" \
  --fold "$4" \
  --output "${10}"
' bash "$GPU" "$PYTHON" "$REPOSITORY_ROOT" "$FOLD" "$EXPERIMENT_ROOT" "$NAME" "$SOUP" "$CANDIDATE" "$AUDIT" "$GROUPED" \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$EXPERIMENT_ROOT/outputs/logs/${NAME}_evaluation.exit"
exit "$status"
