#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
OUTPUT="$ROOT/experiments/v75_pervisual_full_refit/outputs"
COMPLETION="$OUTPUT/checkpoints/v75_pervisual_full_e17/completion.json"
CHECKPOINT="$OUTPUT/checkpoints/v75_pervisual_full_e17/best_int8.pt"
YOLO="$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"
PREDICTION="$OUTPUT/predictions/v75_pervisual_full_raw.npz"
SUBMISSION="$OUTPUT/submissions/v75_pervisual_full_raw.csv"
MANIFEST="$OUTPUT/reports/v75_pervisual_full_raw.json"

mkdir -p "$OUTPUT/predictions" "$OUTPUT/submissions" "$OUTPUT/reports"
until [[ -f "$COMPLETION" ]]; do sleep 60; done
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing V75 checkpoint: $CHECKPOINT" >&2
  exit 1
fi

flock "$ROOT/outputs/logs/gpu-7.lock" env CUDA_VISIBLE_DEVICES=7 \
  "$PYTHON" -B "$ROOT/experiments/v24_full_refit/predict_test.py" \
  --checkpoint "$CHECKPOINT" --yolo "$YOLO" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --batch-size 16 --workers 4 \
  --output "$PREDICTION" --submission "$SUBMISSION" --manifest "$MANIFEST" \
  2>&1 | tee "$OUTPUT/logs/v75_predict_test.log"
