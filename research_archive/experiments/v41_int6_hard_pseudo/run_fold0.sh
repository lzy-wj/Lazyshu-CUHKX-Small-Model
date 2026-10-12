#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
SOURCE="$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f0.pt"
CHECKPOINT="$HERE/outputs/v23_parameter_soup_warm60_l3int6_f0.pt"
PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f0.npz"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
OUTPUT="$HERE/outputs/int6_hard_pseudo_f0.npz"
mkdir -p "$HERE/outputs" "$HERE/logs"

"$PYTHON" -B "$ROOT/scripts/quantize_r2plus1d_hybrid.py" \
  --input "$SOURCE" --output "$CHECKPOINT" --deep-scheme uniform4 --layer3-bits 6
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
  "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
  --checkpoint "$CHECKPOINT" --pseudo-labels "$PSEUDO" --fold 0 \
  --scope heads --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
  --retention-weight 0.25 --batch-size 16 --workers 4 --seed 42 \
  --cache-dir "$CACHE" --frames 32 --tta-hflip --tta-crop-scale 0.75 \
  --output "$OUTPUT" 2>&1 | tee "$HERE/logs/fold0.log"
"$PYTHON" -B "$ROOT/experiments/v30_hard_pseudo_sweep/score_archive.py" \
  --archive "$OUTPUT" --fold 0 --output "$HERE/outputs/fold0_score.json"
"$PYTHON" - "$HERE/outputs/fold0_score.json" <<'PY'
import json
import sys

score = json.load(open(sys.argv[1]))
if score["direct_pairaware_correct"] < 550:
    raise SystemExit("V41 INT6 failed its Fold 0 retention gate")
PY
