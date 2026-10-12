#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
V13="$ROOT/experiments/scene_context_v13/outputs"
V26="$ROOT/experiments/v26_system_distillation"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
mkdir -p "$HERE/outputs" "$HERE/logs"

run_fold() {
  local fold="$1"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
    --checkpoint "$V13/v23_parameter_soup_warm60_f${fold}.pt" --fold "$fold" \
    --cache-dir "$CACHE" --frames 32 --batch-size 16 --workers 4 \
    --tta-hflip --tta-crop-scale 0.875 --adabn none \
    --output "$HERE/outputs/v28_crop0875_f${fold}.npz" \
    2>&1 | tee "$HERE/logs/fold${fold}.log"
}

run_fold 0
"$PYTHON" -B "$V26/evaluate_direct_pairaware.py" \
  --fold0 "$HERE/outputs/v28_crop0875_f0.npz" \
  --output "$HERE/outputs/direct_pairaware_f0.json"
"$PYTHON" - "$HERE/outputs/v28_crop0875_f0.json" "$HERE/outputs/direct_pairaware_f0.json" <<'PY'
import json
import sys
from pathlib import Path

raw = round(json.loads(Path(sys.argv[1]).read_text())["accuracy"] * 652)
decoded = json.loads(Path(sys.argv[2]).read_text())["fold0"]["correct"]
print({"raw": raw, "direct_pairaware": decoded})
if raw <= 487 and decoded <= 540:
    raise SystemExit("V28 Fold0 did not strictly improve; Fold1 remains unrun")
PY

run_fold 1
"$PYTHON" -B "$V26/evaluate_direct_pairaware.py" \
  --fold0 "$HERE/outputs/v28_crop0875_f0.npz" \
  --fold1 "$HERE/outputs/v28_crop0875_f1.npz" \
  --output "$HERE/outputs/direct_pairaware.json"

