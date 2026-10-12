#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
CHECKPOINT="$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f1.pt"
PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f1.npz"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
ARCHIVE="$HERE/outputs/lr300_f1.npz"
mkdir -p "$HERE/outputs" "$HERE/logs"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
  "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
  --checkpoint "$CHECKPOINT" --pseudo-labels "$PSEUDO" --fold 1 \
  --scope heads --epochs 2 --lr 0.0003 --confidence-threshold 0.6 \
  --retention-weight 0.25 --batch-size 16 --workers 4 --cache-dir "$CACHE" --frames 32 \
  --tta-hflip --tta-crop-scale 0.75 --output "$ARCHIVE" \
  2>&1 | tee "$HERE/logs/lr300_f1.log"
"$PYTHON" -B "$HERE/score_archive.py" --archive "$ARCHIVE" --fold 1 \
  --output "$HERE/outputs/lr300_f1_score.json"

"$PYTHON" - "$HERE/outputs/lr300_f0_score.json" "$HERE/outputs/lr300_f1_score.json" \
  "$HERE/outputs/lr300_combined.json" <<'PY'
import json
import sys
from pathlib import Path

fold0 = json.loads(Path(sys.argv[1]).read_text())
fold1 = json.loads(Path(sys.argv[2]).read_text())
correct = fold0["direct_pairaware_correct"] + fold1["direct_pairaware_correct"]
report = {
    "selection": "Highest Fold 0 direct pair-aware, then highest raw accuracy.",
    "config": {"epochs": 2, "lr": 0.0003, "confidence_threshold": 0.6, "retention_weight": 0.25},
    "fold0": fold0,
    "fold1": fold1,
    "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
    "reference": {"correct": 1168, "rows": 1357},
    "promote": correct > 1168,
}
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
