#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
CHECKPOINT="$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f0.pt"
PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f0.npz"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
OUTPUT="$HERE/outputs"
LOGS="$HERE/logs"
mkdir -p "$OUTPUT" "$LOGS"

run_candidate() {
  local name="$1"
  shift
  local archive="$OUTPUT/${name}_f0.npz"
  if [[ ! -f "$archive" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
      "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
      --checkpoint "$CHECKPOINT" --pseudo-labels "$PSEUDO" --fold 0 \
      --scope heads --batch-size 16 --workers 4 --cache-dir "$CACHE" --frames 32 \
      --tta-hflip --tta-crop-scale 0.75 --output "$archive" "$@" \
      2>&1 | tee "$LOGS/${name}_f0.log"
  fi
  "$PYTHON" -B "$HERE/score_archive.py" --archive "$archive" --fold 0 \
    --output "$OUTPUT/${name}_f0_score.json"
}

"$PYTHON" -B "$HERE/score_archive.py" \
  --archive "$ROOT/experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f0.npz" \
  --fold 0 --output "$OUTPUT/baseline_f0_score.json"
run_candidate epoch1 --epochs 1 --lr 0.0002 --confidence-threshold 0.6 --retention-weight 0.25
run_candidate epoch3 --epochs 3 --lr 0.0002 --confidence-threshold 0.6 --retention-weight 0.25
run_candidate conf040 --epochs 2 --lr 0.0002 --confidence-threshold 0.4 --retention-weight 0.25
run_candidate conf050 --epochs 2 --lr 0.0002 --confidence-threshold 0.5 --retention-weight 0.25
run_candidate conf070 --epochs 2 --lr 0.0002 --confidence-threshold 0.7 --retention-weight 0.25
run_candidate conf080 --epochs 2 --lr 0.0002 --confidence-threshold 0.8 --retention-weight 0.25
run_candidate lr100 --epochs 2 --lr 0.0001 --confidence-threshold 0.6 --retention-weight 0.25
run_candidate lr300 --epochs 2 --lr 0.0003 --confidence-threshold 0.6 --retention-weight 0.25
run_candidate retain050 --epochs 2 --lr 0.0002 --confidence-threshold 0.6 --retention-weight 0.5

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*_f0_score.json")):
    item = json.loads(path.read_text())
    item["name"] = path.name.removesuffix("_f0_score.json")
    rows.append(item)
rows.sort(key=lambda item: (item["direct_pairaware_correct"], item["raw_correct"]), reverse=True)
summary = {"reference": {"raw_correct": 493, "direct_pairaware_correct": 550}, "ranking": rows}
(root / "fold0_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
