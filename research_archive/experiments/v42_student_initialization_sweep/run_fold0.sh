#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f0.npz"
CACHE128="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
CACHE160="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160"
OUTPUT="$HERE/outputs"
LOGS="$HERE/logs"
mkdir -p "$OUTPUT" "$LOGS"

names=(fiveway warm progressive noradar)
checkpoints=(
  "$ROOT/experiments/scene_context_v13/outputs/v23_fiveway_parameter_soup_f0.pt"
  "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f0/best_int8.pt"
  "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_progressive160_maskdistill_f0/best_int8.pt"
  "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_noradar_epochs_f0/best_int8.pt"
)
caches=("$CACHE128" "$CACHE128" "$CACHE160" "$CACHE128")

for position in "${!names[@]}"; do
  name="${names[$position]}"
  checkpoint="${checkpoints[$position]}"
  cache="${caches[$position]}"
  archive="$OUTPUT/${name}_f0.npz"
  if [[ ! -f "$archive" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
      "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
      --checkpoint "$checkpoint" --pseudo-labels "$PSEUDO" --fold 0 \
      --scope heads --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
      --retention-weight 0.25 --batch-size 16 --workers 4 --seed 42 \
      --cache-dir "$cache" --frames 32 --tta-hflip --tta-crop-scale 0.75 \
      --output "$archive" 2>&1 | tee "$LOGS/${name}_f0.log"
  fi
  "$PYTHON" -B "$ROOT/experiments/v30_hard_pseudo_sweep/score_archive.py" \
    --archive "$archive" --fold 0 --output "$OUTPUT/${name}_f0_score.json"
done

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in root.glob("*_f0_score.json"):
    item = json.loads(path.read_text())
    item["name"] = path.name.removesuffix("_f0_score.json")
    rows.append(item)
rows.sort(key=lambda item: (item["direct_pairaware_correct"], item["raw_correct"]), reverse=True)
report = {
    "reference": {"name": "warm60_soup", "raw_correct": 493, "direct_pairaware_correct": 550},
    "selected": rows[0],
    "ranking": rows,
    "promote": rows[0]["direct_pairaware_correct"] > 550,
}
(root / "fold0_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not report["promote"]:
    raise SystemExit("No V42 student improves the Fold 0 reference")
PY
