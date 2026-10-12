#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
V13="$ROOT/experiments/scene_context_v13"
V26="$ROOT/experiments/v26_system_distillation"
OUT="$HERE/outputs"
LOGS="$HERE/logs"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
mkdir -p "$OUT/targets" "$OUT/predictions" "$LOGS"

run_fold() {
  local fold="$1"
  "$PYTHON" -B "$HERE/build_targeted_targets.py" \
    --teacher "$V26/outputs/targets/v26_system_teacher_f${fold}.npz" \
    --student "$V13/outputs/v23_parameter_soup_warm60_f${fold}.npz" \
    --output "$OUT/targets/v27_targeted_teacher_f${fold}.npz"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$V13/adapt_pseudo_target.py" \
    --checkpoint "$V13/outputs/v23_parameter_soup_warm60_f${fold}.pt" \
    --pseudo-labels "$OUT/targets/v27_targeted_teacher_f${fold}.npz" \
    --fold "$fold" --scope heads --epochs 2 --lr 0.0002 \
    --confidence-threshold 0.6 --retention-weight 0.25 \
    --batch-size 16 --workers 4 --cache-dir "$CACHE" --frames 32 \
    --tta-hflip --tta-crop-scale 0.75 \
    --output "$OUT/predictions/v27_targeted_distill_f${fold}.npz" \
    2>&1 | tee "$LOGS/fold${fold}.log"
}

run_fold 0
"$PYTHON" -B "$V26/evaluate_direct_pairaware.py" \
  --fold0 "$OUT/predictions/v27_targeted_distill_f0.npz" \
  --output "$OUT/direct_pairaware_f0.json"
"$PYTHON" - "$OUT/direct_pairaware_f0.json" <<'PY'
import json
import sys
from pathlib import Path

correct = json.loads(Path(sys.argv[1]).read_text())["fold0"]["correct"]
print({"fold0_direct_pairaware": correct, "required": 551})
if correct < 551:
    raise SystemExit("V27 Fold0 did not beat V26; Fold1 remains untrained")
PY

run_fold 1
"$PYTHON" -B "$V26/evaluate_direct_pairaware.py" \
  --fold0 "$OUT/predictions/v27_targeted_distill_f0.npz" \
  --fold1 "$OUT/predictions/v27_targeted_distill_f1.npz" \
  --output "$OUT/direct_pairaware.json"
"$PYTHON" - "$OUT/direct_pairaware.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
report["reference_correct"] = 1170
report["promote"] = report["combined"]["correct"] > 1170
path.write_text(json.dumps(report, indent=2) + "\n")
print({"combined": report["combined"], "promote": report["promote"]})
PY

