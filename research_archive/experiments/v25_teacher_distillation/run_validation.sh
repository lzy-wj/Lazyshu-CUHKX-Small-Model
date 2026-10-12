#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
V13="$ROOT/experiments/scene_context_v13"
OUT="$HERE/outputs"
LOGS="$HERE/logs"
mkdir -p "$OUT/targets" "$OUT/predictions" "$LOGS"

build_target() {
  local fold="$1"
  "$PYTHON" -B "$HERE/build_validation_targets.py" \
    --fold "$fold" \
    --pair-aware "$V13/outputs/predictions/v13_pairaware_validation_f${fold}.npz" \
    --output "$OUT/targets/v13_structured_teacher_f${fold}.npz"
}

adapt_fold() {
  local fold="$1"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$HERE/adapt_teacher_target.py" \
    --checkpoint "$V13/outputs/v23_parameter_soup_warm60_f${fold}.pt" \
    --teacher-targets "$OUT/targets/v13_structured_teacher_f${fold}.npz" \
    --fold "$fold" --scope heads --epochs 2 --lr 0.0002 \
    --confidence-threshold 0.6 --structure-weight 0.20 --label-smoothing 0.02 \
    --retention-weight 0.25 --batch-size 16 --workers 4 \
    --cache-dir "$CACHE" --frames 32 --tta-hflip --tta-crop-scale 0.75 \
    --output "$OUT/predictions/v25_teacher_distill_f${fold}.npz" \
    2>&1 | tee "$LOGS/fold${fold}.log"
}

evaluate_domain() {
  local fold1="$1"
  "$PYTHON" -B "$V13/evaluate_candidate_domain_pairaware.py" \
    --name v25_teacher_distill \
    --fold0 "$OUT/predictions/v25_teacher_distill_f0.npz" \
    --fold1 "$fold1" \
    --output "$OUT/v25_teacher_distill_domain_pairaware.json"
}

build_target 0
adapt_fold 0
evaluate_domain "$V13/outputs/v23_parameter_soup_warm60_f1.npz"

"$PYTHON" - "$OUT/predictions/v25_teacher_distill_f0.json" \
  "$OUT/v25_teacher_distill_domain_pairaware.json" <<'PY'
import json
import sys
from pathlib import Path

raw = json.loads(Path(sys.argv[1]).read_text())["correct"]
decoded = json.loads(Path(sys.argv[2]).read_text())["fold0"]["correct"]
print({"fold0_raw": raw, "fold0_domain_pairaware": decoded})
if raw < 475:
    raise SystemExit("Fold0 raw regression exceeds the preregistered 12-row tolerance")
if decoded < 566:
    raise SystemExit("Fold0 does not reach the current V23 second-stage reference")
PY

build_target 1
adapt_fold 1
evaluate_domain "$OUT/predictions/v25_teacher_distill_f1.npz"

"$PYTHON" -B "$V13/evaluate_second_stage_acceptance.py" \
  --base-pattern "$V13/outputs/predictions/v14_fixedsoup_pseudo_c0p6_e2_composite_f{fold}.npz" \
  --base-settings "$V13/outputs/shared6_mixup_fixedsoup_uniform_l3int6_l4b0_domain_pairaware.json" \
  --base-acceptance "$V13/outputs/v14_acceptance_gate.json" \
  --candidate-pattern "$OUT/predictions/v25_teacher_distill_f{fold}.npz" \
  --candidate-settings "$OUT/v25_teacher_distill_domain_pairaware.json" \
  --output "$OUT/v25_teacher_distill_second_stage_acceptance.json"

"$PYTHON" - "$OUT/v25_teacher_distill_second_stage_acceptance.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
correct = report["combined"]["correct"]
report["promotion_reference_correct"] = 1190
report["promote"] = correct > 1190
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print({"combined_correct": correct, "promote": report["promote"]})
PY
