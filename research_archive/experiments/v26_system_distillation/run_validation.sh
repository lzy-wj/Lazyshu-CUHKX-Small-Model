#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
V13="$ROOT/experiments/scene_context_v13"
V24="$ROOT/experiments/v24_pseudo_validation/outputs"
OUT="$HERE/outputs"
LOGS="$HERE/logs"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
mkdir -p "$OUT/targets" "$OUT/predictions" "$LOGS"

build_target() {
  local fold="$1"
  "$PYTHON" -B "$HERE/build_system_targets.py" \
    --fold "$fold" \
    --base-pattern "$V13/outputs/predictions/v14_fixedsoup_pseudo_c0p6_e2_composite_f{fold}.npz" \
    --base-settings "$V13/outputs/shared6_mixup_fixedsoup_uniform_l3int6_l4b0_domain_pairaware.json" \
    --base-acceptance "$V13/outputs/v14_acceptance_gate.json" \
    --candidate-pattern "$V24/v23_warm60_pseudo_composite_f{fold}.npz" \
    --candidate-settings "$V24/v24_warm60_pseudo_domain_pairaware.json" \
    --second-stage "$V24/v24_warm60_pseudo_second_stage_acceptance.json" \
    --output "$OUT/targets/v26_system_teacher_f${fold}.npz"
}

adapt_fold() {
  local fold="$1"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$V13/adapt_pseudo_target.py" \
    --checkpoint "$V13/outputs/v23_parameter_soup_warm60_f${fold}.pt" \
    --pseudo-labels "$OUT/targets/v26_system_teacher_f${fold}.npz" \
    --fold "$fold" --scope heads --epochs 2 --lr 0.0002 \
    --confidence-threshold 0.6 --retention-weight 0.25 \
    --batch-size 16 --workers 4 --cache-dir "$CACHE" --frames 32 \
    --tta-hflip --tta-crop-scale 0.75 \
    --output "$OUT/predictions/v26_system_distill_f${fold}.npz" \
    2>&1 | tee "$LOGS/fold${fold}.log"
}

build_target 0
adapt_fold 0
"$PYTHON" -B "$HERE/evaluate_direct_pairaware.py" \
  --fold0 "$OUT/predictions/v26_system_distill_f0.npz" \
  --output "$OUT/direct_pairaware_f0.json"

"$PYTHON" - "$OUT/direct_pairaware_f0.json" <<'PY'
import json
import sys
from pathlib import Path

correct = json.loads(Path(sys.argv[1]).read_text())["fold0"]["correct"]
print({"fold0_direct_pairaware": correct, "gate": 545})
if correct < 545:
    raise SystemExit("V26 Fold0 gate failed; Fold1 remains untrained")
PY

build_target 1
adapt_fold 1
"$PYTHON" -B "$HERE/evaluate_direct_pairaware.py" \
  --fold0 "$OUT/predictions/v26_system_distill_f0.npz" \
  --fold1 "$OUT/predictions/v26_system_distill_f1.npz" \
  --output "$OUT/direct_pairaware.json"

