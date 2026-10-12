#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
mkdir -p "$HERE/outputs" "$HERE/logs"

run_fold() {
  local fold="$1"
  local checkpoint="$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${fold}.pt"
  local pseudo="$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${fold}.npz"
  local adapted="$HERE/outputs/v23_warm60_pseudo_adapt_f${fold}.npz"
  local base="$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${fold}.npz"
  local composite="$HERE/outputs/v23_warm60_pseudo_composite_f${fold}.npz"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B \
    "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
    --checkpoint "$checkpoint" --pseudo-labels "$pseudo" --fold "$fold" \
    --scope heads --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
    --retention-weight 0.25 --batch-size 16 --workers 4 \
    --cache-dir "$CACHE" --frames 32 --tta-hflip --tta-crop-scale 0.75 \
    --output "$adapted" 2>&1 | tee "$HERE/logs/fold${fold}.log"
  "$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_fixedsoup_pseudo_composite.py" \
    --base "$base" --specialized "$adapted" --fold "$fold" --output "$composite"
}

run_fold 0

"$PYTHON" - "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f0.npz" \
  "$HERE/outputs/v23_warm60_pseudo_composite_f0.npz" <<'PY'
import sys
import numpy as np

base = np.load(sys.argv[1])
candidate = np.load(sys.argv[2])
base_correct = int((base["probabilities"].argmax(1) == base["labels"]).sum())
candidate_correct = int((candidate["probabilities"].argmax(1) == candidate["labels"]).sum())
print({"base_correct": base_correct, "candidate_correct": candidate_correct})
if candidate_correct < base_correct:
    raise SystemExit("Fold0 pseudo adaptation regressed; Fold1 remains blind")
PY

run_fold 1

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/evaluate_candidate_domain_pairaware.py" \
  --name v24_warm60_pseudo \
  --fold0 "$HERE/outputs/v23_warm60_pseudo_composite_f0.npz" \
  --fold1 "$HERE/outputs/v23_warm60_pseudo_composite_f1.npz" \
  --output "$HERE/outputs/v24_warm60_pseudo_domain_pairaware.json"

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/evaluate_second_stage_acceptance.py" \
  --base-pattern "$ROOT/experiments/scene_context_v13/outputs/predictions/v14_fixedsoup_pseudo_c0p6_e2_composite_f{fold}.npz" \
  --base-settings "$ROOT/experiments/scene_context_v13/outputs/shared6_mixup_fixedsoup_uniform_l3int6_l4b0_domain_pairaware.json" \
  --base-acceptance "$ROOT/experiments/scene_context_v13/outputs/v14_acceptance_gate.json" \
  --candidate-pattern "$HERE/outputs/v23_warm60_pseudo_composite_f{fold}.npz" \
  --candidate-settings "$HERE/outputs/v24_warm60_pseudo_domain_pairaware.json" \
  --output "$HERE/outputs/v24_warm60_pseudo_second_stage_acceptance.json"
