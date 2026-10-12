#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${HOME}/miniconda3/envs/Train/bin/python"
CHECKPOINT="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_full_fixedsoup_uniform_l3int6_l4b0.pt"
PSEUDO="$ROOT/experiments/scene_context_v13/outputs/predictions/v14_test_pseudo_labels.npz"
BASE="$ROOT/outputs/predictions/v14_shared6_mixup_full_fixedsoup_uniform_tta.npz"
ADAPTED="$ROOT/experiments/scene_context_v13/outputs/predictions/v14_full_fixedsoup_pseudo_adapt.npz"
ADAPTER="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_full_fixedsoup_pseudo_head_delta.pt"
COMPOSITE="$ROOT/outputs/predictions/v14_full_fixedsoup_pseudo_composite.npz"
BLENDED="$ROOT/outputs/predictions/v14_fixedsoup_pseudo_domain_blend.npz"
SETTINGS="$ROOT/experiments/scene_context_v13/outputs/shared6_mixup_fixedsoup_uniform_l3int6_l4b0_domain_pairaware.json"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/experiments/scene_context_v13/adapt_pseudo_test.py" \
  --checkpoint "$CHECKPOINT" --pseudo-labels "$PSEUDO" --scope heads \
  --epochs 2 --lr 0.0002 --confidence-threshold 0.6 --retention-weight 0.25 \
  --batch-size 16 --workers 4 --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --tta-hflip --tta-crop-scale 0.75 --output "$ADAPTED" --adapter-output "$ADAPTER"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_test_pseudo_composite.py" \
  --base "$BASE" --specialized "$ADAPTED" --output "$COMPOSITE"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/blend_test_domain_candidate.py" \
  --candidate "$COMPOSITE" --settings "$SETTINGS" --output "$BLENDED"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$BLENDED" \
  --output "$ROOT/outputs/predictions/v14_fixedsoup_pseudo_pair_aware.npz" \
  --submission "$ROOT/outputs/submissions/v14_fixedsoup_pseudo_pair_aware.csv" \
  --report "$ROOT/experiments/scene_context_v13/outputs/v14_fixedsoup_pseudo_pair_aware.json"
