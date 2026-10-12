#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${HOME}/miniconda3/envs/Train/bin/python"
SETTINGS="$ROOT/experiments/scene_context_v13/outputs/shared6_mixup_fixedsoup_uniform_l3int6_l4b0_domain_pairaware.json"
for SPEC in l1e4_r025 l3e4_r025 l2e4_r050; do
  case "$SPEC" in
    l1e4_r025) LR=0.0001; RETENTION=0.25 ;;
    l3e4_r025) LR=0.0003; RETENTION=0.25 ;;
    l2e4_r050) LR=0.0002; RETENTION=0.50 ;;
  esac
  TAG="c0p6_e2_${SPEC}"
  for FOLD in 0 1; do
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/experiments/scene_context_v13/adapt_pseudo_target.py" \
      --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_fixedsoup_uniform_l3int6_l4b0_f${FOLD}.pt" \
      --pseudo-labels "$ROOT/experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f${FOLD}.npz" \
      --fold "$FOLD" --scope heads --epochs 2 --lr "$LR" --confidence-threshold 0.6 \
      --retention-weight "$RETENTION" --batch-size 16 --workers 4 \
      --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" --frames 32 \
      --tta-hflip --tta-crop-scale 0.75 \
      --output "$ROOT/experiments/scene_context_v13/outputs/predictions/v14_fixedsoup_uniform_pseudo_${TAG}_f${FOLD}.npz"
    "$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_fixedsoup_pseudo_composite.py" \
      --base "$ROOT/experiments/scene_context_v13/outputs/predictions/shared6_mixup_fixedsoup_quant/uniform_l3int6_l4b0_f${FOLD}.npz" \
      --specialized "$ROOT/experiments/scene_context_v13/outputs/predictions/v14_fixedsoup_uniform_pseudo_${TAG}_f${FOLD}.npz" \
      --fold "$FOLD" \
      --output "$ROOT/experiments/scene_context_v13/outputs/predictions/v14_fixedsoup_pseudo_${TAG}_composite_f${FOLD}.npz"
  done
  "$PYTHON" -B "$ROOT/experiments/scene_context_v13/evaluate_frozen_domain_candidate.py" \
    --name "fixedsoup_pseudo_${TAG}_frozen" \
    --fold0 "$ROOT/experiments/scene_context_v13/outputs/predictions/v14_fixedsoup_pseudo_${TAG}_composite_f0.npz" \
    --fold1 "$ROOT/experiments/scene_context_v13/outputs/predictions/v14_fixedsoup_pseudo_${TAG}_composite_f1.npz" \
    --settings "$SETTINGS" \
    --output "$ROOT/experiments/scene_context_v13/outputs/fixedsoup_pseudo_${TAG}_frozen_domain_pairaware.json"
done
