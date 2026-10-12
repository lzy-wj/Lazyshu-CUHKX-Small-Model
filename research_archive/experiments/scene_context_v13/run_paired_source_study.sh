#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
SCOPE="${2:-heads}"
EPOCHS="${3:-4}"
LR="${4:-0.0003}"
TAG="${SCOPE}_e${EPOCHS}_lr${LR//./p}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${HOME}/miniconda3/envs/Train/bin/python"
OUT="$ROOT/experiments/scene_context_v13/outputs/paired_source/$TAG"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
CHECKPOINT="$ROOT/experiments/blend_aware_v12/outputs/checkpoints/v12_shared6_mixup_f0/best_int8.pt"
FALLBACK0="$ROOT/experiments/blend_aware_v12/outputs/evaluations/v12_shared6_mixup_e18_f0_predictions/epoch_18.npz"
FALLBACK1="$ROOT/experiments/scene_context_v13/outputs/predictions/v14_shared6_mixup_f1_tta.npz"

mkdir -p "$OUT"
for pair in "18 8" "8 18"; do
  read -r SOURCE TARGET <<< "$pair"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/experiments/scene_context_v13/adapt_paired_source.py" \
    --checkpoint "$CHECKPOINT" --fold 0 --source-user "$SOURCE" --target-user "$TARGET" \
    --scope "$SCOPE" --epochs "$EPOCHS" --lr "$LR" --distillation-weight 0.25 \
    --auxiliary-weight 0.1 --batch-size 16 --workers 4 --adabn reset \
    --tta-hflip --tta-crop-scale 0.75 --cache-dir "$CACHE" \
    --output "$OUT/target${TARGET}.npz"
done

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/merge_user_predictions.py" \
  --fallback "$FALLBACK0" \
  --specialized "$OUT/target8.npz" --specialized "$OUT/target18.npz" \
  --output "$OUT/fold0.npz"

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/evaluate_candidate_domain_pairaware.py" \
  --name "paired_source_${TAG}" --fold0 "$OUT/fold0.npz" --fold1 "$FALLBACK1" \
  --output "$OUT/domain_pairaware.json"
