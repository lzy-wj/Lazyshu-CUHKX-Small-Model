#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${HOME}/miniconda3/envs/Train/bin/python"
OUT="$ROOT/experiments/scene_context_v13/outputs/paired_source/heads_e4_lr0p0003_fold1"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
CHECKPOINT="$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_f1/best_int8.pt"
FALLBACK="$ROOT/experiments/scene_context_v13/outputs/predictions/v14_shared6_mixup_f1_tta.npz"
mkdir -p "$OUT"

for pair in "17 7" "7 17"; do
  read -r SOURCE TARGET <<< "$pair"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/experiments/scene_context_v13/adapt_paired_source.py" \
    --checkpoint "$CHECKPOINT" --fold 1 --source-user "$SOURCE" --target-user "$TARGET" \
    --scope heads --epochs 4 --lr 0.0003 --distillation-weight 0.25 --auxiliary-weight 0.1 \
    --batch-size 16 --workers 4 --adabn reset --tta-hflip --tta-crop-scale 0.75 \
    --cache-dir "$CACHE" --output "$OUT/target${TARGET}.npz"
done

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/merge_user_predictions.py" \
  --fallback "$FALLBACK" --specialized "$OUT/target7.npz" --specialized "$OUT/target17.npz" \
  --output "$OUT/fold1.npz"

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/evaluate_candidate_domain_pairaware.py" \
  --name paired_source_heads_e4_true \
  --fold0 "$ROOT/experiments/scene_context_v13/outputs/paired_source/heads_e4_lr0p0003/fold0.npz" \
  --fold1 "$OUT/fold1.npz" --output "$OUT/domain_pairaware.json"
