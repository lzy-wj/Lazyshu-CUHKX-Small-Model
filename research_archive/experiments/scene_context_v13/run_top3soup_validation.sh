#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${HOME}/miniconda3/envs/Train/bin/python"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
OUT="$ROOT/experiments/scene_context_v13/outputs/predictions/shared6_mixup_top3soup"
mkdir -p "$OUT"

for FOLD in 0 1; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
    --checkpoint "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v14_shared6_mixup_top3soup_f${FOLD}.pt" \
    --fold "$FOLD" --tta-hflip --tta-crop-scale 0.75 --batch-size 16 --workers 4 \
    --cache-dir "$CACHE" --frames 32 --output "$OUT/fold${FOLD}.npz"
done

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/evaluate_candidate_domain_pairaware.py" \
  --name shared6_mixup_top3soup --fold0 "$OUT/fold0.npz" --fold1 "$OUT/fold1.npz" \
  --output "$ROOT/experiments/scene_context_v13/outputs/shared6_mixup_top3soup_domain_pairaware.json"
