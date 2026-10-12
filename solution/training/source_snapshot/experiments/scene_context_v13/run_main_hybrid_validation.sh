#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: run_main_hybrid_validation.sh GPU}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="$ROOT/experiments/scene_context_v13/outputs"

export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="$HOME/miniconda3/envs/Train/bin:$PATH"
mkdir -p "$OUTPUT_ROOT/predictions"

for fold in 0 1; do
  python -B "$ROOT/scripts/evaluate_r2plus1d_checkpoint.py" \
    --checkpoint "$OUTPUT_ROOT/checkpoints/v12_mixup_f${fold}_selected_raw_hybrid_compact.pt" \
    --fold "$fold" \
    --tta-hflip \
    --tta-crop-scale 0.75 \
    --batch-size 16 \
    --workers 4 \
    --cache-dir "$ROOT/data/processed/r2plus1d_yolo_32_128" \
    --frames 32 \
    --output "$OUTPUT_ROOT/predictions/v12_mixup_f${fold}_selected_raw_hybrid_tta.npz"
done
