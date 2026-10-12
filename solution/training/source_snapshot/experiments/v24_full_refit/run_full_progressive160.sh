#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

GPU="${1:?Usage: $0 GPU}"
NAME="v24_full_progressive160_e7"
BASE="$OUTPUT_ROOT/checkpoints/v24_full_noradar_e13/best.pt"
require_file "$BASE"
assert_new_training_run "$NAME"

run_logged "$NAME" "$GPU" "$PYTHON" -B "$ROOT/scripts/train_r2plus1d.py" \
  --name "$NAME" --fold -1 \
  --mode shared_multimodal --initialization kinetics \
  --weights "$ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth" \
  --init-model "$BASE" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --frames 32 --image-size 160 --epochs 7 --batch-size 8 --workers 4 \
  --lr 0.00015 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 1 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
  --class-balance-power 0 --seed 43 --patience 10 --eval-every 1 \
  --mixup-alpha 0.3 --visual-strategy joint \
  --output-root "$OUTPUT_ROOT" --swanlab-mode "$SWANLAB_MODE"
