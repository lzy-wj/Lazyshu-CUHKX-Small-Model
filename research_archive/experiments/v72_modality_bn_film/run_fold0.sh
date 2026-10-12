#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
OUTPUT="$ROOT/experiments/v72_modality_bn_film/outputs"
WEIGHTS="$ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth"
mkdir -p "$OUTPUT/logs"

flock "$ROOT/outputs/logs/gpu-7.lock" env CUDA_VISIBLE_DEVICES=7 \
  "$PYTHON" -B "$ROOT/scripts/train_r2plus1d.py" \
  --name v72_modality_bn_film_f0 --fold 0 \
  --mode shared_multimodal --initialization kinetics --weights "$WEIGHTS" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 20 --batch-size 8 --workers 4 \
  --lr 0.0003 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 2 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
  --class-balance-power 0 --seed 42 --patience 20 --eval-every 1 \
  --mixup-alpha 0.3 --visual-strategy modality_bn_film \
  --output-root "$OUTPUT" --swanlab-mode cloud \
  2>&1 | tee "$OUTPUT/logs/v72_modality_bn_film_f0.log"
