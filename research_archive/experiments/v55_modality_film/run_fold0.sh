#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXPERIMENT="$ROOT/experiments/v55_modality_film"
OUTPUT="$EXPERIMENT/outputs"
LOG="$OUTPUT/logs/v55_modality_film_f0.log"
LOCK="$ROOT/outputs/logs/gpu-0.lock"
mkdir -p "$OUTPUT/logs"

flock "$LOCK" bash -c '
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=0
  export PATH="$HOME/miniconda3/envs/Train/bin:$PATH"
  cd "$1"
  exec python -B scripts/train_r2plus1d.py \
    --name v55_modality_film_f0 --fold 0 \
    --mode shared_multimodal --initialization kinetics \
    --weights .tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth \
    --cache-dir data/processed/r2plus1d_yolo_all_visual_32_128 \
    --frames 32 --image-size 128 --epochs 20 --batch-size 8 --workers 4 \
    --lr 0.0003 --backbone-lr-factor 0.1 --weight-decay 0.0001 --warmup-epochs 2 \
    --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 --label-smoothing 0.03 \
    --class-balance-power 0 --seed 42 --patience 20 --eval-every 1 \
    --mixup-alpha 0.3 --visual-strategy joint_film \
    --output-root "$2" --swanlab-mode cloud
' bash "$ROOT" "$OUTPUT" 2>&1 | tee "$LOG"
