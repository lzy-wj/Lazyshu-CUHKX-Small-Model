#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 GPU NAME FOLD" >&2
  exit 2
fi

gpu="$1"
name="$2"
fold="$3"
log_dir="experiments/scene_context_v13/outputs/logs"
mkdir -p "$log_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export TORCH_HOME="$PWD/.tools/torch"

set +e
${HOME}/miniconda3/envs/Train/bin/python -B scripts/train.py \
  --name "$name" \
  --modalities scene_depth \
  --fold "$fold" \
  --seed 42 \
  --epochs 40 \
  --batch-size 16 \
  --lr 0.0003 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 1 \
  --patience 12 \
  --swanlab-mode online \
  --strong-augment \
  --synchronized-temporal-augment \
  --image-size 160 \
  --backbone resnet50 \
  --imagenet \
  --backbone-lr-factor 0.1 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.15 \
  2>&1 | tee "$log_dir/$name.log"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$status" > "$log_dir/$name.exit"
exit "$status"
