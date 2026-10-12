#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 GPU NAME FOLD MODALITIES IMAGE_SIZE" >&2
  exit 2
fi

gpu="$1"
name="$2"
fold="$3"
modalities="$4"
image_size="$5"
log_dir="experiments/scene_context_v13/outputs/logs"
mkdir -p "$log_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export TORCH_HOME="$PWD/.tools/torch"

set +e
${HOME}/miniconda3/envs/Train/bin/python -B scripts/train.py \
  --name "$name" \
  --modalities "$modalities" \
  --fold "$fold" \
  --seed 42 \
  --epochs 60 \
  --batch-size 12 \
  --lr 0.0004 \
  --width 96 \
  --workers 6 \
  --frames 16 \
  --eval-every 2 \
  --patience 20 \
  --swanlab-mode online \
  --strong-augment \
  --synchronized-temporal-augment \
  --canonical-pose \
  --image-size "$image_size" \
  --backbone x3d_s \
  --kinetics \
  --kinetics-source l \
  --backbone-lr-factor 0.12 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.15 \
  2>&1 | tee "$log_dir/$name.log"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$status" > "$log_dir/$name.exit"
exit "$status"
