#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 GPU NAME FOLD" >&2
  exit 2
fi

gpu="$1"
name="$2"
fold="$3"
cache_dir="data/processed/person_16_160"
log_dir="experiments/scene_context_v13/outputs/logs"
mkdir -p "$log_dir"

if [ "$(find "$cache_dir" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)" -ne 3441 ]; then
  echo "person_16_160 cache is incomplete" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export TORCH_HOME="$PWD/.tools/torch"

set +e
${HOME}/miniconda3/envs/Train/bin/python -B scripts/train.py \
  --name "$name" \
  --modalities person_ir,person_thermal,scene_depth,scene_ir,skeleton,imu_world \
  --fold "$fold" \
  --seed 42 \
  --epochs 60 \
  --batch-size 6 \
  --gradient-accumulation 2 \
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
  --image-size 160 \
  --backbone x3d_shared_tokens \
  --fusion transformer \
  --kinetics \
  --kinetics-source l \
  --backbone-lr-factor 0.12 \
  --mixup 0.1 \
  --freeze-backbone-bn \
  --ema-decay 0.97 \
  --auxiliary-weight 0.3 \
  2>&1 | tee "$log_dir/$name.log"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$status" > "$log_dir/$name.exit"
exit "$status"
