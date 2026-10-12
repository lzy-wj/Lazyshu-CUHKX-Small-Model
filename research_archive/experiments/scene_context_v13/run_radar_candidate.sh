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

set +e
${HOME}/miniconda3/envs/Train/bin/python -B \
  experiments/scene_context_v13/train_radar_pointnet.py \
  --name "$name" \
  --fold "$fold" \
  --epochs 80 \
  --batch-size 64 \
  --workers 6 \
  --lr 0.0008 \
  --seed 42 \
  --patience 20 \
  --swanlab-mode online \
  2>&1 | tee "$log_dir/$name.log"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$status" > "$log_dir/$name.exit"
exit "$status"
