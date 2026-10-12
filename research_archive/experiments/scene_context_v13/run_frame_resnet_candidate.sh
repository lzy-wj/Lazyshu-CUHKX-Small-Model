#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 GPU NAME FOLD [MODALITIES]" >&2
  exit 2
fi

gpu="$1"
name="$2"
fold="$3"
modalities="${4:-scene_depth,scene_ir,thermal}"
log_dir="experiments/scene_context_v13/outputs/logs"
mkdir -p "$log_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export TORCH_HOME="$PWD/.tools/torch"

set +e
${HOME}/miniconda3/envs/Train/bin/python -B \
  experiments/scene_context_v13/train_frame_resnet.py \
  --name "$name" \
  --fold "$fold" \
  --modalities "$modalities" \
  --epochs 30 \
  --batch-size 4 \
  --workers 6 \
  --lr 0.0002 \
  --frames 16 \
  --image-size 160 \
  --patience 10 \
  --swanlab-mode online \
  2>&1 | tee "$log_dir/$name.log"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$status" > "$log_dir/$name.exit"
exit "$status"
