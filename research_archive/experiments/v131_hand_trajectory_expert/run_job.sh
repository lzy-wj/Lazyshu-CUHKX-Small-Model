#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU required}"
FOLD="${2:?fold required}"
if [[ ! "$GPU" =~ ^[67]$ ]]; then
  echo "V131 may only use GPUs 6-7" >&2
  exit 2
fi
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$HOME/miniconda3/envs/Train/bin/python" -B \
  "$ROOT/experiments/v131_hand_trajectory_expert/train.py" \
  --fold "$FOLD" --epochs 40 --batch-size 128 --workers 4 --swanlab-mode cloud
