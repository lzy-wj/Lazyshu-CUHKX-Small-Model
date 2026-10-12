#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU index required}"
FOLD="${2:?fold required}"
if [[ ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "V110 may only use GPUs 4-7" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$HOME/miniconda3/envs/Train/bin/python" -B \
  "$ROOT/experiments/v110_object_token_transformer/train.py" \
  --fold "$FOLD" --device "$GPU" --epochs 60 --batch-size 64 --workers 4 \
  --seed 42 --swanlab-mode cloud
