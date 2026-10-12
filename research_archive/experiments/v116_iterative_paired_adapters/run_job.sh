#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU required}"
FOLD="${2:?fold required}"
SOURCE="${3:?source user required}"
TARGET="${4:?target user required}"
if [[ ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "V116 may only use GPUs 4-7" >&2
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
  "$ROOT/experiments/v113_paired_person_adapters/adapt.py" \
  --name "v116_iterative_s${SOURCE}_t${TARGET}_f${FOLD}" \
  --output-root "$ROOT/experiments/v116_iterative_paired_adapters/outputs" \
  --checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f${FOLD}.pt" \
  --pseudo-labels "$ROOT/experiments/v116_iterative_paired_adapters/outputs/v116_v114_pseudo_f${FOLD}.npz" \
  --fold "$FOLD" --source-user "$SOURCE" --target-user "$TARGET" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --epochs 2 --lr 0.0002 --confidence-threshold 0.6 \
  --retention-weight 0.25 --stream-distillation-weight 0.15 \
  --batch-size 16 --workers 4 --frames 32 --image-size 160 \
  --tta-crop-scale 0.75 --seed 42 --swanlab-mode cloud
