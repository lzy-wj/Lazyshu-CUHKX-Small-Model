#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU required}"
SOURCE="${2:?source user required}"
shift 2
DATES=("$@")
if [[ ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "V114 full jobs may only use GPUs 4-7" >&2
  exit 2
fi
if (( ${#DATES[@]} == 0 )); then
  echo "At least one target date is required" >&2
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
  "$ROOT/experiments/v114_paired_person_fusion/adapt_test.py" \
  --name "v114_full_source${SOURCE}" \
  --checkpoint "$ROOT/experiments/v24_full_refit/outputs/final/v24_full_warm60_soup_fp32.pt" \
  --pseudo-labels "$ROOT/experiments/v104_domain_gated_adapters/outputs/v104_refreshed_test_pseudo.npz" \
  --source-user "$SOURCE" --target-dates "${DATES[@]}" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160" \
  --epochs 2 --lr 0.0002 --retention-weight 0.25 \
  --stream-distillation-weight 0.15 --batch-size 16 --workers 4 \
  --seed 42 --swanlab-mode cloud
