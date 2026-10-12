#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU required}"
VARIANT="${2:?variant required}"
if [[ ! "$GPU" =~ ^[45]$ ]]; then
  echo "V129 may only use GPUs 4-5" >&2
  exit 2
fi
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi
case "$VARIANT" in
  conservative)
    NAME="v129_mvit_full_conservative"
    LR="0.0003"
    FACTOR="0.05"
    MIXUP="0.2"
    HORIZON="24"
    ;;
  domainadapt)
    NAME="v129_mvit_full_domainadapt"
    LR="0.0004"
    FACTOR="0.15"
    MIXUP="0.1"
    HORIZON="18"
    ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 4 ;;
esac
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
export TORCH_HOME="$ROOT/.tools/torch"
exec "$HOME/miniconda3/envs/Train/bin/python" -B "$ROOT/scripts/train.py" \
  --name "$NAME" \
  --modalities person_depth,person_thermal,person_ir,skeleton,imu_world \
  --fold -1 --seed 42 --epochs 14 --batch-size 8 --gradient-accumulation 2 \
  --lr "$LR" --backbone-lr-factor "$FACTOR" --schedule-epochs "$HORIZON" --warmup-epochs 2 \
  --workers 6 --frames 16 --image-size 160 --eval-every 2 --patience 6 \
  --backbone mvit_v2_s_shared --fusion transformer --kinetics \
  --strong-augment --synchronized-temporal-augment \
  --mixup "$MIXUP" --ema-decay 0.98 --auxiliary-weight 0.15 --swanlab-mode online
