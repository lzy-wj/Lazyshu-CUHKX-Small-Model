#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU required}"
FOLD="${2:?fold required}"
VARIANT="${3:?variant required}"
if [[ ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "V126 may only use GPUs 4-7" >&2
  exit 2
fi
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi
case "$VARIANT" in
  full) MODALITIES="person_depth,person_thermal,person_ir,skeleton,imu_world" ;;
  visual) MODALITIES="person_depth,person_thermal,person_ir" ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 4 ;;
esac
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
export TORCH_HOME="$ROOT/.tools/torch"
exec "$HOME/miniconda3/envs/Train/bin/python" -B "$ROOT/scripts/train.py" \
  --name "v126_mvit_${VARIANT}_f${FOLD}" \
  --modalities "$MODALITIES" --fold "$FOLD" --seed 42 \
  --epochs 24 --batch-size 8 --gradient-accumulation 2 \
  --lr 0.0003 --backbone-lr-factor 0.05 --schedule-epochs 24 --warmup-epochs 2 \
  --workers 6 --frames 16 --image-size 160 --eval-every 2 --patience 8 \
  --backbone mvit_v2_s_shared --fusion transformer --kinetics \
  --strong-augment --synchronized-temporal-augment \
  --mixup 0.2 --ema-decay 0.98 --auxiliary-weight 0.15 --swanlab-mode online
