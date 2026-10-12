#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU required}"
FOLD="${2:?fold required}"
if [[ ! "$GPU" =~ ^[67]$ ]]; then
  echo "V133 may only use GPUs 6-7" >&2
  exit 2
fi
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
export TORCH_HOME="$ROOT/.tools/torch"
exec "$HOME/miniconda3/envs/Train/bin/python" -B "$ROOT/scripts/train.py" \
  --name "v133_shared_swin3d_f${FOLD}" \
  --modalities person_depth,person_thermal,person_ir,skeleton,imu_world \
  --fold "$FOLD" --seed 42 --epochs 24 --batch-size 8 --gradient-accumulation 2 \
  --lr 0.0003 --backbone-lr-factor 0.1 --schedule-epochs 24 --warmup-epochs 2 \
  --workers 6 --frames 16 --image-size 160 --eval-every 2 --patience 8 \
  --backbone swin3d_t_shared --fusion transformer --kinetics \
  --strong-augment --synchronized-temporal-augment \
  --mixup 0.15 --ema-decay 0.98 --auxiliary-weight 0.15 --swanlab-mode online
