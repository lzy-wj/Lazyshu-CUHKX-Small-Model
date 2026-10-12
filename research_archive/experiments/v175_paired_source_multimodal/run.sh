#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v175_paired_source_multimodal"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
FOLD="${1:?fold required}"
TARGET="${2:?target user required}"
GPU="${3:?GPU required}"

case "$FOLD:$TARGET" in
  0:8|0:18|1:7|1:17) ;;
  *) echo "Expected 0:8, 0:18, 1:7, or 1:17" >&2; exit 2 ;;
esac
if [[ ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "V175 may only use GPUs 4-7" >&2
  exit 2
fi
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi

mkdir -p "$HERE/logs" "$HERE/outputs"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" "$ROOT/scripts/train_r2plus1d.py" \
  --name "v175_user${TARGET}_f${FOLD}" --fold "$FOLD" \
  --validation-users "$TARGET" --mode shared_multimodal \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f${FOLD}/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 5 --batch-size 12 --workers 4 \
  --lr 0.00005 --backbone-lr-factor 0.02 --warmup-epochs 0.5 --patience 5 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
  --label-smoothing 0.03 --mixup-alpha 0.3 \
  --masked-modality-distillation-weight 0.5 \
  --visual-strategy joint --allow-method-combination \
  --output-root "$HERE/outputs" --swanlab-mode cloud
