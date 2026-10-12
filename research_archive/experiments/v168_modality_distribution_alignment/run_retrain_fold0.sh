#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v168_modality_distribution_alignment"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
VARIANT="${1:?variant required}"
GPU="${2:?GPU required}"

if [[ ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "V168 may only use GPUs 4-7" >&2
  exit 2
fi

case "$VARIANT" in
  minimum5)
    MINIMUM=5
    ;;
  full6)
    MINIMUM=6
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
if (( USED > 1024 )); then
  echo "GPU $GPU is busy (${USED} MiB)" >&2
  exit 3
fi

mkdir -p "$HERE/logs" "$HERE/outputs"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" "$ROOT/scripts/train_r2plus1d.py" \
  --name "v168_${VARIANT}_retrain_f0" --fold 0 --mode shared_multimodal \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v18_shared5_joint_noradar_f0/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 10 --batch-size 8 --workers 4 \
  --lr 0.00015 --backbone-lr-factor 0.1 --warmup-epochs 1 --patience 10 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
  --label-smoothing 0.03 --mixup-alpha 0.3 \
  --masked-modality-distillation-weight 0.5 \
  --training-minimum-modalities "$MINIMUM" \
  --selection-minimum-modalities "$MINIMUM" \
  --visual-strategy joint --allow-method-combination \
  --output-root "$HERE/outputs" --swanlab-mode cloud
