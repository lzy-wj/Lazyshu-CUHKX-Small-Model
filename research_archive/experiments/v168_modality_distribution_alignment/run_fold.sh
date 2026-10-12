#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v168_modality_distribution_alignment"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
VARIANT="${1:?variant required}"
FOLD="${2:?fold required}"
GPU="${3:?GPU required}"

if [[ ! "$FOLD" =~ ^[01]$ || ! "$GPU" =~ ^[4-7]$ ]]; then
  echo "Expected fold 0/1 and GPU 4-7" >&2
  exit 2
fi
case "$VARIANT" in
  minimum5)
    EXTRA=(--training-minimum-modalities 5 --selection-minimum-modalities 5 --masked-modality-distillation-weight 0.5)
    ;;
  full6)
    EXTRA=(--training-minimum-modalities 6 --selection-minimum-modalities 6 --masked-modality-distillation-weight 0.5)
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
  --name "v168_${VARIANT}_f${FOLD}" --fold "$FOLD" --mode shared_multimodal \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f${FOLD}/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 7 --batch-size 12 --workers 4 \
  --lr 0.00010 --backbone-lr-factor 0.05 --warmup-epochs 1 --patience 7 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
  --label-smoothing 0.03 --mixup-alpha 0.3 \
  --visual-strategy joint --allow-method-combination \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  "${EXTRA[@]}"
