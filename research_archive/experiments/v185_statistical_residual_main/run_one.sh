#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FOLD GPU VARIANT" >&2
  exit 2
fi

FOLD="$1"
GPU="$2"
VARIANT="$3"
ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v185_statistical_residual_main"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
EXTRA=()
case "$VARIANT" in
  standard) EPOCHS=7; LR=0.00010 ;;
  frozen_visual) EPOCHS=10; LR=0.00030; EXTRA+=(--freeze-visual) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

NAME="v185_statistics_residual_${VARIANT}_f${FOLD}"
mkdir -p "$HERE/logs" "$HERE/outputs"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/scripts/train_r2plus1d.py" \
  --name "$NAME" --fold "$FOLD" --mode shared_multimodal \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f${FOLD}/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs "$EPOCHS" --batch-size 12 --workers 4 \
  --lr "$LR" --backbone-lr-factor 0.05 --warmup-epochs 1 --patience "$EPOCHS" \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
  --label-smoothing 0.03 --mixup-alpha 0.3 \
  --masked-modality-distillation-weight 0.5 --visual-strategy joint \
  --imu-representation world_official --imu-statistics-residual \
  --allow-method-combination "${EXTRA[@]}" \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  > "$HERE/logs/${NAME}.log" 2>&1
