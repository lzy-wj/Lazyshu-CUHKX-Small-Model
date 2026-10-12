#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 FOLD GPU [SEED]" >&2
  exit 2
fi

FOLD="$1"
GPU="$2"
SEED="${3:-42}"
ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v197_device_attention_adapter"
NAME="v197_device_attention_s${SEED}_f${FOLD}"
mkdir -p "$HERE/logs" "$HERE/outputs"
CUDA_VISIBLE_DEVICES="$GPU" "$HOME/miniconda3/envs/Train/bin/python" "$ROOT/scripts/train_r2plus1d.py" \
  --name "$NAME" --fold "$FOLD" --mode shared_multimodal --seed "$SEED" \
  --init-model "$ROOT/experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f${FOLD}/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --image-size 128 --epochs 10 --batch-size 24 --workers 4 \
  --lr 0.001 --backbone-lr-factor 0.05 --warmup-epochs 1 --patience 10 \
  --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
  --label-smoothing 0.03 --mixup-alpha 0.3 \
  --masked-modality-distillation-weight 0.5 --visual-strategy joint \
  --imu-representation world_official --imu-device-attention-residual --freeze-inherited \
  --save-epoch-checkpoints --allow-method-combination \
  --output-root "$HERE/outputs" --swanlab-mode cloud \
  > "$HERE/logs/${NAME}.log" 2>&1
