#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_shared6_ablation.sh METHOD GPU FOLD}"
GPU="${2:?usage: run_shared6_ablation.sh METHOD GPU FOLD}"
FOLD="${3:?usage: run_shared6_ablation.sh METHOD GPU FOLD}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="$ROOT/experiments/scene_context_v13/outputs"
NAME="v14_shared6_${METHOD}_f${FOLD}"
EXTRA=()

case "$METHOD" in
  mixup)
    EXTRA+=(--mixup-alpha 0.3)
    ;;
  mixup_moddrop)
    EXTRA+=(--mixup-alpha 0.3 --structured-modality-dropout 0.1 --allow-method-combination)
    ;;
  mixup_speed)
    EXTRA+=(--mixup-alpha 0.3 --temporal-scale-jitter 0.15 --allow-method-combination)
    ;;
  mixup_speed_resume)
    EXTRA+=(
      --mixup-alpha 0.3
      --temporal-scale-jitter 0.15
      --allow-method-combination
      --init-model "$OUTPUT_ROOT/checkpoints/v14_shared6_mixup_speed_f${FOLD}/epochs/epoch_09.pt"
      --epochs 11
      --lr 0.00018
      --warmup-epochs 0
    )
    ;;
  mixup_temporal)
    EXTRA+=(--mixup-alpha 0.3 --use-temporal-embedding --allow-method-combination)
    ;;
  mixup_stream_gate)
    EXTRA+=(--mixup-alpha 0.3 --fusion-output stream_gated)
    ;;
  *)
    echo "unknown method: $METHOD" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT/logs"
export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="$HOME/miniconda3/envs/Train/bin:$PATH"

python -B "$ROOT/scripts/train_r2plus1d.py" \
  --name "$NAME" \
  --fold "$FOLD" \
  --mode shared_multimodal \
  --initialization kinetics \
  --weights "$ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 \
  --image-size 128 \
  --epochs 20 \
  --batch-size 8 \
  --workers 4 \
  --lr 0.0003 \
  --backbone-lr-factor 0.1 \
  --weight-decay 0.0001 \
  --warmup-epochs 2 \
  --ema-decay 0.99 \
  --dropout 0.3 \
  --auxiliary-weight 0.15 \
  --label-smoothing 0.03 \
  --class-balance-power 0 \
  --seed 42 \
  --patience 20 \
  --eval-every 1 \
  --save-epoch-checkpoints \
  --use-radar \
  --output-root "$OUTPUT_ROOT" \
  --swanlab-mode cloud \
  "${EXTRA[@]}"
