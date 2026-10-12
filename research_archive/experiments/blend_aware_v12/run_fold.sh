#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_fold.sh METHOD GPU [FOLD]}"
GPU="${2:?usage: run_fold.sh METHOD GPU [FOLD]}"
FOLD="${3:-0}"
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$EXPERIMENT_ROOT/../.." && pwd)"
CONDA="$HOME/miniconda3/bin/conda"
NAME="v12_${METHOD}_f${FOLD}"
LOG_DIR="$EXPERIMENT_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

case "$METHOD" in
  control) EXTRA=() ;;
  contrastive) EXTRA=(--cross-user-contrastive-weight 0.05 --contrastive-temperature 0.10 --contrastive-queue-size 1024 --contrastive-class-cap 32 --contrastive-class-user-cap 8 --contrastive-warmup-items 512) ;;
  blend050) EXTRA=(--blend-teacher "$EXPERIMENT_ROOT/outputs/base_teacher_train_f${FOLD}.npz" --blend-loss-weight 0.50) ;;
  blend100) EXTRA=(--blend-teacher "$EXPERIMENT_ROOT/outputs/base_teacher_train_f${FOLD}.npz" --blend-loss-weight 1.00) ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

set +e
flock "$REPOSITORY_ROOT/outputs/logs/gpu-${GPU}.lock" \
  env CUDA_VISIBLE_DEVICES="$GPU" "$CONDA" run --no-capture-output -n Train \
  python "$REPOSITORY_ROOT/scripts/train_r2plus1d.py" \
  --name "$NAME" \
  --fold "$FOLD" \
  --mode multimodal \
  --initialization scratch \
  --weights "$REPOSITORY_ROOT/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth" \
  --init-visual "$REPOSITORY_ROOT/outputs/checkpoints/r2_depth_ir_32f_f${FOLD}/best.pt" \
  --cache-dir "$REPOSITORY_ROOT/data/processed/r2plus1d_yolo_32_128" \
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
  --output-root "$EXPERIMENT_ROOT/outputs" \
  --swanlab-mode cloud \
  "${EXTRA[@]}" 2>&1 | tee "$LOG_DIR/${NAME}.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$LOG_DIR/${NAME}.exit"
exit "$status"
