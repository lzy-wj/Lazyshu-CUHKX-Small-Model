#!/usr/bin/env bash
set -euo pipefail

SHIFT_TOKENS="${1:?usage: run_fold0.sh SHIFT_TOKENS}"
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "$EXPERIMENT_ROOT/../.." && pwd)"
CONDA="$HOME/miniconda3/bin/conda"
NAME="ablation_async_shift_f0_p${SHIFT_TOKENS}"
LOG_DIR="$EXPERIMENT_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

flock "$REPOSITORY_ROOT/outputs/logs/gpu-2.lock" bash -c '
set -o pipefail
export CUDA_VISIBLE_DEVICES=2
"$1" run --no-capture-output -n Train python "$2/scripts/train_r2plus1d.py" \
  --async-shift-tokens "$3" \
  --async-shift-seed 424242 \
  --name "$4" \
  --fold 0 \
  --mode multimodal \
  --initialization scratch \
  --weights "$5/.tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth" \
  --init-visual "$5/outputs/checkpoints/r2_depth_ir_32f_f0/best.pt" \
  --cache-dir "$5/data/processed/r2plus1d_yolo_32_128" \
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
  --patience 8 \
  --eval-every 1 \
  --swanlab-mode cloud 2>&1 | tee "$2/outputs/logs/$4.log"
status=${PIPESTATUS[0]}
printf "%s\n" "$status" > "$2/outputs/logs/$4.exit"
exit "$status"
' bash "$CONDA" "$EXPERIMENT_ROOT" "$SHIFT_TOKENS" "$NAME" "$REPOSITORY_ROOT"
