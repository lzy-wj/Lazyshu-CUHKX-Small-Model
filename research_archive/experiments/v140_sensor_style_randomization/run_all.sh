#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
TRAIN="$ROOT/scripts/train_r2plus1d.py"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
SOURCE="$ROOT/experiments/scene_context_v13/outputs/checkpoints"
OUTPUT="$ROOT/experiments/v140_sensor_style_randomization/outputs"
LOGS="$ROOT/experiments/v140_sensor_style_randomization/logs"
mkdir -p "$OUTPUT" "$LOGS"
cd "$ROOT"

declare -a pids=()
run() {
  local gpu="$1" variant="$2" fold="$3" jitter="$4"
  local name="v140_${variant}_f${fold}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAIN" \
    --name "$name" --fold "$fold" --mode shared_multimodal \
    --init-model "$SOURCE/v23_shared5_joint_warm_maskdistill_f${fold}/best.pt" \
    --cache-dir "$CACHE" --frames 32 --image-size 128 \
    --epochs 5 --batch-size 8 --workers 4 --lr 0.00010 \
    --backbone-lr-factor 0.05 --warmup-epochs 1 --patience 5 \
    --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
    --label-smoothing 0.03 --mixup-alpha 0.3 \
    --masked-modality-distillation-weight 0.5 \
    --photometric-gamma 0.35 --photometric-jitter "$jitter" \
    --visual-strategy joint --allow-method-combination \
    --output-root "$OUTPUT" --swanlab-mode cloud \
    > "$LOGS/${name}.log" 2>&1 &
  pids+=("$!")
}

run 4 gamma 0 0.0
run 5 gamma 1 0.0
run 6 style 0 0.1
run 7 style 1 0.1

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
