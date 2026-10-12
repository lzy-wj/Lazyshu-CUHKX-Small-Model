#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
TRAIN="$ROOT/scripts/train_r2plus1d.py"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
SOURCE="$ROOT/experiments/scene_context_v13/outputs/checkpoints"
OUTPUT="$ROOT/experiments/v136_retained_label_experts/outputs"
LOGS="$ROOT/experiments/v136_retained_label_experts/logs"
mkdir -p "$OUTPUT" "$LOGS"
cd "$ROOT"

DOCUMENTS="17,18,21,22"
DEVICES="19,20,23,24,25,26,27,37,38,39"
declare -a pids=()

run() {
  local gpu="$1" family="$2" fold="$3" labels="$4"
  local name="v136_${family}_f${fold}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAIN" \
    --name "$name" --fold "$fold" --mode shared_multimodal \
    --init-model "$SOURCE/v23_shared5_joint_warm_maskdistill_f${fold}/best.pt" \
    --cache-dir "$CACHE" --frames 32 --image-size 128 \
    --epochs 5 --batch-size 8 --workers 4 --lr 0.00008 \
    --backbone-lr-factor 0.02 --warmup-epochs 1 --patience 5 \
    --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
    --label-smoothing 0.03 --mixup-alpha 0.2 --visual-strategy joint \
    --selection-labels "$labels" --focus-labels "$labels" --focus-weight 4 \
    --output-root "$OUTPUT" --swanlab-mode cloud \
    > "$LOGS/${name}.log" 2>&1 &
  pids+=("$!")
}

run 4 documents 0 "$DOCUMENTS"
run 5 documents 1 "$DOCUMENTS"
run 6 devices 0 "$DEVICES"
run 7 devices 1 "$DEVICES"

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
