#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
TRAIN="$ROOT/scripts/train_r2plus1d.py"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
SOURCE="$ROOT/experiments/scene_context_v13/outputs/checkpoints"
OUTPUT="$ROOT/experiments/v135_end_to_end_label_experts/outputs"
LOGS="$ROOT/experiments/v135_end_to_end_label_experts/logs"
mkdir -p "$OUTPUT" "$LOGS"
cd "$ROOT"

DESK="17,18,19,20,21,22,23,24,25,26,27"
INTERACTION="0,1,2,4,6,7,8,9,10,11,14,17,18,19,20,21,22,23,24,26,27,37,38,39"
declare -a pids=()

run() {
  local gpu="$1" family="$2" fold="$3" labels="$4"
  local name="v135_${family}_f${fold}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$TRAIN" \
    --name "$name" --fold "$fold" --mode shared_multimodal \
    --init-model "$SOURCE/v23_shared5_joint_warm_maskdistill_f${fold}/best.pt" \
    --cache-dir "$CACHE" --frames 32 --image-size 128 \
    --epochs 8 --batch-size 8 --workers 4 --lr 0.00012 \
    --backbone-lr-factor 0.02 --warmup-epochs 1 --patience 8 \
    --ema-decay 0.99 --dropout 0.3 --auxiliary-weight 0.15 \
    --label-smoothing 0.03 --mixup-alpha 0.2 --visual-strategy joint \
    --training-labels "$labels" --selection-labels "$labels" \
    --output-root "$OUTPUT" --swanlab-mode cloud \
    > "$LOGS/${name}.log" 2>&1 &
  pids+=("$!")
}

run 4 desk 0 "$DESK"
run 5 desk 1 "$DESK"
run 6 interaction 0 "$INTERACTION"
run 7 interaction 1 "$INTERACTION"

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
