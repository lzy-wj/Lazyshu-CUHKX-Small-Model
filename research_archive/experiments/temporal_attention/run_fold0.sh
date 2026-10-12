#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="$ROOT/experiments/temporal_attention"
LOCK="$ROOT/outputs/logs/gpu-3.lock"
LOGS="$EXPERIMENT/outputs/logs"
mkdir -p "$LOGS"

run_gpu() {
  local log="$1"
  shift
  flock "$LOCK" bash -c '
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=3
    cd "$1"
    shift
    exec "$@"
  ' bash "$ROOT" "$@" 2>&1 | tee "$log"
}

run_gpu "$LOGS/ablation_temporal_attention_32f_f0.log" \
  conda run --no-capture-output -n Train python "$EXPERIMENT/train.py" \
  --name ablation_temporal_attention_32f_f0 \
  --fold 0 \
  --init-visual "$ROOT/outputs/checkpoints/r2_depth_ir_32f_f0/best.pt" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_32_128" \
  --frames 32 \
  --batch-size 8 \
  --workers 4 \
  --epochs 20 \
  --lr 0.0003 \
  --backbone-lr-factor 0.1 \
  --weight-decay 0.0001 \
  --warmup-epochs 2 \
  --ema-decay 0.99 \
  --dropout 0.3 \
  --auxiliary-weight 0.15 \
  --label-smoothing 0.03 \
  --seed 42 \
  --patience 8 \
  --swanlab-mode cloud

conda run --no-capture-output -n Train python "$EXPERIMENT/interpolate.py" \
  --left "$ROOT/outputs/checkpoints/r2_staged_multimodal_f0/best.pt" \
  --right "$EXPERIMENT/outputs/checkpoints/ablation_temporal_attention_32f_f0/best.pt" \
  --right-weight 0.75 \
  --output "$EXPERIMENT/outputs/checkpoints/ablation_temporal_attention_soup_f0_w075/best_int8.pt"

run_gpu "$LOGS/ablation_temporal_attention_evaluate_f0.log" \
  conda run --no-capture-output -n Train python "$EXPERIMENT/evaluate.py" \
  --checkpoint "$EXPERIMENT/outputs/checkpoints/ablation_temporal_attention_soup_f0_w075/best_int8.pt" \
  --output "$EXPERIMENT/outputs/predictions/fold0_candidate.npz" \
  --batch-size 8 \
  --workers 4 \
  --tta-hflip \
  --tta-crop-scale 0.75

conda run --no-capture-output -n Train python "$ROOT/scripts/evaluate_single_ablation.py" \
  --candidate "$EXPERIMENT/outputs/predictions/fold0_candidate.npz" \
  --fold 0 \
  --output "$EXPERIMENT/outputs/reports/fold0_unified_metrics.json"
