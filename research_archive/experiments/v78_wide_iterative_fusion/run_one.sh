#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU required}"
FOLD="${2:?fold required}"
K="${3:?K required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
NAME="v78_wide768_d4_k${K}_f${FOLD}"
LOG="$HERE/outputs/logs/${NAME}.log"

[[ "$GPU" =~ ^[4-7]$ ]] || { echo "V78 may use only physical GPUs 4-7" >&2; exit 2; }
[[ "$FOLD" =~ ^[01]$ && "$K" =~ ^[12]$ ]] || { echo "fold and K must be 0/1 and 1/2" >&2; exit 2; }
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1==gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$UUID"; then
  echo "GPU $GPU is busy" >&2
  exit 3
fi
mkdir -p "$HERE/outputs/logs"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u "$HERE/train.py" \
  --name "$NAME" --fold "$FOLD" --iterations "$K" \
  --init-checkpoint "$ROOT/experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f${FOLD}.pt" \
  --wide-dimension 768 --wide-depth 4 --wide-heads 12 --feedforward-multiplier 3 \
  --epochs 10 --batch-size 8 --workers 4 --lr 0.0002 --base-lr-factor 0.1 \
  --weight-decay 0.0001 --warmup-epochs 1 --ema-decay 0.99 \
  --auxiliary-weight 0.15 --deep-supervision-decay 0.6 \
  --feedback-temperature 2.0 --label-smoothing 0.03 --mixup-alpha 0.3 \
  --seed 42 --patience 10 --output-root "$HERE/outputs" --swanlab-mode cloud \
  2>&1 | tee "$LOG"

