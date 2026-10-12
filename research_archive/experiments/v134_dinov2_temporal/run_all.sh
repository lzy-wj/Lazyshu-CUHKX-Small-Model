#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
cd "$ROOT"

if [[ ! -f experiments/v134_dinov2_temporal/features/manifest.json ]]; then
  CUDA_VISIBLE_DEVICES=4 TORCH_HOME="$ROOT/.tools/torch" "$PYTHON" \
    experiments/v134_dinov2_temporal/cache_features.py --device cuda \
    > experiments/v134_dinov2_temporal/cache.log 2>&1
fi

mkdir -p experiments/v134_dinov2_temporal/logs
declare -a pids=()
run() {
  local gpu="$1" name="$2" fold="$3" contrast="$4" mixup="$5"
  CUDA_VISIBLE_DEVICES="$gpu" TORCH_HOME="$ROOT/.tools/torch" "$PYTHON" \
    experiments/v134_dinov2_temporal/train.py \
    --name "$name" --fold "$fold" --contrastive-weight "$contrast" --mixup "$mixup" \
    > "experiments/v134_dinov2_temporal/logs/${name}.log" 2>&1 &
  pids+=("$!")
}

run 4 v134_dino_ce_f0 0 0.0 0.2
run 5 v134_dino_ce_f1 1 0.0 0.2
run 6 v134_dino_xuser_f0 0 0.05 0.0
run 7 v134_dino_xuser_f1 1 0.05 0.0

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
