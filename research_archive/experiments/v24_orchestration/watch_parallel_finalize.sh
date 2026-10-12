#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GPU="${1:-1}"
CHECKPOINTS="$ROOT/experiments/v24_full_refit/outputs/checkpoints"
REQUIRED=(
  v24_full_noradar_e13
  v24_full_crossuser_e16
  v24_full_scratch_maskdistill_e16
  v24_full_progressive160_e7
  v24_full_warm_maskdistill_e9
)

all_complete() {
  local name status
  for name in "${REQUIRED[@]}"; do
    [[ -f "$CHECKPOINTS/$name/completion.json" ]] || return 1
    status="$($HOME/miniconda3/envs/Train/bin/python -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' \
      "$CHECKPOINTS/$name/completion.json")"
    [[ "$status" == "complete" ]] || return 1
  done
}

gpu_idle() {
  [[ -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')" ]]
}

while ! all_complete; do
  sleep 60
done
while ! gpu_idle; do
  sleep 60
done
cd "$ROOT"
bash experiments/v24_full_refit/run_finalize.sh "$GPU"

