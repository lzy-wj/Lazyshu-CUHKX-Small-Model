#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_COMPLETION="$ROOT/experiments/v24_full_refit/outputs/checkpoints/v24_full_noradar_e13/completion.json"
GPU_CROSS="${1:-3}"
GPU_SCRATCH="${2:-4}"
GPU_PROGRESSIVE="${3:-5}"
GPU_WARM="${4:-7}"
GPU_FINALIZE="${5:-0}"

gpu_idle() {
  [[ -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')" ]]
}

while [[ ! -f "$BASE_COMPLETION" ]]; do
  sleep 60
done

while true; do
  hour=$((10#$(TZ=Asia/Shanghai date +%H)))
  if (( hour < 8 || hour >= 18 )); then
    break
  fi
  sleep 300
done

for gpu in "$GPU_CROSS" "$GPU_SCRATCH" "$GPU_PROGRESSIVE" "$GPU_WARM"; do
  while ! gpu_idle "$gpu"; do
    sleep 60
  done
done

cd "$ROOT"
pids=()

launch_if_missing() {
  local name="$1"
  local script="$2"
  local gpu="$3"
  local completion="$ROOT/experiments/v24_full_refit/outputs/checkpoints/$name/completion.json"
  local destination="$ROOT/experiments/v24_full_refit/outputs/checkpoints/$name"
  if [[ -f "$completion" ]]; then
    echo "Already complete: $name"
    return
  fi
  if [[ -e "$destination" ]]; then
    echo "Incomplete destination blocks automatic resume: $destination" >&2
    exit 1
  fi
  bash "$script" "$gpu" &
  pids+=("$!")
}

launch_if_missing v24_full_crossuser_e16 experiments/v24_full_refit/run_full_cross_user.sh "$GPU_CROSS"
launch_if_missing v24_full_scratch_maskdistill_e16 experiments/v24_full_refit/run_full_scratch_maskdistill.sh "$GPU_SCRATCH"
launch_if_missing v24_full_progressive160_e7 experiments/v24_full_refit/run_full_progressive160.sh "$GPU_PROGRESSIVE"
launch_if_missing v24_full_warm_maskdistill_e9 experiments/v24_full_refit/run_full_warm_maskdistill.sh "$GPU_WARM"

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"

while ! gpu_idle "$GPU_FINALIZE"; do
  sleep 60
done
bash experiments/v24_full_refit/run_finalize.sh "$GPU_FINALIZE"
