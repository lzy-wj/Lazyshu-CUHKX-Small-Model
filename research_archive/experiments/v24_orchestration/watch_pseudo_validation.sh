#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
READY="$ROOT/experiments/v24_full_refit/outputs/reports/kaggle_submissions_after_v24.csv"
GPU="${1:-1}"

gpu_launch_allowed() {
  [[ -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')" ]] || return 1
  local hour busy
  hour=$((10#$(TZ=Asia/Shanghai date +%H)))
  if (( hour >= 8 && hour < 18 )); then
    busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | sed '/^$/d' | wc -l)"
    (( 8 - busy >= 5 )) || return 1
  fi
}

while [[ ! -f "$READY" ]]; do
  sleep 60
done
while ! gpu_launch_allowed; do
  sleep 60
done
cd "$ROOT"
bash experiments/v24_pseudo_validation/run.sh "$GPU"
