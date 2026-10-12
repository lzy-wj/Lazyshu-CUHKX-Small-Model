#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GPU="${1:-1}"
READY="$ROOT/experiments/v24_full_refit/outputs/reports/kaggle_submissions_after_v24.csv"
mkdir -p "$HERE/logs"

gpu_launch_allowed() {
  [[ -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')" ]] || return 1
  local hour busy
  hour=$((10#$(TZ=Asia/Shanghai date +%H)))
  if (( hour >= 8 && hour < 18 )); then
    busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | sed '/^$/d' | wc -l)"
    (( 8 - busy >= 5 )) || return 1
  fi
}

while [[ ! -f "$READY" ]] || tmux has-session -t v24-pseudo-val-watch 2>/dev/null; do
  sleep 60
done
while ! gpu_launch_allowed; do
  sleep 60
done

cd "$ROOT"
bash experiments/v25_teacher_distillation/run_validation.sh "$GPU" \
  2>&1 | tee "$HERE/logs/validation.log"

