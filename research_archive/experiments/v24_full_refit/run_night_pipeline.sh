#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_BASE="${1:-0}"
GPU_CROSS="${2:-1}"
GPU_SCRATCH="${3:-3}"
GPU_PROGRESSIVE="${4:-4}"
GPU_WARM="${5:-5}"

hour=$((10#$(TZ=Asia/Shanghai date +%H)))
if (( hour >= 8 && hour < 18 )); then
  echo "This orchestrator is night-only; current Asia/Shanghai hour is $hour." >&2
  exit 1
fi

if [[ "$(printf '%s\n' "$GPU_BASE" "$GPU_CROSS" "$GPU_SCRATCH" "$GPU_PROGRESSIVE" "$GPU_WARM" | sort -u | wc -l)" -ne 5 ]]; then
  echo "Five distinct GPU indices are required." >&2
  exit 1
fi

"$DIR/run_full_no_radar.sh" "$GPU_BASE" & base_pid=$!
"$DIR/run_full_cross_user.sh" "$GPU_CROSS" & cross_pid=$!
"$DIR/run_full_scratch_maskdistill.sh" "$GPU_SCRATCH" & scratch_pid=$!

if ! wait "$base_pid"; then
  wait "$cross_pid" || true
  wait "$scratch_pid" || true
  echo "Base training failed; dependent warm-start runs were not launched." >&2
  exit 1
fi

"$DIR/run_full_progressive160.sh" "$GPU_PROGRESSIVE" & progressive_pid=$!
"$DIR/run_full_warm_maskdistill.sh" "$GPU_WARM" & warm_pid=$!

status=0
wait "$cross_pid" || status=1
wait "$scratch_pid" || status=1
wait "$progressive_pid" || status=1
wait "$warm_pid" || status=1
exit "$status"
