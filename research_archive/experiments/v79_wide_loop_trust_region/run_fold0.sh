#!/usr/bin/env bash
set -euo pipefail

GPU_K1="${1:-4}"
GPU_K2="${2:-5}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ "$GPU_K1" != "$GPU_K2" ]] || { echo "Distinct GPUs are required" >&2; exit 2; }
mkdir -p "$HERE/outputs"
bash "$HERE/run_one.sh" "$GPU_K1" 1 > "$HERE/outputs/k1_launcher.log" 2>&1 &
PID_K1=$!
bash "$HERE/run_one.sh" "$GPU_K2" 2 > "$HERE/outputs/k2_launcher.log" 2>&1 &
PID_K2=$!
STATUS=0
wait "$PID_K1" || STATUS=1
wait "$PID_K2" || STATUS=1
exit "$STATUS"

