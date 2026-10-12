#!/usr/bin/env bash
set +e
cd "$(dirname "$0")/.." || exit 1
flock outputs/logs/gpu-3.lock bash scripts/run_v8_x3dl_full24_gpu3.sh \
  > outputs/logs/v8_x3dl_full24_gpu3.log 2>&1
status=$?
printf '%s\n' "$status" > outputs/logs/v8_x3dl_full24_gpu3.exit
exit "$status"
