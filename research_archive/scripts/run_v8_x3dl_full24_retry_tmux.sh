#!/usr/bin/env bash
set +e
cd "$(dirname "$0")/.." || exit 1
flock outputs/logs/gpu-2.lock bash scripts/run_v8_x3dl_full24_retry.sh \
  > outputs/logs/v8_x3dl_full24_retry.log 2>&1
status=$?
printf '%s\n' "$status" > outputs/logs/v8_x3dl_full24_retry.exit
exit "$status"
