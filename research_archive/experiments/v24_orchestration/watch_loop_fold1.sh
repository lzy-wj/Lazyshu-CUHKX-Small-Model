#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SELECTION="$ROOT/experiments/v24_iterative_loop/selection.json"
GPU="${1:-1}"

while [[ ! -f "$SELECTION" ]]; do
  sleep 60
done

cd "$ROOT"
bash experiments/v24_iterative_loop/run_stage2_fold1_selected.sh "$GPU"
