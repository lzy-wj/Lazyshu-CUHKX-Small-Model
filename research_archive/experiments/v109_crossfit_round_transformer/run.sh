#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
for FOLD in 0 1; do
  "$PYTHON" -B "$ROOT/experiments/v109_crossfit_round_transformer/train.py" \
    --fold "$FOLD" --steps 600 --seed 42 --swanlab-mode cloud \
    2>&1 | tee "$ROOT/experiments/v109_crossfit_round_transformer/fold${FOLD}.log"
done
