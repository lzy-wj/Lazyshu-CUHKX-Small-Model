#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v159_ordered_source_alignment"
mkdir -p "$HERE/logs" "$HERE/outputs"
exec "$HOME/miniconda3/envs/Train/bin/python" "$HERE/evaluate.py" \
  > "$HERE/logs/evaluate.log" 2>&1
