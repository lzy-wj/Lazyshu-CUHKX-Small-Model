#!/usr/bin/env bash
set -euo pipefail

ROOT=${LAZYSHU_PROJECT_ROOT}
HERE="$ROOT/experiments/v153_hierarchical_decoder"
mkdir -p "$HERE/logs" "$HERE/outputs"
exec "$HOME/miniconda3/envs/Train/bin/python" "$HERE/evaluate.py" \
  > "$HERE/logs/evaluate.log" 2>&1
