#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
mkdir -p "$HERE/logs"
cd "$ROOT"
exec "$PYTHON" -u -B scripts/prepare_r2plus1d_cache.py \
  --output "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_192" \
  --windows-from "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160/windows.json" \
  --frames 32 --image-size 192 --rgb-modalities Depth_Color,Thermal --workers 12 \
  2>&1 | tee "$HERE/logs/build_cache.log"
