#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
OUTPUT="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_224"

mkdir -p "$HERE/logs"
cd "$ROOT"
exec "$PYTHON" -u -B scripts/prepare_r2plus1d_cache.py \
  --output "$OUTPUT" \
  --windows-from "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_160/windows.json" \
  --frames 32 --image-size 224 --rgb-modalities Depth_Color,Thermal --workers 12 \
  2>&1 | tee "$HERE/logs/build_cache.log"
