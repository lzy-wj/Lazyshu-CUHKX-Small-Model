#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
V24="$ROOT/experiments/v24_full_refit/outputs"
CHECKPOINT="$V24/final/v24_full_warm60_soup_int8.pt"
YOLO="$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"
ADAPTER="$V24/final/v26_system_distill_head_delta.pt"
ADAPTED="$V24/predictions/v26_system_distill_adapt.npz"
CROP="$V24/predictions/v29_crop0875.npz"
FUSED="$V24/predictions/v29_crop_system_fusion.npz"
DECODED="$V24/predictions/v29_crop_system_fusion_pair_aware.npz"
SUBMISSION="$V24/submissions/v29_crop_system_fusion_pair_aware.csv"
REPORT="$V24/reports/v29_crop_system_fusion_pair_aware.json"
MANIFEST="$V24/reports/v29_crop_system_fusion_manifest.json"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"

for path in "$CHECKPOINT" "$YOLO" "$ADAPTER" "$ADAPTED"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/experiments/v24_full_refit/predict_test.py" \
  --checkpoint "$CHECKPOINT" --yolo "$YOLO" --cache-dir "$CACHE" --frames 32 \
  --tta-hflip --tta-crop-scale 0.875 --batch-size 16 --workers 4 --adabn none \
  --output "$CROP" --submission "$V24/submissions/v29_crop0875_raw.csv" \
  --manifest "$V24/reports/v29_crop0875_predict.json"
"$PYTHON" -B "$HERE/blend_archives.py" \
  --adapted "$ADAPTED" --crop "$CROP" --adapted-weight 0.4 --output "$FUSED"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$FUSED" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT"

"$PYTHON" - "$CHECKPOINT" "$YOLO" "$ADAPTER" "$DECODED" "$SUBMISSION" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

checkpoint, yolo, adapter, decoded, submission, manifest = map(Path, sys.argv[1:])
assets = sum(path.stat().st_size for path in (checkpoint, yolo, adapter))
if assets >= 100_000_000:
    raise RuntimeError(f"V29 assets exceed 100 MB: {assets}")
archive = np.load(decoded)
table = pd.read_csv(submission)
if len(archive["indices"]) != 405 or len(table) != 405:
    raise RuntimeError("V29 output does not cover 405 test rows")
if table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V29 submission validation failed")
payload = {
    "checkpoint": str(checkpoint),
    "adapter": str(adapter),
    "yolo": str(yolo),
    "model_asset_bytes": assets,
    "slack_bytes": 100_000_000 - assets,
    "local_direct_pairaware": {"correct": 1177, "rows": 1357},
    "fusion": {"v26_system_distill_crop075": 0.4, "v24_crop0875": 0.6},
    "inference": "Two forwards of one backbone family plus one small head adapter",
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
