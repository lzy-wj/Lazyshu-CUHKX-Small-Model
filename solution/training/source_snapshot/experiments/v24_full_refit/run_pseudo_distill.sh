#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

GPU="${1:?Usage: $0 GPU}"
CHECKPOINT="$OUTPUT_ROOT/final/v24_full_warm60_soup_int8.pt"
YOLO="$ROOT/.tools/checkpoints/yolo11n.pt"
PSEUDO_LABELS="$ROOT/experiments/scene_context_v13/outputs/predictions/v14_test_pseudo_labels.npz"
BASE="$OUTPUT_ROOT/predictions/v24_full_warm60_int8_tta.npz"
ADAPTED="$OUTPUT_ROOT/predictions/v24_full_warm60_pseudo_adapt.npz"
ADAPTER="$OUTPUT_ROOT/final/v24_full_warm60_pseudo_head_delta.pt"
COMPOSITE="$OUTPUT_ROOT/predictions/v24_full_warm60_pseudo_composite.npz"
DECODED="$OUTPUT_ROOT/predictions/v24_full_warm60_pseudo_pair_aware.npz"
SUBMISSION="$OUTPUT_ROOT/submissions/v24_full_warm60_pseudo_pair_aware.csv"
REPORT="$OUTPUT_ROOT/reports/v24_full_warm60_pseudo_pair_aware.json"
MANIFEST="$OUTPUT_ROOT/reports/v24_full_warm60_pseudo_manifest.json"

for path in "$CHECKPOINT" "$YOLO" "$PSEUDO_LABELS" "$BASE"; do
  require_file "$path"
done

run_logged "v24_full_warm60_pseudo_adapt" "$GPU" \
  "$PYTHON" -B "$ROOT/experiments/scene_context_v13/adapt_pseudo_test.py" \
  --checkpoint "$CHECKPOINT" --pseudo-labels "$PSEUDO_LABELS" --scope heads \
  --epochs 2 --lr 0.0002 --confidence-threshold 0.6 --retention-weight 0.25 \
  --batch-size 16 --workers 4 \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" --frames 32 \
  --tta-hflip --tta-crop-scale 0.75 --output "$ADAPTED" --adapter-output "$ADAPTER"

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_test_pseudo_composite.py" \
  --base "$BASE" --specialized "$ADAPTED" --output "$COMPOSITE"

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$ADAPTED" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT"

"$PYTHON" - "$CHECKPOINT" "$YOLO" "$ADAPTER" "$BASE" "$DECODED" "$SUBMISSION" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

checkpoint, yolo, adapter, base, decoded, submission, manifest = map(Path, sys.argv[1:])
assets = sum(path.stat().st_size for path in (checkpoint, yolo, adapter))
if assets >= 100_000_000:
    raise RuntimeError(f"Pseudo-distilled assets exceed 100 MB: {assets}")
base_archive = np.load(base)
decoded_archive = np.load(decoded)
if not np.array_equal(base_archive["indices"], decoded_archive["indices"]):
    raise RuntimeError("Pseudo-distilled prediction indices are misaligned")
table = pd.read_csv(submission)
if len(table) != 405 or table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("Pseudo-distilled submission failed CSV validation")
payload = {
    "checkpoint": str(checkpoint),
    "yolo": str(yolo),
    "adapter": str(adapter),
    "model_asset_bytes": assets,
    "model_limit_bytes": 100_000_000,
    "slack_bytes": 100_000_000 - assets,
    "adaptation_scope": "all test rows selected by paired status or confidence >= 0.6",
    "local_direct_pairaware_reference": {
        "base_correct": 1153,
        "adapted_correct": 1168,
        "rows": 1357,
    },
    "changed_vs_raw_top1": int((
        base_archive["probabilities"].argmax(1)
        != decoded_archive["probabilities"].argmax(1)
    ).sum()),
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
