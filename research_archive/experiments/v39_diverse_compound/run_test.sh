#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
V24="$ROOT/experiments/v24_full_refit/outputs"
CHECKPOINT="$V24/final/v24_full_warm60_soup_int8.pt"
ADAPTER="$V24/final/v24_full_warm60_pseudo_head_delta.pt"
YOLO="$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"
X3DL_MODEL="$ROOT/outputs/checkpoints/v8_x3dl_soup_full_w075/best_int8_compact.pt"
PERSON_DETECTOR="$ROOT/outputs/final_legal_multimodal_v9/models/person_detector_int8_compact.pth"
MAIN="$V24/predictions/v32_domain_adaptation_gate.npz"
X3DL="$ROOT/outputs/predictions/v9_x3dl_soup_full_cropblend50.npz"
SKELETON_AUXILIARY="$V24/predictions/v37_auxiliary_adabn.npz"
INPUT="$V24/predictions/v39_diverse_compound.npz"
DECODED="$V24/predictions/v39_diverse_compound_pair_aware.npz"
SUBMISSION="$V24/submissions/v39_diverse_compound_pair_aware.csv"
REPORT="$V24/reports/v39_diverse_compound_pair_aware.json"
MANIFEST="$V24/reports/v39_diverse_compound_manifest.json"

for path in "$CHECKPOINT" "$ADAPTER" "$YOLO" "$X3DL_MODEL" "$PERSON_DETECTOR" \
  "$MAIN" "$X3DL" "$SKELETON_AUXILIARY"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

"$PYTHON" -B "$HERE/build_test.py" --main "$MAIN" --x3dl "$X3DL" \
  --skeleton-auxiliary "$SKELETON_AUXILIARY" --output "$INPUT"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$INPUT" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT" \
  --template-strength 1.8 --gap-penalty 1.0 --maximum-edits 6 --alignment-beam-size 40

"$PYTHON" - "$CHECKPOINT" "$ADAPTER" "$YOLO" "$X3DL_MODEL" "$PERSON_DETECTOR" \
  "$DECODED" "$SUBMISSION" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

checkpoint, adapter, yolo, x3dl, detector, decoded, submission, manifest = map(Path, sys.argv[1:])
assets = sum(path.stat().st_size for path in (checkpoint, adapter, yolo, x3dl, detector))
if assets >= 100_000_000:
    raise RuntimeError(f"V39 assets exceed 100 MB: {assets}")
archive = np.load(decoded)
table = pd.read_csv(submission)
if len(archive["indices"]) != 405 or len(table) != 405:
    raise RuntimeError("V39 output does not cover 405 test rows")
if table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V39 submission validation failed")
payload = {
    "checkpoint": str(checkpoint),
    "adapter": str(adapter),
    "yolo": str(yolo),
    "x3dl": str(x3dl),
    "person_detector": str(detector),
    "model_asset_bytes": assets,
    "slack_bytes": 100_000_000 - assets,
    "local_direct_pairaware": {"correct": 1179, "rows": 1357},
    "components": ["V32 domain gate", "V38 X3D-L", "V36 skeleton auxiliary", "V31 decoder"],
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
