#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/Train/bin/python}"
V24="$ROOT/experiments/v24_full_refit/outputs"
INPUT="$V24/predictions/v54_repeat_recovered.npz"
DECODED="$V24/predictions/v54_repeat_recovered_pair_aware.npz"
SUBMISSION="$V24/submissions/v54_repeat_recovered_pair_aware.csv"
REPORT="$V24/reports/v54_repeat_recovered_pair_aware.json"
MANIFEST="$V24/reports/v54_repeat_recovered_manifest.json"
VALIDATION="$HERE/outputs/validation.json"
REFERENCE_SUBMISSION="$V24/submissions/v52_domain_logit_residual_pair_aware.csv"
CHECKPOINT="$V24/final/v24_full_warm60_soup_int8.pt"
HEAD_ADAPTER="$V24/final/v24_full_warm60_pseudo_head_delta.pt"
CALIBRATORS="$V24/final/v52_domain_logit_residual_conv1d.pt"
YOLO="$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"

"$PYTHON" -B "$HERE/build_test.py"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$INPUT" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT"

"$PYTHON" - "$CHECKPOINT" "$HEAD_ADAPTER" "$CALIBRATORS" "$YOLO" "$DECODED" "$SUBMISSION" "$MANIFEST" "$VALIDATION" "$REFERENCE_SUBMISSION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

checkpoint, head_adapter, calibrators, yolo, decoded, submission, manifest, validation, reference_submission = map(Path, sys.argv[1:])
assets = sum(path.stat().st_size for path in (checkpoint, head_adapter, calibrators, yolo))
if assets >= 100_000_000:
    raise RuntimeError(f"V54 assets exceed 100 MB: {assets}")
archive = np.load(decoded)
table = pd.read_csv(submission)
reference_table = pd.read_csv(reference_submission)
if len(archive["indices"]) != 405 or len(table) != 405:
    raise RuntimeError("V54 output does not cover 405 test rows")
if table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V54 submission validation failed")
validation_report = json.loads(validation.read_text())
changed_vs_v52 = int((table["prediction"] != reference_table["prediction"]).sum())
payload = {
    "checkpoint": str(checkpoint),
    "head_adapter": str(head_adapter),
    "domain_logit_residuals": str(calibrators),
    "yolo": str(yolo),
    "model_asset_bytes": assets,
    "slack_bytes": 100_000_000 - assets,
    "local_direct_pairaware": validation_report["combined"],
    "repeat_recovery": validation_report["settings"],
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
    "changed_predictions_vs_v52": changed_vs_v52,
    "informative_submission": changed_vs_v52 > 0,
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
