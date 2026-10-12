#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/Train/bin/python}"
V24="$ROOT/experiments/v24_full_refit/outputs"
CALIBRATED="$V24/predictions/v52_domain_logit_residual_calibrated.npz"
FUSED="$V24/predictions/v52_domain_logit_residual_domain_gate.npz"
DECODED="$V24/predictions/v52_domain_logit_residual_pair_aware.npz"
SUBMISSION="$V24/submissions/v52_domain_logit_residual_pair_aware.csv"
REPORT="$V24/reports/v52_domain_logit_residual_pair_aware.json"
MANIFEST="$V24/reports/v52_domain_logit_residual_manifest.json"
CHECKPOINT="$V24/final/v24_full_warm60_soup_int8.pt"
HEAD_ADAPTER="$V24/final/v24_full_warm60_pseudo_head_delta.pt"
CALIBRATORS="$V24/final/v52_domain_logit_residual_conv1d.pt"
YOLO="$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"

"$PYTHON" -B "$HERE/deploy_test.py"
"$PYTHON" -B "$ROOT/experiments/v32_domain_adaptation_gate/build_test.py" \
  --base "$V24/predictions/v24_full_warm60_int8_tta.npz" \
  --adapted "$CALIBRATED" \
  --settings "$ROOT/experiments/v32_domain_adaptation_gate/outputs/search.json" \
  --output "$FUSED"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$FUSED" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT"

"$PYTHON" - "$CHECKPOINT" "$HEAD_ADAPTER" "$CALIBRATORS" "$YOLO" "$DECODED" "$SUBMISSION" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

checkpoint, head_adapter, calibrators, yolo, decoded, submission, manifest = map(Path, sys.argv[1:])
assets = sum(path.stat().st_size for path in (checkpoint, head_adapter, calibrators, yolo))
if assets >= 100_000_000:
    raise RuntimeError(f"V52 assets exceed 100 MB: {assets}")
archive = np.load(decoded)
table = pd.read_csv(submission)
if len(archive["indices"]) != 405 or len(table) != 405:
    raise RuntimeError("V52 output does not cover 405 test rows")
if table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V52 submission validation failed")
payload = {
    "checkpoint": str(checkpoint),
    "head_adapter": str(head_adapter),
    "domain_logit_residuals": str(calibrators),
    "yolo": str(yolo),
    "model_asset_bytes": assets,
    "slack_bytes": 100_000_000 - assets,
    "local_direct_pairaware": {"correct": 1193, "rows": 1357},
    "protocol": "Transductive V26 system-teacher distillation into domain-conditioned 1x1 Conv1d residuals.",
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

