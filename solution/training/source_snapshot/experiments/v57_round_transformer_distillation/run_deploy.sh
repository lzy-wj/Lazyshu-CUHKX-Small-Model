#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
V24="$ROOT/experiments/v24_full_refit/outputs"
SUBMISSION="$V24/submissions/v57_round_transformer_distilled.csv"
REFERENCE="$V24/submissions/v52_domain_logit_residual_pair_aware.csv"
MANIFEST="$V24/reports/v57_round_transformer_distilled_manifest.json"

"$PYTHON" -B "$HERE/deploy_test.py" 2>&1 | tee "$HERE/deploy.log"

"$PYTHON" - "$V24/final/v24_full_warm60_soup_int8.pt" \
  "$V24/final/v24_full_warm60_pseudo_head_delta.pt" \
  "$V24/final/v52_domain_logit_residual_conv1d.pt" \
  "$V24/final/v57_round_residual_transformer.pt" \
  "$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt" \
  "$SUBMISSION" "$REFERENCE" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

checkpoint, head, domain_residual, round_transformer, yolo, submission, reference, manifest = map(Path, sys.argv[1:])
assets = sum(path.stat().st_size for path in (checkpoint, head, domain_residual, round_transformer, yolo))
if assets >= 100_000_000:
    raise RuntimeError(f"V57 assets exceed 100 MB: {assets}")
table = pd.read_csv(submission)
reference_table = pd.read_csv(reference)
if len(table) != 405 or table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V57 submission validation failed")
if not table["path"].equals(reference_table["path"]):
    raise RuntimeError("V57 and V52 submission paths are not aligned")
changed = int((table["prediction"] != reference_table["prediction"]).sum())
if changed == 0:
    raise RuntimeError("V57 is not an informative submission")
payload = {
    "checkpoint": str(checkpoint),
    "head_adapter": str(head),
    "domain_logit_residuals": str(domain_residual),
    "round_transformer": str(round_transformer),
    "round_transformer_sha256": hashlib.sha256(round_transformer.read_bytes()).hexdigest(),
    "yolo": str(yolo),
    "model_asset_bytes": assets,
    "slack_bytes": 100_000_000 - assets,
    "local_direct": {"correct": 1195, "rows": 1357, "accuracy": 1195 / 1357},
    "teacher_role": "training-only pseudo-label generator; not a deployment asset",
    "changed_predictions_vs_v52": changed,
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
