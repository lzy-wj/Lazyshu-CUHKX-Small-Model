#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
V24="$ROOT/experiments/v24_full_refit/outputs"
INPUT="$V24/predictions/v59_teacher_repeat_recovered.npz"
DECODED="$V24/predictions/v59_teacher_sequence_hybrid.npz"
SUBMISSION="$V24/submissions/v59_teacher_sequence_hybrid.csv"
REPORT="$V24/reports/v59_teacher_sequence_hybrid.json"
MANIFEST="$V24/reports/v59_teacher_sequence_hybrid_manifest.json"

"$PYTHON" -B "$HERE/build_test.py"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$INPUT" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT"

"$PYTHON" - "$V24/final/v24_full_warm60_soup_int8.pt" \
  "$V24/final/v24_full_warm60_pseudo_head_delta.pt" \
  "$V24/final/v52_domain_logit_residual_conv1d.pt" \
  "$V24/final/v57_round_residual_transformer.pt" \
  "$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt" \
  "$SUBMISSION" "$V24/submissions/v52_domain_logit_residual_pair_aware.csv" \
  "$V24/submissions/v57_round_transformer_distilled.csv" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

checkpoint, head, domain_residual, transformer, yolo, submission, v52, v57, manifest = map(Path, sys.argv[1:])
assets = sum(path.stat().st_size for path in (checkpoint, head, domain_residual, transformer, yolo))
if assets >= 100_000_000:
    raise RuntimeError(f"V59 assets exceed 100 MB: {assets}")
table = pd.read_csv(submission)
v52_table = pd.read_csv(v52)
v57_table = pd.read_csv(v57)
if len(table) != 405 or table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V59 submission validation failed")
if not table["path"].equals(v52_table["path"]) or not table["path"].equals(v57_table["path"]):
    raise RuntimeError("Candidate paths are not aligned")
changed_v52 = int((table["prediction"] != v52_table["prediction"]).sum())
changed_v57 = int((table["prediction"] != v57_table["prediction"]).sum())
if not changed_v52 or not changed_v57:
    raise RuntimeError("V59 is not an independent candidate")
payload = {
    "model_asset_bytes": assets,
    "slack_bytes": 100_000_000 - assets,
    "model_assets": [str(path) for path in (checkpoint, head, domain_residual, transformer, yolo)],
    "round_transformer_sha256": hashlib.sha256(transformer.read_bytes()).hexdigest(),
    "local_direct": {"correct": 1197, "rows": 1357, "accuracy": 1197 / 1357},
    "changed_predictions_vs_v52": changed_v52,
    "changed_predictions_vs_v57": changed_v57,
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
