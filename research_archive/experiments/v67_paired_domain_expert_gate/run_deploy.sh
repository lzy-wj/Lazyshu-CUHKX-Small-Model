#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
V24="$ROOT/experiments/v24_full_refit/outputs"
INPUT="$V24/predictions/v67_paired_domain_expert_predecode.npz"
DECODED="$V24/predictions/v67_paired_domain_expert_gate.npz"
SUBMISSION="$V24/submissions/v67_paired_domain_expert_gate.csv"
REPORT="$V24/reports/v67_paired_domain_expert_gate.json"
MANIFEST="$V24/reports/v67_paired_domain_expert_gate_manifest.json"

"$PYTHON" -B "$HERE/build_test.py"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$INPUT" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT"

"$PYTHON" - "$V24/reports/v59_teacher_sequence_hybrid_manifest.json" \
  "$ROOT/outputs/checkpoints/v8_x3dl_soup_full_w075/best_int8_compact.pt" \
  "$ROOT/outputs/final_legal_multimodal_v9/models/person_detector_int8_compact.pth" \
  "$V24/final/v66_supervised_expert_transformer.pt" \
  "$SUBMISSION" "$V24/submissions/v59_teacher_sequence_hybrid.csv" \
  "$HERE/outputs/validation.json" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

v59_manifest, x3d, detector, transformer, submission, v59_submission, validation, manifest = map(Path, sys.argv[1:])
base = json.loads(v59_manifest.read_text())
assets = base["model_asset_bytes"] + x3d.stat().st_size + detector.stat().st_size + transformer.stat().st_size
if assets >= 100_000_000:
    raise RuntimeError(f"V67 assets exceed 100 MB: {assets}")
table = pd.read_csv(submission)
reference = pd.read_csv(v59_submission)
if len(table) != 405 or table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V67 submission validation failed")
if not table["path"].equals(reference["path"]):
    raise RuntimeError("V67 and V59 paths are not aligned")
changed = int((table["prediction"] != reference["prediction"]).sum())
local = json.loads(validation.read_text())
payload = {
    "model_asset_bytes": assets,
    "slack_bytes": 100_000_000 - assets,
    "base_assets": base["model_assets"],
    "additional_assets": [str(x3d), str(detector), str(transformer)],
    "local_direct": local["combined"],
    "selected_weights": local["selected_weights"],
    "changed_predictions_vs_v59": changed,
    "informative_submission": changed > 0,
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

