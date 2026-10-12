#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
V24="$ROOT/experiments/v24_full_refit/outputs"
CHECKPOINT="$V24/final/v24_full_warm60_soup_int8.pt"
ADAPTER="$V24/final/v24_full_warm60_pseudo_head_delta.pt"
YOLO="$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"
AUXILIARY="$V24/predictions/v37_auxiliary_adabn.npz"
INPUT="$V24/predictions/v37_aux_domain_gate.npz"
DECODED="$V24/predictions/v37_aux_domain_calibrated_pair_aware.npz"
SUBMISSION="$V24/submissions/v37_aux_domain_calibrated_pair_aware.csv"
REPORT="$V24/reports/v37_aux_domain_calibrated_pair_aware.json"
MANIFEST="$V24/reports/v37_aux_domain_calibrated_manifest.json"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"

for path in "$CHECKPOINT" "$ADAPTER" "$YOLO" "$V24/predictions/v32_domain_adaptation_gate.npz"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$ROOT/experiments/v36_auxiliary_rescue/predict_test.py" \
  --checkpoint "$CHECKPOINT" --cache-dir "$CACHE" --frames 32 --batch-size 16 --workers 4 \
  --tta-hflip --tta-crop-scale 0.75 --output "$AUXILIARY"
"$PYTHON" -B "$HERE/build_test.py" \
  --main "$V24/predictions/v32_domain_adaptation_gate.npz" \
  --auxiliary "$AUXILIARY" --output "$INPUT"
"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$INPUT" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT" \
  --template-strength 1.8 --gap-penalty 1.0 --maximum-edits 6 --alignment-beam-size 40

"$PYTHON" - "$CHECKPOINT" "$ADAPTER" "$YOLO" "$DECODED" "$SUBMISSION" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

checkpoint, adapter, yolo, decoded, submission, manifest = map(Path, sys.argv[1:])
assets = sum(path.stat().st_size for path in (checkpoint, adapter, yolo))
if assets >= 100_000_000:
    raise RuntimeError(f"V37 assets exceed 100 MB: {assets}")
archive = np.load(decoded)
table = pd.read_csv(submission)
if len(archive["indices"]) != 405 or len(table) != 405:
    raise RuntimeError("V37 output does not cover 405 test rows")
if table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V37 submission validation failed")
payload = {
    "checkpoint": str(checkpoint),
    "adapter": str(adapter),
    "yolo": str(yolo),
    "model_asset_bytes": assets,
    "slack_bytes": 100_000_000 - assets,
    "local_direct_pairaware": {"correct": 1176, "rows": 1357},
    "components": ["V32 domain gate", "V36 skeleton auxiliary rescue", "V31 decoder"],
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
