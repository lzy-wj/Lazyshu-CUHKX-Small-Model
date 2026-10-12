#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
GPU="${1:?Usage: $0 GPU}"
V13="$ROOT/experiments/scene_context_v13"
V24="$ROOT/experiments/v24_full_refit/outputs"
CACHE="$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128"
CHECKPOINT="$V24/final/v24_full_warm60_soup_int8.pt"
YOLO="$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"
V24_COMPOSITE="$V24/predictions/v24_full_warm60_pseudo_composite.npz"
V24_BLEND="$V24/predictions/v26_v24_system_domain_blend.npz"
V24_DECODED="$V24/predictions/v26_v24_system_pair_aware.npz"
TARGETS="$V24/predictions/v26_system_teacher_targets.npz"
ADAPTED="$V24/predictions/v26_system_distill_adapt.npz"
ADAPTER="$V24/final/v26_system_distill_head_delta.pt"
DECODED="$V24/predictions/v26_system_distill_pair_aware.npz"
SUBMISSION="$V24/submissions/v26_system_distill_pair_aware.csv"
REPORT="$V24/reports/v26_system_distill_pair_aware.json"
MANIFEST="$V24/reports/v26_system_distill_manifest.json"

for path in \
  "$CHECKPOINT" "$YOLO" "$V24_COMPOSITE" \
  "$ROOT/outputs/predictions/v13_pair_aware.npz" \
  "$ROOT/outputs/predictions/v14_fixedsoup_pseudo_pair_aware.npz" \
  "$ROOT/outputs/predictions/v14_full_fixedsoup_pseudo_composite.npz" \
  "$V13/outputs/v14_acceptance_gate.json" \
  "$ROOT/experiments/v24_pseudo_validation/outputs/v24_warm60_pseudo_domain_pairaware.json" \
  "$ROOT/experiments/v24_pseudo_validation/outputs/v24_warm60_pseudo_second_stage_acceptance.json"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

"$PYTHON" -B "$V13/blend_test_domain_candidate.py" \
  --base "$ROOT/outputs/predictions/final_legal_multimodal_v12_lowrank.npz" \
  --candidate "$V24_COMPOSITE" \
  --settings "$ROOT/experiments/v24_pseudo_validation/outputs/v24_warm60_pseudo_domain_pairaware.json" \
  --output "$V24_BLEND"
"$PYTHON" -B "$V13/build_pair_aware_submission.py" \
  --input "$V24_BLEND" --output "$V24_DECODED" \
  --submission "$V24/submissions/v26_v24_system_pair_aware.csv" \
  --report "$V24/reports/v26_v24_system_pair_aware.json"
"$PYTHON" -B "$HERE/build_test_system_targets.py" \
  --v13-decoded "$ROOT/outputs/predictions/v13_pair_aware.npz" \
  --v14-decoded "$ROOT/outputs/predictions/v14_fixedsoup_pseudo_pair_aware.npz" \
  --v14-candidate "$ROOT/outputs/predictions/v14_full_fixedsoup_pseudo_composite.npz" \
  --base-acceptance "$V13/outputs/v14_acceptance_gate.json" \
  --v24-decoded "$V24_DECODED" --v24-candidate "$V24_COMPOSITE" \
  --second-stage "$ROOT/experiments/v24_pseudo_validation/outputs/v24_warm60_pseudo_second_stage_acceptance.json" \
  --output "$TARGETS"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -B "$V13/adapt_pseudo_test.py" \
  --checkpoint "$CHECKPOINT" --pseudo-labels "$TARGETS" --scope heads \
  --epochs 2 --lr 0.0002 --confidence-threshold 0.6 --retention-weight 0.25 \
  --batch-size 16 --workers 4 --cache-dir "$CACHE" --frames 32 \
  --tta-hflip --tta-crop-scale 0.75 --output "$ADAPTED" --adapter-output "$ADAPTER"
"$PYTHON" -B "$V13/build_pair_aware_submission.py" \
  --input "$ADAPTED" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT"

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
    raise RuntimeError(f"V26 assets exceed 100 MB: {assets}")
archive = np.load(decoded)
table = pd.read_csv(submission)
if len(archive["indices"]) != 405 or len(table) != 405:
    raise RuntimeError("V26 output does not cover 405 test rows")
if table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V26 submission validation failed")
payload = {
    "checkpoint": str(checkpoint),
    "adapter": str(adapter),
    "yolo": str(yolo),
    "model_asset_bytes": assets,
    "slack_bytes": 100_000_000 - assets,
    "local_direct_pairaware": {"correct": 1170, "rows": 1357},
    "training_only_teachers": "V13/V14/V24 system predictions; not inference assets",
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

