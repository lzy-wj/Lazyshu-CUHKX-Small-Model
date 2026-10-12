#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
V24="$ROOT/experiments/v24_full_refit/outputs"
V68="$ROOT/experiments/v68_film_full_refit/outputs"
CHECKPOINT="$V68/checkpoints/v68_film_full_e13/best_int8.pt"
COMPLETION="$V68/checkpoints/v68_film_full_e13/completion.json"
YOLO="$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"
RAW="$V68/predictions/v68_film_full_raw.npz"
PREPOOL="$V24/predictions/v71_gated_film_repeat_recovered.npz"
DECODED="$V24/predictions/v71_gated_film_sequence_hybrid.npz"
SUBMISSION="$V24/submissions/v71_gated_film_sequence_hybrid.csv"
REPORT="$V24/reports/v71_gated_film_sequence_hybrid.json"
TEACHER_PREPOOL="$V24/predictions/v71_film_teacher_repeat_recovered.npz"
TEACHER_DECODED="$V24/predictions/v71_film_teacher_sequence_hybrid.npz"
TEACHER_SUBMISSION="$V24/submissions/v71_film_teacher_sequence_hybrid.csv"
TEACHER_REPORT="$V24/reports/v71_film_teacher_sequence_hybrid.json"
MANIFEST="$V24/reports/v71_gated_film_sequence_hybrid_manifest.json"

until [[ -f "$COMPLETION" ]]; do sleep 60; done
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "V68 completion exists but compact checkpoint is missing" >&2
  exit 1
fi

mkdir -p "$V68/predictions" "$V68/submissions" "$V68/reports"
flock "$ROOT/outputs/logs/gpu-5.lock" env CUDA_VISIBLE_DEVICES=5 \
  "$PYTHON" -B "$ROOT/experiments/v24_full_refit/predict_test.py" \
  --checkpoint "$CHECKPOINT" --yolo "$YOLO" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" \
  --frames 32 --batch-size 16 --workers 4 \
  --output "$RAW" \
  --submission "$V68/submissions/v68_film_full_raw.csv" \
  --manifest "$V68/reports/v68_film_full_raw.json"

CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$HERE/deploy_test.py"

CUDA_VISIBLE_DEVICES='' "$PYTHON" -B \
  "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$PREPOOL" --output "$DECODED" --submission "$SUBMISSION" --report "$REPORT"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B \
  "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$TEACHER_PREPOOL" --output "$TEACHER_DECODED" \
  --submission "$TEACHER_SUBMISSION" --report "$TEACHER_REPORT"

CUDA_VISIBLE_DEVICES='' "$PYTHON" - \
  "$V24/final/v24_full_warm60_soup_int8.pt" \
  "$V24/final/v24_full_warm60_pseudo_head_delta.pt" \
  "$V24/final/v52_domain_logit_residual_conv1d.pt" \
  "$V24/final/v57_round_residual_transformer.pt" \
  "$V24/final/v71_gated_film_transformer.pt" \
  "$YOLO" "$SUBMISSION" "$TEACHER_SUBMISSION" \
  "$V24/submissions/v59_teacher_sequence_hybrid.csv" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

checkpoint, head, residual, v57, v71, yolo, submission, teacher_submission, v59, manifest = map(Path, sys.argv[1:])
assets = (checkpoint, head, residual, v57, v71, yolo)
asset_bytes = sum(path.stat().st_size for path in assets)
if asset_bytes >= 100_000_000:
    raise RuntimeError(f"V71 assets exceed 100 MB: {asset_bytes}")
tables = [pd.read_csv(path) for path in (submission, teacher_submission, v59)]
candidate, teacher, reference = tables
if len(candidate) != 405 or candidate["path"].duplicated().any() or not candidate["prediction"].between(0, 39).all():
    raise RuntimeError("V71 submission validation failed")
if not candidate["path"].equals(teacher["path"]) or not candidate["path"].equals(reference["path"]):
    raise RuntimeError("V71 submission paths are not aligned")
teacher_matches = int((candidate["prediction"] == teacher["prediction"]).sum())
changed_v59 = int((candidate["prediction"] != reference["prediction"]).sum())
payload = {
    "status": "pass" if teacher_matches == len(candidate) and changed_v59 else "review",
    "model_asset_bytes": asset_bytes,
    "slack_bytes": 100_000_000 - asset_bytes,
    "model_assets": [str(path) for path in assets],
    "asset_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in assets},
    "local_direct": {"correct": 1201, "rows": 1357, "accuracy": 1201 / 1357},
    "reference_v59": {"correct": 1197, "rows": 1357, "accuracy": 1197 / 1357},
    "decoded_teacher_student_matches": teacher_matches,
    "changed_predictions_vs_v59": changed_v59,
    "teacher_role": "V68 generates training targets only and is not a deployment asset",
    "submission": str(submission),
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if payload["status"] != "pass":
    raise RuntimeError("V71 final audit requires review")
PY
