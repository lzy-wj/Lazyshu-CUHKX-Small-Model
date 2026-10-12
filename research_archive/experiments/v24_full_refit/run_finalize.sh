#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

GPU="${1:?Usage: $0 GPU}"
CHECKPOINT_ROOT="$OUTPUT_ROOT/checkpoints"
ARTIFACT_ROOT="$OUTPUT_ROOT/final"
PREDICTION_ROOT="$OUTPUT_ROOT/predictions"
SUBMISSION_ROOT="$OUTPUT_ROOT/submissions"
REPORT_ROOT="$OUTPUT_ROOT/reports"
YOLO="$ROOT/outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"

BASE="$CHECKPOINT_ROOT/v24_full_noradar_e13/best.pt"
CROSS="$CHECKPOINT_ROOT/v24_full_crossuser_e16/best.pt"
PROGRESSIVE="$CHECKPOINT_ROOT/v24_full_progressive160_e7/best.pt"
SCRATCH_MASK="$CHECKPOINT_ROOT/v24_full_scratch_maskdistill_e16/best.pt"
WARM_MASK="$CHECKPOINT_ROOT/v24_full_warm_maskdistill_e9/best.pt"
SOURCES=("$BASE" "$CROSS" "$PROGRESSIVE" "$SCRATCH_MASK" "$WARM_MASK")
WEIGHTS=(0.082416 0.079504 0.148508 0.089572 0.600000)

for path in "${SOURCES[@]}" "$YOLO"; do
  require_file "$path"
done
for name in v24_full_noradar_e13 v24_full_crossuser_e16 v24_full_progressive160_e7 v24_full_scratch_maskdistill_e16 v24_full_warm_maskdistill_e9; do
  require_file "$CHECKPOINT_ROOT/$name/completion.json"
done
"$PYTHON" - "${SOURCES[@]}" <<'PY'
import json
import sys
from pathlib import Path

for checkpoint_text in sys.argv[1:]:
    checkpoint = Path(checkpoint_text)
    completion = checkpoint.with_name("completion.json")
    payload = json.loads(completion.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"Training run is not complete: {completion}")
PY

mkdir -p "$ARTIFACT_ROOT" "$PREDICTION_ROOT" "$SUBMISSION_ROOT" "$REPORT_ROOT"
FP32="$ARTIFACT_ROOT/v24_full_warm60_soup_fp32.pt"
INT8="$ARTIFACT_ROOT/v24_full_warm60_soup_int8.pt"
INT4="$ARTIFACT_ROOT/v24_full_warm60_soup_int4.pt"
RAW_NPZ="$PREDICTION_ROOT/v24_full_warm60_int8_tta.npz"
RAW_CSV="$SUBMISSION_ROOT/v24_full_warm60_int8_tta_raw.csv"
DECODED_NPZ="$PREDICTION_ROOT/v24_full_warm60_int8_pair_aware.npz"
DECODED_CSV="$SUBMISSION_ROOT/v24_full_warm60_int8_pair_aware.csv"
PREDICT_MANIFEST="$REPORT_ROOT/v24_full_warm60_int8_predict.json"
DECODER_REPORT="$REPORT_ROOT/v24_full_warm60_int8_pair_aware.json"

"$PYTHON" -B "$ROOT/scripts/soup_r2plus1d_checkpoints.py" \
  --checkpoints "${SOURCES[@]}" --weights "${WEIGHTS[@]}" \
  --storage fp32 --output "$FP32"
"$PYTHON" -B "$ROOT/scripts/soup_r2plus1d_checkpoints.py" \
  --checkpoints "${SOURCES[@]}" --weights "${WEIGHTS[@]}" \
  --storage int8 --output "$INT8"
"$PYTHON" -B "$ROOT/scripts/quantize_r2plus1d_hybrid.py" \
  --input "$FP32" --output "$INT4" --deep-scheme uniform4 --layer3-bits 4

run_logged "v24_full_warm60_int8_predict" "$GPU" \
  "$PYTHON" -B "$V24_DIR/predict_test.py" \
  --checkpoint "$INT8" --yolo "$YOLO" \
  --cache-dir "$ROOT/data/processed/r2plus1d_yolo_all_visual_32_128" --frames 32 \
  --tta-hflip --tta-crop-scale 0.75 --batch-size 16 --workers 4 --adabn none \
  --output "$RAW_NPZ" --submission "$RAW_CSV" --manifest "$PREDICT_MANIFEST"

"$PYTHON" -B "$ROOT/experiments/scene_context_v13/build_pair_aware_submission.py" \
  --input "$RAW_NPZ" --output "$DECODED_NPZ" \
  --submission "$DECODED_CSV" --report "$DECODER_REPORT"

"$PYTHON" -B "$V24_DIR/audit_release.py" \
  --checkpoint "$INT8" --fallback-checkpoint "$INT4" --yolo "$YOLO" \
  --sources "${SOURCES[@]}" \
  --prediction "$RAW_NPZ" --decoded-prediction "$DECODED_NPZ" \
  --raw-submission "$RAW_CSV" \
  --decoded-submission "$DECODED_CSV" --decoder-report "$DECODER_REPORT" \
  --test-csv "$ROOT/data/downloads/CUHK-X_Small_Model_Track/Small-Model-Track/Testing/test_file/test.csv" \
  --output "$REPORT_ROOT/release_audit.json"

echo "Finalize complete. No Kaggle submission was performed."
