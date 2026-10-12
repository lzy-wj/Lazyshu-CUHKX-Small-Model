#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash inference.sh <data_dir> <output_csv> [submission_id]" >&2
  exit 2
fi

DATA_DIR="$1"
OUTPUT_CSV="$2"
TARGET_SUBMISSION_ID="${3:-primary}"
PACKAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# DATA_DIR is selected by the Organizing Committee at runtime. It contains the
# relevant track's official label-free Testing data in the published layout.
# Do not replace it with a hard-coded participant or server path.

if [ ! -d "$DATA_DIR" ]; then
  echo "Input data directory does not exist: $DATA_DIR" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$OUTPUT_CSV")"
export PYTHONPATH="$PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

exec python3 -B "$PACKAGE_ROOT/src/predict.py" \
  --data-dir "$DATA_DIR" \
  --output-csv "$OUTPUT_CSV" \
  --submission-id "$TARGET_SUBMISSION_ID" \
  --config "$PACKAGE_ROOT/configs/final.yaml"
