#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
while [[ ! -f outputs/logs/spatial_study_queue.exit ]]; do
    sleep 5
done
test "$(cat outputs/logs/spatial_study_queue.exit)" = 0
conda run --no-capture-output -n Train python scripts/run_scratch_study.py \
    --config configs/spatial_token_study_s43.json --output outputs/study_spatial_s43 \
    --audit outputs/study_spatial_s43/audit.json
conda run --no-capture-output -n Train python scripts/summarize_spatial_study.py
conda run --no-capture-output -n Train python scripts/summarize_spatial_study.py \
    --config configs/spatial_token_study_s43.json --output outputs/study_spatial_s43 \
    --markdown docs/SPATIAL_STUDY_S43_RESULTS_20260907.md
conda run --no-capture-output -n Train python scripts/summarize_spatial_replication.py
