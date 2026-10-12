#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export SUBMISSION_OVERRIDE="$ROOT/experiments/v24_full_refit/outputs/submissions/v59_teacher_sequence_hybrid.csv"
export MANIFEST_OVERRIDE="$ROOT/experiments/v24_full_refit/outputs/reports/v59_teacher_sequence_hybrid_manifest.json"
export REPORT_OVERRIDE="$ROOT/experiments/v24_full_refit/outputs/reports/kaggle_submissions_after_v59.csv"
export MESSAGE_OVERRIDE="V59 teacher compression plus repeat-aware sequence decoding; local 1197/1357; 71.32MB"
exec bash "$ROOT/experiments/v57_round_transformer_distillation/submit_after_reset.sh"
