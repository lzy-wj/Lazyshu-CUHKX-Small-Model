#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
KAGGLE="$HOME/miniconda3/envs/Train/bin/kaggle"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
COMPETITION="cuhk-x-competition-small-model-track"
SUBMISSION="${SUBMISSION_OVERRIDE:-$ROOT/experiments/v24_full_refit/outputs/submissions/v57_round_transformer_distilled.csv}"
MANIFEST="${MANIFEST_OVERRIDE:-$ROOT/experiments/v24_full_refit/outputs/reports/v57_round_transformer_distilled_manifest.json}"
REPORT="${REPORT_OVERRIDE:-$ROOT/experiments/v24_full_refit/outputs/reports/kaggle_submissions_after_v57.csv}"
MESSAGE="${MESSAGE_OVERRIDE:-V57 round-Transformer compression of system teacher; local 1195/1357; 71.32MB}"

"$PYTHON" - "$SUBMISSION" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

submission, manifest = map(Path, sys.argv[1:])
audit = json.loads(manifest.read_text())
table = pd.read_csv(submission)
digest = hashlib.sha256(submission.read_bytes()).hexdigest()
if digest != audit["submission_sha256"]:
    raise RuntimeError("V57 submission changed after release audit")
if audit["model_asset_bytes"] >= 100_000_000 or audit["changed_predictions_vs_v52"] <= 0:
    raise RuntimeError("V57 release manifest does not pass size/information gates")
if len(table) != 405 or table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("V57 CSV validation failed")
PY

target="$(date -u -d 'tomorrow 00:05 UTC' +%s)"
now="$(date -u +%s)"
if (( now < target )); then
  echo "WAIT_UNTIL_UTC=$(date -u -d "@$target" --iso-8601=seconds)"
  sleep "$((target - now))"
fi

submission_rows() {
  local attempt output
  for attempt in $(seq 1 20); do
    if output="$($KAGGLE competitions submissions -c "$COMPETITION" -v 2>/dev/null)"; then
      printf '%s\n' "$output"
      return 0
    fi
    echo "Kaggle query failed on attempt $attempt/20" >&2
    sleep 30
  done
  return 1
}

name="$(basename "$SUBMISSION")"
rows="$(submission_rows)"
if printf '%s\n' "$rows" | grep -Fq ",$name,"; then
  echo "ALREADY_SUBMITTED=$name"
  printf '%s\n' "$rows" > "$REPORT"
  exit 0
fi
utc_date="$(date -u +%F)"
count="$(printf '%s\n' "$rows" | awk -F, -v date="$utc_date" 'NR > 1 && index($3, date) == 1 {count++} END {print count + 0}')"
if (( count >= 5 )); then
  echo "DAILY_LIMIT_REACHED=$count" >&2
  exit 1
fi

for attempt in $(seq 1 5); do
  if "$PYTHON" -B "$ROOT/scripts/submit_kaggle_curl.py" \
    --competition "$COMPETITION" --file "$SUBMISSION" \
    --message "$MESSAGE"; then
    break
  fi
  if (( attempt == 5 )); then exit 1; fi
  echo "Kaggle submit failed on attempt $attempt/5" >&2
  sleep 30
done

for _ in $(seq 1 60); do
  rows="$(submission_rows)"
  row="$(printf '%s\n' "$rows" | grep -F ",$name," | head -1 || true)"
  echo "$row"
  if [[ "$row" == *"SubmissionStatus.COMPLETE"* ]]; then
    printf '%s\n' "$rows" > "$REPORT"
    exit 0
  fi
  if [[ "$row" == *"SubmissionStatus.ERROR"* || "$row" == *"SubmissionStatus.CANCELLED"* ]]; then
    exit 1
  fi
  sleep 15
done
exit 1
