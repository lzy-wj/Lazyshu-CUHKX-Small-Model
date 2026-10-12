#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
KAGGLE="$HOME/miniconda3/envs/Train/bin/kaggle"
COMPETITION="cuhk-x-competition-small-model-track"
SUBMISSION="$ROOT/experiments/v24_full_refit/outputs/submissions/v71_gated_film_sequence_hybrid.csv"
MANIFEST="$ROOT/experiments/v24_full_refit/outputs/reports/v71_gated_film_sequence_hybrid_manifest.json"
REPORT="$ROOT/experiments/v24_full_refit/outputs/reports/kaggle_submissions_after_v71.csv"

"$PYTHON" -c 'import hashlib,json,sys; from pathlib import Path; import pandas as pd; s,m=map(Path,sys.argv[1:]); a=json.loads(m.read_text()); t=pd.read_csv(s); assert a["status"]=="pass" and a["model_asset_bytes"]<100000000; assert hashlib.sha256(s.read_bytes()).hexdigest()==a["submission_sha256"]; assert len(t)==405 and not t["path"].duplicated().any() and t["prediction"].between(0,39).all() and a["changed_predictions_vs_v59"]>0' "$SUBMISSION" "$MANIFEST"

target="$(date -u -d 'tomorrow 00:20 UTC' +%s)"
now="$(date -u +%s)"
if (( now < target )); then sleep "$((target-now))"; fi

submission_rows() {
  local attempt output
  for attempt in $(seq 1 20); do
    if output="$($KAGGLE competitions submissions -c "$COMPETITION" -v 2>/dev/null)"; then
      printf '%s\n' "$output"
      return 0
    fi
    sleep 30
  done
  return 1
}

rows="$(submission_rows)"
name="$(basename "$SUBMISSION")"
if printf '%s\n' "$rows" | grep -Fq ",$name,"; then printf '%s\n' "$rows" > "$REPORT"; exit 0; fi
count="$(printf '%s\n' "$rows" | awk -F, -v date="$(date -u +%F)" 'NR>1 && index($3,date)==1 {n++} END{print n+0}')"
if (( count >= 5 )); then exit 1; fi
for attempt in $(seq 1 5); do
  if "$PYTHON" -B "$ROOT/scripts/submit_kaggle_curl.py" --competition "$COMPETITION" --file "$SUBMISSION" --message "V71 gated FiLM; local 1201/1357; 75.4MB assets"; then
    break
  fi
  if (( attempt == 5 )); then exit 1; fi
  sleep 30
done

for _ in $(seq 1 60); do
  rows="$(submission_rows)"
  printf '%s\n' "$rows" > "$REPORT"
  row="$(printf '%s\n' "$rows" | grep -F ",$name," | head -1 || true)"
  echo "$row"
  if [[ "$row" == *"SubmissionStatus.COMPLETE"* ]]; then exit 0; fi
  if [[ "$row" == *"SubmissionStatus.ERROR"* || "$row" == *"SubmissionStatus.CANCELLED"* ]]; then exit 1; fi
  sleep 15
done
exit 1
