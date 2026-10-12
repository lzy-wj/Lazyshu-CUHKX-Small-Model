#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMPETITION="cuhk-x-competition-small-model-track"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
KAGGLE="$HOME/miniconda3/envs/Train/bin/kaggle"
AUDIT="$ROOT/experiments/v24_full_refit/outputs/reports/release_audit.json"
PRIMARY="$ROOT/experiments/v24_full_refit/outputs/submissions/v24_full_warm60_int8_pair_aware.csv"
PSEUDO="$ROOT/experiments/v24_full_refit/outputs/submissions/v24_full_warm60_pseudo_pair_aware.csv"
FUSED="$ROOT/experiments/v24_full_refit/outputs/submissions/v29_crop_system_fusion_pair_aware.csv"
GPU="${1:-1}"

submission_rows() {
  local attempt output
  for attempt in $(seq 1 5); do
    if output="$("$KAGGLE" competitions submissions -c "$COMPETITION" -v 2>/dev/null)"; then
      printf '%s\n' "$output"
      return
    fi
    echo "Kaggle submission query failed on attempt $attempt/5" >&2
    sleep 5
  done
  return 1
}

submitted_filename() {
  submission_rows | grep -F ",$1," | head -1 || true
}

daily_submission_count() {
  local utc_date
  utc_date="$(date -u +%F)"
  submission_rows | awk -F, -v date="$utc_date" 'NR > 1 && index($3, date) == 1 {count++} END {print count + 0}'
}

submit_candidate() {
  local file="$1"
  local message="$2"
  local name
  name="$(basename "$file")"
  if [[ -n "$(submitted_filename "$name")" ]]; then
    echo "Already submitted: $name"
    return
  fi
  local count
  count="$(daily_submission_count)"
  if (( count >= 5 )); then
    echo "Daily Kaggle limit reached before $name" >&2
    return 1
  fi
  "$PYTHON" -B "$ROOT/scripts/submit_kaggle_curl.py" \
    --competition "$COMPETITION" --file "$file" --message "$message"
  for _ in $(seq 1 30); do
    row="$(submitted_filename "$name")"
    echo "$row"
    if [[ "$row" == *"SubmissionStatus.COMPLETE"* ]]; then
      return
    fi
    if [[ "$row" == *"SubmissionStatus.ERROR"* || "$row" == *"SubmissionStatus.CANCELLED"* ]]; then
      return 1
    fi
    sleep 10
  done
  return 1
}

gpu_launch_allowed() {
  [[ -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')" ]] || return 1
  local hour busy
  hour=$((10#$(TZ=Asia/Shanghai date +%H)))
  if (( hour >= 8 && hour < 18 )); then
    busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | sed '/^$/d' | wc -l)"
    (( 8 - busy >= 5 )) || return 1
  fi
}

while [[ ! -f "$AUDIT" ]]; do
  sleep 60
done

"$PYTHON" - "$AUDIT" "$PRIMARY" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

audit = json.loads(Path(sys.argv[1]).read_text())
submission = Path(sys.argv[2])
assets = audit.get("primary", {}).get("model_asset_bytes", audit.get("model_asset_bytes"))
if assets is None or assets >= 100_000_000:
    raise RuntimeError(f"Release audit does not prove a legal asset size: {assets}")
table = pd.read_csv(submission)
if len(table) != 405 or table["path"].duplicated().any() or not table["prediction"].between(0, 39).all():
    raise RuntimeError("Primary CSV validation failed")
PY

submit_candidate "$PRIMARY" \
  "V24 full-data five-source INT8 parameter soup plus pair-aware decoder; audited under 100MB"

while ! gpu_launch_allowed; do
  sleep 60
done
bash "$ROOT/experiments/v24_full_refit/run_pseudo_distill.sh" "$GPU"

changed="$($PYTHON - "$PRIMARY" "$PSEUDO" <<'PY'
import sys
import pandas as pd

left = pd.read_csv(sys.argv[1]).set_index("path")["prediction"]
right = pd.read_csv(sys.argv[2]).set_index("path")["prediction"]
if not left.index.equals(right.index):
    raise RuntimeError("Candidate CSV paths are misaligned")
print(int((left != right).sum()))
PY
)"
echo "Pseudo candidate changes $changed rows versus primary"
if (( changed >= 2 && changed <= 120 )); then
  submit_candidate "$PSEUDO" \
    "V24 full-data INT8 soup plus all-row V13-teacher self-training and pair-aware decoder; local direct gain 15; under 100MB"
else
  echo "Pseudo candidate rejected by information gate: changed=$changed"
fi

local_promotion="$($PYTHON - <<'PY'
import json
from pathlib import Path

path = Path("experiments/v29_crop_system_fusion/outputs/direct_pairaware.json")
print(int(path.is_file() and json.loads(path.read_text()).get("promote", False)))
PY
)"
if (( local_promotion == 1 )); then
  while ! gpu_launch_allowed; do
    sleep 60
  done
  bash "$ROOT/experiments/v26_system_distillation/run_test_distill.sh" "$GPU"
  bash "$ROOT/experiments/v29_crop_system_fusion/run_test_fusion.sh" "$GPU"
  fused_changed="$($PYTHON - "$PSEUDO" "$FUSED" <<'PY'
import sys
import pandas as pd

left = pd.read_csv(sys.argv[1]).set_index("path")["prediction"]
right = pd.read_csv(sys.argv[2]).set_index("path")["prediction"]
if not left.index.equals(right.index):
    raise RuntimeError("V24 and V29 submissions are misaligned")
print(int((left != right).sum()))
PY
)"
  echo "Fused candidate changes $fused_changed rows versus hard pseudo"
  if (( fused_changed >= 2 && fused_changed <= 120 )); then
    submit_candidate "$FUSED" \
      "V29 system-distilled and wider-crop multi-forward fusion; local direct 1177/1357; audited under 100MB"
  else
    echo "Fused candidate rejected by information gate: changed=$fused_changed"
  fi
fi

submission_rows > "$ROOT/experiments/v24_full_refit/outputs/reports/kaggle_submissions_after_v24.csv"
