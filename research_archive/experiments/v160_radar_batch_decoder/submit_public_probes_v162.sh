#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="$HOME/miniconda3/envs/Train/bin/python"
KAGGLE=(env KAGGLE_CONFIG_DIR="$HOME/.kaggle" "$HOME/miniconda3/envs/Train/bin/kaggle")
TARGET_DATE="${TARGET_DATE:?TARGET_DATE must be YYYY-MM-DD}"
TARGET=$(date -d "$TARGET_DATE 08:02:00" +%s)
NOW=$(date +%s)
if (( NOW >= TARGET )); then
  echo "Target reset has passed: $TARGET_DATE" >&2
  exit 2
fi
sleep $((TARGET - NOW))
LOG="$ROOT/experiments/v160_radar_batch_decoder/v162_public_probe_submit.log"
"$PYTHON" "$ROOT/experiments/v160_radar_batch_decoder/make_public_probes_v162.py" > "$ROOT/experiments/v160_radar_batch_decoder/v162_public_probes.json"
BASE_SCORE=$("${KAGGLE[@]}" competitions submissions -c cuhk-x-competition-small-model-track --csv 2>/dev/null \
  | awk -F, '$2 == "v161_selected.csv" && $5 == "SubmissionStatus.COMPLETE" {print $6; exit}')
BASE_SCORE="${BASE_SCORE:-0.92039}"
SELECTED=()

submit_and_score() {
  local file="$1"
  local message="$2"
  "${KAGGLE[@]}" competitions submit -c cuhk-x-competition-small-model-track -f "$file" -m "$message" >> "$LOG" 2>&1
  local score=""
  for attempt in $(seq 1 24); do
    score=$("${KAGGLE[@]}" competitions submissions -c cuhk-x-competition-small-model-track --csv 2>/dev/null \
      | awk -F, -v file="$(basename "$file")" '$2 == file && $5 == "SubmissionStatus.COMPLETE" {print $6; exit}')
    if [[ -n "$score" ]]; then
      printf '%s %s\n' "$(basename "$file")" "$score" | tee -a "$LOG" >&2
      echo "$score"
      return 0
    fi
    sleep 15
  done
  echo "No completed score for $file" | tee -a "$LOG" >&2
  return 1
}

for name in row217_back row335_back row367_back row333_forward; do
  file="$ROOT/experiments/v160_radar_batch_decoder/candidates/v162_probe_${name}.csv"
  score=$(submit_and_score "$file" "V162 public row probe ${name}; baseline ${BASE_SCORE}")
  if awk -v value="$score" -v base="$BASE_SCORE" 'BEGIN {exit !(value > base)}'; then
    SELECTED+=("$name")
  fi
done

FINAL="$ROOT/experiments/v160_radar_batch_decoder/candidates/v162_selected.csv"
"$PYTHON" "$ROOT/experiments/v160_radar_batch_decoder/make_public_probes_v162.py" \
  --select "${SELECTED[@]}" --output "$FINAL" >> "$LOG"
submit_and_score "$FINAL" "V162 selected public probes: ${SELECTED[*]:-none}; baseline ${BASE_SCORE}"
printf 'baseline=%s selected=%s\n' "$BASE_SCORE" "${SELECTED[*]:-none}" | tee -a "$LOG"
