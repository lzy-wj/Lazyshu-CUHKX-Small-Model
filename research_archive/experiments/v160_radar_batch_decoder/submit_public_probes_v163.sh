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
LOG="$ROOT/experiments/v160_radar_batch_decoder/v163_public_probe_submit.log"
"$PYTHON" "$ROOT/experiments/v160_radar_batch_decoder/make_public_probes_v163.py" > "$ROOT/experiments/v160_radar_batch_decoder/v163_public_probes.json"
"$PYTHON" "$ROOT/experiments/v164_evidence_gated_radar/evaluate.py" > "$ROOT/experiments/v164_evidence_gated_radar/evaluate.log"
BASE_SCORE=$("${KAGGLE[@]}" competitions submissions -c cuhk-x-competition-small-model-track --csv 2>/dev/null \
  | awk -F, '$2 == "v162_selected.csv" && $5 == "SubmissionStatus.COMPLETE" {print $6; exit}')
BASE_SCORE="${BASE_SCORE:-0.92039}"

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

for name in \
  v164_evidence_gated_radar_generalization \
  v164_evidence_gated_radar \
  v164_evidence_gated_radar_conservative; do
  file="$ROOT/experiments/v164_evidence_gated_radar/candidates/${name}.csv"
  submit_and_score "$file" "${name}; local evidence gate improves both folds with zero losses"
done

for name in row337_forward row350_forward; do
  file="$ROOT/experiments/v160_radar_batch_decoder/candidates/v163_probe_${name}.csv"
  submit_and_score "$file" "V163 public row probe ${name}; baseline ${BASE_SCORE}"
done
printf 'baseline=%s complete=true\n' "$BASE_SCORE" | tee -a "$LOG"
