#!/usr/bin/env bash
set -euo pipefail
cd ${LAZYSHU_PROJECT_ROOT}
candidate="experiments/v127_supervision_routed_system/candidates/v127_medium_source_coverage.csv"
log="experiments/v127_supervision_routed_system/submit_medium.log"
target=$(date -d "today 08:02:00" +%s)
now=$(date +%s)
if (( now > target )); then target=$(date -d "tomorrow 08:02:00" +%s); fi
sleep $((target-now))
KAGGLE_CONFIG_DIR="$HOME/.kaggle" "$HOME/miniconda3/envs/Train/bin/kaggle" competitions submit \
  -c cuhk-x-competition-small-model-track -f "$candidate" \
  -m "V127 supervision-routed medium candidate; 1205/1357 local; 11 model-derived changes" \
  > "$log" 2>&1
sleep 90
KAGGLE_CONFIG_DIR="$HOME/.kaggle" "$HOME/miniconda3/envs/Train/bin/kaggle" competitions submissions \
  -c cuhk-x-competition-small-model-track | head -8 >> "$log" 2>&1
