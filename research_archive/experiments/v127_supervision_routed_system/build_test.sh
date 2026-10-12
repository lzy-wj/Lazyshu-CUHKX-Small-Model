#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$HOME/miniconda3/envs/Train/bin/python" \
  "$ROOT/experiments/v125_domain_complement/build_test_candidate.py" \
  --unmatched-system V71 \
  --unmatched "$ROOT/experiments/v24_full_refit/outputs/submissions/v71_gated_film_sequence_hybrid.csv" \
  --output "$ROOT/experiments/v127_supervision_routed_system/candidates/v127_supervision_routed.csv"
