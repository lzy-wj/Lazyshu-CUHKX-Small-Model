#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU index required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$HERE/selection.json" ]] || { echo "Run Fold0 stage and selection first" >&2; exit 2; }
K="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_iterations"])' "$HERE/selection.json")"
"$HERE/run_one.sh" "$GPU" 1 "$K"
python "$HERE/summarize_results.py"
