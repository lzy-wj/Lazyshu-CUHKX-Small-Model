#!/usr/bin/env bash

set -euo pipefail

V24_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$V24_DIR/../.." && pwd)"
PYTHON="${PYTHON:-${HOME}/miniconda3/envs/Train/bin/python}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
OUTPUT_ROOT="$V24_DIR/outputs"

export PATH="$(dirname "$PYTHON"):$PATH"

mkdir -p "$OUTPUT_ROOT/logs" "$V24_DIR/run_metadata"

gpu_processes() {
  local gpu="$1"
  nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null | awk 'NF'
}

gpu_is_idle() {
  local gpu="$1"
  [[ -z "$(gpu_processes "$gpu")" ]]
}

assert_gpu_launch_allowed() {
  local gpu="$1"
  local gpu_count target_was_idle=0
  gpu_count="$(nvidia-smi --list-gpus | wc -l)"
  if ! [[ "$gpu" =~ ^[0-9]+$ ]] || (( gpu < 0 || gpu >= gpu_count )); then
    echo "Invalid GPU index: $gpu" >&2
    return 1
  fi
  if gpu_is_idle "$gpu"; then
    target_was_idle=1
  elif [[ "${ALLOW_SHARED_GPU:-0}" != "1" ]]; then
    echo "GPU $gpu already has a compute process; refusing to launch:" >&2
    gpu_processes "$gpu" >&2
    return 1
  else
    echo "GPU $gpu is shared by explicit user permission:" >&2
    gpu_processes "$gpu" >&2
  fi

  local hour free=0 index required_free
  hour=$((10#$(TZ=Asia/Shanghai date +%H)))
  if (( hour >= 8 && hour < 18 )); then
    for ((index = 0; index < gpu_count; index++)); do
      if gpu_is_idle "$index"; then
        free=$((free + 1))
      fi
    done
    required_free=$((target_was_idle == 1 ? 5 : 4))
    if (( free < required_free )); then
      echo "Daytime launch would leave fewer than four GPUs idle (currently free: $free)." >&2
      return 1
    fi
  fi
}

assert_new_training_run() {
  local name="$1"
  local destination="$OUTPUT_ROOT/checkpoints/$name"
  if [[ -e "$destination" ]]; then
    echo "Training destination already exists; archive it before rerunning: $destination" >&2
    return 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file is missing: $path" >&2
    return 1
  fi
}

run_logged() {
  local run_name="$1"
  local gpu="$2"
  shift 2
  local -a command=("$@")

  assert_gpu_launch_allowed "$gpu"

  local timestamp log_path metadata_path command_text pid status
  timestamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  log_path="$OUTPUT_ROOT/logs/${run_name}_${timestamp}.log"
  metadata_path="$V24_DIR/run_metadata/${run_name}_${timestamp}.json"
  printf -v command_text '%q ' "${command[@]}"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "${command[@]}"
  ) > >(tee "$log_path") 2>&1 &
  pid=$!

  "$PYTHON" - "$metadata_path" "$pid" "$gpu" "$log_path" "$command_text" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

path, pid, gpu, log_path, command = sys.argv[1:]
payload = {
    "status": "running",
    "pid": int(pid),
    "gpu": int(gpu),
    "command": command.strip(),
    "start_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "log_path": log_path,
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY

  set +e
  wait "$pid"
  status=$?
  set -e

  "$PYTHON" - "$metadata_path" "$status" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": "complete" if int(sys.argv[2]) == 0 else "failed",
    "exit_code": int(sys.argv[2]),
    "end_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
})
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
  return "$status"
}
