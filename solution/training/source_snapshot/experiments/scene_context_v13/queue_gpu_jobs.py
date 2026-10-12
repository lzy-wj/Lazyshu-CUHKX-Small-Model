#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "experiments/scene_context_v13/outputs/gpu_queue_state.json"
LOG = ROOT / "experiments/scene_context_v13/outputs/gpu_queue_events.jsonl"


JOBS = [
    {
        "name": "v13_x3dl_sync_f0",
        "arguments": ["v13_x3dl_sync_f0", "0", "person_ir,person_thermal,skeleton,imu_world", "224"],
    },
    {
        "name": "v13_x3dl_sync_f1",
        "arguments": ["v13_x3dl_sync_f1", "1", "person_ir,person_thermal,skeleton,imu_world", "224"],
    },
    {
        "name": "v13_x3dl_detail_f0",
        "arguments": ["v13_x3dl_detail_f0", "0", "person_ir_detail,person_thermal_detail,skeleton,imu_world", "160"],
    },
    {
        "name": "v13_x3dl_detail_f1",
        "arguments": ["v13_x3dl_detail_f1", "1", "person_ir_detail,person_thermal_detail,skeleton,imu_world", "160"],
    },
]


def event(kind, **values):
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind} | values
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def gpu_inventory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    inventory = {}
    for line in output.strip().splitlines():
        index, uuid = [value.strip() for value in line.split(",", 1)]
        inventory[int(index)] = uuid
    return inventory


def occupied_uuids():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {line.split(",", 1)[0].strip() for line in output.strip().splitlines() if line.strip()}


def save_state(pending, running, completed, failed):
    state = {
        "pending": [job["name"] for job in pending],
        "running": {name: {"gpu": item["gpu"], "pid": item["process"].pid} for name, item in running.items()},
        "completed": completed,
        "failed": failed,
    }
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def main():
    pending = []
    completed = []
    for job in JOBS:
        exit_path = ROOT / f"experiments/scene_context_v13/outputs/logs/{job['name']}.exit"
        checkpoint = ROOT / f"outputs/checkpoints/{job['name']}/best.pt"
        if exit_path.is_file() and exit_path.read_text().strip() == "0" and checkpoint.is_file():
            completed.append(job["name"])
        else:
            pending.append(job)
    running = {}
    failed = {}
    inventory = gpu_inventory()
    event("queue_started", pending=[job["name"] for job in pending], completed=completed)
    while pending or running:
        for name, item in list(running.items()):
            status = item["process"].poll()
            if status is None:
                continue
            if status == 0:
                completed.append(name)
                event("job_completed", name=name, gpu=item["gpu"])
            else:
                failed[name] = status
                event("job_failed", name=name, gpu=item["gpu"], status=status)
            del running[name]
        occupied = occupied_uuids()
        reserved = {item["gpu"] for item in running.values()}
        free = [index for index, uuid in inventory.items() if uuid not in occupied and index not in reserved]
        while pending and free:
            gpu = free.pop(0)
            job = pending.pop(0)
            command = [
                "bash",
                "experiments/scene_context_v13/run_x3dl_candidate.sh",
                str(gpu),
                *job["arguments"],
            ]
            process = subprocess.Popen(command, cwd=ROOT)
            running[job["name"]] = {"process": process, "gpu": gpu}
            event("job_started", name=job["name"], gpu=gpu, pid=process.pid)
        save_state(pending, running, completed, failed)
        if pending or running:
            time.sleep(30)
    event("queue_finished", completed=completed, failed=failed)


if __name__ == "__main__":
    main()
