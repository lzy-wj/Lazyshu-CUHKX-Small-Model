#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "experiments/scene_context_v13/outputs/followup_queue_state.json"
LOG = ROOT / "experiments/scene_context_v13/outputs/followup_queue_events.jsonl"
JOBS = [
    ("v13_radar_point_f0", "run_radar_candidate.sh", ["v13_radar_point_f0", "0"]),
    ("v13_radar_point_f1", "run_radar_candidate.sh", ["v13_radar_point_f1", "1"]),
    ("v13_resnet50_f0", "run_resnet_candidate.sh", ["v13_resnet50_f0", "0"]),
    ("v13_resnet50_f1", "run_resnet_candidate.sh", ["v13_resnet50_f1", "1"]),
]


def event(kind, **values):
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind} | values
    with LOG.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def gpu_inventory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    )
    return {
        int(line.split(",", 1)[0]): line.split(",", 1)[1].strip()
        for line in output.strip().splitlines()
    }


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


def main():
    pending = list(JOBS)
    running = {}
    completed = []
    failed = {}
    inventory = gpu_inventory()
    event("queue_started", pending=[job[0] for job in pending])
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
            name, runner, arguments = pending.pop(0)
            command = [
                "bash",
                f"experiments/scene_context_v13/{runner}",
                str(gpu),
                *arguments,
            ]
            process = subprocess.Popen(command, cwd=ROOT)
            running[name] = {"process": process, "gpu": gpu}
            event("job_started", name=name, gpu=gpu, pid=process.pid)
        STATE.write_text(
            json.dumps(
                {
                    "pending": [job[0] for job in pending],
                    "running": {
                        name: {"gpu": item["gpu"], "pid": item["process"].pid}
                        for name, item in running.items()
                    },
                    "completed": completed,
                    "failed": failed,
                },
                indent=2,
            )
            + "\n"
        )
        if pending or running:
            time.sleep(30)
    event("queue_finished", completed=completed, failed=failed)


if __name__ == "__main__":
    main()
