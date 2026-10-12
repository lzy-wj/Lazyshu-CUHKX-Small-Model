#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "experiments/scene_context_v13/outputs/frame_resnet_queue_state.json"
LOG = ROOT / "experiments/scene_context_v13/outputs/frame_resnet_queue_events.jsonl"
JOBS = [("v13_frame_resnet_f0", "0"), ("v13_frame_resnet_f1", "1")]


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
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {line.split(",", 1)[0].strip() for line in output.strip().splitlines() if line.strip()}


def write_event(kind, **values):
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind} | values
    with LOG.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def main():
    pending = list(JOBS)
    running = {}
    completed = []
    failed = {}
    inventory = gpu_inventory()
    while pending or running:
        for name, item in list(running.items()):
            status = item["process"].poll()
            if status is None:
                continue
            if status == 0:
                completed.append(name)
                write_event("job_completed", name=name, gpu=item["gpu"])
            else:
                failed[name] = status
                write_event("job_failed", name=name, gpu=item["gpu"], status=status)
            del running[name]
        busy = occupied_uuids()
        reserved = {item["gpu"] for item in running.values()}
        free = [
            index for index, uuid in inventory.items() if uuid not in busy and index not in reserved
        ]
        while pending and free:
            gpu = free.pop(0)
            name, fold = pending.pop(0)
            process = subprocess.Popen(
                [
                    "bash", "experiments/scene_context_v13/run_frame_resnet_candidate.sh",
                    str(gpu), name, fold,
                ],
                cwd=ROOT,
            )
            running[name] = {"process": process, "gpu": gpu}
            write_event("job_started", name=name, gpu=gpu, pid=process.pid)
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


if __name__ == "__main__":
    main()
