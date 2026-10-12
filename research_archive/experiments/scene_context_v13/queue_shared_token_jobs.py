#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "experiments/scene_context_v13/outputs/shared_token_queue_state.json"
LOG = ROOT / "experiments/scene_context_v13/outputs/shared_token_queue_events.jsonl"
CACHE = ROOT / "data/processed/person_16_160"
JOBS = [
    ("v13_shared_tokens_f0", "0", "shared"),
    ("v13_shared_tokens_f1", "1", "shared"),
    ("v13_scene_resnet50_f0", "0", "resnet"),
    ("v13_scene_resnet50_f1", "1", "resnet"),
]


def inventory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    )
    return {
        int(line.split(",", 1)[0]): line.split(",", 1)[1].strip()
        for line in output.strip().splitlines()
    }


def occupied():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {line.split(",", 1)[0].strip() for line in output.strip().splitlines() if line.strip()}


def event(kind, **values):
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind} | values
    with LOG.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def main():
    pending = list(JOBS)
    running = {}
    completed = []
    failed = {}
    gpus = inventory()
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
        cache_complete = CACHE.is_dir() and len(list(CACHE.glob("*.npy"))) == 3441
        busy = occupied()
        reserved = {item["gpu"] for item in running.values()}
        free = [index for index, uuid in gpus.items() if uuid not in busy and index not in reserved]
        while cache_complete and pending and free:
            gpu = free.pop(0)
            name, fold, kind = pending.pop(0)
            runner = (
                "experiments/scene_context_v13/run_shared_token_candidate.sh"
                if kind == "shared"
                else "experiments/scene_context_v13/run_scene_resnet_candidate.sh"
            )
            command = ["bash", runner, str(gpu), name, fold]
            process = subprocess.Popen(command, cwd=ROOT)
            running[name] = {"process": process, "gpu": gpu}
            event("job_started", name=name, gpu=gpu, pid=process.pid)
        STATE.write_text(
            json.dumps(
                {
                    "cache_complete": cache_complete,
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
