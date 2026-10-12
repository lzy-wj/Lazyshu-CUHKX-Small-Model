import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT
from src.scratch_models import validate_scratch_config


def free_gpus(allowed, reserved):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    available = []
    for row in output.splitlines():
        gpu, memory = map(int, row.split(","))
        if gpu not in allowed or gpu in reserved or memory > 1024:
            continue
        processes = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if processes.strip():
            continue
        with (ROOT / f"outputs/logs/gpu-{gpu}.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            available.append(gpu)
    return available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scratch_ablation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/study_scratch")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/logs").mkdir(parents=True, exist_ok=True)
    with (destination / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = json.loads((args.audit or destination / "audit.json").read_text())
        if audit["status"] != "passed":
            raise RuntimeError("Scratch audit must pass before training")
        for name, digest in audit["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Source changed since scratch audit: {name}")
        pending, running, results = [], [], {}
        for job in config["jobs"]:
            validate_scratch_config(config | job)
            checkpoint = ROOT / "outputs/checkpoints" / job["name"]
            exit_path = ROOT / "outputs/logs" / (job["name"] + ".exit")
            if exit_path.exists() and exit_path.read_text().strip() == "0" and (checkpoint / "completion.json").exists():
                results[job["name"]] = json.loads((checkpoint / "completion.json").read_text())
                continue
            if checkpoint.exists() or exit_path.exists():
                raise RuntimeError(f"Incomplete existing experiment {job['name']}; inspect rather than silently overwrite")
            pending.append(job)
        failure = None
        deadline = time.monotonic() + config.get("queue_hours", 6) * 3600
        while pending or running:
            for active in running[:]:
                status = active["process"].poll()
                if status is None:
                    continue
                active["handle"].close()
                running.remove(active)
                name = active["job"]["name"]
                completion = ROOT / f"outputs/checkpoints/{name}/completion.json"
                if status != 0 or not completion.exists():
                    if not completion.parent.exists() and time.monotonic() < deadline:
                        pending.insert(0, active["job"])
                        print("GPU reservation refused; requeued", name, flush=True)
                        continue
                    failure = f"{name} failed with status {status}; no further jobs launched"
                    results[name] = {"status": "failed", "exit": status}
                    pending.clear()
                else:
                    results[name] = json.loads(completion.read_text())
                    detail = ROOT / f"outputs/checkpoints/{name}/ablation_metrics.json"
                    if (config | active["job"]).get("fold", 0) >= 0:
                        results[name]["validation"] = json.loads(detail.read_text())
                    else:
                        results[name]["validation"] = None
                    print("Completed", name, results[name]["best_accuracy"], flush=True)
            if time.monotonic() > deadline and pending:
                failure = "Queue deadline exceeded; no further jobs launched"
                pending.clear()
            if not failure:
                for gpu in free_gpus(config["gpus"], {active["gpu"] for active in running}):
                    if not pending:
                        break
                    ready = next((position for position, candidate in enumerate(pending)
                                  if all((ROOT / source).exists() and ((ROOT / source).parent / "completion.json").exists()
                                         for source in (config | candidate).get("encoder_sources", []))), None)
                    if ready is None:
                        break
                    job = pending.pop(ready)
                    options = config | job
                    command = ["bash", "scripts/run_training.sh", str(gpu), job["name"], "--strict-scratch",
                               "--canonical-pose", "--modalities", ",".join(job["modalities"]), "--fusion", job["fusion"]]
                    for key in ["fold", "seed", "epochs", "batch_size", "frames", "image_size", "lr", "workers", "eval_every", "patience",
                                "visual_multiplier", "schedule_epochs", "warmup_epochs", "thermal_view", "imu_representation", "skeleton_representation",
                                "visual_encoder", "visual_feedforward", "diagnostic_split", "swanlab_mode", "auxiliary_weight"]:
                        if key in options:
                            command.extend(["--" + key.replace("_", "-"), str(options[key])])
                    if options.get("encoder_sources"):
                        command.extend(["--encoder-sources", *options["encoder_sources"]])
                    handle = (ROOT / f"outputs/logs/{job['name']}.launcher.log").open("w")
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                    running.append({"job": job, "gpu": gpu, "process": process, "handle": handle})
                    print("Started", job["name"], "GPU", gpu, flush=True)
            report = {"status": "failed" if failure else "running" if running or pending else "complete",
                      "failure": failure, "pending": [job["name"] for job in pending],
                      "running": [{"name": active["job"]["name"], "gpu": active["gpu"]} for active in running],
                      "results": results, "note": config.get("protocol", "Exploratory user-isolated validation scores; not official test accuracy")}
            temporary = destination / "report.tmp.json"
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(destination / "report.json")
            if running or pending:
                time.sleep(15)
        if failure:
            raise RuntimeError(failure)


if __name__ == "__main__":
    main()
