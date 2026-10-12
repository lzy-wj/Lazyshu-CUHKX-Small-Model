import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "outputs/study_validation"
MODALITIES = "scene_ir,thermal,scene_depth,skeleton,imu,radar"


def free_gpus(reserved):
    candidates = []
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    for row in output.splitlines():
        gpu, memory = map(int, row.split(","))
        if gpu not in [2, 3, 7] or gpu in reserved or memory > 1024:
            continue
        active = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if active.strip():
            continue
        with (ROOT / f"outputs/logs/gpu-{gpu}.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            candidates.append(gpu)
    return candidates


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with (DESTINATION / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = json.loads((DESTINATION / "cpu_pipeline_audit.json").read_text())
        if audit["status"] != "passed":
            raise RuntimeError("CPU pipeline audit must pass")
        for path, digest in audit["source_sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Source changed after audit: {path}")
        pending = [
            {"name": "validation_smoke_concat", "smoke": True},
            {"name": "validation_v1_skeleton_f0", "modalities": "skeleton"},
            {"name": "validation_v1_concat_f0", "modalities": MODALITIES},
        ]
        for job in pending:
            if (ROOT / f"outputs/checkpoints/{job['name']}").exists():
                raise RuntimeError(f"Existing diagnostic run requires inspection: {job['name']}")
        running, completed = [], {}
        smoke_passed, failure = False, None
        deadline = time.monotonic() + 6 * 3600
        while pending or running:
            for active in running[:]:
                status = active["process"].poll()
                if status is None:
                    continue
                active["handle"].close()
                running.remove(active)
                job = active["job"]
                if status != 0:
                    if not (ROOT / f"outputs/checkpoints/{job['name']}").exists():
                        pending.insert(0, job)
                        print("GPU reservation refused; requeued", job["name"], flush=True)
                    else:
                        failure = f"{job['name']} failed; no new jobs will start"
                        pending.clear()
                elif job.get("smoke"):
                    smoke_passed = True
                    completed[job["name"]] = {"status": "passed"}
                    print("CUDA nested-validation smoke passed", flush=True)
                else:
                    folder = ROOT / f"outputs/checkpoints/{job['name']}"
                    completed[job["name"]] = json.loads((folder / "diagnostic_best.json").read_text())
                    print("Completed", job["name"], json.dumps(completed[job["name"]]), flush=True)
            if time.monotonic() > deadline:
                failure = "Six-hour queue budget exceeded; no further jobs will start"
                pending.clear()
            if not failure:
                for gpu in free_gpus({active["gpu"] for active in running}):
                    if not pending or not smoke_passed and not pending[0].get("smoke"):
                        break
                    job = pending.pop(0)
                    if job.get("smoke"):
                        command = ["bash", "scripts/run_validation_smoke.sh", str(gpu)]
                    else:
                        command = ["bash", "scripts/run_training.sh", str(gpu), job["name"], "--strict-scratch", "--canonical-pose",
                                   "--diagnostic-split", "configs/same_user_diagnostic_split.json", "--modalities", job["modalities"],
                                   "--fold", "0", "--seed", "42", "--epochs", "60", "--batch-size", "24", "--frames", "16",
                                   "--image-size", "160", "--lr", "0.001", "--workers", "4", "--eval-every", "2", "--patience", "60"]
                    handle = (ROOT / f"outputs/logs/{job['name']}.launcher.log").open("a")
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                    running.append({"job": job, "gpu": gpu, "process": process, "handle": handle})
                    print("Started", job["name"], "GPU", gpu, flush=True)
            status = "failed" if failure else "running" if running else "waiting_for_gpu" if pending else "complete"
            report = {"status": status, "failure": failure, "cuda_smoke_passed": smoke_passed,
                      "pending": [job["name"] for job in pending],
                      "running": [{"name": active["job"]["name"], "gpu": active["gpu"]} for active in running],
                      "completed": completed, "selection_partition": "same-user held-out clips, not the cross-user diagnostic set"}
            temporary = DESTINATION / "report.tmp.json"
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(DESTINATION / "report.json")
            if pending or running:
                time.sleep(20)
        (ROOT / "outputs/logs/validation_study.exit").write_text("1\n" if failure else "0\n")
        if failure:
            raise RuntimeError(failure)


if __name__ == "__main__":
    main()
