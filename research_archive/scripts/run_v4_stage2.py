import json
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPUS = [2, 3, 7]


def completed(name):
    path = ROOT / f"outputs/logs/{name}.exit"
    if not path.exists():
        return False
    if path.read_text().strip() != "0":
        raise RuntimeError(f"Experiment {name} failed; inspect its log rather than silently retrying")
    assert (ROOT / f"outputs/checkpoints/{name}/best.pt").exists()
    return True


def free_gpus():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    available = []
    for row in output.splitlines():
        gpu, memory = map(int, row.split(","))
        if gpu not in GPUS or memory > 1024:
            continue
        with (ROOT / f"outputs/logs/gpu-{gpu}.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            fcntl.flock(handle, fcntl.LOCK_UN)
        available.append(gpu)
    return available


def run_script(script, *arguments):
    subprocess.run([sys.executable, f"scripts/{script}", *arguments], cwd=ROOT, check=True)


def main():
    deadline = time.monotonic() + 4 * 3600
    print("Waiting for both fold-0 video candidates; no new fold-1 metrics are read before selection.", flush=True)
    while not all(completed(name) for name in ["v4_detail_f0", "v4_jointstats_f0"]):
        if time.monotonic() > deadline:
            raise TimeoutError("Fold-0 training did not finish within four hours")
        time.sleep(30)
    selected_path = ROOT / "configs/v4_selected_ensemble.json"
    if not selected_path.exists():
        run_script("calibrate_v4.py")
    selected = json.loads(selected_path.read_text())
    print("Selection frozen:", json.dumps(selected), flush=True)
    jobs = []
    if "v4_jointstats_f0" in selected["names"]:
        jobs.append({"name": "v4_jointstats_f1", "kind": "fold1"})
    for source, final in zip(selected["names"], selected["final_models"]):
        if final.startswith("v4_"):
            jobs.append({"name": final, "kind": "refit", "source": source})
    pending = [job for job in jobs if not completed(job["name"])]
    running = []
    failure = None
    while pending or running:
        if time.monotonic() > deadline:
            raise TimeoutError("V4 jobs exceeded the four-hour orchestration budget")
        for job in running[:]:
            status = job["process"].poll()
            if status is not None:
                job["handle"].close()
                running.remove(job)
                if status != 0:
                    if not (ROOT / f"outputs/logs/{job['name']}.log").exists():
                        pending.append({key: value for key, value in job.items() if key in ["name", "kind", "source"]})
                        print("GPU reservation was refused before training; requeueing", job["name"], flush=True)
                    else:
                        failure = f"Job {job['name']} failed; other running jobs will finish before exit"
                        pending.clear()
                        print(failure, flush=True)
                elif not completed(job["name"]):
                    failure = f"Job {job['name']} did not record successful completion"
                    pending.clear()
                else:
                    print("Completed", job["name"], flush=True)
        if failure:
            if not running:
                raise RuntimeError(failure)
            time.sleep(30)
            continue
        available = [gpu for gpu in free_gpus() if gpu not in [job["gpu"] for job in running]]
        for gpu in available:
            if not pending:
                break
            job = pending.pop(0)
            if job["kind"] == "refit":
                command = ["bash", "scripts/run_recipe.sh", str(gpu), job["source"], job["name"]]
            else:
                command = ["bash", "scripts/run_training.sh", str(gpu), job["name"], "--fold", "1",
                           "--modalities", "person_ir,person_thermal,skeleton,geometry,imu_world", "--temporal-backbone", "statistics",
                           "--backbone", "x3d_s", "--kinetics", "--image-size", "224", "--frames", "13",
                           "--epochs", "60", "--patience", "24", "--batch-size", "16", "--workers", "6",
                           "--lr", "0.0005", "--backbone-lr-factor", "0.15", "--canonical-pose", "--strong-augment",
                           "--mixup", "0.1", "--ema-decay", "0.97", "--freeze-backbone-bn"]
            handle = (ROOT / f"outputs/logs/{job['name']}_launcher.log").open("a")
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            running.append(job | {"gpu": gpu, "process": process, "handle": handle})
            print("Launched", job["name"], "on GPU", gpu, flush=True)
        if pending or running:
            time.sleep(30)
    for name in selected["names"]:
        confirmation = name.replace("_f0", "_f1")
        while not completed(confirmation):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Waiting too long for {confirmation}")
            time.sleep(30)
    run_script("confirm_ensemble.py", "--config", str(selected_path), "--output-prefix", "v4_ensemble")
    while not free_gpus():
        if time.monotonic() > deadline:
            raise TimeoutError("No idle GPU for inference")
        time.sleep(30)
    gpu = free_gpus()[0]
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "1",
                               "TORCH_HOME": str(ROOT / ".tools/torch")}
    subprocess.run(["flock", "-n", f"outputs/logs/gpu-{gpu}.lock", sys.executable, "scripts/predict_v4.py"],
                   cwd=ROOT, env=environment, check=True)
    print("V4 confirmation, full-data refits, prediction and packaged-weight recheck completed.", flush=True)


if __name__ == "__main__":
    main()
