import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_validation_queue import free_gpus
from src.data import ROOT, get_records
from src.validation_splits import make_diagnostic_split


def main():
    destination = ROOT / "outputs/study_representations"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([sys.executable, "scripts/audit_representations.py"], cwd=ROOT, check=True)
        split = make_diagnostic_split(get_records(), fold=1)
        split_path = ROOT / "configs/same_user_diagnostic_split_f1.json"
        if split_path.exists() and json.loads(split_path.read_text()) != split:
            raise RuntimeError("Refusing to replace a different frozen fold-1 split")
        split_path.write_text(json.dumps(split, indent=2) + "\n")
        jobs = [{"name": f"representation_v1_{representation}_f{fold}", "fold": fold, "representation": representation}
                for fold in [0, 1] for representation in ["coordinates", "centered", "invariant"]
                if (fold, representation) != (0, "coordinates")]
        sources = ["src/scratch_models.py", "src/data.py", "scripts/train.py", "src/validation_splits.py",
                   "scripts/run_representation_study.py", "scripts/audit_representations.py", "scripts/predict.py",
                   "configs/same_user_diagnostic_split.json", "configs/same_user_diagnostic_split_f1.json"]
        specification = {
            "jobs": jobs, "baseline_f0": "validation_v1_skeleton_f0", "seed": 42, "epochs": 60,
            "selection": "Each run selects its checkpoint on same-user inner validation only. Compare all three prespecified representations on both folds; no outer-based early stopping.",
            "promotion": "No production replacement unless outer accuracy improves on both folds, with no inner-accuracy collapse. Both outer folds have historical exposure and are not pristine tests.",
            "hypothesis": "Centering removes absolute root height/trajectory; pair distances and bone angles remove rigid coordinate and uniform size differences. These may also discard action cues.",
            "confounds": "Invariant input has 136 rather than 51 channels per stream and recomputes derivatives after input augmentation; this is a representation recipe comparison, not an exactly parameter-matched isolated intervention.",
            "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}}
        if (destination / "specification.json").exists():
            raise RuntimeError("Study already exists; inspect before rerunning")
        (destination / "specification.json").write_text(json.dumps(specification, indent=2) + "\n")
        for job in jobs:
            if (ROOT / f"outputs/checkpoints/{job['name']}").exists():
                raise RuntimeError(f"Existing checkpoint directory: {job['name']}")
        pending, running, completed = list(jobs), [], {}
        failure = None
        deadline = time.monotonic() + 3600
        while pending or running:
            for active in running[:]:
                status = active["process"].poll()
                if status is None:
                    continue
                active["handle"].close()
                running.remove(active)
                job = active["job"]
                folder = ROOT / f"outputs/checkpoints/{job['name']}"
                if status != 0:
                    if not folder.exists() and time.monotonic() < deadline:
                        pending.insert(0, job)
                    else:
                        failure = f"Failed: {job['name']}"
                        pending.clear()
                else:
                    completion = json.loads((folder / "completion.json").read_text())
                    assert completion["status"] == "complete"
                    completed[job["name"]] = json.loads((folder / "diagnostic_best.json").read_text())
                    print("Completed", job["name"], completed[job["name"]], flush=True)
            if time.monotonic() > deadline and pending:
                failure = "GPU queue deadline exceeded"
                pending.clear()
            if not failure:
                for gpu in free_gpus({active["gpu"] for active in running}):
                    if not pending:
                        break
                    job = pending.pop(0)
                    suffix = "_f1" if job["fold"] else ""
                    command = ["bash", "scripts/run_training.sh", str(gpu), job["name"], "--strict-scratch", "--canonical-pose",
                               "--diagnostic-split", f"configs/same_user_diagnostic_split{suffix}.json", "--modalities", "skeleton",
                               "--skeleton-representation", job["representation"], "--fold", str(job["fold"]), "--seed", "42",
                               "--epochs", "60", "--batch-size", "24", "--frames", "16", "--image-size", "160", "--lr", "0.001",
                               "--workers", "4", "--eval-every", "2", "--patience", "60"]
                    handle = (ROOT / f"outputs/logs/{job['name']}.launcher.log").open("a")
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                    running.append({"job": job, "gpu": gpu, "process": process, "handle": handle})
                    print("Started", job["name"], "GPU", gpu, flush=True)
            report = {"status": "failed" if failure else "running" if running or pending else "complete", "failure": failure,
                      "pending": [job["name"] for job in pending],
                      "running": [{"name": active["job"]["name"], "gpu": active["gpu"]} for active in running], "completed": completed}
            temporary = destination / "report.tmp.json"
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(destination / "report.json")
            if pending or running:
                time.sleep(10)
        if failure:
            raise RuntimeError(failure)


if __name__ == "__main__":
    main()
