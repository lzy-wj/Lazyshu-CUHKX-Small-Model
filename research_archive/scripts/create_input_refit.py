import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_indices, get_records
from src.scratch_models import validate_scratch_config
from src.scratch_protocol import modality_presence


def build_job(specification, prior_jobs):
    names = specification["sources"]
    if len(names) != 2:
        raise ValueError("A full-data recipe requires exactly two user-fold sources")
    sources = []
    for name in names:
        folder = ROOT / "outputs/checkpoints" / name
        config = json.loads((folder / "config.json").read_text())
        completion = json.loads((folder / "completion.json").read_text())
        metrics = json.loads((folder / "metrics.json").read_text())
        validate_scratch_config(config)
        if completion["status"] != "complete" or config.get("diagnostic_split") or not config["strict_scratch"]:
            raise ValueError("Only completed original user-fold scratch experiments can select full-data training")
        if config.get("source_sha256", {}).get("data/metadata/samples.json") != hashlib.sha256((ROOT / "data/metadata/samples.json").read_bytes()).hexdigest():
            raise ValueError("Source metadata hash mismatch")
        if metrics["epoch"] != completion["best_epoch"] or metrics["accuracy"] != completion["best_accuracy"]:
            raise ValueError("Source selection records disagree")
        if config.get("mixup", 0) or config.get("single_modality_mixup") or config.get("strong_augment") or config.get("spatial_augment"):
            raise ValueError("This input-study refit does not silently transfer additional augmentation recipes")
        if not config.get("canonical_pose") or config.get("ema_decay", 0.99) != 0.99 or config.get("auxiliary_weight", 0.15) != 0.15:
            raise ValueError("Source differs from the fixed input-study training protocol")
        train, valid = get_indices(get_records(), config["fold"])
        split = json.loads((folder / "split.json").read_text())
        presence = modality_presence(config["modalities"]).any(1)
        if split["train"] != [index for index in train if presence[index]] or split["valid"] != valid:
            raise ValueError("Source split mismatch")
        if completion["checkpoint_bytes"] != (folder / "best.pt").stat().st_size:
            raise ValueError("Source checkpoint changed")
        sources.append((config, metrics, hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest()))
    if {source[0]["fold"] for source in sources} != {0, 1}:
        raise ValueError("Use one source from each user fold")
    reference = sources[0][0]
    defaults = {"skeleton_representation": "coordinates", "imu_representation": "raw", "thermal_view": "full", "visual_multiplier": 1}
    keys = ["modalities", "seed", "batch_size", "frames", "image_size", "lr", "fusion", *defaults]
    for key in keys:
        if sources[0][0].get(key, defaults.get(key)) != sources[1][0].get(key, defaults.get(key)):
            raise ValueError(f"Fold recipes differ in {key}")
    full, _ = get_indices(get_records(), -1)
    presence = modality_presence(reference["modalities"]).any(1)
    full_clips = sum(bool(presence[index]) for index in full)
    full_steps = full_clips // reference["batch_size"]
    selected_updates, scheduled_updates, warmup_updates = [], [], []
    for config, metrics, _ in sources:
        steps = config["train_clips"] // config["batch_size"]
        selected_updates.append(metrics["epoch"] * steps)
        scheduled_updates.append((config.get("schedule_epochs") or config["epochs"]) * steps)
        warmup_updates.append(config.get("warmup_epochs", 5) * steps)
    job = {key: reference.get(key, defaults.get(key)) for key in keys}
    job.update({"name": specification["name"], "fold": -1,
                "epochs": max(1, round(float(np.mean(selected_updates)) / full_steps)),
                "schedule_epochs": max(1, round(float(np.mean(scheduled_updates)) / full_steps)),
                "warmup_epochs": float(np.mean(warmup_updates)) / full_steps})
    initialized = [bool(config.get("encoder_sources")) for config, _, _ in sources]
    if initialized[0] != initialized[1]:
        raise ValueError("Fold initialization protocols differ")
    full_sources = specification.get("encoder_sources", [])
    if initialized[0] != bool(full_sources):
        raise ValueError("Full-data recipe must preserve the selected initialization stage explicitly")
    if full_sources:
        if len(full_sources) != len(job["modalities"]):
            raise ValueError("Provide one full-data source per modality")
        for source in full_sources:
            expected_name = Path(source).parent.name
            if expected_name not in prior_jobs:
                raise ValueError("Full encoder must be a preceding job in this recipe")
        job["encoder_sources"] = full_sources
    provenance = {"name": job["name"], "sources": names, "source_sha256": [source[2] for source in sources],
                  "source_best_epochs": [source[1]["epoch"] for source in sources],
                  "selected_optimizer_updates": selected_updates, "full_training_clips": full_clips, "full_steps_per_epoch": full_steps,
                  "full_epochs": job["epochs"], "full_schedule_epochs": job["schedule_epochs"], "full_warmup_epochs": job["warmup_epochs"]}
    return job, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads(args.specification.read_text())
    jobs, provenance = [], []
    for entry in specification["jobs"]:
        job, audit = build_job(entry, {job["name"] for job in jobs})
        jobs.append(job)
        provenance.append(audit)
    config = {"name": specification["name"], "workers": 4, "eval_every": 2, "patience": 120, "gpus": [2, 3, 7],
              "queue_hours": 3, "jobs": jobs,
              "protocol": "Full-data refit. Match mean best optimizer-update budget from two user folds, preserving warmup and cosine horizon. No validation or test-based checkpoint selection during refit.",
              "provenance": provenance}
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    paths = ["src/data.py", "src/scratch_models.py", "src/encoder_initialization.py", "scripts/train.py", "scripts/run_scratch_study.py",
             "scripts/create_input_refit.py", str(args.output), str(args.specification)]
    audit = {"status": "passed", "scope": "Completed two-fold recipe, split, initialization and update-budget validation; model forward/backward covered by input/fusion audits and completed training",
             "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
