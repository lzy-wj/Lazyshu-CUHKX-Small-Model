import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset, get_indices, get_records, prepare_batch
from src.encoder_initialization import load_encoder_sources
from src.scratch_models import ScratchMultiModalModel, validate_scratch_config
from src.scratch_protocol import modality_presence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/input_fusion_study.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/study_input_fusion"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    configuration = json.loads((ROOT / args.config).read_text())
    records = get_records()
    job = next(job for job in configuration["jobs"] if job["fusion"] == "transformer" and job.get("fold", configuration.get("fold", 0)) == 0)
    options = {"thermal_view": "motion_crop", "imu_representation": "world"}
    model = ScratchMultiModalModel(job["modalities"], fusion="transformer", **options)
    train, valid = get_indices(records, 0)
    presence = modality_presence(job["modalities"]).any(1)
    train = [index for index in train if presence[index]]
    provenance = load_encoder_sources(model, job["encoder_sources"], records, train, valid, 0)
    assert len(provenance) == 3
    incompatible = ScratchMultiModalModel(job["modalities"], fusion="transformer")
    try:
        load_encoder_sources(incompatible, job["encoder_sources"], records, train, valid, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Representation mismatch accepted during encoder loading")
    dataset = ClipDataset(train[:2], job["modalities"], frames=4, image_size=160, canonical_pose=True, **options)
    batch = {name: torch.as_tensor(np.stack([dataset[position][name] for position in range(2)])) for name in dataset[0]}
    batch = prepare_batch(batch, torch.device("cpu"))
    output, auxiliary = model(batch)
    loss = output.square().mean() + sum(values.square().mean() for values in auxiliary.values())
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for candidate in configuration["jobs"]:
        validate_scratch_config(configuration | candidate)
    sources = ["src/data.py", "src/scratch_models.py", "src/encoder_initialization.py", "scripts/train.py", "scripts/run_scratch_study.py",
               str(args.config), "scripts/audit_fusion_inputs.py", "data/metadata/thermal_motion.json", "data/processed/imu_world.npy"]
    report = {"status": "passed", "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "checks": ["same-fold encoder loading", "mismatched input rejection", "finite fused gradients", "preset config validation"],
              "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}}
    destination = ROOT / args.output
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Passed", report["checks"], "parameters", report["parameters"])


if __name__ == "__main__":
    main()
