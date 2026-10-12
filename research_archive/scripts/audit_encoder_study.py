import copy
import hashlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import learning_rate_at_epoch
from src.data import ROOT, get_indices, get_records
from src.encoder_initialization import load_encoder_sources
from src.scratch_models import ScratchMultiModalModel, validate_scratch_config


def main():
    torch.set_num_threads(4)
    config_path = ROOT / "configs/encoder_study.json"
    config = json.loads(config_path.read_text())
    records = get_records()
    train, valid = get_indices(records, 0)
    outcomes = {}
    for job in config["jobs"]:
        options = config | job
        validate_scratch_config(options)
        model = ScratchMultiModalModel(job["modalities"], job["fusion"], options["visual_multiplier"])
        initial_heads = {name: values.clone() for name, values in model.state_dict().items() if not name.startswith("encoders.")}
        if job.get("encoder_sources"):
            lineage = load_encoder_sources(model, job["encoder_sources"], records, train, valid, 0)
            assert len(lineage) == len(job["modalities"])
            for name, values in initial_heads.items():
                torch.testing.assert_close(model.state_dict()[name], values, rtol=0, atol=0)
            for entry in lineage:
                checkpoint = torch.load(ROOT / entry["checkpoint"], map_location="cpu", weights_only=False)
                prefix = f"encoders.{entry['modality']}."
                for name, values in model.encoders[entry["modality"]].state_dict().items():
                    torch.testing.assert_close(values, checkpoint["model"][prefix + name], rtol=0, atol=0)
        batch = {name: torch.randn(2, 32, channels) for name, channels in [("skeleton", 153), ("imu", 85), ("radar", 28)]}
        batch.update({name: torch.randn(2, 4, 3, 32, 32) for name in ["scene_ir", "thermal", "scene_depth"]})
        batch.update({name + "_mask": torch.tensor([1., 0.]) for name in job["modalities"]})
        output, _ = model(batch)
        torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
        assert output.shape == (2, 40) and torch.isfinite(output).all()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        model.eval()
        with torch.no_grad():
            expected = model(batch)[0]
            for name in job["modalities"]:
                batch[name][1] = float("nan")
            torch.testing.assert_close(model(batch)[0], expected)
            archive = io.BytesIO()
            torch.save(model.state_dict(), archive)
            size = archive.tell()
            assert size < 100_000_000
            archive.seek(0)
            restored = ScratchMultiModalModel(job["modalities"], job["fusion"], options["visual_multiplier"]).eval()
            restored.load_state_dict(torch.load(archive, weights_only=True), strict=True)
            torch.testing.assert_close(restored(batch)[0], expected, rtol=0, atol=0)
        outcomes[job["name"]] = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                                  "serialized_state_bytes": size, "same_fold_encoder_sources": len(job.get("encoder_sources", []))}
    source_path = ROOT / "outputs/checkpoints/scratch_v1_depth_f0/best.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_split = json.loads((source_path.parent / "split.json").read_text())
    target = ScratchMultiModalModel(["scene_depth"])
    rejected = []
    for mutation in ["fold", "pretrained", "imagenet", "chain", "leaked_train", "wrong_valid", "metadata", "tensor_shape", "not_strict"]:
        with tempfile.TemporaryDirectory(prefix="encoder_audit_", dir=ROOT / "outputs/checkpoints") as directory:
            path = Path(directory)
            modified = copy.deepcopy(source)
            split = copy.deepcopy(source_split)
            if mutation == "fold":
                modified["config"]["fold"] = 1
            elif mutation == "pretrained":
                modified["config"]["pretrained"] = True
            elif mutation == "imagenet":
                modified["config"]["imagenet"] = True
            elif mutation == "chain":
                modified["config"]["encoder_sources"] = [str(source_path)]
            elif mutation == "leaked_train":
                split["train"].append(valid[0])
            elif mutation == "wrong_valid":
                split["valid"] = split["valid"][1:]
            elif mutation == "metadata":
                modified["config"]["source_sha256"]["data/metadata/samples.json"] = "wrong"
            elif mutation == "tensor_shape":
                key = next(name for name in modified["model"] if name.startswith("encoders."))
                modified["model"][key] = torch.zeros(1)
            elif mutation == "not_strict":
                modified["config"]["strict_scratch"] = False
            torch.save(modified, path / "best.pt")
            (path / "config.json").write_text(json.dumps(modified["config"]))
            (path / "split.json").write_text(json.dumps(split))
            (path / "completion.json").write_text(json.dumps({"status": "complete", "checkpoint_bytes": (path / "best.pt").stat().st_size}))
            before = {name: values.clone() for name, values in target.state_dict().items()}
            try:
                load_encoder_sources(target, [path / "best.pt"], records, train, valid, 0)
            except ValueError:
                rejected.append(mutation)
            else:
                raise AssertionError(f"Unsafe initialization accepted: {mutation}")
            for name, values in before.items():
                torch.testing.assert_close(target.state_dict()[name], values, rtol=0, atol=0)
    staged = next(job for job in config["jobs"] if job.get("encoder_sources"))
    duplicate = list(staged["encoder_sources"])
    duplicate[1] = duplicate[0]
    try:
        load_encoder_sources(ScratchMultiModalModel(staged["modalities"]), duplicate, records, train, valid, 0)
    except ValueError:
        rejected.append("duplicate_modality")
    else:
        raise AssertionError("Duplicate source accepted")
    for epoch in range(1, 61):
        previous = .001 * min(1, epoch / 5) * (.05 + .95 * (1 + math.cos(math.pi * epoch / 60)) / 2)
        assert learning_rate_at_epoch(.001, epoch, 60, 5) == previous
    assert all(learning_rate_at_epoch(.001, epoch, 60, 5) == .00005 for epoch in range(61, 121))
    for name in ["scratch_v1_depth_f0", "scratch_v1_concat_f0", "scratch_v1_transformer_f0"]:
        checkpoint = torch.load(ROOT / f"outputs/checkpoints/{name}/best.pt", map_location="cpu", weights_only=False)
        restored = ScratchMultiModalModel(checkpoint["config"]["modalities"], checkpoint["config"]["fusion"])
        restored.load_state_dict(checkpoint["model"], strict=True)
    paths = ["src/scratch_models.py", "src/encoder_initialization.py", "src/scratch_protocol.py", "src/data.py",
             "scripts/train.py", "scripts/predict.py", "scripts/train_from_recipe.py", "scripts/audit_encoder_study.py",
             "scripts/run_scratch_study.py", "configs/encoder_study.json"]
    report = {"status": "passed", "models": outcomes, "rejected": rejected,
              "scope": "Source provenance, exact encoder-only loading, no partial mutation on rejection, gradients, NaN masking, serialization, legacy state compatibility and duration schedule",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}}
    destination = ROOT / "outputs/study_encoders"
    destination.mkdir(exist_ok=True)
    (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
