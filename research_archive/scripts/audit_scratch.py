import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset, get_indices, get_records, prepare_batch
from src.scratch_models import MODALITIES, ScratchMultiModalModel, validate_scratch_config
from src.scratch_protocol import ablation_metrics, modality_presence


def main():
    torch.set_num_threads(4)
    configuration = json.loads((ROOT / "configs/scratch_ablation.json").read_text())
    for job in configuration["jobs"]:
        validate_scratch_config(configuration | job)
    rejected = []
    for forbidden in [{"imagenet": True}, {"kinetics": True}, {"init": "anything.pt"},
                      {"skeleton_backbone": "stgcn"}, {"skeleton_backbone": "attention"},
                      {"temporal_backbone": "statistics"}, {"temporal_backbone": "pyramid"},
                      {"backbone": "x3d_s"}, {"modalities": ["person_ir"]},
                      {"modalities": ["ir_features"]}, {"modalities": ["geometry"]},
                      {"fusion": "pointnet"}, {"mixup": 0.2}]:
        try:
            validate_scratch_config({"modalities": ["skeleton"]} | forbidden)
        except ValueError:
            rejected.append(forbidden)
        else:
            raise AssertionError(f"Forbidden configuration accepted: {forbidden}")
    batch = {name: torch.randn(2, 32, channels) for name, channels in [("skeleton", 153), ("imu", 85), ("radar", 28)]}
    batch.update({name: torch.randn(2, 4, 3, 32, 32) for name in MODALITIES[:3]})
    batch.update({name + "_mask": torch.tensor([1., 0.]) for name in MODALITIES})
    outcomes = {}
    for job in configuration["jobs"]:
        torch.manual_seed(42)
        model = ScratchMultiModalModel(job["modalities"], job["fusion"])
        output, _ = model(batch)
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1]))
        loss.backward()
        assert output.shape == (2, 40) and torch.isfinite(output).all()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        model.eval()
        with torch.no_grad():
            expected = model(batch)[0]
            modified = {name: values.clone() for name, values in batch.items()}
            for name in job["modalities"]:
                modified[name][1] = float("nan")
            torch.testing.assert_close(model(modified)[0], expected)
            torch.testing.assert_close(expected[1], model.class_prior_logits)
            archive = io.BytesIO()
            torch.save(model.state_dict(), archive)
            assert archive.tell() < 100_000_000
            outcomes[job["name"]] = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                                      "serialized_state_bytes": archive.tell()}
            archive.seek(0)
            restored = ScratchMultiModalModel(job["modalities"], job["fusion"]).eval()
            restored.load_state_dict(torch.load(archive, weights_only=True))
            torch.testing.assert_close(restored(batch)[0], expected, rtol=0, atol=0)
            if len(job["modalities"]) == 6:
                for name in job["modalities"]:
                    partial = {key: values.clone() for key, values in batch.items()}
                    partial[name + "_mask"][0] = 0
                    reference = model(partial)[0]
                    partial[name][0] = float("nan")
                    torch.testing.assert_close(model(partial)[0], reference)
    records = get_records()
    train, valid = get_indices(records, 0)
    assert not ({records[index]["user"] for index in train} & {records[index]["user"] for index in valid})
    dataset = ClipDataset([train[0], valid[0]], list(MODALITIES), frames=4, image_size=160, canonical_pose=True)
    sample = prepare_batch(next(iter(DataLoader(dataset, batch_size=2))), "cpu")
    assert all(torch.isfinite(values).all() for values in sample.values())
    assert sample["scene_depth"].shape == (2, 4, 3, 160, 160)
    presence = modality_presence(list(MODALITIES))
    np.testing.assert_array_equal(sample["scene_depth_mask"].numpy(), presence[[train[0], valid[0]], 2])
    labels = np.array([records[index]["label"] for index in valid])
    predictions = {"indices": np.array(valid), "labels": labels, "probabilities": np.eye(40)[labels]}
    metrics = ablation_metrics(predictions, records, ["radar"])
    assert metrics["all_validation"]["accuracy"] == 1.0 and metrics["all_validation"]["macro_f1"] == 1.0
    assert metrics["available_count"] == presence[valid, -1].sum()
    torch.manual_seed(42)
    concat = ScratchMultiModalModel(list(MODALITIES), "concat")
    torch.manual_seed(42)
    transformer = ScratchMultiModalModel(list(MODALITIES), "transformer")
    for name, values in concat.encoders.state_dict().items():
        torch.testing.assert_close(values, transformer.encoders.state_dict()[name], rtol=0, atol=0)
    old_manifest = ROOT / "outputs/final_v3/manifest.json"
    if old_manifest.exists():
        for artifact in json.loads(old_manifest.read_text())["artifacts"]:
            assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    report = {"status": "passed", "rejected": rejected, "models": outcomes,
              "training_clips": len(train), "validation_clips": len(valid),
              "training_presence": presence[train].sum(0).tolist(),
              "validation_presence": presence[valid].sum(0).tolist(),
              "scope": "CPU gradients, missing/NaN mask invariance, exact reload, identical fusion encoder initialization, user isolation, raw-scene loading, coverage metrics, preserved old artifacts"}
    destination = ROOT / "outputs/study_scratch"
    report["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
        "src/scratch_models.py", "src/scratch_protocol.py", "src/data.py", "scripts/train.py",
        "scripts/audit_scratch.py", "scripts/run_scratch_study.py", "configs/scratch_ablation.json"]}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
