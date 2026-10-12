import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_person_detail import detail_box
from src.data import ROOT, ClipDataset, get_indices, get_records, prepare_batch
from src.models import JointAttentionEncoder, MultiModalModel


def main():
    torch.set_num_threads(4)
    records = get_records()
    report = {"folds": {}, "source_sha256": {}}
    for fold in [0, 1]:
        train, valid = get_indices(records, fold)
        assert not set(train) & set(valid)
        assert all(records[index]["split"] == "train" for index in train + valid)
        report["folds"][str(fold)] = {"train": len(train), "valid": len(valid)}
    assert sum(record["split"] == "test" for record in records) == 405
    assert all(record["label"] == -1 for record in records if record["split"] == "test")
    np.testing.assert_allclose(detail_box([0.1, 0.2, 0.9, 1], 0.8), [0.1, 0.2, 0.9, 0.776])
    np.testing.assert_array_equal(detail_box([0, 0, 1, 1], 0), [0, 0, 1, 1])
    masks = np.load(ROOT / "data/processed/person_detail_mask.npy")
    assert masks.shape == (3441, 2) and np.isin(masks, [0, 1]).all()
    digest = hashlib.sha256()
    for record in records:
        path = ROOT / f"data/processed/person_detail_16_160/{record['index']:05d}.npy"
        values = np.load(path, mmap_mode="r")
        assert values.dtype == np.uint8 and values.shape == (16, 6, 160, 160)
        for modality_index in range(2):
            if not masks[record["index"], modality_index]:
                assert not values[:, modality_index * 3:(modality_index + 1) * 3].any()
        with path.open("rb") as handle:
            file_digest = hashlib.file_digest(handle, "sha256").digest()
        digest.update(file_digest)
    report["detail_cache"] = {"clips": len(records), "available": masks.sum(0).tolist(),
                              "ordered_file_digest_sha256": digest.hexdigest()}
    modalities = ["person_ir_detail", "person_thermal_detail", "skeleton", "imu_world"]
    dataset = ClipDataset([0, 1], modalities, frames=13, canonical_pose=True, image_size=160)
    batch = prepare_batch(next(iter(DataLoader(dataset, batch_size=2))), "cpu")
    for name in modalities[:2]:
        assert batch[name].shape == (2, 13, 3, 160, 160)
        assert batch[name].min() >= -1 and batch[name].max() <= 1
    model = MultiModalModel(["skeleton", "geometry", "imu_world"], skeleton_backbone="attention",
                            temporal_backbone="statistics")
    assert isinstance(model.encoders["skeleton"], JointAttentionEncoder)
    sensors = {name: torch.randn(2, 64, size) for name, size in [("skeleton", 153), ("geometry", 408), ("imu_world", 85)]}
    sensors.update({name + "_mask": torch.ones(2) for name in model.modalities})
    output, _ = model(sensors)
    output.sum().backward()
    assert output.shape == (2, 40) and torch.isfinite(output).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    previous = json.loads((ROOT / "outputs/final_v3/manifest.json").read_text())
    for artifact in previous["artifacts"]:
        assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    report["v3_artifact_hashes_unchanged"] = True
    for name in ["scripts/prepare_person_detail.py", "src/data.py", "src/models.py", "scripts/train.py", "scripts/audit_v4.py"]:
        report["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    report["status"] = "passed"
    report["scope"] = "Fixed label-independent crop/fallback, complete cache shapes/masks/hashes, detail normalization, attention gradients, user isolation, preserved V3 artifacts."
    (ROOT / "outputs/predictions/v4_data_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
