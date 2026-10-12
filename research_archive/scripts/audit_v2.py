import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_imu_world import rotate
from src.data import ROOT, get_indices, get_records


def main():
    records = get_records()
    assert len(records) == 3441
    assert all(record["index"] == index for index, record in enumerate(records))
    assert len({record["id"] for record in records}) == len(records)
    assert all(record["label"] == -1 for record in records if record["split"] == "test")
    report = {"clips": len(records), "folds": {}, "cache_shapes": {}, "hashes": {}}
    for fold in [0, 1]:
        train, valid = get_indices(records, fold)
        assert not set(train) & set(valid)
        assert all(records[index]["split"] == "train" for index in train + valid)
        report["folds"][str(fold)] = {
            "train": len(train), "valid": len(valid),
            "train_users": sorted({records[index]["user"] for index in train}),
            "valid_users": sorted({records[index]["user"] for index in valid}),
        }
    vectors = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(rotate(vectors, np.array([[1.0, 0.0, 0.0, 0.0]])), vectors)
    np.testing.assert_allclose(rotate(vectors, np.array([[1.0, 0.0, 0.0, 1.0]])),
                               [[0.0, 1.0, 0.0]], atol=1e-6)
    paths = ["data/metadata/samples.json", "data/metadata/person_boxes.npz",
             "scripts/prepare_person_crops.py", "scripts/prepare_imu_world.py",
             "scripts/extract_visual_features.py"]
    for name, channels in [("skeleton", 153), ("geometry", 408), ("imu_world", 85)]:
        path = ROOT / f"data/processed/{name}.npy"
        array = np.load(path, mmap_mode="r")
        assert array.shape == (3441, 64, channels)
        assert np.isfinite(array).all()
        report["cache_shapes"][name] = list(array.shape)
        paths.append(str(path.relative_to(ROOT)))
    boxes = np.load(ROOT / "data/metadata/person_boxes.npz")
    assert boxes["boxes"].shape == (3441, 2, 4)
    assert np.isfinite(boxes["boxes"]).all()
    assert ((boxes["boxes"] >= 0) & (boxes["boxes"] <= 1)).all()
    assert (boxes["boxes"][..., 2:] > boxes["boxes"][..., :2]).all()
    report["detected_ir_thermal"] = (boxes["confidence"] > 0).sum(0).tolist()
    mask = np.load(ROOT / "data/processed/person_mask.npy")
    assert mask.shape == (3441, 3)
    assert np.isin(mask, [0, 1]).all()
    report["present_ir_thermal_depth"] = mask.sum(0).astype(int).tolist()
    for record in records:
        array = np.load(ROOT / f"data/processed/person_16_224/{record['index']:05d}.npy", mmap_mode="r")
        assert array.shape == (16, 9, 224, 224) and array.dtype == np.uint8
    for name in paths:
        with (ROOT / name).open("rb") as handle:
            report["hashes"][name] = hashlib.file_digest(handle, "sha256").hexdigest()
    report["status"] = "passed"
    report["note"] = "Cache shape/finiteness, subject isolation and rotation arithmetic checks; not a test-accuracy estimate."
    (ROOT / "outputs/predictions/v2_data_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
