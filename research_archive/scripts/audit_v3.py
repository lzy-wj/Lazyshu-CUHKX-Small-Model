import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_relative_imu import multiply
from src.data import ROOT, get_indices, get_records
from src.models import MobileVideoEncoder, StatisticalEncoder


def main():
    torch.set_num_threads(1)
    report = {"folds": {}, "caches": {}, "source_sha256": {}}
    records = get_records()
    assert len(records) == 3441 and all(record["index"] == index for index, record in enumerate(records))
    for fold in [0, 1]:
        train, valid = get_indices(records, fold)
        assert not set(train) & set(valid)
        assert all(records[index]["split"] == "train" for index in train + valid)
        report["folds"][str(fold)] = {"train": len(train), "valid": len(valid)}
    assert sum(record["split"] == "test" for record in records) == 405
    assert all(record["label"] == -1 for record in records if record["split"] == "test")
    for name, channels in [("imu_timed", 105), ("imu_relative", 95)]:
        path = ROOT / f"data/processed/{name}.npy"
        values = np.load(path, mmap_mode="r")
        mask = np.load(ROOT / f"data/processed/{name}_mask.npy")
        assert values.shape == (3441, 64, channels) and np.isfinite(values).all()
        assert mask.shape == (3441,) and np.isin(mask, [0, 1]).all()
        assert not values[mask == 0].any()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        report["caches"][name] = {"shape": list(values.shape), "present": int(mask.sum()), "sha256": digest}
    generator = np.random.default_rng(42)
    chest, arm, global_rotation = [generator.normal(size=(16, 4)) for _ in range(3)]
    for values in [chest, arm, global_rotation]:
        values /= np.linalg.norm(values, axis=1, keepdims=True)
    inverse = chest.copy()
    inverse[:, 1:] *= -1
    expected = multiply(inverse, arm)
    rotated_chest = multiply(global_rotation, chest)
    rotated_chest[:, 1:] *= -1
    actual = multiply(rotated_chest, multiply(global_rotation, arm))
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    values = torch.arange(6 * 16 * 4.0).reshape(6, 16, 2, 2).requires_grad_()
    shifted = MobileVideoEncoder.shift(values, 3).reshape(2, 3, 16, 2, 2)
    original = values.reshape(2, 3, 16, 2, 2)
    torch.testing.assert_close(shifted[:, :-1, :2], original[:, 1:, :2])
    torch.testing.assert_close(shifted[:, :, 4:], original[:, :, 4:])
    assert not shifted[:, -1, :2].any()
    shifted.sum().backward()
    assert torch.isfinite(values.grad).all()
    encoder = StatisticalEncoder(95, pyramid=True).eval()
    assert encoder(torch.zeros(2, 64, 95)).shape == (2, 192)
    for name in ["src/data.py", "src/models.py", "scripts/train.py", "scripts/predict.py",
                 "scripts/prepare_imu_timed.py", "scripts/prepare_relative_imu.py", "configs/video-requirements.txt"]:
        report["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    report["status"] = "passed"
    report["scope"] = "User isolation, masks and finite caches, relative-rotation invariance, temporal-shift boundaries/gradients, pyramid shape. Not a test-score estimate."
    (ROOT / "outputs/predictions/v3_data_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
