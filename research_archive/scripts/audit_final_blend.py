import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_indices, get_records


def rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    bundle = ROOT / "outputs/final_scratch_blend_v2"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert len(manifest["models"]) == 4
    assert [entry["weight"] for entry in manifest["models"]] == [0.375, 0.375, 0.125, 0.125]
    for name, metadata in manifest["files"].items():
        path = bundle / name
        assert path.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    inventory = {str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()}
    assert inventory == set(manifest["files"]) | {"manifest.json"}
    cached_rows = rows(ROOT / "outputs/submissions/scratch_blend_v2.csv")
    raw_rows = rows(ROOT / "outputs/predictions/scratch_blend_v2_raw.csv")
    assert rows(bundle / "submission.csv") == cached_rows == raw_rows
    assert len(cached_rows) == len({row["path"] for row in cached_rows}) == 405
    test_csv = ROOT / "data/downloads/CUHK-X_Small_Model_Track/Small-Model-Track/Testing/test_file/test.csv"
    assert [row["path"] for row in cached_rows] == [row["path"] for row in rows(test_csv)]
    assert all(0 <= int(row["prediction"]) < 40 for row in cached_rows)
    with np.load(ROOT / "outputs/predictions/scratch_blend_v2.npz") as cached, np.load(ROOT / "outputs/predictions/scratch_blend_v2_raw.npz") as raw:
        difference = np.abs(cached["probabilities"] - raw["probabilities"])
        max_difference = float(difference.max())
        changed = int((cached["probabilities"].argmax(1) != raw["probabilities"].argmax(1)).sum())
        np.testing.assert_allclose(cached["probabilities"], raw["probabilities"], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(raw["probabilities"].sum(1), 1, atol=1e-6)
        assert changed == 0
    full_indices, _ = get_indices(get_records(), -1)
    models = []
    expected = [("full_input_v1_staged42", "transformer", 42, 20),
                ("full_input_v1_staged43", "transformer", 43, 9),
                ("full_input_v1_cross42", "cross_transformer", 42, 9),
                ("full_input_v1_cross43", "cross_transformer", 43, 7)]
    for entry, (name, fusion, seed, epoch) in zip(manifest["models"], expected):
        state = torch.load(bundle / entry["file"], map_location="cpu", weights_only=False)
        config = state["config"]
        assert config["name"] == name and config["fusion"] == fusion and config["seed"] == seed and state["epoch"] == epoch
        assert config["fold"] == -1 and config["train_clips"] == 3036 and config["valid_clips"] == 0
        assert config["strict_scratch"] and not config["pretrained"] and not config.get("weights_source")
        assert config["modalities"] == ["skeleton", "imu", "thermal"]
        assert config["thermal_view"] == "motion_crop" and config["imu_representation"] == "world"
        assert len(config["encoder_provenance"]) == 3 and all(source["fold"] == -1 for source in config["encoder_provenance"])
        split = json.loads((ROOT / "outputs/checkpoints" / name / "split.json").read_text())
        assert split == {"train": full_indices, "valid": []}
        dtypes = {str(value.dtype) for value in state["model"].values() if value.is_floating_point()}
        assert dtypes == {"torch.float32"}
        models.append({"name": name, "fusion": fusion, "seed": seed, "epoch": epoch,
                       "parameters": config["parameters"], "bytes": entry["bytes"], "dtypes": sorted(dtypes)})
    validation = json.loads((ROOT / "outputs/study_staged_spatial/fullfold_cross.json").read_text())
    assert validation["status"] == "complete" and validation["decision"] == "promote_to_full_refit"
    assert validation["blend"]["all_validation"]["accuracy"] > validation["self"]["all_validation"]["accuracy"]
    assert all(validation["folds"][str(fold)]["blend"]["all_validation"]["accuracy"] >= validation["folds"][str(fold)]["self"]["all_validation"]["accuracy"] for fold in [0, 1])
    refit = json.loads((ROOT / "configs/full_cross_refit.json").read_text())
    assert refit["official_test_labels_used"] is False and refit["ensemble"]["weights"] == [0.375, 0.375, 0.125, 0.125]
    total = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    assert manifest["model_bytes"] == sum(entry["bytes"] for entry in manifest["models"])
    assert total < 100_000_000
    archive = ROOT / "outputs/final_scratch_blend_v2.zip"
    with zipfile.ZipFile(archive) as handle:
        archived = {name: handle.read(name) for name in handle.namelist() if not name.endswith("/")}
    expected_archive = {f"final_scratch_blend_v2/{path.relative_to(bundle)}": path.read_bytes()
                        for path in bundle.rglob("*") if path.is_file()}
    assert archived.keys() == expected_archive.keys()
    assert all(hashlib.sha256(archived[name]).digest() == hashlib.sha256(value).digest() for name, value in expected_archive.items())
    report = {"status": "passed", "test_rows": 405, "raw_vs_cached_changed_predictions": changed,
              "max_probability_delta": max_difference, "model_bytes": manifest["model_bytes"], "bundle_bytes": total,
              "archive_bytes": archive.stat().st_size, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "models": models, "development_validation": {"old_accuracy": validation["self"]["all_validation"]["accuracy"],
                  "new_accuracy": validation["blend"]["all_validation"]["accuracy"],
                  "delta": validation["paired_blend_vs_self"]["accuracy_delta"],
                  "macro_f1": validation["blend"]["all_validation"]["macro_f1"]},
              "checks": ["payload hashes and exact inventory", "full-data split and source provenance", "scratch-only architecture",
                         "FP32 checkpoint storage", "fixed validation-selected weights", "raw versus cached probability replay",
                         "test CSV order, uniqueness and class range", "100 MB payload limit", "zip contents match bundle"],
              "official_test_accuracy": None, "submission_sent_to_kaggle": False}
    destination = ROOT / "outputs/study_staged_spatial/final_blend_audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
