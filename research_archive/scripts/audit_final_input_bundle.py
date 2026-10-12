import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_indices, get_records


def rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    bundle = ROOT / "outputs/final_scratch_input"
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, metadata in manifest["files"].items():
        path = bundle / name
        assert path.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    expected = set(manifest["files"]) | {"manifest.json"}
    assert {str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()} == expected
    cached = rows(ROOT / "outputs/submissions/scratch_input_v1.csv")
    independent = rows(ROOT / "outputs/predictions/scratch_input_v1_raw.csv")
    assert rows(bundle / "submission.csv") == cached == independent
    assert len(cached) == len({row["path"] for row in cached}) == 405
    source = ROOT / "data/downloads/CUHK-X_Small_Model_Track/Small-Model-Track/Testing/test_file/test.csv"
    assert [row["path"] for row in cached] == [row["path"] for row in rows(source)]
    assert all(0 <= int(row["prediction"]) < 40 for row in cached)
    with np.load(ROOT / "outputs/predictions/scratch_input_v1.npz") as original, np.load(ROOT / "outputs/predictions/scratch_input_v1_raw.npz") as raw:
        difference = float(np.abs(original["probabilities"] - raw["probabilities"]).max())
        np.testing.assert_allclose(original["probabilities"], raw["probabilities"], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(raw["probabilities"].sum(1), 1, atol=1e-6)
    full, _ = get_indices(get_records(), -1)
    models = []
    for entry in manifest["models"]:
        state = torch.load(bundle / entry["file"], map_location="cpu", weights_only=False)
        config = state["config"]
        assert config["fold"] == -1 and config["train_clips"] == 3036 and config["valid_clips"] == 0
        split = json.loads((ROOT / "outputs/checkpoints" / config["name"] / "split.json").read_text())
        assert split["train"] == full and split["valid"] == []
        assert not config["pretrained"] and len(config["encoder_provenance"]) == 3
        assert all(source["fold"] == -1 for source in config["encoder_provenance"])
        dtypes = {str(value.dtype) for value in state["model"].values() if value.is_floating_point()}
        assert dtypes == {"torch.float32"}
        models.append({"name": config["name"], "seed": config["seed"], "epochs": state["epoch"], "parameters": config["parameters"], "storage_dtypes": sorted(dtypes)})
    total = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    assert total < 100_000_000
    stages = ["study_inputs", "study_input_fusion", "study_input_replication", "study_full_inputs", "study_full_staged42", "study_full_staged43"]
    for stage in stages:
        assert json.loads((ROOT / "outputs" / stage / "report.json").read_text())["status"] == "complete"
    report = {"status": "passed", "test_rows": 405, "raw_vs_cached_changed_predictions": 0, "max_probability_delta": difference,
              "model_bytes": manifest["model_bytes"], "bundle_bytes": total, "models": models,
              "checks": ["all payload hashes and inventory", "full-data training indices", "no external pretrained weights",
                         "same full-data encoder provenance", "FP32 on-disk storage", "raw/cached probability replay", "CSV order and class range", "all training stages completed"],
              "official_test_accuracy": None, "submission_sent_to_kaggle": False}
    (ROOT / "outputs/study_inputs/final_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
