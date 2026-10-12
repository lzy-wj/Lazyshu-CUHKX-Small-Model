import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    bundle = ROOT / "outputs/final_v2"
    manifest = json.loads((bundle / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    original = np.load(ROOT / "outputs/predictions/research_multimodal_v2.npz")
    recheck = np.load(ROOT / "outputs/predictions/research_multimodal_v2_recheck.npz")
    np.testing.assert_array_equal(original["indices"], recheck["indices"])
    np.testing.assert_allclose(original["probabilities"], recheck["probabilities"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(original["probabilities"].sum(1), 1, atol=1e-6, rtol=0)
    np.testing.assert_allclose(np.average(original["per_model"], axis=0, weights=original["weights"]),
                               original["probabilities"], atol=1e-8, rtol=0)
    assert original["probabilities"].shape == (405, 40)
    assert np.isfinite(original["probabilities"]).all()
    csv_path = bundle / "research_multimodal_v2.csv"
    repeated_csv = ROOT / "outputs/submissions/research_multimodal_v2_recheck.csv"
    assert csv_path.read_bytes() == repeated_csv.read_bytes()
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    np.testing.assert_array_equal([int(row["prediction"]) for row in rows],
                                  original["probabilities"].argmax(1))
    first_version = json.loads((ROOT / "outputs/final/manifest.json").read_text())
    for model in first_version["models"]:
        path = ROOT / "outputs/final/models" / f"{model['config']['name']}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == model["sha256"]
    report = {"status": "passed", "csv_exact_match": True,
              "probability_max_absolute_difference": float(np.abs(original["probabilities"] - recheck["probabilities"]).max()),
              "rows": len(rows), "model_hashes_verified": True, "v1_model_hashes_unchanged": True,
              "test_accuracy": None,
              "scope": "Reloaded packaged classifier weights with existing audited preprocessing caches; not a raw-data detector rerun."}
    (bundle / "reproducibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
