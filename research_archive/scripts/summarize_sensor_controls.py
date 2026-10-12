import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.validation_splits import load_diagnostic_split


def read_run(name, fold, records):
    folder = ROOT / "outputs/checkpoints" / name
    completion = json.loads((folder / "completion.json").read_text())
    config = json.loads((folder / "config.json").read_text())
    assert completion["status"] == "complete" and not config["pretrained"] and not config["encoder_sources"]
    assert (folder / "best.pt").stat().st_size == completion["checkpoint_bytes"] < 100_000_000
    expected_train, expected_inner, expected_outer = load_diagnostic_split(ROOT / config["diagnostic_split"], records, fold)
    split = json.loads((folder / "split.json").read_text())
    mask = np.load(ROOT / "data/processed/sensor_mask.npy")[:, 0] > 0
    assert split["train"] == [index for index in expected_train if mask[index]]
    assert split["valid"] == expected_inner and split["outer_valid"] == expected_outer
    diagnostic = json.loads((folder / "diagnostic_best.json").read_text())
    history = [json.loads(line) for line in (folder / "diagnostic_history.jsonl").read_text().splitlines()]
    assert diagnostic["inner_valid"]["accuracy"] == max(row["inner_valid"]["accuracy"] for row in history)
    predictions = {}
    for partition, filename, indices in [("clean_train", "clean_train.npz", split["train"]),
                                         ("inner_valid", "validation.npz", expected_inner),
                                         ("outer_valid", "outer_valid.npz", expected_outer)]:
        with np.load(folder / filename) as values:
            np.testing.assert_array_equal(values["indices"], indices)
            np.testing.assert_array_equal(values["labels"], [records[index]["label"] for index in indices])
            probability = values["probabilities"]
            assert np.isfinite(probability).all()
            np.testing.assert_allclose(probability.sum(1), 1, atol=1e-5)
            correct = probability.argmax(1) == values["labels"]
            assert abs(correct.mean() - diagnostic[partition]["accuracy"]) < 1e-12
            predictions[partition] = correct
    users = np.asarray([records[index]["user"] for index in expected_outer])
    result = {"name": name, "fold": fold, "epoch": diagnostic["epoch"],
              "accuracy": {partition: diagnostic[partition]["accuracy"] for partition in predictions},
              "outer_balanced_accuracy": diagnostic["outer_valid"]["balanced_accuracy"],
              "outer_available_accuracy": float(predictions["outer_valid"][mask[expected_outer]].mean()),
              "outer_per_user": {str(user): float(predictions["outer_valid"][users == user].mean()) for user in np.unique(users)},
              "checkpoint_bytes": completion["checkpoint_bytes"],
              "checkpoint_sha256": hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest()}
    return result, predictions


def main():
    records = get_records()
    results, comparisons = [], {}
    for fold in [0, 1]:
        baseline_name = "validation_v1_skeleton_f0" if fold == 0 else "representation_v1_coordinates_f1"
        baseline, baseline_predictions = read_run(baseline_name, fold, records)
        results.append(baseline)
        for recipe, name in [("centered", f"representation_v1_centered_f{fold}"),
                             ("invariant", f"representation_v1_invariant_f{fold}"),
                             ("mixup", f"regularization_v1_mixup_f{fold}")]:
            result, predictions = read_run(name, fold, records)
            results.append(result)
            comparison = {}
            for partition in predictions:
                reference, candidate = baseline_predictions[partition], predictions[partition]
                comparison[partition] = {"gain_percentage_points": float((candidate.mean() - reference.mean()) * 100),
                                         "corrected": int((candidate & ~reference).sum()),
                                         "regressed": int((~candidate & reference).sum())}
            comparisons.setdefault(recipe, {})[str(fold)] = comparison
    report = {"status": "complete", "runs": results, "paired_comparisons": comparisons,
              "notes": ["All checkpoints selected by inner holdout only, not outer maximum",
                        "Whole-clip indices, raw split, prediction labels and reported metrics verified",
                        "Per-user support is small; paired clip counts are descriptive, not independent-user significance",
                        "Both folds have historical exposure; these are not official test results"]}
    destination = ROOT / "outputs/study_representations/sensor_controls_summary.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(result["name"], {name: round(value * 100, 2) for name, value in result["accuracy"].items()})
    print("Verified and saved", destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
