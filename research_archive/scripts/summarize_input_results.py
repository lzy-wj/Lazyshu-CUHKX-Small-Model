import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_indices, get_records
from src.scratch_protocol import classification_metrics, modality_presence


def read(name, fold, records):
    folder = ROOT / "outputs/checkpoints" / name
    configuration = json.loads((folder / "config.json").read_text())
    completion = json.loads((folder / "completion.json").read_text())
    assert completion["status"] == "complete" and configuration["fold"] == fold
    assert configuration["strict_scratch"] and not configuration["pretrained"] and not configuration.get("diagnostic_split")
    train, valid = get_indices(records, fold)
    split = json.loads((folder / "split.json").read_text())
    presence = modality_presence(configuration["modalities"]).any(1)
    assert split["train"] == [index for index in train if presence[index]] and split["valid"] == valid
    with np.load(folder / "validation.npz") as data:
        np.testing.assert_array_equal(data["indices"], valid)
        labels = np.asarray([records[index]["label"] for index in valid])
        np.testing.assert_array_equal(data["labels"], labels)
        probabilities = data["probabilities"]
        correct = probabilities.argmax(1) == labels
        assert np.isfinite(probabilities).all()
        assert abs(correct.mean() - completion["best_accuracy"]) < 1e-12
    users = np.asarray([records[index]["user"] for index in valid])
    summary = {"name": name, "fold": fold, "accuracy": float(correct.mean()), "best_epoch": completion["best_epoch"],
               "parameters": completion["parameters"], "checkpoint_bytes": completion["checkpoint_bytes"],
               "per_user": {str(user): float(correct[users == user].mean()) for user in np.unique(users)}}
    return summary, labels, probabilities


def main():
    records = get_records()
    groups = {
        "imu_raw": ["scratch_v1_imu_f0", "input_v1_imu_raw_f1"],
        "imu_world": ["input_v1_imu_world_f0", "input_v1_imu_world_f1"],
        "thermal_full": ["scratch_v1_thermal_f0", "input_v1_thermal_full_f1"],
        "thermal_motion": ["input_v1_thermal_motion_f0", "input_v1_thermal_motion_f1"],
    }
    fusion = {"fusion_original": [f"fusion_input_v1_original_f{fold}" for fold in [0, 1]],
              "fusion_improved": [f"fusion_input_v1_improved_f{fold}" for fold in [0, 1]],
              "fusion_staged": [f"fusion_input_v1_staged_f{fold}" for fold in [0, 1]]}
    if all((ROOT / f"outputs/checkpoints/{name}/completion.json").exists() for names in fusion.values() for name in names):
        groups.update(fusion)
    repeated = [f"fusion_input_v1_staged_s43_f{fold}" for fold in [0, 1]]
    if "fusion_staged" in groups and all((ROOT / f"outputs/checkpoints/{name}/completion.json").exists() for name in repeated):
        groups["fusion_staged_seed43"] = repeated
    report = {"status": "complete", "groups": {}, "comparisons": {},
              "protocol": "Two historically inspected user-isolated validation folds; per-fold best checkpoints, not a pristine test estimate"}
    predictions = {}
    for group, names in groups.items():
        runs = [read(name, fold, records) for fold, name in enumerate(names)]
        labels = np.concatenate([run[1] for run in runs])
        probability = np.concatenate([run[2] for run in runs])
        predictions[group] = (labels, probability)
        metrics = classification_metrics(labels, probability.argmax(1))
        report["groups"][group] = {"folds": [run[0] for run in runs], "pooled_accuracy": metrics["accuracy"],
                                   "pooled_balanced_accuracy": metrics["balanced_accuracy"], "pooled_macro_f1": metrics["macro_f1"], "validation_clips": len(labels)}
    if "fusion_staged_seed43" in predictions:
        labels, original = predictions["fusion_staged"]
        np.testing.assert_array_equal(labels, predictions["fusion_staged_seed43"][0])
        probability = (original + predictions["fusion_staged_seed43"][1]) / 2
        metrics = classification_metrics(labels, probability.argmax(1))
        folds, start = [], 0
        for fold in [0, 1]:
            _, valid = get_indices(records, fold)
            end = start + len(valid)
            folds.append({"fold": fold, "accuracy": float((probability[start:end].argmax(1) == labels[start:end]).mean())})
            start = end
        report["groups"]["fusion_staged_equal_average"] = {"folds": folds, "pooled_accuracy": metrics["accuracy"],
                "pooled_balanced_accuracy": metrics["balanced_accuracy"], "pooled_macro_f1": metrics["macro_f1"], "validation_clips": len(labels),
                "weights": [0.5, 0.5], "note": "Prespecified equal average; no weight or temperature search"}
    comparisons = [("imu_raw", "imu_world"), ("thermal_full", "thermal_motion")]
    if "fusion_improved" in groups:
        comparisons += [("fusion_original", "fusion_improved"), ("fusion_improved", "fusion_staged")]
    for reference, candidate in comparisons:
        labels, baseline = predictions[reference]
        np.testing.assert_array_equal(labels, predictions[candidate][0])
        original = baseline.argmax(1) == labels
        improved = predictions[candidate][1].argmax(1) == labels
        report["comparisons"][f"{reference}->{candidate}"] = {
            "gain_percentage_points": float((improved.mean() - original.mean()) * 100),
            "corrected": int((improved & ~original).sum()), "regressed": int((~improved & original).sum()),
            "fold_gains_percentage_points": [100 * (report["groups"][candidate]["folds"][fold]["accuracy"] - report["groups"][reference]["folds"][fold]["accuracy"]) for fold in [0, 1]]}
    destination = ROOT / "outputs/study_inputs/comparison.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    for name, values in report["groups"].items():
        print(name, "folds", [round(fold["accuracy"] * 100, 2) for fold in values["folds"]],
              "pooled", round(values["pooled_accuracy"] * 100, 2), "balanced", round(values["pooled_balanced_accuracy"] * 100, 2))
    print("Saved", destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
