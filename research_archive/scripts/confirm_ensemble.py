import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from evaluate_experiments import metrics


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/selected_ensemble.json")
    parser.add_argument("--output-prefix", default="ensemble")
    args = parser.parse_args()
    selected = json.loads(args.config.read_text())
    reports = {}
    records = json.loads((ROOT / "data/metadata/samples.json").read_text())
    for fold in [0, 1]:
        names = [name.replace("_f0", f"_f{fold}") for name in selected["names"]]
        arrays = [dict(np.load(ROOT / "outputs/checkpoints" / name / "validation.npz")) for name in names]
        if not all(np.array_equal(array["indices"], arrays[0]["indices"]) for array in arrays):
            raise RuntimeError("Validation sample indices differ")
        probabilities = np.average([softmax(array["logits"] / temperature, axis=1)
                                    for array, temperature in zip(arrays, selected["temperatures"])],
                                   axis=0, weights=selected["weights"])
        labels = arrays[0]["labels"]
        indices = arrays[0]["indices"]
        users = np.array([records[index]["user"] for index in indices])
        report = metrics(probabilities, labels)
        report["clips"] = len(labels)
        report["per_user"] = {int(user): metrics(probabilities[users == user], labels[users == user])
                              for user in np.unique(users)}
        prediction = probabilities.argmax(1)
        confusion = np.zeros((40, 40), dtype=np.int64)
        np.add.at(confusion, (labels, prediction), 1)
        report["confusion_matrix"] = confusion.tolist()
        reports[f"fold{fold}"] = report
        np.savez(ROOT / "outputs/predictions" / f"{args.output_prefix}_fold{fold}.npz", probabilities=probabilities,
                 labels=labels, indices=indices)
        print(f"Fold {fold}:", {name: value for name, value in report.items() if name not in ["confusion_matrix", "per_user"]})
    reports["protocol"] = "Weights and temperatures selected on fold 0, transferred unchanged to fold 1; checkpoints selected within each fold."
    reports["weighted_accuracy"] = sum(reports[f"fold{fold}"]["accuracy"] * reports[f"fold{fold}"]["clips"]
                                         for fold in [0, 1]) / sum(reports[f"fold{fold}"]["clips"] for fold in [0, 1])
    (ROOT / "outputs/predictions" / f"{args.output_prefix}_validation_report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
