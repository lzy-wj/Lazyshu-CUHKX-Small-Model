import argparse
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def metrics(probabilities, labels):
    prediction = probabilities.argmax(1)
    confusion = np.zeros((40, 40), dtype=np.int64)
    np.add.at(confusion, (labels, prediction), 1)
    recall = confusion.diagonal() / confusion.sum(1).clip(1)
    precision = confusion.diagonal() / confusion.sum(0).clip(1)
    f1 = 2 * precision * recall / (precision + recall).clip(1e-12)
    supported = confusion.sum(1) > 0
    return {"accuracy": float((prediction == labels).mean()), "balanced_accuracy": float(recall[supported].mean()),
            "supported_classes": int(supported.sum()),
            "macro_f1": float(f1.mean()),
            "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-8)).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--output", default="experiment_comparison")
    args = parser.parse_args()
    names = args.names or [path.parent.name for path in (ROOT / "outputs/checkpoints").glob("*/validation.npz")
                           if not path.parent.name.startswith("smoke")]
    rows = []
    arrays = {}
    for name in names:
        folder = ROOT / "outputs/checkpoints" / name
        data = np.load(folder / "validation.npz")
        config = json.loads((folder / "config.json").read_text())
        row = {"name": name, "fold": config["fold"], "parameters": config["parameters"],
               "pretrained": config["pretrained"], **metrics(data["probabilities"], data["labels"])}
        arrays[name] = dict(data)
        rows.append(row)
    rows.sort(key=lambda row: row["accuracy"], reverse=True)
    for row in rows:
        print(json.dumps(row), flush=True)
    pairs = []
    for first, second in itertools.combinations(names, 2):
        left, right = arrays[first], arrays[second]
        if not np.array_equal(left["indices"], right["indices"]):
            continue
        candidates = []
        for weight in np.arange(0.1, 1, 0.1):
            probabilities = left["probabilities"] * weight + right["probabilities"] * (1 - weight)
            candidates.append({"names": [first, second], "weights": [float(weight), float(1 - weight)],
                               **metrics(probabilities, left["labels"])})
        pairs.append(max(candidates, key=lambda row: (row["accuracy"], -row["nll"])))
    pairs.sort(key=lambda row: (row["accuracy"], -row["nll"]), reverse=True)
    print("Best validation blends (selection estimates, not independent test scores):")
    for row in pairs[:8]:
        print(json.dumps(row), flush=True)
    destination = ROOT / "outputs/predictions" / f"{args.output}.json"
    destination.write_text(json.dumps({"single_models": rows, "validation_selected_blends": pairs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
