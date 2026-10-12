import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from evaluate_experiments import metrics


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", default="calibrated_selection")
    args = parser.parse_args()
    arrays = [dict(np.load(ROOT / "outputs/checkpoints" / name / "validation.npz")) for name in args.names]
    if not all(np.array_equal(array["indices"], arrays[0]["indices"]) for array in arrays):
        raise ValueError("Calibration requires identical validation clips")
    temperatures = []
    calibrated = []
    labels = arrays[0]["labels"]
    for name, array in zip(args.names, arrays):
        def objective(temperature):
            probabilities = softmax(array["logits"] / temperature, axis=1)
            return float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-8)).mean())
        result = minimize_scalar(objective, bounds=(0.5, 4), method="bounded")
        temperatures.append(float(result.x))
        calibrated.append(softmax(array["logits"] / result.x, axis=1))
        print(name, "temperature", result.x, "nll", result.fun)
    candidates = []
    for parts in itertools.product(range(11), repeat=len(args.names)):
        if sum(parts) != 10:
            continue
        weights = np.asarray(parts) / 10
        probability = np.average(calibrated, axis=0, weights=weights)
        candidates.append({"weights": weights.tolist(), **metrics(probability, labels)})
    candidates.sort(key=lambda row: (row["accuracy"], -row["nll"]), reverse=True)
    best = candidates[0]
    probability = np.average(calibrated, axis=0, weights=best["weights"])
    report = {"names": args.names, "temperatures": temperatures, "selection": best,
              "top_candidates": candidates[:10], "note": "Calibration and weights fit on validation fold 0; confirm on fold 1."}
    (ROOT / "outputs/predictions" / f"{args.output}.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(ROOT / "outputs/predictions" / f"{args.output}.npz", probabilities=probability,
             labels=labels, indices=arrays[0]["indices"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
