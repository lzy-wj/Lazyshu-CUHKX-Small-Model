import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from evaluate_experiments import metrics


ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "configs/v3_selected_ensemble.json"
    if destination.exists():
        raise RuntimeError("V3 selection is already frozen; reuse the existing configuration")
    exit_path = ROOT / "outputs/logs/v3_x3d_f0.exit"
    if not exit_path.exists() or exit_path.read_text().strip() != "0":
        raise RuntimeError("Finish the fold-0 X3D experiment before freezing selection")
    previous = json.loads((ROOT / "configs/v2_selected_ensemble.json").read_text())
    names = ["v3_x3d_f0"] + previous["names"]
    arrays = [np.load(ROOT / "outputs/checkpoints" / name / "validation.npz") for name in names]
    labels = arrays[0]["labels"]
    assert all(np.array_equal(array["indices"], arrays[0]["indices"]) and np.array_equal(array["labels"], labels)
               for array in arrays)

    def objective(temperature):
        probability = softmax(arrays[0]["logits"] / temperature, axis=1)
        return float(-np.log(probability[np.arange(len(labels)), labels].clip(1e-8)).mean())

    temperature = float(minimize_scalar(objective, bounds=(0.5, 4), method="bounded").x)
    new_probability = softmax(arrays[0]["logits"] / temperature, axis=1)
    old_probability = np.average([softmax(array["logits"] / value, axis=1)
                                  for array, value in zip(arrays[1:], previous["temperatures"])],
                                 axis=0, weights=previous["weights"])
    candidates = []
    for alpha in np.linspace(0, 1, 21):
        probability = alpha * new_probability + (1 - alpha) * old_probability
        candidates.append({"new_model_weight": float(alpha), **metrics(probability, labels)})
    candidates.sort(key=lambda row: (row["accuracy"], -row["nll"]), reverse=True)
    best = candidates[0]
    alpha = best["new_model_weight"]
    weights = [alpha] + [(1 - alpha) * value for value in previous["weights"]]
    temperatures = [temperature] + previous["temperatures"]
    source_epoch = json.loads((ROOT / "outputs/checkpoints/v3_x3d_f0/metrics.json").read_text())["epoch"]
    matching_refits = [name for name in ["v3_final_x3d", "v3_final_x3d_refined"]
                       if (ROOT / f"configs/{name}.json").exists()
                       and json.loads((ROOT / f"configs/{name}.json").read_text())["source_best_epoch"] == source_epoch]
    if not matching_refits:
        raise RuntimeError("Launch a full-data refit matching the finalized fold-0 selected epoch")
    final_models = [matching_refits[-1]] + previous["final_models"]
    active = [index for index, weight in enumerate(weights) if weight > 0]
    selected = {"names": [names[index] for index in active], "final_models": [final_models[index] for index in active],
                "weights": [weights[index] for index in active], "temperatures": [temperatures[index] for index in active],
                "selection_fold": 0, "selection_accuracy": best["accuracy"],
                "frozen_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
                "protocol": "Preserve the complete V2 blend and its temperatures. Fit only X3D temperature and a single 21-point mixing weight on fold 0, transfer unchanged to fold 1. Checkpoints selected within each fold.",
                "eligibility": "AI-assisted research, external small-model pretraining; competition eligibility unresolved. No automatic submission."}
    destination.write_text(json.dumps(selected, indent=2) + "\n")
    report = {"selection": selected, "top_candidates": candidates[:10]}
    (ROOT / "outputs/predictions/v3_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
