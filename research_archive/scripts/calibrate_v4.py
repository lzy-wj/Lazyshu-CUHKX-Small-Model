import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from evaluate_experiments import metrics


ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "configs/v4_selected_ensemble.json"
    if destination.exists():
        raise RuntimeError("V4 selection is frozen; do not overwrite it")
    previous = json.loads((ROOT / "configs/v3_selected_ensemble.json").read_text())
    new_names = ["v4_detail_f0", "v4_jointstats_f0"]
    for name in new_names:
        path = ROOT / f"outputs/logs/{name}.exit"
        if not path.exists() or path.read_text().strip() != "0":
            raise RuntimeError(f"Finish {name} before selection")
    names = list(dict.fromkeys(previous["names"] + new_names))
    arrays = {name: np.load(ROOT / f"outputs/checkpoints/{name}/validation.npz") for name in names}
    reference = arrays[names[0]]
    labels = reference["labels"]
    for array in arrays.values():
        np.testing.assert_array_equal(array["indices"], reference["indices"])
        np.testing.assert_array_equal(array["labels"], labels)
    temperatures = dict(zip(previous["names"], previous["temperatures"]))
    for name in new_names:
        def objective(temperature):
            probability = softmax(arrays[name]["logits"] / temperature, axis=1)
            return float(-np.log(probability[np.arange(len(labels)), labels].clip(1e-8)).mean())
        temperatures[name] = float(minimize_scalar(objective, bounds=(0.5, 4), method="bounded").x)
    probabilities = {name: softmax(arrays[name]["logits"] / temperatures[name], axis=1) for name in names}
    old_probability = np.average([probabilities[name] for name in previous["names"]], axis=0, weights=previous["weights"])
    candidates = [{"names": previous["names"], "weights": previous["weights"],
                   "baseline": True, **metrics(old_probability, labels)}]
    detector = ROOT / ".tools/torch/hub/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
    costs = {name: (ROOT / f"outputs/checkpoints/{name}/best.pt").stat().st_size for name in names}
    seen = set()
    for pair in itertools.combinations(["v3_x3d_f0"] + new_names, 2):
        for alpha in np.linspace(0, 1, 21):
            weights = [0.85 * float(alpha), 0.85 * (1 - float(alpha)), 0.15]
            candidate_names = list(pair) + ["v2_statistics_f0"]
            active = [index for index, weight in enumerate(weights) if weight > 1e-8]
            candidate_names = [candidate_names[index] for index in active]
            weights = [weights[index] for index in active]
            key = tuple(sorted(zip(candidate_names, np.round(weights, 8))))
            if key in seen:
                continue
            seen.add(key)
            estimated_bytes = sum(costs[name] for name in candidate_names) + detector.stat().st_size
            if estimated_bytes > 99_000_000:
                continue
            probability = np.average([probabilities[name] for name in candidate_names], axis=0, weights=weights)
            candidates.append({"names": candidate_names, "weights": weights, "baseline": False,
                               "estimated_model_bytes": estimated_bytes, **metrics(probability, labels)})
    candidates.sort(key=lambda row: (row["accuracy"], -row["nll"]), reverse=True)
    best = candidates[0]
    final_mapping = dict(zip(previous["names"], previous["final_models"]))
    final_mapping.update({"v4_detail_f0": "v4_final_detail", "v4_jointstats_f0": "v4_final_jointstats"})
    selected = {"names": best["names"], "final_models": [final_mapping[name] for name in best["names"]],
                "weights": best["weights"], "temperatures": [temperatures[name] for name in best["names"]],
                "selection_fold": 0, "selection_accuracy": best["accuracy"], "baseline_retained": best["baseline"],
                "frozen_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
                "protocol": "Fold-0-only temperatures and predeclared 21-point pair mixing, fixed 0.15 statistical weight; unchanged V3 is a fallback. Transfer unchanged to fold 1. Each fold selects its checkpoint on its own validation set; not an unbiased test estimate.",
                "eligibility": previous["eligibility"]}
    destination.write_text(json.dumps(selected, indent=2) + "\n")
    report = {"selection": selected, "candidate_count": len(candidates), "candidates": candidates,
              "attention_rejected": "50.46% fold-0 accuracy, below the 53.07% existing statistical expert."}
    (ROOT / "outputs/predictions/v4_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selection": selected, "candidate_count": len(candidates), "top_candidates": candidates[:5]}, indent=2))


if __name__ == "__main__":
    main()
