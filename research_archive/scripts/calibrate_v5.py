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
    output = ROOT / "outputs/predictions/v5_selection.json"
    if output.exists():
        raise RuntimeError("V5 selection has already been recorded; do not overwrite it")
    new_names = ["v5_minit_f0", "v5_scene_f0"]
    for name in new_names:
        assert (ROOT / f"outputs/logs/{name}.exit").read_text().strip() == "0"
    previous = json.loads((ROOT / "configs/v4_selected_ensemble.json").read_text())
    original = json.loads((ROOT / "configs/v3_selected_ensemble.json").read_text())
    baseline = max(previous["selection_accuracy"], original["selection_accuracy"])
    temperatures = dict(zip(original["names"], original["temperatures"]))
    temperatures.update(dict(zip(previous["names"], previous["temperatures"])))
    videos = ["v3_x3d_f0", "v4_jointstats_f0"] + new_names
    names = videos + ["v2_statistics_f0"]
    arrays = {name: np.load(ROOT / f"outputs/checkpoints/{name}/validation.npz") for name in names}
    labels = arrays[names[0]]["labels"]
    for array in arrays.values():
        np.testing.assert_array_equal(array["labels"], labels)
        np.testing.assert_array_equal(array["indices"], arrays[names[0]]["indices"])
    for name in new_names:
        def objective(temperature):
            probability = softmax(arrays[name]["logits"] / temperature, axis=1)
            return float(-np.log(probability[np.arange(len(labels)), labels].clip(1e-8)).mean())
        temperatures[name] = float(minimize_scalar(objective, bounds=(0.5, 4), method="bounded").x)
    probabilities = {name: softmax(arrays[name]["logits"] / temperatures[name], axis=1) for name in names}
    candidates = []
    for pair in itertools.combinations(videos, 2):
        if not set(pair) & set(new_names):
            continue
        for alpha in np.linspace(0, 1, 21):
            candidate_names = list(pair) + ["v2_statistics_f0"]
            weights = [float(alpha) * 0.85, (1 - float(alpha)) * 0.85, 0.15]
            active = [index for index, weight in enumerate(weights) if weight > 1e-8]
            candidate_names = [candidate_names[index] for index in active]
            weights = [weights[index] for index in active]
            if not set(candidate_names) & set(new_names):
                continue
            size = sum((ROOT / f"outputs/checkpoints/{name}/best.pt").stat().st_size for name in candidate_names)
            if any(name != "v5_scene_f0" and name != "v2_statistics_f0" for name in candidate_names):
                size += (ROOT / ".tools/torch/hub/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth").stat().st_size
            assert size <= 99_000_000
            probability = np.average([probabilities[name] for name in candidate_names], axis=0, weights=weights)
            candidates.append({"names": candidate_names, "weights": weights, "estimated_model_bytes": size,
                               **metrics(probability, labels)})
    candidates.sort(key=lambda row: (row["accuracy"], -row["nll"]), reverse=True)
    best = candidates[0]
    approved = best["accuracy"] >= baseline + 0.01
    report = {"baseline_fold0_accuracy": baseline, "minimum_improvement": 0.01, "promote_to_confirmation": approved,
              "candidate_count": len(candidates), "candidates": candidates, "temperatures": temperatures,
              "protocol": "Fold-0-only predeclared pair grid with fixed 0.15 statistical expert. The 1-point gate limits marginal validation tuning; it is not a guarantee of generalization."}
    output.write_text(json.dumps(report, indent=2) + "\n")
    if approved:
        final_mapping = dict(zip(original["names"], original["final_models"])) | dict(zip(previous["names"], previous["final_models"]))
        final_mapping.update({"v5_minit_f0": "v5_final_minit", "v5_scene_f0": "v5_final_scene"})
        selected = {"names": best["names"], "final_models": [final_mapping[name] for name in best["names"]],
                    "weights": best["weights"], "temperatures": [temperatures[name] for name in best["names"]],
                    "selection_fold": 0, "selection_accuracy": best["accuracy"],
                    "frozen_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
                    "protocol": report["protocol"], "eligibility": original["eligibility"]}
        (ROOT / "configs/v5_selected_ensemble.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"} | {"top_candidates": candidates[:5]}, indent=2))


if __name__ == "__main__":
    main()
