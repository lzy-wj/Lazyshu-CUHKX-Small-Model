#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v32_domain_adaptation_gate.search import apply_weights, decode
from src.data import get_records


FAMILIES = {
    "ends_0875": ("early_0875", "late_0875"),
    "center_0875": ("center_0875",),
    "ends_0750": ("early_0750", "late_0750"),
    "all_0875": ("early_0875", "center_0875", "late_0875"),
}
WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)


def load_fold(fold):
    base = np.load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz")
    adapted = np.load(ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz")
    views = np.load(ROOT / f"experiments/v46_temporal_tta/outputs/fold{fold}_views.npz")
    if not np.array_equal(base["indices"], adapted["indices"]) or not np.array_equal(base["indices"], views["indices"]):
        raise ValueError(f"Fold {fold} archives are not aligned")
    return base, adapted, views


def temporal_probabilities(adapted, views, family, weight):
    extra = np.mean([views[f"probabilities_{name}"] for name in FAMILIES[family]], axis=0)
    return (1 - weight) * adapted["probabilities"] + weight * extra


def evaluate_candidate(fold, loaded, family, weight, records, settings):
    base, adapted, views = loaded
    candidate = temporal_probabilities(adapted, views, family, weight)
    probabilities = apply_weights(base, {"probabilities": candidate, "indices": adapted["indices"]}, records, fold, settings)
    result = decode(probabilities, adapted, fold, records)
    result["raw_correct"] = int((probabilities.argmax(1) == adapted["labels"]).sum())
    result["nll"] = float(-np.log(np.clip(probabilities[np.arange(len(probabilities)), adapted["labels"]], 1e-8, 1)).mean())
    return result, probabilities


def main():
    records = get_records()
    settings = json.loads(
        (ROOT / "experiments/v32_domain_adaptation_gate/outputs/search.json").read_text()
    )["settings"]
    folds = [load_fold(fold) for fold in (0, 1)]
    historical_full = folds[0][1]["probabilities"]
    reproduced_full = folds[0][2]["probabilities_full"]
    reproduction = {
        "historical_correct": int((historical_full.argmax(1) == folds[0][1]["labels"]).sum()),
        "reproduced_correct": int((reproduced_full.argmax(1) == folds[0][1]["labels"]).sum()),
        "mean_absolute_probability_difference": float(np.abs(historical_full - reproduced_full).mean()),
        "maximum_absolute_probability_difference": float(np.abs(historical_full - reproduced_full).max()),
        "changed_top1": int((historical_full.argmax(1) != reproduced_full.argmax(1)).sum()),
    }

    candidates = []
    for family in FAMILIES:
        for weight in WEIGHTS:
            result, _ = evaluate_candidate(0, folds[0], family, weight, records, settings)
            candidate = {"family": family, "weight": weight, "fold0": result}
            candidates.append(candidate)
            print(json.dumps({"family": family, "weight": weight, "correct": result["correct"], "raw": result["raw_correct"]}), flush=True)
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["raw_correct"],
            -item["fold0"]["nll"],
            -item["weight"],
        ),
    )
    fold1, fold1_probabilities = evaluate_candidate(
        1, folds[1], selected["family"], selected["weight"], records, settings
    )
    selected["fold1"] = fold1
    _, fold0_probabilities = evaluate_candidate(
        0, folds[0], selected["family"], selected["weight"], records, settings
    )
    correct = selected["fold0"]["correct"] + fold1["correct"]
    output_root = ROOT / "experiments/v46_temporal_tta/outputs"
    np.savez(output_root / "selected_f0.npz", probabilities=fold0_probabilities, indices=folds[0][1]["indices"], labels=folds[0][1]["labels"])
    np.savez(output_root / "selected_f1.npz", probabilities=fold1_probabilities, indices=folds[1][1]["indices"], labels=folds[1][1]["labels"])
    report = {
        "protocol": "Temporal TTA family and weight selected on Fold 0; Fold 1 evaluated once blind.",
        "v24_adaptation_reproduction": reproduction,
        "settings": settings,
        "selected": selected,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference": {"correct": 1174, "rows": 1357},
        "promote": selected["fold0"]["correct"] >= 551 and fold1["correct"] > 623,
        "candidates": candidates,
    }
    (output_root / "search.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

