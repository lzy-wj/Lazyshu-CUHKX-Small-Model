#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import prepool, run
from src.data import get_records


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        archive = np.load(ROOT / f"experiments/v52_domain_logit_residual/outputs/gated_f{fold}.npz")
        folds.append((archive["probabilities"], archive["indices"], archive["labels"]))
    settings_grid = [
        {"enabled": False, "maximum_gap": 120, "minimum_similarity": 0.7, "power": 1.5}
    ]
    for maximum_gap in (90, 120, 150, 180):
        for minimum_similarity in (0.4, 0.5, 0.6, 0.7, 0.8):
            for power in (1.0, 1.5, 2.0):
                settings_grid.append(
                    {
                        "enabled": True,
                        "maximum_gap": maximum_gap,
                        "minimum_similarity": minimum_similarity,
                        "power": power,
                    }
                )
    candidates = []
    for settings in settings_grid:
        result = run(0, folds[0], records, settings)
        candidates.append(settings | {"fold0": result})
        print(json.dumps({"settings": settings, "correct": result["correct"]}), flush=True)
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -item["fold0"]["prepool_groups"],
            item["minimum_similarity"],
        ),
    )
    settings = {
        key: selected[key]
        for key in ("enabled", "maximum_gap", "minimum_similarity", "power")
    }
    selected["fold1"] = run(1, folds[1], records, settings)
    output_root = ROOT / "experiments/v54_recovered_repeats/outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    for fold, values in enumerate(folds):
        probabilities, groups = prepool(
            values[0], values[1], records,
            settings["maximum_gap"], settings["minimum_similarity"], settings["power"],
        )
        np.savez(
            output_root / f"prepooled_f{fold}.npz",
            probabilities=probabilities,
            indices=values[1],
            labels=values[2],
        )
        if groups != selected[f"fold{fold}"]["prepool_groups"]:
            raise RuntimeError("Repeat-group count changed between scoring and export")
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": "Repeat settings selected on Fold 0 from timestamps and probabilities only; Fold 1 blind.",
        "settings": settings,
        "fold0": selected["fold0"],
        "fold1": selected["fold1"],
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference": {"name": "V52", "correct": 1193, "rows": 1357},
        "promote": selected["fold0"]["correct"] > 568 and selected["fold1"]["correct"] > 625,
        "candidates": candidates,
    }
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

