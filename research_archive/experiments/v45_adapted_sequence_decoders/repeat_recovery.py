#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from src.data import get_records


def values(fold):
    archive = np.load(ROOT / f"experiments/v32_domain_adaptation_gate/outputs/v32_domain_gate_f{fold}.npz")
    return archive["probabilities"], archive["indices"], archive["labels"]


def main():
    records = get_records()
    folds = [values(fold) for fold in (0, 1)]
    settings_grid = [
        {"enabled": False, "maximum_gap": 120, "minimum_similarity": 0.7, "power": 1.5}
    ]
    for maximum_gap in (120, 180, 240):
        for minimum_similarity in (0.4, 0.5, 0.6, 0.7):
            for power in (1.0, 1.5):
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
    settings = {key: selected[key] for key in ("enabled", "maximum_gap", "minimum_similarity", "power")}
    selected["fold1"] = run(1, folds[1], records, settings)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": "Repeat recovery selected on Fold 0 V32 probabilities; Fold 1 blind.",
        "selected": selected,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference": {"fold0": 551, "fold1": 623, "correct": 1174, "rows": 1357},
        "promote": selected["fold0"]["correct"] >= 551 and selected["fold1"]["correct"] > 623,
        "candidates": candidates,
    }
    output = ROOT / "experiments/v45_adapted_sequence_decoders/outputs/repeat_recovery.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
