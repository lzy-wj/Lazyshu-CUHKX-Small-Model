#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from src.data import get_records


SETTINGS = {
    "enabled": True,
    "maximum_gap": 150,
    "minimum_similarity": 0.7,
    "power": 1.0,
}


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        archive = np.load(
            ROOT / f"experiments/v57_round_transformer_distillation/outputs/predictions_f{fold}.npz"
        )
        folds.append(
            run(
                fold,
                (archive["probabilities"], archive["indices"], archive["labels"]),
                records,
                SETTINGS,
            )
        )
    correct = folds[0]["correct"] + folds[1]["correct"]
    report = {
        "protocol": (
            "V57 is followed by inherited label-free repeat recovery and the frozen pair-aware decoder. "
            "No V59 setting is selected on Fold 1."
        ),
        "settings": SETTINGS,
        "fold0": folds[0],
        "fold1": folds[1],
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "references": {"V52": {"correct": 1193}, "V57": {"correct": 1195}},
        "promote": folds[0]["correct"] >= 567 and folds[1]["correct"] >= 628 and correct > 1195,
    }
    output = HERE / "outputs/validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
