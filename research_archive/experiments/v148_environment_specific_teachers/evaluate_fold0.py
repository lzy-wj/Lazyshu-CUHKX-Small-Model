#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import align, components, decode, load
from src.data import get_records


WEIGHTS = (0.02, 0.05, 0.10, 0.20, 0.40)


def main():
    records = get_records()
    reference, v71, v114, domain = components(0, records)
    users = np.asarray([records[int(index)]["user"] for index in reference["indices"]])
    specialist = np.zeros_like(v71)
    diagnostics = {}
    for environment in (0, 1):
        values = align(
            reference,
            load(HERE / f"outputs/checkpoints/v148_env{environment}_f0/validation.npz"),
        )
        selected = (users > 15) == bool(environment)
        specialist[selected] = values[selected]
        diagnostics[str(environment)] = {
            "correct": int((values[selected].argmax(1) == reference["labels"][selected]).sum()),
            "rows": int(selected.sum()),
        }
    control = scores(decode(0, reference, v71, v114, domain, records), reference, 0, records)
    grid = [{"weight": 0.0, "fold0": control}]
    for weight in WEIGHTS:
        prediction = decode(
            0,
            reference,
            (1 - weight) * v71 + weight * specialist,
            (1 - weight) * v114 + weight * specialist,
            domain,
            records,
        )
        grid.append({"weight": weight, "fold0": scores(prediction, reference, 0, records)})
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["weight"],
        ),
    )
    report = {
        "protocol": "Environment specialists and one shared weight selected on Fold0 only.",
        "diagnostics": diagnostics,
        "grid": grid,
        "selected": selected,
        "advance": selected["weight"] > 0,
    }
    output = HERE / "outputs/fold0_gate.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"diagnostics": diagnostics, "selected": selected, "advance": report["advance"]}, indent=2))


if __name__ == "__main__":
    main()
