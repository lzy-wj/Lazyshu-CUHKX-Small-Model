#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from src.data import get_records


def main():
    records = get_records()
    output = HERE / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    for fold in (0, 1):
        reference, _, probabilities, _ = v104_probabilities(fold, records)
        path = output / f"v124_v104_soft_f{fold}.npz"
        np.savez(
            path,
            probabilities=probabilities.astype(np.float32),
            indices=reference["indices"],
        )
        report = {
            "fold": fold,
            "rows": len(reference["indices"]),
            "mean_confidence": float(probabilities.max(axis=1).mean()),
            "teacher": "V104 domain-gated predecode probabilities",
        }
        path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
