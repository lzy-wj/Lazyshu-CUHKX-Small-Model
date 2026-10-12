#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import prepool
from src.data import get_records


def main():
    records = get_records()
    input_path = ROOT / "experiments/v24_full_refit/outputs/predictions/v57_round_transformer_distilled.npz"
    output_path = ROOT / "experiments/v24_full_refit/outputs/predictions/v59_teacher_repeat_recovered.npz"
    archive = np.load(input_path)
    probabilities, groups = prepool(
        archive["probabilities"], archive["indices"], records, 150, 0.7, 1.0
    )
    np.savez(output_path, probabilities=probabilities, indices=archive["indices"])
    report = {
        "input": str(input_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "settings": {"maximum_gap": 150, "minimum_similarity": 0.7, "power": 1.0},
        "recovered_groups": groups,
        "changed_top1_before_decoder": int(
            (probabilities.argmax(1) != archive["probabilities"].argmax(1)).sum()
        ),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
