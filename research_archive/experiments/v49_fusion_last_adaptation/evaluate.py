#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v32_domain_adaptation_gate.search import apply_weights, decode
from src.data import get_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    records = get_records()
    settings = json.loads(
        (ROOT / "experiments/v32_domain_adaptation_gate/outputs/search.json").read_text()
    )["settings"]
    base = np.load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{args.fold}.npz")
    adapted = np.load(ROOT / f"experiments/v49_fusion_last_adaptation/outputs/fold{args.fold}.npz")
    if not np.array_equal(base["indices"], adapted["indices"]):
        raise ValueError("Base and fusion-adapted predictions are not aligned")
    probabilities = apply_weights(base, adapted, records, args.fold, settings)
    result = decode(probabilities, adapted, args.fold, records)
    result.update(
        raw_correct=int((probabilities.argmax(1) == adapted["labels"]).sum()),
        adapted_raw_correct=int((adapted["probabilities"].argmax(1) == adapted["labels"]).sum()),
    )
    output = ROOT / f"experiments/v49_fusion_last_adaptation/outputs/evaluation_f{args.fold}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

