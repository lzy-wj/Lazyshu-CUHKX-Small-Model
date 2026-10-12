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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    settings = json.loads(
        (ROOT / "experiments/v32_domain_adaptation_gate/outputs/search.json").read_text()
    )["settings"]
    base = np.load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{args.fold}.npz")
    candidate = np.load(ROOT / f"experiments/v47_environment_adabn/outputs/fold{args.fold}.npz")
    if not np.array_equal(base["indices"], candidate["indices"]):
        raise ValueError("Base and environment-AdaBN predictions are not aligned")
    probabilities = apply_weights(base, candidate, records, args.fold, settings)
    result = decode(probabilities, candidate, args.fold, records)
    result.update(
        raw_correct=int((probabilities.argmax(1) == candidate["labels"]).sum()),
        raw_accuracy=float((probabilities.argmax(1) == candidate["labels"]).mean()),
    )
    output = ROOT / f"experiments/v47_environment_adabn/outputs/evaluation_f{args.fold}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

