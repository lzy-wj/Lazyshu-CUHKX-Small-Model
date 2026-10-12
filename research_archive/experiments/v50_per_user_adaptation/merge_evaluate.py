#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v32_domain_adaptation_gate.search import apply_weights, decode
from src.data import FOLDS, get_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    records = get_records()
    settings = json.loads(
        (ROOT / "experiments/v32_domain_adaptation_gate/outputs/search.json").read_text()
    )["settings"]
    base = np.load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{args.fold}.npz")
    row_by_index = {}
    user_reports = {}
    for user in FOLDS[args.fold]:
        path = ROOT / f"experiments/v50_per_user_adaptation/outputs/fold{args.fold}_user{user}.npz"
        archive = np.load(path)
        for index, label, probability in zip(archive["indices"], archive["labels"], archive["probabilities"]):
            row_by_index[int(index)] = (int(label), probability)
        report = json.loads(path.with_suffix(".json").read_text())
        user_reports[str(user)] = {
            "rows": len(archive["indices"]),
            "selected_pseudo_rows": report["selected_pseudo_rows"],
            "raw_correct": int((archive["probabilities"].argmax(1) == archive["labels"]).sum()),
        }
    indices = base["indices"]
    if set(row_by_index) != set(indices.tolist()):
        raise RuntimeError("Per-user archives do not cover the fold exactly")
    labels = np.asarray([row_by_index[int(index)][0] for index in indices])
    adapted_probabilities = np.stack([row_by_index[int(index)][1] for index in indices])
    adapted = {"probabilities": adapted_probabilities, "indices": indices, "labels": labels}
    probabilities = apply_weights(base, adapted, records, args.fold, settings)
    result = decode(probabilities, adapted, args.fold, records)
    result.update(
        raw_correct=int((probabilities.argmax(1) == labels).sum()),
        adapted_raw_correct=int((adapted_probabilities.argmax(1) == labels).sum()),
        users=user_reports,
    )
    output_root = ROOT / "experiments/v50_per_user_adaptation/outputs"
    np.savez(output_root / f"merged_f{args.fold}.npz", probabilities=probabilities, indices=indices, labels=labels)
    (output_root / f"evaluation_f{args.fold}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

