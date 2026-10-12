#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v114_paired_person_fusion.evaluate import PAIRINGS
from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import components, decode
from experiments.v154_triplicate_template_matching.evaluate import decode_fold
from src.data import get_records


RETENTION = 0.70
SEEDS = (1570, 1571, 1572, 1573, 1574)


def subset_reference(reference, selected):
    rows = len(reference["indices"])
    return {
        key: value[selected] if isinstance(value, np.ndarray) and len(value) == rows else value
        for key, value in reference.items()
    }


def main():
    records = get_records()
    results = []
    for seed in SEEDS:
        for fold in (0, 1):
            reference, v71, v114, domain = components(fold, records)
            rng = np.random.default_rng(seed + fold * 100)
            selected = rng.random(len(reference["indices"])) < RETENTION
            users = np.asarray(
                [records[int(index)]["user"] for index in reference["indices"]]
            )
            for user in np.unique(users):
                positions = np.flatnonzero(users == user)
                if not selected[positions].any():
                    selected[positions[0]] = True
            partial = subset_reference(reference, selected)
            partial_v71 = v71[selected]
            partial_v114 = v114[selected]
            partial_domain = domain[selected]
            baseline_prediction = decode(
                fold,
                partial,
                partial_v71,
                partial_v114,
                partial_domain,
                records,
            )
            capacity_prediction, _ = decode_fold(
                fold, partial, partial_v114, records
            )
            partial_users = users[selected]
            paired = np.isin(partial_users, list(PAIRINGS[fold]))
            routed_capacity = baseline_prediction.copy()
            routed_capacity[paired] = capacity_prediction[paired]
            baseline = scores(baseline_prediction, partial, fold, records)
            capacity = scores(routed_capacity, partial, fold, records)
            results.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "rows": int(selected.sum()),
                    "baseline": baseline,
                    "capacity": capacity,
                    "delta": capacity["correct"] - baseline["correct"],
                }
            )
    deltas = [item["delta"] for item in results]
    by_fold = {
        str(fold): [item["delta"] for item in results if item["fold"] == fold]
        for fold in (0, 1)
    }
    report = {
        "protocol": "Five fixed 70% partial validation views; no test labels or fitted threshold.",
        "retention": RETENTION,
        "seeds": list(SEEDS),
        "results": results,
        "summary": {
            "mean_delta": float(np.mean(deltas)),
            "minimum_delta": int(min(deltas)),
            "positive_views": int(sum(delta > 0 for delta in deltas)),
            "nonnegative_views": int(sum(delta >= 0 for delta in deltas)),
            "by_fold": by_fold,
        },
        "promote_capacity_rule": min(deltas) >= 0 and sum(delta > 0 for delta in deltas) >= 6,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "promote": report["promote_capacity_rule"]}, indent=2))


if __name__ == "__main__":
    main()
