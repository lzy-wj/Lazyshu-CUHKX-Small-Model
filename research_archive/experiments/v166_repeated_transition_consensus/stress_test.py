#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v164_evidence_gated_radar.evaluate import accept
from experiments.v165_balanced_cycle_radar.evaluate import cycle_mask
from experiments.v166_repeated_transition_consensus.evaluate import (
    fold_inputs,
    repeated_transition_mask,
)
from src.data import get_records


def main():
    records = get_records()
    cached = [fold_inputs(fold, records) for fold in (0, 1)]
    retention_reports = []
    for retention in (0.5, 0.7, 0.85):
        runs = []
        for seed in range(1660, 1680):
            generator = np.random.default_rng(seed)
            fold_reports = []
            for fold, (reference, evidence, incumbent, challenger) in enumerate(cached):
                indices = reference["indices"]
                labels = reference["labels"]
                users = np.asarray([records[int(index)]["user"] for index in indices])
                retained = np.zeros(len(indices), dtype=bool)
                for user in sorted(set(users)):
                    positions = np.flatnonzero(users == user)
                    count = max(1, round(len(positions) * retention))
                    retained[
                        generator.choice(positions, count, replace=False)
                    ] = True
                selected_indices = indices[retained]
                selected_labels = labels[retained]
                selected_evidence = evidence[retained]
                selected_incumbent = incumbent[retained]
                selected_challenger = challenger[retained]
                selected_users = users[retained]
                selected = accept(
                    selected_incumbent,
                    selected_challenger,
                    selected_evidence,
                )
                selected |= cycle_mask(
                    selected_indices,
                    selected_incumbent,
                    selected_challenger,
                    records,
                )
                selected |= repeated_transition_mask(
                    selected_users,
                    selected_incumbent,
                    selected_challenger,
                    2,
                )
                prediction = selected_incumbent.copy()
                prediction[selected] = selected_challenger[selected]
                fold_reports.append(
                    {
                        "fold": fold,
                        "rows": len(prediction),
                        "accepted": int(selected.sum()),
                        "delta": int(
                            (prediction == selected_labels).sum()
                            - (selected_incumbent == selected_labels).sum()
                        ),
                        "gains": int(
                            (
                                selected
                                & (selected_challenger == selected_labels)
                                & (selected_incumbent != selected_labels)
                            ).sum()
                        ),
                        "losses": int(
                            (
                                selected
                                & (selected_challenger != selected_labels)
                                & (selected_incumbent == selected_labels)
                            ).sum()
                        ),
                    }
                )
            runs.append({"seed": seed, "folds": fold_reports})
        deltas = [sum(fold["delta"] for fold in run["folds"]) for run in runs]
        losses = [sum(fold["losses"] for fold in run["folds"]) for run in runs]
        retention_reports.append(
            {
                "retention": retention,
                "runs": runs,
                "summary": {
                    "minimum_delta": min(deltas),
                    "mean_delta": float(np.mean(deltas)),
                    "maximum_delta": max(deltas),
                    "maximum_losses": max(losses),
                    "both_folds_positive": sum(
                        all(fold["delta"] > 0 for fold in run["folds"])
                        for run in runs
                    ),
                },
            }
        )
    report = {
        "protocol": (
            "Twenty fixed user-stratified random partial-script views at each "
            "retention; V166 selection is recomputed after row removal."
        ),
        "retentions": retention_reports,
    }
    output = HERE / "outputs/stress_test.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        str(item["retention"]): item["summary"] for item in retention_reports
    }, indent=2))


if __name__ == "__main__":
    main()
