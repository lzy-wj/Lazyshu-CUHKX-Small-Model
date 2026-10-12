#!/usr/bin/env python3
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v164_evidence_gated_radar.evaluate import accept
from experiments.v165_balanced_cycle_radar.evaluate import cycle_mask
from experiments.v84_residual_invariant_skeleton.evaluate import domains
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


TEST_GROUP_BY_DATE = {
    "2025-05-31": "source20",
    "2025-06-01": "source20",
    "2025-06-02": "source21",
    "2025-06-12": "unmatched",
    "2025-06-13": "unmatched",
    "2025-06-16": "source16",
    "2025-06-17": "source16",
}


def repeated_transition_mask(groups, incumbent, challenger, minimum_repeats):
    counts = Counter(
        (groups[position], int(incumbent[position]), int(challenger[position]))
        for position in range(len(incumbent))
        if incumbent[position] != challenger[position]
    )
    return np.asarray(
        [
            incumbent[position] != challenger[position]
            and counts[
                (
                    groups[position],
                    int(incumbent[position]),
                    int(challenger[position]),
                )
            ]
            >= minimum_repeats
            for position in range(len(incumbent))
        ],
        dtype=bool,
    )


def fold_inputs(fold, records):
    reference, _, v104, _ = v104_probabilities(fold, records)
    v71 = np.load(
        ROOT / f"experiments/v71_gated_film_distillation/outputs/student_f{fold}.npz"
    )["probabilities"].astype(np.float64)
    v114, _ = fused_probabilities(fold, reference, v104, records, weight=0.6)
    route = domains(reference["indices"], fold, records) >= 2
    evidence = np.where(route[:, None], v114, v71)
    incumbent = np.load(
        ROOT / f"experiments/v127_supervision_routed_system/outputs/v127_validation_f{fold}.npz"
    )["pseudo_labels"].astype(np.int64)
    challenger = np.load(
        ROOT / f"experiments/v160_radar_batch_decoder/outputs/validation_f{fold}.npz"
    )["pseudo_labels"].astype(np.int64)
    return reference, evidence, incumbent, challenger


def validation(records, minimum_repeats):
    fold_reports = []
    for fold in (0, 1):
        reference, evidence, incumbent, challenger = fold_inputs(fold, records)
        groups = [records[int(index)]["user"] for index in reference["indices"]]
        selected = cycle_mask(
            reference["indices"], incumbent, challenger, records
        )
        selected |= accept(incumbent, challenger, evidence)
        selected |= repeated_transition_mask(
            groups, incumbent, challenger, minimum_repeats
        )
        prediction = incumbent.copy()
        prediction[selected] = challenger[selected]
        labels = reference["labels"].astype(np.int64)
        user_reports = {}
        for user in sorted(set(groups)):
            user_rows = np.asarray([group == user for group in groups])
            user_reports[str(user)] = {
                "accepted": int((selected & user_rows).sum()),
                "delta": int(
                    ((prediction == labels) & user_rows).sum()
                    - ((incumbent == labels) & user_rows).sum()
                ),
            }
        fold_reports.append(
            {
                "fold": fold,
                "incumbent": scores(incumbent, reference, fold, records),
                "candidate": scores(prediction, reference, fold, records),
                "accepted": int(selected.sum()),
                "gains": int((selected & (challenger == labels) & (incumbent != labels)).sum()),
                "losses": int((selected & (challenger != labels) & (incumbent == labels)).sum()),
                "users": user_reports,
            }
        )
    return fold_reports


def test_groups(test_indices, records):
    output = []
    for index in test_indices:
        interval = record_interval(records[int(index)])
        if interval is None:
            raise ValueError(f"Missing test timestamp for {index}")
        date = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")
        output.append(TEST_GROUP_BY_DATE[date])
    return output


def build_candidate(name, incumbent_path, minimum_repeats, records):
    challenger_path = (
        ROOT
        / "experiments/v160_radar_batch_decoder/candidates/v160_radar_batch_source_coverage.csv"
    )
    incumbent = pd.read_csv(incumbent_path)
    challenger = pd.read_csv(challenger_path)
    if not incumbent["path"].equals(challenger["path"]):
        raise ValueError("Candidate paths do not align")
    test_indices = np.asarray(
        [record["index"] for record in records if record["split"] == "test"],
        dtype=np.int64,
    )
    evidence = np.load(
        ROOT / "experiments/v160_radar_batch_decoder/outputs/test_radar_batch.npz"
    )["raw_probabilities"].astype(np.float64)
    incumbent_labels = incumbent["prediction"].to_numpy(dtype=np.int64)
    challenger_labels = challenger["prediction"].to_numpy(dtype=np.int64)
    selected = cycle_mask(test_indices, incumbent_labels, challenger_labels, records)
    selected |= accept(incumbent_labels, challenger_labels, evidence)
    selected |= repeated_transition_mask(
        test_groups(test_indices, records),
        incumbent_labels,
        challenger_labels,
        minimum_repeats,
    )
    output = incumbent.copy()
    output.loc[selected, "prediction"] = challenger.loc[selected, "prediction"]
    destination = HERE / f"candidates/{name}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    report = {
        "output": str(destination.relative_to(ROOT)),
        "incumbent": str(incumbent_path.relative_to(ROOT)),
        "minimum_repeats": minimum_repeats,
        "rows": len(output),
        "accepted": int(selected.sum()),
        "changed_rows": [
            {
                "position": int(position),
                "path": str(output.iloc[position]["path"]),
                "incumbent": int(incumbent_labels[position]),
                "challenger": int(challenger_labels[position]),
            }
            for position in np.flatnonzero(selected)
        ],
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def summarize(folds):
    return {
        "incumbent_correct": sum(item["incumbent"]["correct"] for item in folds),
        "candidate_correct": sum(item["candidate"]["correct"] for item in folds),
        "rows": sum(item["candidate"]["rows"] for item in folds),
        "accepted": sum(item["accepted"] for item in folds),
        "gains": sum(item["gains"] for item in folds),
        "losses": sum(item["losses"] for item in folds),
    }


def main():
    records = get_records()
    public_base = (
        ROOT / "experiments/v160_radar_batch_decoder/candidates/v127_without_0531.csv"
    )
    local_base = (
        ROOT
        / "experiments/v127_supervision_routed_system/candidates/v127_medium_source_coverage.csv"
    )
    variants = {}
    for minimum_repeats in (2, 3):
        folds = validation(records, minimum_repeats)
        name = f"v166_repeat{minimum_repeats}"
        variants[name] = {
            "folds": folds,
            "summary": summarize(folds),
            "public_test": build_candidate(
                name, public_base, minimum_repeats, records
            ),
            "generalization_test": build_candidate(
                f"{name}_generalization",
                local_base,
                minimum_repeats,
                records,
            ),
        }
    report = {
        "protocol": (
            "Union V164 evidence, V165 balanced cycles, and same-user repeated "
            "V127-to-V160 transitions."
        ),
        "test_groups": TEST_GROUP_BY_DATE,
        "variants": variants,
    }
    (HERE / "outputs").mkdir(parents=True, exist_ok=True)
    (HERE / "outputs/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
