#!/usr/bin/env python3
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v160_radar_batch_decoder.radar_batch import radar_batch_interval
from experiments.v164_evidence_gated_radar.evaluate import accept
from experiments.v84_residual_invariant_skeleton.evaluate import domains
from src.data import get_records


def temporal_groups(indices, records, gap_seconds=30):
    values = [
        (radar_batch_interval(records[int(index)]), int(index)) for index in indices
    ]
    missing = [index for interval, index in values if interval is None]
    values = sorted((interval, index) for interval, index in values if interval is not None)
    output = []
    for interval, index in values:
        if not output or interval[0] - output[-1][-1][0][1] > gap_seconds:
            output.append([])
        output[-1].append((interval, index))
    return [[index for _, index in group] for group in output] + [
        [index] for index in missing
    ]


def maximum_balanced_subset(positions, incumbent, challenger):
    changed = [
        int(position)
        for position in positions
        if incumbent[position] != challenger[position]
    ]
    for size in range(len(changed), 1, -1):
        for subset in itertools.combinations(changed, size):
            selected = list(subset)
            if Counter(incumbent[selected]) == Counter(challenger[selected]):
                return selected
    return []


def cycle_mask(indices, incumbent, challenger, records):
    positions = {int(index): position for position, index in enumerate(indices)}
    selected = np.zeros(len(indices), dtype=bool)
    for group in temporal_groups(indices, records):
        group_positions = [positions[index] for index in group]
        selected[
            maximum_balanced_subset(group_positions, incumbent, challenger)
        ] = True
    return selected


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


def validation(records, use_evidence):
    fold_reports = []
    for fold in (0, 1):
        reference, evidence, incumbent, challenger = fold_inputs(fold, records)
        selected = cycle_mask(
            reference["indices"], incumbent, challenger, records
        )
        if use_evidence:
            selected |= accept(incumbent, challenger, evidence)
        prediction = incumbent.copy()
        prediction[selected] = challenger[selected]
        labels = reference["labels"].astype(np.int64)
        user_reports = {}
        for user in sorted({records[int(index)]["user"] for index in reference["indices"]}):
            user_rows = np.asarray(
                [records[int(index)]["user"] == user for index in reference["indices"]]
            )
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


def build_candidate(name, incumbent_path, use_evidence, records):
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
    archive = np.load(
        ROOT / "experiments/v160_radar_batch_decoder/outputs/test_radar_batch.npz"
    )
    evidence = archive["raw_probabilities"].astype(np.float64)
    incumbent_labels = incumbent["prediction"].to_numpy(dtype=np.int64)
    challenger_labels = challenger["prediction"].to_numpy(dtype=np.int64)
    selected = cycle_mask(test_indices, incumbent_labels, challenger_labels, records)
    if use_evidence:
        selected |= accept(incumbent_labels, challenger_labels, evidence)
    output = incumbent.copy()
    output.loc[selected, "prediction"] = challenger.loc[selected, "prediction"]
    destination = HERE / f"candidates/{name}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    report = {
        "output": str(destination.relative_to(ROOT)),
        "incumbent": str(incumbent_path.relative_to(ROOT)),
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
    cycle_folds = validation(records, use_evidence=False)
    union_folds = validation(records, use_evidence=True)
    report = {
        "protocol": (
            "Within each acquisition round, accept the largest V160 change subset that "
            "preserves the V127 label multiset; union it with the fixed V164 evidence gate."
        ),
        "cycle_only": {
            "folds": cycle_folds,
            "summary": summarize(cycle_folds),
            "test": build_candidate(
                "v165_balanced_cycle_only", public_base, False, records
            ),
        },
        "union": {
            "folds": union_folds,
            "summary": summarize(union_folds),
            "public_test": build_candidate(
                "v165_balanced_cycle_union", public_base, True, records
            ),
            "generalization_test": build_candidate(
                "v165_balanced_cycle_union_generalization", local_base, True, records
            ),
        },
    }
    (HERE / "outputs").mkdir(parents=True, exist_ok=True)
    (HERE / "outputs/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
