#!/usr/bin/env python3
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v84_residual_invariant_skeleton.evaluate import domains
from src.data import get_records


def ranks(probabilities):
    output = np.empty_like(probabilities, dtype=np.int64)
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    output[np.arange(len(probabilities))[:, None], order] = np.arange(1, probabilities.shape[1] + 1)
    return output


def accept(
    incumbent,
    challenger,
    probabilities,
    challenger_max_rank=2,
    incumbent_min_rank=3,
):
    model_ranks = ranks(probabilities)
    rows = np.arange(len(incumbent))
    return (
        (incumbent != challenger)
        & (model_ranks[rows, challenger] <= challenger_max_rank)
        & (model_ranks[rows, incumbent] >= incumbent_min_rank)
    )


def validation(records, challenger_max_rank=2, incumbent_min_rank=3):
    fold_reports = []
    for fold in (0, 1):
        reference, _, v104, _ = v104_probabilities(fold, records)
        v71 = np.load(
            ROOT / f"experiments/v71_gated_film_distillation/outputs/student_f{fold}.npz"
        )["probabilities"].astype(np.float64)
        v114, _ = fused_probabilities(fold, reference, v104, records, weight=0.6)
        route = domains(reference["indices"], fold, records) >= 2
        evidence = np.where(route[:, None], v114, v71)
        incumbent_archive = np.load(
            ROOT
            / f"experiments/v127_supervision_routed_system/outputs/v127_validation_f{fold}.npz"
        )
        challenger_archive = np.load(
            ROOT / f"experiments/v160_radar_batch_decoder/outputs/validation_f{fold}.npz"
        )
        incumbent = incumbent_archive["pseudo_labels"].astype(np.int64)
        challenger = challenger_archive["pseudo_labels"].astype(np.int64)
        labels = reference["labels"].astype(np.int64)
        selected = accept(
            incumbent,
            challenger,
            evidence,
            challenger_max_rank,
            incumbent_min_rank,
        )
        prediction = incumbent.copy()
        prediction[selected] = challenger[selected]
        user_reports = {}
        for user in sorted({records[int(index)]["user"] for index in reference["indices"]}):
            user_rows = np.asarray(
                [records[int(index)]["user"] == user for index in reference["indices"]]
            )
            user_reports[str(user)] = {
                "rows": int(user_rows.sum()),
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


def build_test_candidate(
    name,
    incumbent_path,
    challenger_max_rank=2,
    incumbent_min_rank=3,
):
    challenger_path = (
        ROOT
        / "experiments/v160_radar_batch_decoder/candidates/v160_radar_batch_source_coverage.csv"
    )
    incumbent = pd.read_csv(incumbent_path)
    challenger = pd.read_csv(challenger_path)
    if not incumbent["path"].equals(challenger["path"]):
        raise ValueError("Candidate paths do not align")
    archive = np.load(
        ROOT / "experiments/v160_radar_batch_decoder/outputs/test_radar_batch.npz"
    )
    evidence = archive["raw_probabilities"].astype(np.float64)
    incumbent_labels = incumbent["prediction"].to_numpy(dtype=np.int64)
    challenger_labels = challenger["prediction"].to_numpy(dtype=np.int64)
    selected = accept(
        incumbent_labels,
        challenger_labels,
        evidence,
        challenger_max_rank,
        incumbent_min_rank,
    )
    output = incumbent.copy()
    output.loc[selected, "prediction"] = challenger.loc[selected, "prediction"]
    destination = HERE / f"candidates/{name}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    model_ranks = ranks(evidence)
    rows = np.arange(len(output))
    report = {
        "output": str(destination.relative_to(ROOT)),
        "incumbent": str(incumbent_path.relative_to(ROOT)),
        "rows": len(output),
        "accepted": int(selected.sum()),
        "challenger_max_rank": challenger_max_rank,
        "incumbent_min_rank": incumbent_min_rank,
        "changed_rows": [
            {
                "position": int(position),
                "path": str(output.iloc[position]["path"]),
                "incumbent": int(incumbent_labels[position]),
                "challenger": int(challenger_labels[position]),
                "incumbent_rank": int(model_ranks[position, incumbent_labels[position]]),
                "challenger_rank": int(model_ranks[position, challenger_labels[position]]),
            }
            for position in np.flatnonzero(selected)
        ],
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    (destination.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    records = get_records()
    settings = {
        "v164_evidence_gated_radar": (2, 3, "public"),
        "v164_evidence_gated_radar_conservative": (1, 4, "public"),
        "v164_evidence_gated_radar_strict": (1, 5, "public"),
        "v164_evidence_gated_radar_generalization": (2, 3, "local"),
    }
    variants = {}
    for name, (challenger_max_rank, incumbent_min_rank, base) in settings.items():
        folds = validation(records, challenger_max_rank, incumbent_min_rank)
        incumbent_path = (
            ROOT / "experiments/v127_supervision_routed_system/candidates/v127_medium_source_coverage.csv"
            if base == "local"
            else ROOT / "experiments/v160_radar_batch_decoder/candidates/v127_without_0531.csv"
        )
        test = build_test_candidate(
            name,
            incumbent_path,
            challenger_max_rank,
            incumbent_min_rank,
        )
        variants[name] = {
            "selection": {
                "challenger_max_rank": challenger_max_rank,
                "incumbent_min_rank": incumbent_min_rank,
            },
            "folds": folds,
            "summary": {
                "incumbent_correct": sum(item["incumbent"]["correct"] for item in folds),
                "candidate_correct": sum(item["candidate"]["correct"] for item in folds),
                "rows": sum(item["candidate"]["rows"] for item in folds),
                "accepted": sum(item["accepted"] for item in folds),
                "gains": sum(item["gains"] for item in folds),
                "losses": sum(item["losses"] for item in folds),
            },
            "test": test,
        }
    report = {
        "protocol": (
            "Accept a Radar-group challenger only when it is top-2 under the original "
            "date-routed multimodal evidence and the V127 incumbent is outside top-2."
        ),
        "variants": variants,
    }
    (HERE / "outputs").mkdir(parents=True, exist_ok=True)
    (HERE / "outputs/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
