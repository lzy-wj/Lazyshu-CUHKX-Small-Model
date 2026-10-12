#!/usr/bin/env python3
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.evaluate_repeat_decoder as repeat_decoder
from experiments.scene_context_v13.research_flexible_pair_decoder import (
    flexible_assignment,
    model_score,
)
from experiments.scene_context_v13.research_paired_script_decoder import user_templates
from experiments.scene_context_v13.research_repeat_recovery_decoder import prepool, run
from experiments.scene_context_v13.research_sequence_decoder import repeat_pool
from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v70_film_margin_distillation.validate import SETTINGS as V71_SETTINGS
from experiments.v84_residual_invariant_skeleton.evaluate import PAIR_SETTINGS, domains
from src.data import FOLDS, get_records


TIMESTAMP_PATTERN = re.compile(r"(20\d\d-\d\d-\d\d_\d\d-\d\d-\d\d\.\d+)")
SOURCE_BY_DATE = {
    "2025-05-31": 20,
    "2025-06-01": 20,
    "2025-06-02": 21,
    "2025-06-12": None,
    "2025-06-13": None,
    "2025-06-16": 16,
    "2025-06-17": 16,
}
ORIGINAL_RECORD_INTERVAL = repeat_decoder.record_interval
RADAR_INTERVAL_CACHE = {}


def radar_batch_interval(record):
    index = int(record["index"])
    if index in RADAR_INTERVAL_CACHE:
        return RADAR_INTERVAL_CACHE[index]
    folder = Path(record["modalities"].get("Radar", ""))
    values = []
    if folder.is_dir():
        for path in folder.iterdir():
            match = TIMESTAMP_PATTERN.search(path.name)
            if match:
                values.append(
                    datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S.%f").timestamp()
                )
    if values:
        value = min(values)
        interval = (value, max(values))
    else:
        interval = ORIGINAL_RECORD_INTERVAL(record)
    RADAR_INTERVAL_CACHE[index] = interval
    return interval


def install_radar_batch_intervals():
    repeat_decoder.record_interval = radar_batch_interval


def date_for_record(record):
    interval = ORIGINAL_RECORD_INTERVAL(record)
    if interval is None:
        raise ValueError(f"Missing acquisition interval for {record['id']}")
    return datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")


def ordered_probabilities(archive_path, indices):
    archive = np.load(archive_path)
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    return np.stack([archive["probabilities"][positions[int(index)]] for index in indices]).astype(np.float64)


def decode_test_route(probabilities, indices, records, route, fold=0):
    install_radar_batch_intervals()
    if route == "v71":
        probabilities, _ = prepool(
            probabilities,
            indices,
            records,
            V71_SETTINGS["maximum_gap"],
            V71_SETTINGS["minimum_similarity"],
            V71_SETTINGS["power"],
        )
    probability_by_index, round_values, groups = repeat_pool(probabilities, indices, records)
    transitions = [repeat_decoder.transition_matrix(records, set(), environment=value) for value in (0, 1)]
    source_users = {value for value in SOURCE_BY_DATE.values() if value is not None}
    templates = {
        user: sorted({sequence for values in user_templates(records, user).values() for sequence in values})
        for user in source_users
    }
    decoded = {int(index): probability.copy() for index, probability in zip(indices, probabilities)}
    changed_by_date = Counter()
    selected_by_date = Counter()
    rounds_by_date = Counter()
    for item in round_values:
        date = date_for_record(records[item["rows"][0]])
        if date not in SOURCE_BY_DATE:
            raise ValueError(f"Unexpected test acquisition date: {date}")
        environment = int(date >= "2025-06-09")
        transition = transitions[environment]
        values = np.stack([probability_by_index[index] for index in item["rows"]])
        free = repeat_decoder.assign_transition_unique(values, transition, 0.4, beam_size=100).argmax(axis=1)
        unary = np.log(np.clip(values, 1e-12, 1.0))
        free_score = model_score(free, unary, transition, 0.4)
        selected = tuple(int(value) for value in free)
        source = SOURCE_BY_DATE[date]
        best = (free_score, free_score, selected, 0, 0)
        if source is not None:
            for template in templates[source]:
                candidate = flexible_assignment(
                    values,
                    template,
                    transition,
                    transition_strength=0.4,
                    template_strength=1.8,
                    gap_penalty=0.8,
                    maximum_edits=5,
                    beam_size=40,
                )
                if candidate is not None and candidate[0] > best[0]:
                    best = candidate
        rounds_by_date[date] += 1
        if best[2] != selected:
            selected_by_date[date] += 1
            changed_by_date[date] += sum(left != right for left, right in zip(best[2], selected))
            selected = best[2]
        constrained = np.full_like(values, 1e-12)
        constrained[np.arange(len(selected)), selected] = 1.0
        for position, index in enumerate(item["rows"]):
            decoded[index] = constrained[position]
    ordered = np.stack([decoded[int(index)] for index in indices])
    report = {
        "route": route,
        "rows": len(indices),
        "rounds": len(round_values),
        "repeat_groups": len(groups),
        "rounds_by_date": dict(sorted(rounds_by_date.items())),
        "selected_rounds_by_date": dict(sorted(selected_by_date.items())),
        "changed_predictions": int((ordered.argmax(axis=1) != probabilities.argmax(axis=1)).sum()),
        "changed_predictions_by_date": dict(sorted(changed_by_date.items())),
    }
    return ordered, report


def validation_report():
    install_radar_batch_intervals()
    records = get_records()
    fold_reports = []
    for fold in (0, 1):
        reference, _, v104, _ = v104_probabilities(fold, records)
        v71_archive = np.load(
            ROOT / "experiments/v71_gated_film_distillation/outputs" / f"student_f{fold}.npz"
        )
        v114, _ = fused_probabilities(fold, reference, v104, records, weight=0.6)
        domain = domains(reference["indices"], fold, records)
        v71_result = run(
            fold,
            (v71_archive["probabilities"], reference["indices"], reference["labels"]),
            records,
            V71_SETTINGS,
        )
        v114_result = run(
            fold,
            (v114, reference["indices"], reference["labels"]),
            records,
            PAIR_SETTINGS,
        )
        prediction = np.where(
            domain >= 2,
            np.asarray(v114_result["predictions"], dtype=np.int64),
            np.asarray(v71_result["predictions"], dtype=np.int64),
        )
        output = ROOT / "experiments/v160_radar_batch_decoder/outputs"
        output.mkdir(parents=True, exist_ok=True)
        np.savez(
            output / f"validation_f{fold}.npz",
            indices=reference["indices"],
            labels=reference["labels"],
            pseudo_labels=prediction,
        )
        incumbent = np.load(
            ROOT / "experiments/v127_supervision_routed_system/outputs" / f"v127_validation_f{fold}.npz"
        )["pseudo_labels"]
        fold_reports.append(
            {
                "fold": fold,
                "rows": len(prediction),
                "correct": int((prediction == reference["labels"]).sum()),
                "accuracy": float((prediction == reference["labels"]).mean()),
                "incumbent_correct": int((incumbent == reference["labels"]).sum()),
                "delta": int((prediction == reference["labels"]).sum() - (incumbent == reference["labels"]).sum()),
                "metrics": scores(prediction, reference, fold, records),
                "v71_rounds": v71_result["rounds"],
                "v114_rounds": v114_result["rounds"],
                "v71_repeat_groups": v71_result["prepool_groups"],
                "v114_repeat_groups": v114_result["prepool_groups"],
            }
        )
    correct = sum(item["correct"] for item in fold_reports)
    rows = sum(item["rows"] for item in fold_reports)
    report = {
        "protocol": "Replace timestamp-based grouping with the timestamped Radar filename batch key; all model probabilities, source labels, and decoder strengths remain fixed.",
        "interval_policy": "Radar filename timestamp is the batch key; visual/CSV interval is used only when Radar is unavailable and for date routing.",
        "folds": fold_reports,
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "incumbent_v127": {"correct": sum(item["incumbent_correct"] for item in fold_reports), "rows": rows},
        "promote": all(item["delta"] > 0 for item in fold_reports),
    }
    output = ROOT / "experiments/v160_radar_batch_decoder/outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_test_candidate():
    install_radar_batch_intervals()
    records = get_records()
    test_indices = np.asarray([record["index"] for record in records if record["split"] == "test"], dtype=np.int64)
    v71 = ordered_probabilities(ROOT / "experiments/v24_full_refit/outputs/predictions/v71_gated_film_distilled.npz", test_indices)
    v114 = ordered_probabilities(ROOT / "experiments/v114_paired_person_fusion/candidates/v114_paired_person_raw.npz", test_indices)
    dates = [date_for_record(records[int(index)]) for index in test_indices]
    paired = np.asarray([SOURCE_BY_DATE[date] is not None for date in dates], dtype=bool)
    v71_decoded, v71_report = decode_test_route(v71, test_indices, records, "v71")
    v114_decoded, v114_report = decode_test_route(v114, test_indices, records, "v114")
    raw_probabilities = np.where(paired[:, None], v114, v71)
    decoded_probabilities = np.where(paired[:, None], v114_decoded, v71_decoded)
    predictions = decoded_probabilities.argmax(axis=1).astype(np.int64)
    incumbent_path = ROOT / "experiments/v24_full_refit/outputs/submissions/v24_full_warm60_pseudo_pair_aware.csv"
    incumbent_table = csv.reader(incumbent_path.open(newline=""))
    next(incumbent_table)
    incumbent = np.asarray([int(row[1]) for row in incumbent_table], dtype=np.int64)
    v127_path = ROOT / "experiments/v127_supervision_routed_system/candidates/v127_medium_source_coverage.csv"
    v127_table = csv.reader(v127_path.open(newline=""))
    next(v127_table)
    v127_predictions = np.asarray([int(row[1]) for row in v127_table], dtype=np.int64)
    source_support = {
        source: {record["label"] for record in records if record["split"] == "train" and record["user"] == source}
        for source in {source for source in SOURCE_BY_DATE.values() if source is not None}
    }
    protected = np.asarray(
        [paired[position] and int(prediction) not in source_support[SOURCE_BY_DATE[dates[position]]]
         for position, prediction in enumerate(predictions)],
        dtype=bool,
    )
    protected_predictions = predictions.copy()
    protected_predictions[protected] = incumbent[protected]
    output_dir = ROOT / "experiments/v160_radar_batch_decoder/candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_archive = ROOT / "experiments/v160_radar_batch_decoder/outputs/test_radar_batch.npz"
    np.savez(raw_archive, probabilities=decoded_probabilities, raw_probabilities=raw_probabilities, indices=test_indices)
    path_by_index = {record["index"]: record["id"] for record in records}
    for name, values in (("v160_radar_batch.csv", predictions), ("v160_radar_batch_source_coverage.csv", protected_predictions)):
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "prediction"])
            for index, prediction in zip(test_indices, values):
                writer.writerow([path_by_index[int(index)], int(prediction)])
    report = {
        "protocol": "Radar batch-key decoder with date-routed source templates; source-unsupported paired labels are protected by the public V24 incumbent.",
        "rows": len(test_indices),
        "paired_rows": int(paired.sum()),
        "unmatched_rows": int((~paired).sum()),
        "protected_source_coverage_rows": int(protected.sum()),
        "changed_vs_v127_medium": int((protected_predictions != v127_predictions).sum()),
        "changed_vs_v24": int((predictions != incumbent).sum()),
        "v71": v71_report,
        "v114": v114_report,
        "sha256": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in ("v160_radar_batch.csv", "v160_radar_batch_source_coverage.csv")
        },
        "submitted": True,
        "public_score": 0.84079,
    }
    (output_dir / "v160_radar_batch.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = validation_report()
    print(json.dumps({"validation": result["summary"], "folds": result["folds"], "promote": result["promote"]}, indent=2))
    test_result = build_test_candidate()
    print(json.dumps({"test": {key: value for key, value in test_result.items() if key not in ("v71", "v114")}}, indent=2))
