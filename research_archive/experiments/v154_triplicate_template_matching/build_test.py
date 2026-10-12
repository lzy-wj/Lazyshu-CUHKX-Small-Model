#!/usr/bin/env python3
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import flexible_assignment, model_score
from experiments.scene_context_v13.research_paired_script_decoder import user_templates
from experiments.scene_context_v13.research_sequence_decoder import repeat_pool
from experiments.v114_paired_person_fusion.build_candidate import SOURCE_BY_DATE
from scripts.evaluate_repeat_decoder import assign_transition_unique, record_interval, transition_matrix
from src.data import get_records


REPEATS = 3
UNMATCHED_DATES = {"2025-06-12", "2025-06-13"}


def read_submission(path):
    table = pd.read_csv(path)
    if list(table.columns) != ["path", "prediction"] or len(table) != 405:
        raise ValueError(f"Invalid submission: {path}")
    return table


def source_templates(records, source):
    return sorted(
        {
            sequence
            for sequences in user_templates(records, source).values()
            for sequence in sequences
        }
    )


def main():
    records = get_records()
    archive = np.load(HERE.parent / "v114_paired_person_fusion/candidates/v114_paired_person_raw.npz")
    probabilities = archive["probabilities"].astype(np.float64)
    indices = archive["indices"].astype(np.int64)
    probability_by_index, round_values, repeat_groups = repeat_pool(
        probabilities, indices, records
    )
    transitions = [transition_matrix(records, set(), environment=value) for value in (0, 1)]
    decoded = {
        int(index): int(probability.argmax())
        for index, probability in zip(indices, probabilities)
    }
    rounds_by_source = {source: [] for source in set(SOURCE_BY_DATE.values())}
    round_dates = {}
    for item in round_values:
        date = datetime.fromtimestamp(item["start"]).strftime("%Y-%m-%d")
        source = SOURCE_BY_DATE.get(date)
        if source is not None:
            rounds_by_source[source].append(item)
            round_dates[id(item)] = date

    source_reports = {}
    for source, target_rounds in sorted(rounds_by_source.items()):
        unique_templates = source_templates(records, source)
        templates = [template for template in unique_templates for _ in range(REPEATS)]
        gains = np.full(
            (len(target_rounds), len(templates) + len(target_rounds)),
            -1e6,
            dtype=np.float64,
        )
        gains[:, len(templates):] = 0
        candidates = {}
        free_sequences = []
        for target_position, item in enumerate(target_rounds):
            date = round_dates[id(item)]
            environment = int(date >= "2025-06-09")
            transition = transitions[environment]
            values = np.stack([probability_by_index[index] for index in item["rows"]])
            free = assign_transition_unique(values, transition, 0.4, beam_size=100).argmax(1)
            unary = np.log(np.clip(values, 1e-12, 1))
            free_score = model_score(free, unary, transition, 0.4)
            free_sequences.append(tuple(int(value) for value in free))
            for template_position, template in enumerate(templates):
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
                if candidate is None:
                    continue
                gains[target_position, template_position] = candidate[0] - free_score
                candidates[(target_position, template_position)] = candidate
        rows, columns = linear_sum_assignment(-gains)
        assignment = dict(zip(rows.tolist(), columns.tolist()))
        usage = np.zeros(len(unique_templates), dtype=np.int64)
        selected_rounds = changed = 0
        for target_position, item in enumerate(target_rounds):
            template_position = assignment[target_position]
            selected = free_sequences[target_position]
            if template_position < len(templates) and gains[target_position, template_position] > 0:
                selected = candidates[(target_position, template_position)][2]
                usage[template_position // REPEATS] += 1
                selected_rounds += 1
            changed += sum(left != right for left, right in zip(free_sequences[target_position], selected))
            decoded.update(zip(item["rows"], selected))
        source_reports[str(source)] = {
            "rounds": len(target_rounds),
            "templates": len(unique_templates),
            "selected_rounds": selected_rounds,
            "changed_from_free": changed,
            "template_usage": usage.tolist(),
        }

    v24 = read_submission(
        ROOT / "experiments/v24_full_refit/outputs/submissions/v24_full_warm60_pseudo_pair_aware.csv"
    )
    v114 = read_submission(
        ROOT / "experiments/v114_paired_person_fusion/candidates/v114_paired_person_do_not_submit.csv"
    )
    v127 = read_submission(
        ROOT / "experiments/v127_supervision_routed_system/candidates/v127_medium_source_coverage.csv"
    )
    unmatched = read_submission(
        ROOT / "experiments/v24_full_refit/outputs/submissions/v71_gated_film_sequence_hybrid.csv"
    )
    index_by_id = {record["id"]: record["index"] for record in records}
    source_support = {
        source: {
            record["label"]
            for record in records
            if record["split"] == "train" and record["user"] == source
        }
        for source in rounds_by_source
    }
    predictions = []
    dates = []
    protected = []
    for row, path in enumerate(v24["path"]):
        index = index_by_id[path]
        date = datetime.fromtimestamp(record_interval(records[index])[0]).strftime("%Y-%m-%d")
        dates.append(date)
        source = SOURCE_BY_DATE.get(date)
        preserve = source is not None and int(v24.iloc[row]["prediction"]) not in source_support[source]
        protected.append(preserve)
        if date in UNMATCHED_DATES:
            prediction = int(unmatched.iloc[row]["prediction"])
        elif preserve:
            prediction = int(v24.iloc[row]["prediction"])
        else:
            prediction = int(decoded[index])
        predictions.append(prediction)
    output = pd.DataFrame({"path": v24["path"], "prediction": predictions})
    destination = HERE / "candidates/v154_triplicate_source_coverage.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    report = {
        "protocol": "Three source-template slots, V71 unmatched route, and V127 source-coverage protection.",
        "rounds": len(round_values),
        "repeat_groups": len(repeat_groups),
        "sources": source_reports,
        "protected_rows": int(sum(protected)),
        "changes_vs_v24": int((output["prediction"] != v24["prediction"]).sum()),
        "changes_vs_v114": int((output["prediction"] != v114["prediction"]).sum()),
        "changes_vs_v127": int((output["prediction"] != v127["prediction"]).sum()),
        "distribution": {
            str(label): count for label, count in sorted(Counter(predictions).items())
        },
        "submission_eligible": False,
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
