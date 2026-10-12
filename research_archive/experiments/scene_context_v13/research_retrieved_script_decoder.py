#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/retrieved_script_decoder.json",
    )
    return parser.parse_args()


def candidate_sources(records, excluded_users):
    return tuple(
        sorted(
            {
                record["user"]
                for record in records
                if record["split"] == "train" and record["user"] not in excluded_users
            }
        )
    )


def retrieve_sources(prepared, target_users, source_users):
    rankings = {}
    selected = {}
    for target in target_users:
        environment = int(target > 15)
        candidates = []
        for source in source_users:
            if int(source > 15) != environment:
                continue
            result = evaluate(
                prepared,
                template_strength=0.8,
                gap_penalty=0.3,
                maximum_edits=1,
                allowed_users={target},
                source_by_user={target: source},
            )
            values = result["per_user"][str(target)]
            candidates.append(
                {
                    "source": source,
                    "margin": values["margin"],
                    "selected_rounds": values["selected_rounds"],
                    "changed_predictions": values["changed_predictions"],
                }
            )
        candidates.sort(
            key=lambda item: (item["margin"], item["selected_rounds"]), reverse=True
        )
        rankings[str(target)] = candidates
        selected[target] = candidates[0]["source"]
    return selected, rankings


def run_fold(fold, records):
    probabilities = selected_probabilities(fold)
    sources = candidate_sources(records, set(FOLDS[fold]))
    by_environment = (
        tuple(user for user in sources if user <= 15),
        tuple(user for user in sources if user > 15),
    )
    prepared = prepare_fold(
        *probabilities,
        records,
        by_environment,
        FOLDS[fold],
    )
    mapping, rankings = retrieve_sources(prepared, FOLDS[fold], sources)
    result = evaluate(
        prepared,
        template_strength=1.6,
        gap_penalty=0.6,
        maximum_edits=5,
        allowed_users=set(FOLDS[fold]),
        alignment_beam_size=40,
        source_by_user=mapping,
    )
    return {
        "mapping": {str(key): value for key, value in sorted(mapping.items())},
        "rankings": rankings,
        "result": result,
    }


def main():
    args = parse_args()
    records = get_records()
    fold0 = run_fold(0, records)
    fold1 = run_fold(1, records)
    correct = fold0["result"]["correct"] + fold1["result"]["correct"]
    rows = 1357
    report = {
        "protocol": (
            "For every validation user, a strict matcher retrieves one same-environment source from fold-local "
            "training users without target labels. The frozen flexible decoder is then applied."
        ),
        "fold0": fold0,
        "fold1": fold1,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "baseline": {"correct": 1103, "rows": rows, "accuracy": 1103 / rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
