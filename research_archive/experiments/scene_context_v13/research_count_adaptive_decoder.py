#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_pair_identification_decoder import identify_strict_pairs
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_roundwise_pair_decoder import decode, source_rounds
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/count_adaptive_decoder.json",
    )
    return parser.parse_args()


def independent(prepared, paired):
    return evaluate(
        prepared,
        template_strength=1.8,
        gap_penalty=0.8,
        maximum_edits=5,
        allowed_users=paired,
        alignment_beam_size=40,
    )


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    prepared = [
        prepare_fold(*folds[fold], records, PAIR_SOURCES[fold], FOLDS[fold])
        for fold in (0, 1)
    ]
    paired = [identify_strict_pairs(values)[0] for values in prepared]
    reports = []
    for fold in (0, 1):
        mapping = {
            user: PAIR_SOURCES[fold][int(user > 15)][0]
            for user in paired[fold]
        }
        direct = independent(prepared[fold], paired[fold])
        ordered = decode(prepared[fold], records, mapping, skip_penalty=0.0, minimum_gain=0.0)
        per_user = {}
        for user in sorted(direct["per_user"], key=int):
            numeric_user = int(user)
            if numeric_user not in mapping:
                mode = "independent"
                values = direct["per_user"][user]
            else:
                source = mapping[numeric_user]
                target_rounds = sum(item["user"] == numeric_user for item in prepared[fold]["rounds"])
                source_count = len(source_rounds(records, source))
                mode = "ordered" if abs(target_rounds - source_count) > 2 else "independent"
                values = (ordered if mode == "ordered" else direct)["per_user"][user]
                values = values | {"target_rounds": target_rounds, "source_rounds": source_count}
            per_user[user] = values | {"mode": mode}
        correct = sum(values["correct"] for values in per_user.values())
        rows = sum(values["rows"] for values in per_user.values())
        reports.append(
            {
                "correct": correct,
                "rows": rows,
                "accuracy": correct / rows,
                "per_user": per_user,
                "independent": direct,
                "ordered": ordered,
            }
        )
    correct = reports[0]["correct"] + reports[1]["correct"]
    rows = reports[0]["rows"] + reports[1]["rows"]
    report = {
        "protocol": (
            "Use independent template editing when target/source round counts differ by at most two; otherwise "
            "use chronological cyclic alignment. The threshold is structural and fixed before evaluation."
        ),
        "fold0": reports[0],
        "fold1": reports[1],
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "deployed_reference": {"correct": 1161, "rows": rows, "accuracy": 1161 / rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
