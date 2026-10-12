#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/pair_identification_decoder.json",
    )
    return parser.parse_args()


def identify_paired_users(prepared, result):
    users_by_environment = defaultdict(set)
    for item in prepared["rounds"]:
        users_by_environment[item["environment"]].add(item["user"])
    selected = set()
    diagnostics = {}
    for environment, users in sorted(users_by_environment.items()):
        ranking = sorted(
            (
                {
                    "user": user,
                    "margin": result["per_user"][str(user)]["margin"],
                    "selected_rounds": result["per_user"][str(user)]["selected_rounds"],
                    "changed_predictions": result["per_user"][str(user)]["changed_predictions"],
                }
                for user in users
            ),
            key=lambda item: (item["margin"], item["selected_rounds"]),
            reverse=True,
        )
        selected.add(ranking[0]["user"])
        diagnostics[str(environment)] = ranking
    return selected, diagnostics


def identify_strict_pairs(prepared):
    unrestricted = evaluate(prepared, template_strength=0.8, gap_penalty=0.3, maximum_edits=1)
    return identify_paired_users(prepared, unrestricted)


def run_setting(
    prepared,
    paired_users,
    diagnostics,
    template_strength,
    gap_penalty,
    maximum_edits,
    alignment_beam_size,
):
    restricted = evaluate(
        prepared,
        template_strength,
        gap_penalty,
        maximum_edits,
        allowed_users=paired_users,
        alignment_beam_size=alignment_beam_size,
    )
    return {
        "identified_users": sorted(paired_users),
        "identification": diagnostics,
        "result": restricted,
    }


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    prepared = {
        mode: [
            prepare_fold(
                *folds[fold],
                records,
                PAIR_SOURCES[fold],
                FOLDS[fold],
                aligned_repeats=mode == "aligned",
            )
            for fold in (0, 1)
        ]
        for mode in ("standard", "aligned")
    }
    pair_identity = {
        mode: [identify_strict_pairs(values) for values in prepared[mode]]
        for mode in prepared
    }
    candidates = []
    for maximum_edits in (4, 5):
        for gap_penalty in (0.4, 0.6):
            for template_strength in (1.4, 1.6):
                for alignment_beam_size in (40,):
                    for repeat_mode in prepared:
                        candidates.append(
                            {
                                "maximum_edits": maximum_edits,
                                "gap_penalty": gap_penalty,
                                "template_strength": template_strength,
                                "alignment_beam_size": alignment_beam_size,
                                "repeat_mode": repeat_mode,
                                "fold0": run_setting(
                                    prepared[repeat_mode][0],
                                    pair_identity[repeat_mode][0][0],
                                    pair_identity[repeat_mode][0][1],
                                    template_strength,
                                    gap_penalty,
                                    maximum_edits,
                                    alignment_beam_size,
                                ),
                            }
                        )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["result"]["correct"],
            -item["fold0"]["result"]["changed_predictions"],
            -item["maximum_edits"],
            -item["template_strength"],
            -item["alignment_beam_size"],
        ),
    )
    selected["fold1"] = run_setting(
        prepared[selected["repeat_mode"]][1],
        pair_identity[selected["repeat_mode"]][1][0],
        pair_identity[selected["repeat_mode"]][1][1],
        selected["template_strength"],
        selected["gap_penalty"],
        selected["maximum_edits"],
        selected["alignment_beam_size"],
    )
    correct = selected["fold0"]["result"]["correct"] + selected["fold1"]["result"]["correct"]
    rows = len(folds[0][2]) + len(folds[1][2])
    report = {
        "protocol": (
            "Fold0 selects edit settings. A fixed strict one-edit matcher identifies one paired target in each "
            "environment from aggregate model-template margin without labels; fold1 is evaluated once blind."
        ),
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "baseline": {"correct": 1103, "rows": rows, "accuracy": 1103 / rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
