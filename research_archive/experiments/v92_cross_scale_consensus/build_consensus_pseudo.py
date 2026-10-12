#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import flexible_assignment, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--lowres-pseudo", type=Path, required=True)
    parser.add_argument("--highres-probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pair_predictions(probabilities, indices, labels, fold, records):
    prepared = prepare_fold(
        probabilities,
        indices,
        labels,
        records,
        PAIR_SOURCES[fold],
        FOLDS[fold],
    )
    prediction = {int(index): int(values.argmax()) for index, values in zip(indices, probabilities)}
    for item in prepared["rounds"]:
        selected = item["free"]
        best = (item["free_score"], item["free_score"], selected, 0, 0)
        if item["user"] in PAIRED_USERS[fold]:
            for source in prepared["sources"][item["environment"]]:
                for template in prepared["templates"][source]:
                    candidate = flexible_assignment(
                        item["values"],
                        template,
                        item["transition"],
                        transition_strength=0.4,
                        template_strength=1.8,
                        gap_penalty=0.8,
                        maximum_edits=5,
                        beam_size=40,
                    )
                    if candidate is not None and candidate[0] > best[0]:
                        best = candidate
            selected = best[2]
        prediction.update(zip(item["rows"], selected))
    return np.asarray([prediction[int(index)] for index in indices], dtype=np.int64)


def main():
    args = parse_args()
    records = get_records()
    lowres = np.load(args.lowres_pseudo)
    highres = np.load(args.highres_probabilities)
    positions = {int(index): position for position, index in enumerate(highres["indices"])}
    order = np.asarray([positions[int(index)] for index in lowres["indices"]])
    if not np.array_equal(lowres["labels"], highres["labels"][order]):
        raise ValueError("Low- and high-resolution validation labels do not align")
    highres_pseudo = pair_predictions(
        highres["probabilities"][order],
        lowres["indices"],
        lowres["labels"],
        args.fold,
        records,
    )
    original_selected = lowres["paired"] | (lowres["confidence"] >= 0.6)
    agreement = lowres["pseudo_labels"] == highres_pseudo
    selected = original_selected & agreement
    confidence = selected.astype(np.float32)
    paired = np.zeros(len(selected), dtype=bool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        probabilities=lowres["probabilities"],
        pseudo_labels=lowres["pseudo_labels"],
        confidence=confidence,
        paired=paired,
        indices=lowres["indices"],
        labels=lowres["labels"],
        highres_pseudo_labels=highres_pseudo,
        cross_scale_agreement=agreement,
    )
    labels = lowres["labels"]
    removed = original_selected & ~agreement
    report = {
        "fold": args.fold,
        "protocol": "Original V24 selection intersected with 128/160 pair-aware agreement.",
        "original_selected": int(original_selected.sum()),
        "consensus_selected": int(selected.sum()),
        "removed": int(removed.sum()),
        "audit_only_original_pseudo_accuracy": float(
            (lowres["pseudo_labels"][original_selected] == labels[original_selected]).mean()
        ),
        "audit_only_consensus_pseudo_accuracy": float(
            (lowres["pseudo_labels"][selected] == labels[selected]).mean()
        ),
        "audit_only_removed_pseudo_accuracy": float(
            (lowres["pseudo_labels"][removed] == labels[removed]).mean()
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
