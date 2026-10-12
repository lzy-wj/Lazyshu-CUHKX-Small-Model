#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


TIMESTAMP_PATTERN = re.compile(r"(20\d\d-\d\d-\d\d_\d\d-\d\d-\d\d\.\d+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/samples.json"))
    parser.add_argument("--gap-seconds", type=float, default=30.0)
    return parser.parse_args()


def timestamp(record):
    for modality in ("Depth_Color", "IR"):
        folder = Path(record["modalities"].get(modality, ""))
        if not folder.is_dir():
            continue
        values = []
        for file_path in folder.iterdir():
            match = TIMESTAMP_PATTERN.search(file_path.name)
            if match:
                values.append(
                    datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S.%f").timestamp()
                )
        if values:
            return min(values), max(values)
    return None


def segment(records, gap_seconds):
    stamped = [(timestamp(record), record) for record in records]
    stamped = [(stamp, record) for stamp, record in stamped if stamp is not None]
    stamped.sort(key=lambda value: (value[0][0], value[1]["index"]))
    groups = []
    for stamp, record in stamped:
        if not groups or stamp[0] - groups[-1][-1][0][1] > gap_seconds:
            groups.append([])
        groups[-1].append((stamp, record))
    return groups


def main():
    args = parse_args()
    records = json.loads(args.metadata.read_text())
    train = [record for record in records if record["split"] == "train"]
    by_user = defaultdict(list)
    for record in train:
        by_user[record["user"]].append(record)

    all_segments = []
    for user, user_records in sorted(by_user.items()):
        for group in segment(user_records, args.gap_seconds):
            all_segments.append((user, group))

    purity = []
    exact_trials = 0
    descriptors = []
    for user, group in all_segments:
        trials = Counter(item[1]["trial"] for item in group)
        dominant, count = trials.most_common(1)[0]
        purity.append(count / len(group))
        exact_trials += len(trials) == 1
        descriptors.append(
            {
                "user": user,
                "trial": dominant,
                "trial_count": len(trials),
                "start": group[0][0][0],
                "end": max(item[0][1] for item in group),
                "labels": [item[1]["label"] for item in group],
            }
        )

    triplet_orders = Counter()
    by_prefix = defaultdict(list)
    for descriptor in descriptors:
        parts = descriptor["trial"].split("-")
        by_prefix[(descriptor["user"], "-".join(parts[:2]))].append(descriptor)
    for values in by_prefix.values():
        ordered = sorted(values, key=lambda value: value["start"])
        triplet_orders[tuple(value["trial"].split("-")[-1] for value in ordered)] += 1

    chronological = defaultdict(list)
    for descriptor in descriptors:
        chronological[descriptor["user"]].append(descriptor)
    consecutive_triplets = 0
    total_prefixes = 0
    examples = []
    same_prefix_gaps = []
    different_prefix_gaps = []
    same_length_same_prefix = Counter()
    same_length_different_prefix = Counter()
    for user, values in chronological.items():
        ordered = sorted(values, key=lambda value: value["start"])
        for left, right in zip(ordered, ordered[1:]):
            left_prefix = "-".join(left["trial"].split("-")[:2])
            right_prefix = "-".join(right["trial"].split("-")[:2])
            gap = right["start"] - left["end"]
            if left_prefix == right_prefix:
                same_prefix_gaps.append(gap)
                same_length_same_prefix[len(left["labels"]) == len(right["labels"])] += 1
            else:
                different_prefix_gaps.append(gap)
                same_length_different_prefix[len(left["labels"]) == len(right["labels"])] += 1
        positions = {id(value): position for position, value in enumerate(ordered)}
        grouped = defaultdict(list)
        for value in ordered:
            grouped["-".join(value["trial"].split("-")[:2])].append(value)
        for prefix, prefix_values in grouped.items():
            if len(prefix_values) != 3:
                continue
            total_prefixes += 1
            pos = sorted(positions[id(value)] for value in prefix_values)
            if pos == list(range(pos[0], pos[0] + 3)):
                consecutive_triplets += 1
            elif len(examples) < 10:
                examples.append({"user": user, "prefix": prefix, "positions": pos})

    def percentiles(values):
        values = sorted(values)
        return {
            str(percentile): values[round((len(values) - 1) * percentile / 100)]
            for percentile in (0, 10, 25, 50, 75, 90, 100)
        }

    print(
        json.dumps(
            {
                "records": len(train),
                "segments": len(all_segments),
                "exact_trial_segments": exact_trials,
                "minimum_purity": min(purity),
                "prefixes": len(by_prefix),
                "three_segment_prefixes": total_prefixes,
                "chronologically_consecutive_triplets": consecutive_triplets,
                "triplet_orders": {str(key): value for key, value in triplet_orders.items()},
                "same_prefix_gap_seconds": percentiles(same_prefix_gaps),
                "different_prefix_gap_seconds": percentiles(different_prefix_gaps),
                "same_prefix_adjacent_equal_length": dict(same_length_same_prefix),
                "different_prefix_adjacent_equal_length": dict(same_length_different_prefix),
                "nonconsecutive_examples": examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
