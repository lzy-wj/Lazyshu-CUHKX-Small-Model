#!/usr/bin/env python3
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v114_paired_person_fusion.build_candidate import SOURCE_BY_DATE
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


UNMATCHED_DATES = {"2025-06-12", "2025-06-13"}


def read(path):
    table = pd.read_csv(path)
    if list(table.columns) != ["path", "prediction"] or len(table) != 405:
        raise ValueError(f"Invalid submission table: {path}")
    return table


def main():
    incumbent = read(
        ROOT
        / "experiments/v24_full_refit/outputs/submissions/"
        "v24_full_warm60_pseudo_pair_aware.csv"
    )
    paired = read(
        ROOT
        / "experiments/v114_paired_person_fusion/candidates/"
        "v114_paired_person_do_not_submit.csv"
    )
    unmatched = read(
        ROOT
        / "experiments/v24_full_refit/outputs/submissions/"
        "v71_gated_film_sequence_hybrid.csv"
    )
    if not incumbent["path"].equals(paired["path"]) or not incumbent["path"].equals(
        unmatched["path"]
    ):
        raise ValueError("Submission paths do not align")

    records = get_records()
    index_by_id = {record["id"]: record["index"] for record in records}
    source_support = {
        source: {
            record["label"]
            for record in records
            if record["split"] == "train" and record["user"] == source
        }
        for source in set(SOURCE_BY_DATE.values())
    }
    dates = []
    protected = []
    for path, incumbent_label in zip(incumbent["path"], incumbent["prediction"]):
        interval = record_interval(records[index_by_id[path]])
        if interval is None:
            raise ValueError(f"Missing timestamp for {path}")
        date = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")
        dates.append(date)
        source = SOURCE_BY_DATE.get(date)
        protected.append(
            source is not None and int(incumbent_label) not in source_support[source]
        )
    protected = pd.Series(protected)
    use_unmatched = pd.Series([date in UNMATCHED_DATES for date in dates])
    output = paired.copy()
    output.loc[use_unmatched, "prediction"] = unmatched.loc[
        use_unmatched, "prediction"
    ]
    output.loc[protected, "prediction"] = incumbent.loc[protected, "prediction"]

    destination = HERE / "candidates/v127_medium_source_coverage.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    changed = output["prediction"] != incumbent["prediction"]
    report = {
        "protocol": (
            "V114 paired domains plus V71 unmatched domains; preserve incumbent "
            "labels absent from the paired source user's labeled class coverage."
        ),
        "protected_source_coverage_rows": int(protected.sum()),
        "changes_vs_public_v24": int(changed.sum()),
        "changed_rows": [
            {
                "position": int(position),
                "path": str(output.iloc[position]["path"]),
                "date": dates[position],
                "incumbent": int(incumbent.iloc[position]["prediction"]),
                "candidate": int(output.iloc[position]["prediction"]),
            }
            for position in changed[changed].index
        ],
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "submitted": False,
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
