#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v114_paired_person_fusion.build_candidate import SOURCE_BY_DATE
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-adaptation", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path):
    table = pd.read_csv(path)
    if list(table.columns) != ["path", "prediction"] or len(table) != 405:
        raise ValueError(f"Invalid submission table: {path}")
    return table


def main():
    args = parse_args()
    pre_adaptation = read(args.pre_adaptation)
    incumbent = read(args.incumbent)
    challenger = read(args.challenger)
    if not (
        pre_adaptation["path"].equals(incumbent["path"])
        and incumbent["path"].equals(challenger["path"])
    ):
        raise ValueError("Submission paths do not align")
    protected_adaptation = pre_adaptation["prediction"] != incumbent["prediction"]
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
    protected_unsupported = []
    for path, label in zip(incumbent["path"], incumbent["prediction"]):
        index = index_by_id[path]
        interval = record_interval(records[index])
        if interval is None:
            raise ValueError(f"Missing timestamp for {path}")
        date = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")
        source = SOURCE_BY_DATE.get(date)
        protected_unsupported.append(
            source is not None and int(label) not in source_support[source]
        )
    protected_unsupported = pd.Series(protected_unsupported)
    protected = protected_adaptation | protected_unsupported
    output = challenger.copy()
    output.loc[protected, "prediction"] = incumbent.loc[protected, "prediction"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    changed = output["prediction"] != incumbent["prediction"]
    report = {
        "protocol": (
            "Preserve every established V24 self-training correction; accept "
            "paired-person challenger labels only on all other rows."
        ),
        "protected_rows": int(protected.sum()),
        "protected_prior_adaptation_rows": int(protected_adaptation.sum()),
        "protected_source_unsupported_rows": int(protected_unsupported.sum()),
        "challenger_changes_before_protection": int(
            (challenger["prediction"] != incumbent["prediction"]).sum()
        ),
        "accepted_changes": int(changed.sum()),
        "changed_rows": [
            {
                "position": int(position),
                "path": str(output.iloc[position]["path"]),
                "incumbent": int(incumbent.iloc[position]["prediction"]),
                "challenger": int(output.iloc[position]["prediction"]),
            }
            for position in changed[changed].index
        ],
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
