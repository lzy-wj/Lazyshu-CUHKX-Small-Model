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

from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


UNMATCHED_DATES = {"2025-06-12", "2025-06-13"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pre-adaptation",
        type=Path,
        default=ROOT
        / "experiments/v24_full_refit/outputs/submissions/"
        "v24_full_warm60_int8_pair_aware.csv",
    )
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=ROOT
        / "experiments/v24_full_refit/outputs/submissions/"
        "v24_full_warm60_pseudo_pair_aware.csv",
    )
    parser.add_argument(
        "--paired",
        type=Path,
        default=ROOT
        / "experiments/v114_paired_person_fusion/candidates/"
        "v114_protected_candidate.csv",
    )
    parser.add_argument(
        "--unmatched",
        type=Path,
        default=ROOT / "outputs/submissions/v14_fixedsoup_pseudo_pair_aware.csv",
    )
    parser.add_argument("--unmatched-system", default="V14")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "candidates/v125_domain_complement.csv",
    )
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
    paired = read(args.paired)
    unmatched = read(args.unmatched)
    if not all(
        incumbent["path"].equals(table["path"])
        for table in (pre_adaptation, paired, unmatched)
    ):
        raise ValueError("Submission paths do not align")

    records = get_records()
    index_by_id = {record["id"]: record["index"] for record in records}
    dates = []
    for path in incumbent["path"]:
        interval = record_interval(records[index_by_id[path]])
        if interval is None:
            raise ValueError(f"Missing timestamp for {path}")
        dates.append(datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d"))
    use_unmatched = pd.Series([date in UNMATCHED_DATES for date in dates])

    scientific = paired.copy()
    scientific.loc[use_unmatched, "prediction"] = unmatched.loc[
        use_unmatched, "prediction"
    ]
    protected = scientific.copy()
    prior_correction = pre_adaptation["prediction"] != incumbent["prediction"]
    protected.loc[prior_correction, "prediction"] = incumbent.loc[
        prior_correction, "prediction"
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scientific.to_csv(args.output, index=False)
    protected_path = args.output.with_name(args.output.stem + "_protected.csv")
    protected.to_csv(protected_path, index=False)
    report = {
        "protocol": (
            f"V114 on dates with a labeled same-person source; {args.unmatched_system} on the two "
            "unmatched dates. Protected output preserves all V24 pseudo-adaptation "
            "corrections."
        ),
        "unmatched_dates": sorted(UNMATCHED_DATES),
        "unmatched_rows": int(use_unmatched.sum()),
        "scientific_changes_vs_v114": int(
            (scientific["prediction"] != paired["prediction"]).sum()
        ),
        "protected_changes_vs_v114": int(
            (protected["prediction"] != paired["prediction"]).sum()
        ),
        "protected_prior_rows": int(prior_correction.sum()),
        "outputs": {
            "scientific": {
                "path": str(args.output.relative_to(ROOT)),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            "protected": {
                "path": str(protected_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(protected_path.read_bytes()).hexdigest(),
            },
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    main()
