#!/usr/bin/env python3
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v160_radar_batch_decoder.radar_batch import SOURCE_BY_DATE, date_for_record
from src.data import get_records


def read(path):
    table = pd.read_csv(path)
    if list(table.columns) != ["path", "prediction"] or len(table) != 405:
        raise ValueError(f"Invalid submission: {path}")
    return table


def main():
    records = get_records()
    index_by_id = {record["id"]: record["index"] for record in records}
    incumbent = read(ROOT / "experiments/v24_full_refit/outputs/submissions/v24_full_warm60_pseudo_pair_aware.csv")
    candidate = read(ROOT / "experiments/v160_radar_batch_decoder/candidates/v160_radar_batch_source_coverage.csv")
    dates = [date_for_record(records[index_by_id[path]]) for path in incumbent["path"]]
    families = {
        "source20": {"2025-05-31", "2025-06-01"},
        "source21": {"2025-06-02"},
        "source16": {"2025-06-16", "2025-06-17"},
    }
    output_dir = ROOT / "experiments/v160_radar_batch_decoder/candidates"
    report = {"families": {}}
    for name, selected_dates in families.items():
        values = incumbent.copy()
        active = pd.Series([date in selected_dates for date in dates])
        values.loc[active, "prediction"] = candidate.loc[active, "prediction"]
        destination = output_dir / f"v160_{name}.csv"
        values.to_csv(destination, index=False)
        report["families"][name] = {
            "dates": sorted(selected_dates),
            "rows": int(active.sum()),
            "changes_vs_v24": int((values["prediction"] != incumbent["prediction"]).sum()),
            "changed_rows": int((values["prediction"] != candidate["prediction"]).sum()),
        }
    (output_dir / "partition_candidates.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
