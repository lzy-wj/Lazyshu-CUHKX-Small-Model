import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.validation_splits import make_diagnostic_split, validate_diagnostic_split


def main():
    records = get_records()
    split = make_diagnostic_split(records)
    assert split == make_diagnostic_split(records)
    rejected = []
    for case in ["overlap", "missing_clip", "outer_change", "metadata", "fold", "duplicate"]:
        broken = copy.deepcopy(split)
        if case == "overlap":
            broken["train"].append(broken["valid"][0])
        elif case == "missing_clip":
            broken["train"].pop()
        elif case == "outer_change":
            broken["outer_valid"].pop()
        elif case == "metadata":
            broken["metadata_sha256"] = "incorrect"
        elif case == "fold":
            broken["fold"] = 1
        elif case == "duplicate":
            broken["valid"].append(broken["valid"][0])
        try:
            validate_diagnostic_split(broken, records, 0)
        except ValueError:
            rejected.append(case)
        else:
            raise AssertionError(f"Invalid split accepted: {case}")
    destination = ROOT / "configs/same_user_diagnostic_split.json"
    if destination.exists() and json.loads(destination.read_text()) != split:
        raise RuntimeError("Refusing to overwrite a different frozen diagnostic split")
    destination.write_text(json.dumps(split, indent=2) + "\n")
    report = {"status": "passed", "rejected": rejected, "partitions": {}}
    for name in ["train", "valid", "outer_valid"]:
        selected = [records[index] for index in split[name]]
        report["partitions"][name] = {"clips": len(selected), "users": sorted({record["user"] for record in selected}),
                                      "supported_classes": len({record["label"] for record in selected})}
    folder = ROOT / "outputs/study_validation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
