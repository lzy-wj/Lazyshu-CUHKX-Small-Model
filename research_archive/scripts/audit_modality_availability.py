import io
import json
import sys
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_data import DEVICES
from src.data import ROOT, get_indices, get_records
from src.scratch_protocol import modality_presence


def archive_csv_entries():
    entries = {}
    current = None
    with (ROOT / "data/metadata/train_archive_listing.txt").open() as handle:
        for line in handle:
            key, separator, value = line.rstrip("\n").partition(" = ")
            if key == "Path" and separator:
                current = value if value.endswith(".csv") and any(f"/{name}/" in value for name in ["Radar", "IMU"]) else None
                if current:
                    entries[current] = {}
            elif current and key in ["Size", "CRC"]:
                entries[current][key] = value
    return entries


def inspect_sensor(arguments):
    record, modality, archive = arguments
    result = {"index": record["index"], "id": record["id"], "user": record["user"],
              "modality": modality, "csv_files": 0, "rows": 0, "matching_rows": 0,
              "archive_checks": Counter(), "examples": []}
    if modality not in record["modalities"]:
        return result | {"reason": "no_modality_directory"}
    folder = ROOT / record["modalities"][modality]
    direct = sorted(folder.glob("*.csv"))
    recursive = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() == ".csv")
    result["csv_files"] = len(direct)
    result["extra_csv_not_read_by_current_parser"] = len(set(recursive) - set(direct))
    if not direct:
        return result | {"reason": "no_direct_csv"}
    errors = []
    for path in direct:
        raw = path.read_bytes()
        archive_path = str(path.relative_to(ROOT / "data/raw"))
        expected = archive.get(archive_path)
        if expected and "Size" in expected and "CRC" in expected:
            matches = len(raw) == int(expected["Size"]) and f"{zlib.crc32(raw) & 0xffffffff:08X}" == expected["CRC"].upper()
            result["archive_checks"]["match" if matches else "mismatch"] += 1
        else:
            result["archive_checks"]["not_in_listing_or_no_crc"] += 1
        try:
            table = pd.read_csv(io.BytesIO(raw))
            result["rows"] += len(table)
            if modality == "Radar":
                result["matching_rows"] += int(table["timestamp"].notna().sum())
            elif len(table):
                names = table.iloc[:, 1].astype(str)
                result["matching_rows"] += int(names.str.startswith(tuple(device + "(" for device in DEVICES)).sum())
            if table.empty:
                result["examples"].append({"path": str(path.relative_to(ROOT)), "bytes": len(raw), "columns": table.columns.tolist()})
        except (ValueError, KeyError, pd.errors.ParserError) as error:
            errors.append({"path": str(path.relative_to(ROOT)), "error": type(error).__name__})
    if errors:
        return result | {"reason": "parse_error", "errors": errors}
    if not result["rows"]:
        return result | {"reason": "csv_headers_only"}
    if not result["matching_rows"]:
        return result | {"reason": "all_timestamps_missing" if modality == "Radar" else "no_recognized_imu_device"}
    return result | {"reason": "usable"}


def main():
    records = get_records()
    selected = [record for record in records if record["split"] == "train"]
    train, valid = get_indices(records, 0)
    modalities = ["scene_ir", "thermal", "scene_depth", "skeleton", "imu", "radar"]
    raw_names = ["IR", "Thermal", "Depth_Color", "Skeleton", "IMU", "Radar"]
    masks = modality_presence(modalities)
    report = {"audited_at": datetime.now().astimezone().isoformat(), "total_labeled_clips": len(selected),
              "fold0_training_clips": len(train), "fold0_validation_clips": len(valid), "modalities": {}}
    for position, name in enumerate(raw_names):
        report["modalities"][name] = {
            "directories": sum(name in record["modalities"] for record in selected),
            "cached_available": int(masks[[record["index"] for record in selected], position].sum()),
            "fold0_train_available": int(masks[train, position].sum()),
            "fold0_valid_available": int(masks[valid, position].sum()),
        }
    archive = archive_csv_entries()
    work = [(record, name, archive) for name in ["Radar", "IMU"] for record in selected]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(inspect_sensor, work))
    for modality, column in [("Radar", 5), ("IMU", 4)]:
        subset = [result for result in results if result["modality"] == modality]
        reasons = Counter(result["reason"] for result in subset)
        checks = Counter()
        by_user = defaultdict(Counter)
        examples = {}
        mismatches = []
        for result in subset:
            checks.update(result["archive_checks"])
            by_user[str(result["user"])][result["reason"]] += 1
            if (result["reason"] == "usable") != bool(masks[result["index"], column]):
                mismatches.append(result)
            if result["reason"] != "usable" and result["reason"] not in examples:
                examples[result["reason"]] = result
        report["modalities"][modality].update({"raw_reasons": dict(reasons), "archive_crc_and_size": dict(checks),
            "parser_cache_mismatches": mismatches,
            "extra_csv_not_read": sum(result.get("extra_csv_not_read_by_current_parser", 0) for result in subset),
            "per_user_reasons": dict(by_user), "examples": examples})
    report["all_six_available"] = int(masks[[record["index"] for record in selected]].all(1).sum())
    report["any_modality_available"] = int(masks[[record["index"] for record in selected]].any(1).sum())
    report["scope"] = "All labeled clips: raw IMU/radar CSV reread, nested CSV check, cached presence comparison, extracted CSV size/CRC against downloaded archive listing. Does not establish why upstream recordings are empty."
    destination = ROOT / "outputs/study_scratch/availability_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, values in report["modalities"].items():
        print(name, json.dumps({key: value for key, value in values.items() if key not in ["per_user_reasons", "examples", "parser_cache_mismatches"]}, ensure_ascii=False))
        if "parser_cache_mismatches" in values:
            print("Parser/cache mismatches:", len(values["parser_cache_mismatches"]))
    print("All six:", report["all_six_available"], "Any:", report["any_modality_available"])
    print(destination)


if __name__ == "__main__":
    main()
