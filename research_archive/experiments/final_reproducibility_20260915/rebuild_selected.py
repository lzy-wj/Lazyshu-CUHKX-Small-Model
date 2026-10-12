#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v160_radar_batch_decoder.radar_batch import date_for_record
from src.data import get_records


BASE_PATH = ROOT / (
    "experiments/v24_full_refit/outputs/submissions/"
    "v24_full_warm60_pseudo_pair_aware.csv"
)
FULL_V127_PATH = ROOT / (
    "experiments/v127_supervision_routed_system/candidates/"
    "v127_medium_source_coverage.csv"
)
PRIMARY_PATH = ROOT / (
    "experiments/v160_radar_batch_decoder/candidates/v127_without_0531.csv"
)
RADAR_PATH = ROOT / (
    "experiments/v160_radar_batch_decoder/candidates/"
    "v160_radar_batch_source_coverage.csv"
)
RADAR_EVIDENCE_PATH = ROOT / (
    "experiments/v160_radar_batch_decoder/outputs/test_radar_batch.npz"
)
HEDGE_PATH = ROOT / (
    "experiments/v164_evidence_gated_radar/candidates/"
    "v164_evidence_gated_radar.csv"
)
ASSET_AUDIT_PATH = ROOT / (
    "experiments/v127_supervision_routed_system/asset_audit.json"
)
EXPECTED_HASHES = {
    "v127_without_0531.csv": (
        "76210e13f1f63a3d63c356f2462aa09917a6d9c0c919a0e8025098ab4408acb0"
    ),
    "v164_evidence_gated_radar.csv": (
        "aaf3ac9dc0fe71fe5a21b78d2243ac330dee3bec0aad69ebb52f11acb7989639"
    ),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_submission(path):
    table = pd.read_csv(path)
    if list(table.columns) != ["path", "prediction"] or len(table) != 405:
        raise ValueError(f"Invalid submission table: {path}")
    if not table["path"].is_unique:
        raise ValueError(f"Duplicate paths in submission table: {path}")
    labels = table["prediction"].to_numpy(dtype=np.int64)
    if not np.array_equal(labels, table["prediction"].to_numpy()):
        raise ValueError(f"Non-integral predictions in submission table: {path}")
    if np.any((labels < 0) | (labels >= 40)):
        raise ValueError(f"Out-of-range predictions in submission table: {path}")
    return table


def assert_aligned(reference, *tables):
    for table in tables:
        if not reference["path"].equals(table["path"]):
            raise ValueError("Submission paths do not align")


def ranks(probabilities):
    output = np.empty_like(probabilities, dtype=np.int64)
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    output[np.arange(len(probabilities))[:, None], order] = np.arange(
        1, probabilities.shape[1] + 1
    )
    return output


def audit_assets():
    report = json.loads(ASSET_AUDIT_PATH.read_text())
    total = 0
    for asset in report["assets"]:
        path = ROOT / asset["path"]
        if path.stat().st_size != asset["bytes"]:
            raise RuntimeError(f"Asset size mismatch: {path}")
        if digest(path) != asset["sha256"]:
            raise RuntimeError(f"Asset hash mismatch: {path}")
        total += path.stat().st_size
    if total != report["total_bytes"] or total >= report["competition_limit_bytes"]:
        raise RuntimeError("Model asset budget audit failed")
    return {
        "count": len(report["assets"]),
        "total_bytes": total,
        "limit_bytes": report["competition_limit_bytes"],
        "slack_bytes": report["competition_limit_bytes"] - total,
    }


def rebuild_primary():
    base = read_submission(BASE_PATH)
    full_v127 = read_submission(FULL_V127_PATH)
    assert_aligned(base, full_v127)
    records = get_records()
    record_by_id = {record["id"]: record for record in records}
    dates = pd.Series(
        [date_for_record(record_by_id[path]) for path in full_v127["path"]]
    )
    rollback = dates.eq("2025-05-31") & full_v127["prediction"].ne(
        base["prediction"]
    )
    output = full_v127.copy()
    output.loc[rollback, "prediction"] = base.loc[rollback, "prediction"]
    return output, np.flatnonzero(rollback.to_numpy()).tolist()


def rebuild_hedge(primary):
    challenger = read_submission(RADAR_PATH)
    assert_aligned(primary, challenger)
    archive = np.load(RADAR_EVIDENCE_PATH)
    evidence = archive["raw_probabilities"].astype(np.float64)
    indices = archive["indices"].astype(np.int64)
    records = get_records()
    test_indices = np.asarray(
        [record["index"] for record in records if record["split"] == "test"],
        dtype=np.int64,
    )
    if evidence.shape != (405, 40) or not np.array_equal(indices, test_indices):
        raise ValueError("Radar evidence is not aligned with the test set")
    incumbent_labels = primary["prediction"].to_numpy(dtype=np.int64)
    challenger_labels = challenger["prediction"].to_numpy(dtype=np.int64)
    model_ranks = ranks(evidence)
    rows = np.arange(len(primary))
    selected = (
        (incumbent_labels != challenger_labels)
        & (model_ranks[rows, challenger_labels] <= 2)
        & (model_ranks[rows, incumbent_labels] >= 3)
    )
    output = primary.copy()
    output.loc[selected, "prediction"] = challenger.loc[selected, "prediction"]
    return output, np.flatnonzero(selected).tolist()


def verify_output(table, name, expected_path, output_dir):
    destination = output_dir / name
    table.to_csv(destination, index=False)
    actual_hash = digest(destination)
    expected_hash = EXPECTED_HASHES[name]
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Rebuilt hash mismatch for {name}: {actual_hash} != {expected_hash}"
        )
    if destination.read_bytes() != expected_path.read_bytes():
        raise RuntimeError(f"Rebuilt bytes differ from frozen candidate: {name}")
    return {
        "path": str(destination),
        "rows": len(table),
        "sha256": actual_hash,
        "byte_identical_to": str(expected_path.relative_to(ROOT)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "rebuilt",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    asset_report = audit_assets()
    primary, rollback_positions = rebuild_primary()
    hedge, hedge_positions = rebuild_hedge(primary)
    report = {
        "pass": True,
        "assets": asset_report,
        "primary": verify_output(
            primary,
            "v127_without_0531.csv",
            PRIMARY_PATH,
            args.output_dir,
        ),
        "hedge": verify_output(
            hedge,
            "v164_evidence_gated_radar.csv",
            HEDGE_PATH,
            args.output_dir,
        ),
        "primary_rollback_positions": rollback_positions,
        "hedge_changed_positions": hedge_positions,
    }
    (args.output_dir / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
