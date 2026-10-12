#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TARGET_LIMIT = 95_000_000
COMPETITION_LIMIT = 100_000_000
EXPECTED_WEIGHTS = [0.082416, 0.079504, 0.148508, 0.089572, 0.600000]
EXPECTED_EPOCHS = [13, 16, 7, 16, 9]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fallback-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs=5, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--decoded-prediction", type=Path, required=True)
    parser.add_argument("--raw-submission", type=Path, required=True)
    parser.add_argument("--decoded-submission", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_submission(path, test_csv):
    submission = pd.read_csv(path)
    if list(submission.columns) != ["path", "prediction"]:
        raise ValueError(f"Unexpected submission columns in {path}")
    if len(submission) != len(test_csv) or submission["path"].tolist() != test_csv["path"].tolist():
        raise ValueError(f"Submission row order does not match test.csv: {path}")
    if submission["prediction"].isna().any():
        raise ValueError(f"Submission contains missing predictions: {path}")
    predictions = submission["prediction"].to_numpy()
    if not np.equal(predictions, predictions.astype(np.int64)).all():
        raise ValueError(f"Submission contains non-integer predictions: {path}")
    if not ((0 <= predictions) & (predictions < 40)).all():
        raise ValueError(f"Submission contains predictions outside [0, 39]: {path}")
    return {"rows": len(submission), "classes": int(submission["prediction"].nunique())}


def main():
    args = parse_args()
    required = [
        args.checkpoint,
        args.fallback_checkpoint,
        args.yolo,
        *args.sources,
        args.prediction,
        args.decoded_prediction,
        args.raw_submission,
        args.decoded_submission,
        args.decoder_report,
        args.test_csv,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    source_reports = []
    for path, expected_epoch in zip(args.sources, EXPECTED_EPOCHS):
        archive = torch.load(path, map_location="cpu", weights_only=False)
        config = archive["config"]
        if int(config["fold"]) != -1 or int(config["train_clips"]) != 3036:
            raise ValueError(f"Source is not a 3036-clip full-data fit: {path}")
        if bool(config.get("use_radar", False)):
            raise ValueError(f"Source unexpectedly enables radar: {path}")
        if config.get("mode") != "shared_multimodal" or config.get("visual_strategy") != "joint":
            raise ValueError(f"Source architecture is not the frozen shared/joint model: {path}")
        if archive.get("epoch") != expected_epoch:
            raise ValueError(f"Source epoch mismatch for {path}: {archive.get('epoch')} != {expected_epoch}")
        source_reports.append({
            "path": str(path),
            "sha256": sha256(path),
            "epoch": archive.get("epoch"),
            "train_clips": int(config["train_clips"]),
        })

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if int(config["fold"]) != -1 or int(config["train_clips"]) != 3036:
        raise ValueError("Final checkpoint does not identify a full-data fit")
    if bool(config.get("use_radar", False)):
        raise ValueError("Final checkpoint unexpectedly enables radar")
    if config.get("mode") != "shared_multimodal" or config.get("visual_strategy") != "joint":
        raise ValueError("Final checkpoint is not the frozen shared/joint architecture")
    soup = checkpoint.get("soup")
    if not soup or len(soup.get("sources", [])) != 5:
        raise ValueError("Final checkpoint does not contain the frozen five-source soup manifest")
    actual_weights = [float(source["weight"]) for source in soup["sources"]]
    if not np.allclose(actual_weights, EXPECTED_WEIGHTS, atol=1e-9):
        raise ValueError(f"Soup weights changed: {actual_weights}")
    for expected, actual in zip(source_reports, soup["sources"]):
        if expected["sha256"] != actual["sha256"]:
            raise ValueError("Soup source hash does not match audited full-data checkpoint")

    model_asset_bytes = args.checkpoint.stat().st_size + args.yolo.stat().st_size
    fallback_asset_bytes = args.fallback_checkpoint.stat().st_size + args.yolo.stat().st_size
    if model_asset_bytes > TARGET_LIMIT:
        raise ValueError(f"Model assets exceed V24 target: {model_asset_bytes} > {TARGET_LIMIT}")
    if model_asset_bytes >= COMPETITION_LIMIT:
        raise ValueError(f"Model assets exceed competition limit: {model_asset_bytes}")
    if fallback_asset_bytes > TARGET_LIMIT:
        raise ValueError(f"Fallback model assets exceed V24 target: {fallback_asset_bytes} > {TARGET_LIMIT}")

    archive = np.load(args.prediction)
    if archive["probabilities"].shape != (405, 40) or archive["indices"].shape != (405,):
        raise ValueError("Prediction archive must contain exactly 405 rows and 40 classes")
    decoded_archive = np.load(args.decoded_prediction)
    if decoded_archive["probabilities"].shape != (405, 40):
        raise ValueError("Decoded prediction archive must contain exactly 405 rows and 40 classes")
    if not np.array_equal(decoded_archive["indices"], archive["indices"]):
        raise ValueError("Decoded prediction indices do not match raw prediction indices")
    test_csv = pd.read_csv(args.test_csv)
    raw = check_submission(args.raw_submission, test_csv)
    decoded = check_submission(args.decoded_submission, test_csv)
    decoder = json.loads(args.decoder_report.read_text())
    protocol = decoder.get("protocol", {})
    expected_decoder = {
        "maximum_edits": 5,
        "gap_penalty": 0.8,
        "template_strength": 1.8,
        "alignment_beam_size": 40,
        "transition_strength": 0.4,
    }
    for key, value in expected_decoder.items():
        if protocol.get(key) != value:
            raise ValueError(f"Frozen decoder setting changed for {key}")

    files = {}
    for path in required:
        files[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    report = {
        "status": "pass",
        "protocol": "docs/V24_EXPERIMENT_PROTOCOL_20260910.md",
        "selection_data": "Fold 0 only; Fold 1 blind confirmation; no test labels",
        "checkpoint": str(args.checkpoint),
        "model_asset_bytes": model_asset_bytes,
        "fallback_model_asset_bytes": fallback_asset_bytes,
        "target_limit_bytes": TARGET_LIMIT,
        "competition_limit_bytes": COMPETITION_LIMIT,
        "target_slack_bytes": TARGET_LIMIT - model_asset_bytes,
        "competition_slack_bytes": COMPETITION_LIMIT - model_asset_bytes,
        "soup_weights": actual_weights,
        "sources": source_reports,
        "raw_submission": raw,
        "decoded_submission": decoded,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
