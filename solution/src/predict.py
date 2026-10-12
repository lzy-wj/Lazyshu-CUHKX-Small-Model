import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from src.pipeline import run_pipeline
from src.runtime_data import prepare_runtime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    return parser.parse_args()


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_artifacts(package_root):
    expected = json.loads((package_root / "configs/artifact_hashes.json").read_text())
    total = 0
    for relative, metadata in expected["files"].items():
        path = package_root / relative
        if path.stat().st_size != metadata["bytes"]:
            raise RuntimeError(f"Artifact size mismatch: {relative}")
        if sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"Artifact SHA-256 mismatch: {relative}")
        total += path.stat().st_size
    if total != expected["total_bytes"] or total >= 100_000_000:
        raise RuntimeError("Small-track model artifact budget check failed")
    return total


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable NVIDIA GPU is required")
    package_root = Path(__file__).resolve().parents[1]
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    work_dir = args.work_dir or args.output_csv.parent / ".lazyshu_inference_cache"
    work_dir.mkdir(parents=True, exist_ok=True)
    total_assets = verify_artifacts(package_root)
    started = time.monotonic()
    records, test_csv, sensors, videos = prepare_runtime(
        args.data_dir,
        work_dir,
        package_root / "artifacts/yolo11n_compact.pt",
        "0",
        workers=4,
        detection_batch_size=64,
    )
    preprocessing_seconds = time.monotonic() - started
    report = run_pipeline(
        package_root,
        records,
        test_csv,
        sensors,
        videos,
        args.output_csv,
        args.submission_id,
        batch_size=16,
        workers=4,
        device=torch.device("cuda"),
        work_dir=work_dir,
    )
    report.update(
        preprocessing_seconds=preprocessing_seconds,
        elapsed_seconds=time.monotonic() - started,
        model_asset_bytes=total_assets,
    )
    (work_dir / "inference_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
