#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.predict_r2plus1d import predict_checkpoint
from src.data import get_records


TARGET_LIMIT = 95_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    parser.add_argument("--adabn", choices=("none", "global"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Prediction requires CUDA")
    if not 0.5 <= args.tta_crop_scale <= 1:
        raise ValueError("TTA crop scale must be in [0.5, 1]")
    model_asset_bytes = args.checkpoint.stat().st_size + args.yolo.stat().st_size
    if model_asset_bytes > TARGET_LIMIT:
        raise ValueError(f"Model assets exceed the V24 target: {model_asset_bytes} > {TARGET_LIMIT}")

    records = get_records()
    indices = [record["index"] for record in records if record["split"] == "test"]
    prediction_args = SimpleNamespace(
        cache_dir=args.cache_dir,
        frames=args.frames,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        tta_hflip=args.tta_hflip,
        tta_crop_scale=args.tta_crop_scale,
        adabn=args.adabn,
    )
    probabilities = predict_checkpoint(
        args.checkpoint,
        indices,
        prediction_args,
        torch.device("cuda"),
    )
    predictions = probabilities.argmax(axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, indices=np.asarray(indices))
    test_csv = pd.read_csv(
        ROOT / "data/downloads/CUHK-X_Small_Model_Track/Small-Model-Track/Testing/test_file/test.csv"
    )
    predicted = pd.DataFrame({
        "path": [records[index]["id"] for index in indices],
        "prediction": predictions,
    })
    if set(test_csv["path"]) != set(predicted["path"]):
        raise ValueError("Prediction paths do not match test.csv")
    submission = test_csv[["path"]].merge(predicted, on="path", how="left")
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission, index=False)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "yolo": str(args.yolo),
        "yolo_sha256": sha256(args.yolo),
        "model_asset_bytes": model_asset_bytes,
        "target_limit_bytes": TARGET_LIMIT,
        "rows": len(submission),
        "classes_predicted": int(np.unique(predictions).size),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "tta_hflip": args.tta_hflip,
        "tta_crop_scale": args.tta_crop_scale,
        "adabn": args.adabn,
        "output": str(args.output),
        "submission": str(args.submission),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
