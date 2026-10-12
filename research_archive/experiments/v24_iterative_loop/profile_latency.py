#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from iterative_model import build_from_source
from scripts.train_r2plus1d import prepare_batch
from src.data import get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=HERE / "latency_profile.json")
    return parser.parse_args()


def inputs(batch):
    return {
        "video": batch["video"],
        "skeleton": batch["skeleton"],
        "imu": batch["imu"],
        "radar": batch.get("radar"),
        "sensor_mask": batch["sensor_mask"],
        "visual_mask": batch["visual_mask"],
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Latency profiling requires CUDA")
    model, config, _ = build_from_source(args.checkpoint, iterations=1)
    model.cuda().eval()
    _, valid_indices = get_indices(get_records(), args.fold)
    dataset = R2Plus1DDataset(
        valid_indices[:args.batch_size],
        Path(config["cache_dir"]),
        training=False,
        include_sensors=True,
        include_radar=bool(config.get("use_radar", False)),
        frames=int(config.get("frames", 32)),
        image_size=int(config.get("image_size", 128)),
        canonical_pose=bool(config.get("canonical_pose", False)),
    )
    raw_batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0)))
    batch = prepare_batch(raw_batch, torch.device("cuda"), multimodal=True)
    results = {}
    for iterations in (1, 2, 3):
        model.iterations = iterations
        elapsed = []
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for repeat in range(args.warmup + args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                model(**inputs(batch))
                torch.cuda.synchronize()
                if repeat >= args.warmup:
                    elapsed.append((time.perf_counter() - started) * 1000)
        results[str(iterations)] = {
            "mean_batch_ms": statistics.mean(elapsed),
            "median_batch_ms": statistics.median(elapsed),
            "mean_sample_ms": statistics.mean(elapsed) / args.batch_size,
            "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
    report = {
        "checkpoint": str(args.checkpoint),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
