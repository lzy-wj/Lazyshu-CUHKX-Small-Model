#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--points", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/radar_points.npy")
    return parser.parse_args()


def point_features(frame, points):
    values = frame[["x", "y", "z", "v", "snr", "noise"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    values = values[np.isfinite(values).all(axis=1)]
    output = np.zeros((points, 10), dtype=np.float32)
    if not len(values):
        return output
    score = np.abs(values[:, 3]) * 5 + np.clip((values[:, 4] - values[:, 5] + 500) / 500, 0, 2)
    if len(values) > points:
        values = values[np.argsort(score)[-points:]]
    center = np.median(values[:, :3], axis=0)
    count = len(values)
    output[:count, :6] = values / np.asarray([4, 6, 4, 1, 400, 700], dtype=np.float32)
    output[:count, 6:9] = (values[:, :3] - center) / np.asarray([2, 3, 2], dtype=np.float32)
    output[:count, 9] = 1
    return np.clip(output, -2, 2)


def process(task):
    index, folder, steps, points = task
    output = np.zeros((steps, points, 10), dtype=np.float16)
    folder = Path(folder)
    tables = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.csv")):
            if path.stat().st_size <= 50:
                continue
            try:
                table = pd.read_csv(path)
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            if not table.empty and {"frame", "x", "y", "z", "v", "snr", "noise"} <= set(table):
                tables.append(table)
    if not tables:
        return index, output, 0
    table = pd.concat(tables, ignore_index=True)
    grouped = [frame for _, frame in table.groupby(["timestamp", "frame"], sort=True)]
    if not grouped:
        return index, output, 0
    selected = np.linspace(0, len(grouped) - 1, steps).round().astype(int)
    cache = {}
    for position, frame_index in enumerate(selected):
        frame_index = int(frame_index)
        if frame_index not in cache:
            cache[frame_index] = point_features(grouped[frame_index], points)
        output[position] = cache[frame_index].astype(np.float16)
    return index, output, len(grouped)


def main():
    args = parse_args()
    if args.steps < 8 or args.points < 4:
        raise ValueError("Radar cache requires at least 8 steps and 4 points")
    records = get_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(records), args.steps, args.points, 10)
    values = open_memmap(args.output, mode="w+", dtype=np.float16, shape=shape)
    frame_counts = np.zeros(len(records), dtype=np.int32)
    tasks = [
        (record["index"], record["modalities"].get("Radar", ""), args.steps, args.points)
        for record in records
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, array, frame_count in tqdm(
            pool.map(process, tasks, chunksize=4), total=len(tasks), desc="Radar points"
        ):
            values[index] = array
            frame_counts[index] = frame_count
    values.flush()
    mask_path = args.output.with_name("radar_points_mask.npy")
    np.save(mask_path, (frame_counts > 0).astype(np.float32))
    frame_count_path = args.output.with_name("radar_points_frame_count.npy")
    np.save(frame_count_path, frame_counts)
    metadata_path = ROOT / "data/metadata/samples.json"
    report = {
        "complete": True,
        "shape": list(shape),
        "dtype": "float16",
        "available": int((frame_counts > 0).sum()),
        "steps": args.steps,
        "points": args.points,
        "frame_count_quantiles": np.quantile(
            frame_counts[frame_counts > 0], [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
        "features": ["x", "y", "z", "velocity", "snr", "noise", "centered_x", "centered_y", "centered_z", "valid"],
        "selection": "Top points by Doppler magnitude and signal quality when a frame exceeds the point budget.",
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest = args.output.with_name("radar_points_manifest.json")
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
