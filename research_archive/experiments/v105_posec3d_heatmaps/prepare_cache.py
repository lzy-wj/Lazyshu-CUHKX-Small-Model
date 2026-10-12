#!/usr/bin/env python3
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ((0, 7, 8, 9, 10), (1, 2, 3), (4, 5, 6), (11, 12, 13), (14, 15, 16))
PLANES = ((0, 2), (1, 2), (0, 1))
GAUSSIAN = None
SKELETON = None
PRESENT = None
CONFIG = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/posec3d_32_48")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--coordinate-limit", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=1.25)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def initialize_worker(skeleton_path, mask_path, config):
    global GAUSSIAN, SKELETON, PRESENT, CONFIG
    SKELETON = np.load(skeleton_path, mmap_mode="r")
    PRESENT = np.load(mask_path, mmap_mode="r")[:, 0].astype(bool)
    CONFIG = config
    radius = int(np.ceil(config["sigma"] * 3))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    GAUSSIAN = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * config["sigma"] ** 2))


def canonical_pose(index):
    pose = np.asarray(SKELETON[index, :, :51], dtype=np.float32).reshape(64, 17, 3)
    positions = np.linspace(0, len(pose) - 1, CONFIG["frames"]).round().astype(np.int64)
    pose = pose[positions].copy()
    pose -= pose[:, :1]
    lateral = (
        pose[:, 1, :2]
        - pose[:, 4, :2]
        + pose[:, 14, :2]
        - pose[:, 11, :2]
    ).mean(axis=0)
    angle = np.arctan2(lateral[1], lateral[0])
    horizontal = pose[..., 0] * np.cos(angle) + pose[..., 1] * np.sin(angle)
    depth = -pose[..., 0] * np.sin(angle) + pose[..., 1] * np.cos(angle)
    pose[..., 0] = horizontal
    pose[..., 1] = depth
    scale = np.median(np.linalg.norm(pose[:, 8] - pose[:, 0], axis=-1))
    pose /= np.clip(scale, 0.15, 1.2)
    return np.clip(pose, -CONFIG["coordinate_limit"], CONFIG["coordinate_limit"])


def render(index):
    shape = (CONFIG["frames"], len(GROUPS) * len(PLANES), CONFIG["size"], CONFIG["size"])
    output = np.zeros(shape, dtype=np.uint8)
    if not PRESENT[index]:
        return index, output
    pose = canonical_pose(index)
    radius = GAUSSIAN.shape[0] // 2
    limit = CONFIG["coordinate_limit"]
    size = CONFIG["size"]
    for frame in range(CONFIG["frames"]):
        for plane_index, (horizontal_axis, vertical_axis) in enumerate(PLANES):
            for group_index, joints in enumerate(GROUPS):
                channel = plane_index * len(GROUPS) + group_index
                for joint in joints:
                    horizontal = int(round((pose[frame, joint, horizontal_axis] / limit + 1) * 0.5 * (size - 1)))
                    vertical = int(round((1 - (pose[frame, joint, vertical_axis] / limit + 1) * 0.5) * (size - 1)))
                    left = max(horizontal - radius, 0)
                    right = min(horizontal + radius + 1, size)
                    top = max(vertical - radius, 0)
                    bottom = min(vertical + radius + 1, size)
                    patch = GAUSSIAN[
                        top - (vertical - radius):GAUSSIAN.shape[0] - ((vertical + radius + 1) - bottom),
                        left - (horizontal - radius):GAUSSIAN.shape[1] - ((horizontal + radius + 1) - right),
                    ]
                    values = np.rint(patch * 255).astype(np.uint8)
                    np.maximum(output[frame, channel, top:bottom, left:right], values, out=output[frame, channel, top:bottom, left:right])
    return index, output


def main():
    args = parse_args()
    if args.frames < 16 or args.size < 32 or args.coordinate_limit <= 0 or args.sigma <= 0:
        raise ValueError("Invalid PoseC3D cache geometry")
    skeleton_path = ROOT / "data/processed/skeleton.npy"
    mask_path = ROOT / "data/processed/sensor_mask.npy"
    skeleton = np.load(skeleton_path, mmap_mode="r")
    shape = (len(skeleton), args.frames, len(GROUPS) * len(PLANES), args.size, args.size)
    args.output.mkdir(parents=True, exist_ok=True)
    cache_path = args.output / "heatmaps.uint8.memmap"
    manifest_path = args.output / "manifest.json"
    expected_bytes = int(np.prod(shape))
    if manifest_path.is_file() and cache_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") and manifest.get("shape") == list(shape) and cache_path.stat().st_size == expected_bytes:
            print(json.dumps(manifest, indent=2))
            return
    cache = np.memmap(cache_path, dtype=np.uint8, mode="w+", shape=shape)
    config = {
        "frames": args.frames,
        "size": args.size,
        "coordinate_limit": args.coordinate_limit,
        "sigma": args.sigma,
    }
    with Pool(
        args.workers,
        initializer=initialize_worker,
        initargs=(str(skeleton_path), str(mask_path), config),
    ) as pool:
        for index, values in tqdm(pool.imap_unordered(render, range(len(skeleton)), chunksize=2), total=len(skeleton)):
            cache[index] = values
    cache.flush()
    manifest = {
        "complete": True,
        "shape": list(shape),
        "bytes": cache_path.stat().st_size,
        "dtype": "uint8",
        "frames": args.frames,
        "size": args.size,
        "coordinate_limit": args.coordinate_limit,
        "sigma": args.sigma,
        "groups": [list(group) for group in GROUPS],
        "planes": [list(plane) for plane in PLANES],
        "normalization": "root centered, median torso scale, lateral-axis yaw canonicalization",
        "skeleton_sha256": hashlib.sha256(skeleton_path.read_bytes()).hexdigest(),
        "sensor_mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
