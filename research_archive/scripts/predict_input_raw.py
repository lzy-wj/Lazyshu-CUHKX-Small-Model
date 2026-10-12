import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_data import sensor_clip
from scripts.prepare_imu_world import features as world_features
from scripts.prepare_thermal_motion import motion_box
from src.data import prepare_batch
from src.scratch_models import ScratchMultiModalModel, validate_scratch_config


def canonical_pose(features):
    pose = features[:, :51].reshape(64, 17, 3).copy()
    pose[..., :2] -= pose[:, :1, :2]
    lateral = (pose[:, 1, :2] - pose[:, 4, :2] + pose[:, 14, :2] - pose[:, 11, :2]).mean(0)
    angle = np.arctan2(lateral[1], lateral[0])
    horizontal = pose[..., 0] * np.cos(angle) + pose[..., 1] * np.sin(angle)
    vertical = -pose[..., 0] * np.sin(angle) + pose[..., 1] * np.cos(angle)
    pose[..., 0], pose[..., 1] = horizontal, vertical
    scale = np.clip(np.median(np.linalg.norm(pose[:, 8] - pose[:, 0], axis=-1)), 0.15, 1.2)
    pose = pose / scale * 0.5
    bones = pose - pose[:, [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]]
    velocity = np.diff(pose, axis=0, prepend=pose[:1]) * 10
    return np.concatenate([pose.reshape(64, -1), bones.reshape(64, -1), velocity.reshape(64, -1)], axis=1)


def read_clip(arguments):
    index, folder, frames = arguments
    record = {"index": index, "modalities": {name: str((folder / name).resolve())
                                              for name in ["Skeleton", "IMU", "Thermal"] if (folder / name).is_dir()}}
    skeleton_record = dict(record, modalities={name: path for name, path in record["modalities"].items() if name == "Skeleton"})
    skeleton, _, _, presence = sensor_clip(skeleton_record)
    imu = world_features(record)
    thermal = np.zeros((24, 3, 160, 160), dtype=np.uint8)
    mask = 0.0
    if "Thermal" in record["modalities"]:
        files = sorted(path for path in Path(record["modalities"]["Thermal"]).iterdir()
                       if path.suffix.lower() in [".png", ".jpg", ".jpeg"] and not path.name.startswith("."))
        if files:
            selected = np.linspace(0, len(files) - 1, 24).round().astype(int)
            decoded = {}
            for position in np.unique(selected):
                try:
                    with Image.open(files[position]) as image:
                        decoded[position] = image.convert("RGB")
                except (OSError, ValueError):
                    continue
            if decoded:
                nearest = [min(decoded, key=lambda valid: abs(valid - position)) for position in selected]
                thermal[:] = np.stack([np.asarray(decoded[position].resize((160, 160), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
                                       for position in nearest])
                box = motion_box(thermal)
                for step, position in enumerate(nearest):
                    image = decoded[position]
                    width, height = image.size
                    pixels = (int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height))
                    thermal[step] = np.asarray(image.crop(pixels).resize((160, 160), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
                mask = 1.0
    positions = np.linspace(0, 23, frames).round().astype(int)
    return {"skeleton": canonical_pose(skeleton), "skeleton_mask": np.float32(presence[0]),
            "imu": imu, "imu_mask": np.float32(imu[:, 80:].any()),
            "thermal": thermal[positions], "thermal_mask": np.float32(mask), "index": index, "label": -1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(6)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    states = []
    for model in manifest["models"]:
        path = args.bundle / model["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError("Bundled model hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = state["config"]
        validate_scratch_config(config)
        if set(config["modalities"]) != {"skeleton", "imu", "thermal"} or config["thermal_view"] != "motion_crop" or config["imu_representation"] != "world":
            raise ValueError("Raw entry point supports the selected three-modality input recipe only")
        if not config["canonical_pose"] or config["image_size"] != 160 or config["fold"] != -1:
            raise ValueError("Unexpected full-data preprocessing configuration")
        states.append(state)
    if len({state["config"]["frames"] for state in states}) != 1:
        raise ValueError("Bundled models must use matching frame sampling")
    with args.test_csv.open(encoding="utf-8-sig") as handle:
        paths = [row["path"] for row in csv.DictReader(handle)]
    if len(paths) != 405 or len(set(paths)) != 405:
        raise ValueError("Expected 405 unique competition test paths")
    root = args.raw_root.resolve()
    folders = [(root / path).resolve() for path in paths]
    if any(not folder.is_relative_to(root) or not folder.is_dir() for folder in folders):
        raise ValueError("Test directory is missing or outside raw-root")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        clips = list(pool.map(read_clip, [(index, folder, states[0]["config"]["frames"]) for index, folder in enumerate(folders)]))
    print("Prepared", len(clips), "clips directly from raw files", flush=True)
    predictions = []
    for state, entry in zip(states, manifest["models"]):
        config = state["config"]
        model = ScratchMultiModalModel(config["modalities"], fusion=config["fusion"], visual_multiplier=config.get("visual_multiplier", 1),
                                       visual_encoder=config.get("visual_encoder", "legacy"),
                                       visual_feedforward=config.get("visual_feedforward", 384),
                                       skeleton_representation=config.get("skeleton_representation", "coordinates"),
                                       scratch_skeleton_encoder=config.get("scratch_skeleton_encoder", "tcn"),
                                       skeleton_dimension=config.get("skeleton_dimension", 192),
                                       skeleton_depth=config.get("skeleton_depth", 3),
                                       skeleton_heads=config.get("skeleton_heads", 6),
                                       thermal_view=config["thermal_view"], imu_representation=config["imu_representation"])
        model.load_state_dict(state["model"], strict=True)
        model.cuda().eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(clips), args.batch_size):
                selected = clips[start:start + args.batch_size]
                batch = {name: torch.as_tensor(np.stack([clip[name] for clip in selected])) for name in selected[0]}
                batch = prepare_batch(batch, torch.device("cuda"))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(batch)
                batches.append((logits.float() / entry["temperature"]).softmax(1).cpu().numpy())
        predictions.append(np.concatenate(batches))
        del model
        torch.cuda.empty_cache()
    probability = np.average(np.stack(predictions), axis=0, weights=[model["weight"] for model in manifest["models"]])
    if probability.shape != (405, 40) or not np.isfinite(probability).all():
        raise ValueError("Invalid raw inference output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "prediction"])
        writer.writerows(zip(paths, probability.argmax(1).tolist()))
    np.savez(args.output.with_suffix(".npz"), probabilities=probability, paths=np.asarray(paths))
    print("Wrote", args.output, "405 predictions; no labels or prepared caches were read", flush=True)


if __name__ == "__main__":
    main()
