#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from src.data import get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import R2Plus1DClassifier, dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checkpoint_state(checkpoint):
    for key in ("model_int4", "model_int7", "model_int6", "model_int8"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = R2Plus1DClassifier(
        mode=config["mode"],
        weights=None,
        dropout=float(config["dropout"]),
        input_channels=int(config.get("input_channels", 4)),
        mask_joint_visual=bool(config.get("mask_joint_visual", False)),
        use_radar=bool(config.get("use_radar", False)),
        skeleton_representation=config.get("skeleton_representation", "coordinates"),
        mixstyle_probability=float(config.get("mixstyle_probability", 0.0)),
        mixstyle_alpha=float(config.get("mixstyle_alpha", 0.1)),
        fusion_depth=int(config.get("fusion_depth", 2)),
        use_temporal_embedding=bool(config.get("use_temporal_embedding", False)),
        fusion_output=config.get("fusion_output", "residual"),
    ).cuda()
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model.eval()

    records = get_records()
    _, validation_indices = get_indices(records, args.fold)
    source_users = {user for group in PAIR_SOURCES[args.fold] for user in group}
    source_indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] in source_users
    ]
    source_index_set = set(source_indices)
    indices = source_indices + validation_indices
    dataset = R2Plus1DDataset(
        indices,
        args.cache_dir,
        training=False,
        include_sensors=True,
        include_radar=bool(config.get("use_radar", False)),
        frames=int(config.get("frames", 32)),
        image_size=int(config.get("image_size", 128)),
        canonical_pose=bool(config.get("canonical_pose", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    captured = {}
    hook = model.normalization.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("features", output.detach())
    )
    feature_rows = []
    probability_rows = []
    with torch.no_grad():
        for batch in tqdm(loader):
            video = batch["video"].cuda(non_blocking=True).permute(0, 2, 1, 3, 4)
            keyword = {
                "skeleton": batch["skeleton"].cuda(non_blocking=True),
                "imu": batch["imu"].cuda(non_blocking=True),
                "sensor_mask": batch["sensor_mask"].cuda(non_blocking=True),
                "visual_mask": batch["visual_mask"].cuda(non_blocking=True),
            }
            if "radar" in batch:
                keyword["radar"] = batch["radar"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(video, **keyword)
            feature_rows.append(captured["features"].float().cpu().numpy())
            probability_rows.append(logits.float().softmax(1).cpu().numpy())
    hook.remove()
    ordered_indices = np.asarray(indices, dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        features=np.concatenate(feature_rows),
        probabilities=np.concatenate(probability_rows),
        indices=ordered_indices,
        labels=np.asarray([records[index]["label"] for index in indices]),
        users=np.asarray([records[index]["user"] for index in indices]),
        source=np.asarray([index in source_index_set for index in indices]),
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "fold": args.fold,
        "source_users": sorted(source_users),
        "source_rows": len(source_indices),
        "validation_rows": len(validation_indices),
        "feature_dimension": int(feature_rows[0].shape[1]),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
