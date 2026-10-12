#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import R2Plus1DClassifier, apply_low_rank_delta, dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/r2plus1d_yolo_16_128")
    parser.add_argument("--yolo", type=Path, default=ROOT / ".tools/checkpoints/yolo11n.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--adabn", choices=("none", "global"), default="none")
    parser.add_argument("--low-rank-adapter", type=Path)
    parser.add_argument("--low-rank-target-weight", type=float, default=0.125)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


@torch.no_grad()
def predict_checkpoint(path, indices, args, device, adapter=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"].copy()
    if adapter is not None:
        config.update(adapter.get("config_overrides", {}))
    mode = config["mode"]
    model = R2Plus1DClassifier(
        mode=mode,
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
        use_phase_temporal_embedding=bool(config.get("use_phase_temporal_embedding", False)),
        fusion_output=config.get("fusion_output", "residual"),
        visual_motion_channels=bool(config.get("visual_motion_channels", False)),
        visual_strategy=config.get("visual_strategy", "sequential"),
        per_visual_auxiliary=bool(config.get("per_visual_auxiliary", False)),
        environment_moe=float(config.get("environment_auxiliary_weight", 0.0)) > 0,
        visual_multiscale_tokens=bool(config.get("visual_multiscale_tokens", False)),
        normalize_modality_tokens=bool(config.get("normalize_modality_tokens", False)),
        skeleton_encoder=config.get("skeleton_encoder", "tcn"),
        radar_representation=config.get("radar_representation", "statistics"),
        imu_inputs={"world": 85, "relative": 95, "combined": 180, "world_magnetic": 100, "world_official": 135}[
            config.get("imu_representation", "world")
        ],
        imu_statistics_residual=bool(config.get("imu_statistics_residual", False)),
        imu_statistics_logit_residual=bool(config.get("imu_statistics_logit_residual", False)),
        imu_device_attention_residual=bool(config.get("imu_device_attention_residual", False)),
        imu_statistics_inputs=50 if config.get("imu_representation") == "world_official" else None,
    ).to(device)
    quantized = (
        checkpoint["model_int4"] if "model_int4" in checkpoint else
        checkpoint["model_int7"] if "model_int7" in checkpoint else
        checkpoint["model_int6"] if "model_int6" in checkpoint else checkpoint["model_int8"]
    )
    state = dequantize_state_dict(quantized)
    if adapter is not None:
        state = apply_low_rank_delta(state, adapter)
    model.load_state_dict(state, strict=True)
    model.eval()
    dataset = R2Plus1DDataset(
        indices,
        args.cache_dir,
        training=False,
        include_sensors=mode in ("multimodal", "shared_multimodal"),
        frames=args.frames or int(config.get("frames", 16)),
        image_size=getattr(args, "image_size", None) or int(config.get("image_size", 128)),
        include_radar=bool(config.get("use_radar", False)),
        radar_representation=config.get("radar_representation", "statistics"),
        imu_representation=config.get("imu_representation", "world"),
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
    if args.adabn == "global":
        batch_norms = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.reset_running_stats()
                layer.momentum = None
                layer.train()
                batch_norms.append(layer)
        if not batch_norms:
            raise RuntimeError("AdaBN requested for a model without batch normalization")
        for batch in tqdm(loader, desc=f"{path.parent.name} AdaBN"):
            video = batch["video"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4)
            keyword = {"visual_mask": batch["visual_mask"].to(device, non_blocking=True)}
            if mode in ("multimodal", "shared_multimodal"):
                keyword.update(
                    {
                        "skeleton": batch["skeleton"].to(device, non_blocking=True),
                        "imu": batch["imu"].to(device, non_blocking=True),
                        "sensor_mask": batch["sensor_mask"].to(device, non_blocking=True),
                    }
                )
                if "radar" in batch:
                    keyword["radar"] = batch["radar"].to(device, non_blocking=True)
                if config.get("ignore_radar", False) and keyword["sensor_mask"].shape[1] >= 3:
                    keyword["sensor_mask"] = keyword["sensor_mask"].clone()
                    keyword["sensor_mask"][:, 2] = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(video, **keyword)
        model.eval()
    probabilities = []
    for batch in tqdm(loader, desc=path.parent.name):
        video = batch["video"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4)
        keyword = {"visual_mask": batch["visual_mask"].to(device, non_blocking=True)}
        if mode in ("multimodal", "shared_multimodal"):
            keyword.update({
                "skeleton": batch["skeleton"].to(device, non_blocking=True),
                "imu": batch["imu"].to(device, non_blocking=True),
                "sensor_mask": batch["sensor_mask"].to(device, non_blocking=True),
            })
            if "radar" in batch:
                keyword["radar"] = batch["radar"].to(device, non_blocking=True)
            if config.get("ignore_radar", False) and keyword["sensor_mask"].shape[1] >= 3:
                keyword["sensor_mask"] = keyword["sensor_mask"].clone()
                keyword["sensor_mask"][:, 2] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(video, **keyword)
            views = [logits]
            if args.tta_hflip:
                flipped, _ = model(torch.flip(video, dims=(-1,)), **keyword)
                views.append(flipped)
            if args.tta_crop_scale < 1:
                side = round(video.shape[-1] * args.tta_crop_scale)
                top = (video.shape[-2] - side) // 2
                left = (video.shape[-1] - side) // 2
                cropped = functional.interpolate(
                    video[..., top:top + side, left:left + side],
                    size=video.shape[-3:], mode="trilinear", align_corners=False,
                )
                cropped_logits, _ = model(cropped, **keyword)
                views.append(cropped_logits)
                if args.tta_hflip:
                    cropped_flipped, _ = model(torch.flip(cropped, dims=(-1,)), **keyword)
                    views.append(cropped_flipped)
            logits = torch.stack(views).mean(0)
        probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(probabilities)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Prediction requires CUDA")
    if not 0.5 <= args.tta_crop_scale <= 1:
        raise ValueError("TTA crop scale must be in [0.5, 1]")
    if not 0 <= args.low_rank_target_weight <= 1:
        raise ValueError("Low-rank target weight must be in [0, 1]")
    if args.low_rank_adapter is not None and len(args.checkpoints) != 1:
        raise ValueError("Low-rank inference requires exactly one base checkpoint")
    if args.low_rank_adapter is not None and args.weights is not None:
        raise ValueError("Use --low-rank-target-weight instead of --weights with an adapter")
    if args.weights is None:
        args.weights = [1 / len(args.checkpoints)] * len(args.checkpoints)
    if len(args.weights) != len(args.checkpoints) or any(weight < 0 for weight in args.weights):
        raise ValueError("Checkpoint weights must be nonnegative and match checkpoints")
    total_weight = sum(args.weights)
    args.weights = [weight / total_weight for weight in args.weights]
    model_bytes = sum(path.stat().st_size for path in args.checkpoints) + args.yolo.stat().st_size
    if args.low_rank_adapter is not None:
        model_bytes += args.low_rank_adapter.stat().st_size
    if model_bytes >= 100_000_000:
        raise ValueError(f"Submitted model assets exceed 100 MB: {model_bytes}")
    records = get_records()
    indices = [record["index"] for record in records if record["split"] == "test"]
    device = torch.device("cuda")
    components = [predict_checkpoint(path, indices, args, device) for path in args.checkpoints]
    if args.low_rank_adapter is not None:
        adapter = torch.load(args.low_rank_adapter, map_location="cpu", weights_only=False)
        target = predict_checkpoint(args.checkpoints[0], indices, args, device, adapter=adapter)
        probabilities = (
            (1 - args.low_rank_target_weight) * components[0]
            + args.low_rank_target_weight * target
        )
    else:
        probabilities = sum(weight * values for weight, values in zip(args.weights, components))
    predictions = probabilities.argmax(axis=1)
    output_dir = ROOT / "outputs/predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"{args.name}.npz", probabilities=probabilities, indices=np.asarray(indices))
    submission = pd.DataFrame(
        {
            "path": [records[index]["id"] for index in indices],
            "prediction": predictions,
        }
    )
    test_csv = pd.read_csv(
        ROOT / "data/downloads/CUHK-X_Small_Model_Track/Small-Model-Track/Testing/test_file/test.csv"
    )
    if set(test_csv["path"]) != set(submission["path"]):
        raise ValueError("Prediction paths do not match test.csv")
    submission = test_csv[["path"]].merge(submission, on="path", how="left")
    csv_path = ROOT / "outputs/submissions" / f"{args.name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(csv_path, index=False)
    manifest = {
        "name": args.name,
        "checkpoints": [str(path) for path in args.checkpoints],
        "weights": args.weights,
        "low_rank_adapter": str(args.low_rank_adapter) if args.low_rank_adapter else None,
        "low_rank_target_weight": args.low_rank_target_weight if args.low_rank_adapter else None,
        "tta_hflip": args.tta_hflip,
        "tta_crop_scale": args.tta_crop_scale,
        "frames": args.frames,
        "adabn": args.adabn,
        "model_asset_bytes": model_bytes,
        "rows": len(submission),
        "classes_predicted": int(np.unique(predictions).size),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "submission": str(csv_path),
    }
    (output_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
