#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import R2Plus1DClassifier, dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    parser.add_argument("--tta-upper-body-fraction", type=float, default=1.0)
    parser.add_argument(
        "--adabn",
        choices=("none", "global", "user", "global_keep", "user_keep"),
        default="none",
    )
    parser.add_argument("--adabn-passes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/r2plus1d_yolo_16_128")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--save-auxiliary", action="store_true")
    return parser.parse_args()


def build_model(config, state):
    mode = config["mode"]
    model = R2Plus1DClassifier(
        mode=mode,
        weights=None,
        dropout=float(config["dropout"]),
        input_channels=int(config.get("input_channels", 4)),
        mask_joint_visual=bool(config.get("mask_joint_visual", False)),
        use_radar=bool(config.get("use_radar", False)),
        skeleton_representation=config.get("skeleton_representation", "coordinates"),
        skeleton_encoder=config.get("skeleton_encoder", "tcn"),
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
        radar_representation=config.get("radar_representation", "statistics"),
        imu_inputs={"world": 85, "relative": 95, "combined": 180, "world_magnetic": 100, "world_official": 135}[
            config.get("imu_representation", "world")
        ],
        imu_statistics_residual=bool(config.get("imu_statistics_residual", False)),
        imu_statistics_logit_residual=bool(config.get("imu_statistics_logit_residual", False)),
        imu_device_attention_residual=bool(config.get("imu_device_attention_residual", False)),
        imu_statistics_inputs=50 if config.get("imu_representation") == "world_official" else None,
    ).cuda()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_loader(indices, args, config, mode):
    dataset = R2Plus1DDataset(
        indices, args.cache_dir, training=False, include_sensors=mode in ("multimodal", "shared_multimodal"),
        include_radar=bool(config.get("use_radar", False)),
        radar_representation=config.get("radar_representation", "statistics"),
        imu_representation=config.get("imu_representation", "world"),
        canonical_pose=bool(config.get("canonical_pose", False)),
        frames=args.frames or int(config.get("frames", 16)),
        image_size=getattr(args, "image_size", None) or int(config.get("image_size", 128)),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )


def model_inputs(batch, mode, ignore_radar=False):
    video = batch["video"].cuda(non_blocking=True).permute(0, 2, 1, 3, 4)
    keyword = {}
    keyword = {"visual_mask": batch["visual_mask"].cuda(non_blocking=True)}
    if mode in ("multimodal", "shared_multimodal"):
        sensor_mask = batch["sensor_mask"].cuda(non_blocking=True)
        if ignore_radar and sensor_mask.shape[1] >= 3:
            sensor_mask = sensor_mask.clone()
            sensor_mask[:, 2] = 0
        keyword.update({
            "skeleton": batch["skeleton"].cuda(non_blocking=True),
            "imu": batch["imu"].cuda(non_blocking=True),
            "sensor_mask": sensor_mask,
        })
        if "radar" in batch:
            keyword["radar"] = batch["radar"].cuda(non_blocking=True)
    return video, keyword


@torch.no_grad()
def adapt_batch_norm(model, loader, mode, passes, reset, ignore_radar=False):
    model.eval()
    batch_norms = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            if reset:
                layer.reset_running_stats()
                layer.momentum = None
            layer.train()
            batch_norms.append(layer)
    if not batch_norms:
        raise RuntimeError("AdaBN requested for a model without batch normalization")
    for _ in range(passes):
        for batch in loader:
            video, keyword = model_inputs(batch, mode, ignore_radar)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(video, **keyword)
    model.eval()


@torch.no_grad()
def predict(
    model,
    loader,
    mode,
    tta_hflip,
    crop_scale,
    ignore_radar=False,
    upper_body_fraction=1.0,
):
    rows = []
    for batch in tqdm(loader):
        video, keyword = model_inputs(batch, mode, ignore_radar)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, auxiliary = model(video, **keyword)
            views = [logits]
            auxiliary_views = {name: [values] for name, values in auxiliary.items()}
            upper_views = []
            upper_auxiliary_views = {name: [] for name in auxiliary}
            if tta_hflip:
                flipped, flipped_auxiliary = model(torch.flip(video, dims=(-1,)), **keyword)
                views.append(flipped)
                for name, values in flipped_auxiliary.items():
                    auxiliary_views[name].append(values)
            if crop_scale < 1:
                side = round(video.shape[-1] * crop_scale)
                top = (video.shape[-2] - side) // 2
                left = (video.shape[-1] - side) // 2
                cropped = functional.interpolate(
                    video[..., top:top + side, left:left + side],
                    size=video.shape[-3:], mode="trilinear", align_corners=False,
                )
                cropped_logits, cropped_auxiliary = model(cropped, **keyword)
                views.append(cropped_logits)
                for name, values in cropped_auxiliary.items():
                    auxiliary_views[name].append(values)
                if tta_hflip:
                    cropped_flipped, cropped_flipped_auxiliary = model(torch.flip(cropped, dims=(-1,)), **keyword)
                    views.append(cropped_flipped)
                    for name, values in cropped_flipped_auxiliary.items():
                        auxiliary_views[name].append(values)
            if upper_body_fraction < 1:
                upper_height = round(video.shape[-2] * upper_body_fraction)
                upper = functional.interpolate(
                    video[..., :upper_height, :],
                    size=video.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                upper_logits, upper_auxiliary = model(upper, **keyword)
                upper_views.append(upper_logits)
                for name, values in upper_auxiliary.items():
                    upper_auxiliary_views[name].append(values)
                if tta_hflip:
                    upper_flipped, upper_flipped_auxiliary = model(
                        torch.flip(upper, dims=(-1,)), **keyword
                    )
                    upper_views.append(upper_flipped)
                    for name, values in upper_flipped_auxiliary.items():
                        upper_auxiliary_views[name].append(values)
            logits = torch.stack(views).mean(0)
            auxiliary = {
                name: torch.stack(values).mean(0) for name, values in auxiliary_views.items()
            }
            if upper_views:
                logits = 0.5 * (logits + torch.stack(upper_views).mean(0))
                auxiliary = {
                    name: 0.5 * (values + torch.stack(upper_auxiliary_views[name]).mean(0))
                    for name, values in auxiliary.items()
                }
        probability = logits.float().softmax(1).cpu().numpy()
        auxiliary_probability = {
            name: values.float().softmax(1).cpu().numpy() for name, values in auxiliary.items()
        }
        for position, (index, label, values) in enumerate(
            zip(batch["index"].numpy(), batch["label"].numpy(), probability)
        ):
            branches = {name: branch[position] for name, branch in auxiliary_probability.items()}
            rows.append((int(index), int(label), values, branches))
    return rows


@torch.no_grad()
def main():
    args = parse_args()
    if args.adabn_passes < 1:
        raise ValueError("AdaBN passes must be positive")
    if not 0.5 <= args.tta_crop_scale <= 1:
        raise ValueError("TTA crop scale must be in [0.5, 1]")
    if not 0.5 <= args.tta_upper_body_fraction <= 1:
        raise ValueError("Upper-body fraction must be in [0.5, 1]")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mode = config["mode"]
    state = (
        dequantize_state_dict(checkpoint["model_int4"])
        if "model_int4" in checkpoint else
        dequantize_state_dict(checkpoint["model_int7"])
        if "model_int7" in checkpoint else
        dequantize_state_dict(checkpoint["model_int6"])
        if "model_int6" in checkpoint else
        dequantize_state_dict(checkpoint["model_int8"])
        if "model_int8" in checkpoint else checkpoint["model"]
    )
    records = get_records()
    _, indices = get_indices(records, args.fold)
    groups = [indices]
    if args.adabn.startswith("user"):
        groups = [[index for index in indices if records[index]["user"] == user]
                  for user in sorted({records[index]["user"] for index in indices})]
    rows = []
    for group in groups:
        model = build_model(config, copy.deepcopy(state))
        loader = make_loader(group, args, config, mode)
        if args.adabn != "none":
            adapt_batch_norm(
                model,
                loader,
                mode,
                args.adabn_passes,
                reset=not args.adabn.endswith("_keep"),
                ignore_radar=bool(config.get("ignore_radar", False)),
            )
        rows.extend(
            predict(
                model,
                loader,
                mode,
                args.tta_hflip,
                args.tta_crop_scale,
                ignore_radar=bool(config.get("ignore_radar", False)),
                upper_body_fraction=args.tta_upper_body_fraction,
            )
        )
        del model
        torch.cuda.empty_cache()
    ordered = {index: (label, values, branches) for index, label, values, branches in rows}
    ordered_indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray([ordered[int(index)][0] for index in ordered_indices])
    probabilities = np.stack([ordered[int(index)][1] for index in ordered_indices])
    auxiliary = {
        f"probabilities_{name}": np.stack([ordered[int(index)][2][name] for index in ordered_indices])
        for name in next(iter(ordered.values()))[2]
    } if args.save_auxiliary else {}
    available = np.load(args.cache_dir / "available.npy").astype(bool)[ordered_indices]
    metrics = {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "available_accuracy": float((probabilities[available].argmax(1) == labels[available]).mean()),
        "coverage": float(available.mean()),
        "tta_hflip": args.tta_hflip,
        "tta_crop_scale": args.tta_crop_scale,
        "adabn": args.adabn,
        "adabn_passes": args.adabn_passes,
        "frames": args.frames or int(config.get("frames", 16)),
        "image_size": getattr(args, "image_size", None) or int(config.get("image_size", 128)),
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "save_auxiliary": args.save_auxiliary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=ordered_indices, **auxiliary)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
