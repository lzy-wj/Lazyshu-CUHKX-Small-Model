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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ClipDataset, get_indices, get_records, prepare_batch
from src.models import GroupedX3DTokenModel, MultiModalModel, SharedX3DTokenModel
from src.r2plus1d_models import dequantize_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--adabn-scope", choices=("none", "environment", "user"), default="none")
    parser.add_argument("--adabn-passes", type=int, default=1)
    parser.add_argument("--adabn-reset", action="store_true")
    parser.add_argument("--save-auxiliary", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--indices-scope", choices=("train", "valid"), default="valid")
    parser.add_argument("--temporal-crop", type=float, default=1.0)
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.5 <= args.tta_crop_scale <= 1:
        raise ValueError("TTA crop scale must be in [0.5, 1]")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if int(config["fold"]) != args.fold:
        raise ValueError("Checkpoint fold mismatch")
    model = GroupedX3DTokenModel(
        config["modalities"], kinetics=False, kinetics_source=config.get("kinetics_source", "s")
    ) if config.get("backbone") == "x3d_grouped_tokens" else SharedX3DTokenModel(
        config["modalities"], kinetics=False, kinetics_source=config.get("kinetics_source", "s")
    ) if config.get("backbone") == "x3d_shared_tokens" else MultiModalModel(
        config["modalities"],
        width=config["width"],
        spatial_pool=config.get("spatial_pool", 1),
        backbone=config.get("backbone", "scratch"),
        skeleton_backbone=config.get("skeleton_backbone", "tcn"),
        temporal_backbone=config.get("temporal_backbone", "tcn"),
        temporal_shift=config.get("temporal_shift", False),
        kinetics_source=config.get("kinetics_source", "s"),
        fusion=config.get("fusion", "concat"),
    )
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
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    indices = train_indices if args.indices_scope == "train" else valid_indices
    dataset_options = {
        "frames": config["frames"],
        "canonical_pose": config.get("canonical_pose", False),
        "image_size": config.get("image_size", 96),
        "thermal_view": config.get("thermal_view", "full"),
        "imu_representation": config.get("imu_representation", "raw"),
        "temporal_crop": args.temporal_crop,
    }
    groups = {"all": indices}
    if args.adabn_scope == "environment":
        groups = {
            name: [index for index in indices if (records[index]["user"] > 15) == second]
            for name, second in (("a", False), ("b", True))
        }
    elif args.adabn_scope == "user":
        groups = {
            str(user): [index for index in indices if records[index]["user"] == user]
            for user in sorted({records[index]["user"] for index in indices})
        }
    source_state = copy.deepcopy(model.state_dict())
    probability_by_index = {}
    label_by_index = {}
    auxiliary_by_name = {name: {} for name in config["modalities"]}
    embedding_by_index = {}
    for group_indices in groups.values():
        model.load_state_dict(source_state, strict=True)
        dataset = ClipDataset(group_indices, config["modalities"], **dataset_options)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        if args.adabn_scope != "none":
            model.eval()
            batch_norms = [
                layer for name, layer in model.named_modules()
                if ".frames." in name and isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
            ]
            for layer in batch_norms:
                if args.adabn_reset:
                    layer.reset_running_stats()
                    layer.momentum = None
                layer.train()
            with torch.inference_mode():
                for _ in range(args.adabn_passes):
                    for raw_batch in loader:
                        batch = prepare_batch(raw_batch, torch.device("cuda"))
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            model(batch)
            model.eval()
        with torch.inference_mode():
            for raw_batch in loader:
                batch = prepare_batch(raw_batch, torch.device("cuda"))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, auxiliary, embeddings = model(batch, return_embedding=True)
                    logit_views = [logits]
                    embedding_views = [embeddings]
                    auxiliary_views = {name: [values] for name, values in auxiliary.items()}
                    if args.tta_hflip:
                        flipped = {
                            name: torch.flip(values, dims=(-1,))
                            if name in config["modalities"] and values.ndim == 5 else values
                            for name, values in batch.items()
                        }
                        flipped_logits, flipped_auxiliary, flipped_embeddings = model(flipped, return_embedding=True)
                        logit_views.append(flipped_logits)
                        embedding_views.append(flipped_embeddings)
                        for name, values in flipped_auxiliary.items():
                            auxiliary_views[name].append(values)
                    if args.tta_crop_scale < 1:
                        cropped = dict(batch)
                        for name in config["modalities"]:
                            values = batch[name]
                            if values.ndim != 5:
                                continue
                            side = round(min(values.shape[-2:]) * args.tta_crop_scale)
                            top = (values.shape[-2] - side) // 2
                            left = (values.shape[-1] - side) // 2
                            cropped[name] = functional.interpolate(
                                values[..., top:top + side, left:left + side].flatten(0, 1),
                                size=values.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            ).reshape_as(values)
                        cropped_logits, cropped_auxiliary, cropped_embeddings = model(cropped, return_embedding=True)
                        logit_views.append(cropped_logits)
                        embedding_views.append(cropped_embeddings)
                        for name, values in cropped_auxiliary.items():
                            auxiliary_views[name].append(values)
                        if args.tta_hflip:
                            cropped_flipped = {
                                name: torch.flip(values, dims=(-1,))
                                if name in config["modalities"] and values.ndim == 5 else values
                                for name, values in cropped.items()
                            }
                            values = model(cropped_flipped, return_embedding=True)
                            logit_views.append(values[0])
                            embedding_views.append(values[2])
                            for name, branch in values[1].items():
                                auxiliary_views[name].append(branch)
                    logits = torch.stack(logit_views).mean(0)
                    embeddings = torch.stack(embedding_views).mean(0)
                    auxiliary = {name: torch.stack(values).mean(0) for name, values in auxiliary_views.items()}
                auxiliary_probabilities = {
                    name: values.float().softmax(1).cpu().numpy()
                    for name, values in auxiliary.items()
                } if args.save_auxiliary else {}
                for index, label, probability in zip(
                    batch["index"].cpu().numpy(),
                    batch["label"].cpu().numpy(),
                    logits.float().softmax(1).cpu().numpy(),
                ):
                    probability_by_index[int(index)] = probability
                    label_by_index[int(index)] = label
                if args.save_auxiliary:
                    for name, values in auxiliary_probabilities.items():
                        for index, probability in zip(batch["index"].cpu().numpy(), values):
                            auxiliary_by_name[name][int(index)] = probability
                if args.save_embeddings:
                    for index, embedding in zip(batch["index"].cpu().numpy(), embeddings.float().cpu().numpy()):
                        embedding_by_index[int(index)] = embedding
    ordered_indices = np.asarray(indices)
    probabilities = np.stack([probability_by_index[int(index)] for index in ordered_indices])
    labels = np.asarray([label_by_index[int(index)] for index in ordered_indices])
    metrics = {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "tta_hflip": args.tta_hflip,
        "adabn_scope": args.adabn_scope,
        "adabn_passes": args.adabn_passes,
        "adabn_reset": args.adabn_reset,
        "save_auxiliary": args.save_auxiliary,
        "save_embeddings": args.save_embeddings,
        "indices_scope": args.indices_scope,
        "temporal_crop": args.temporal_crop,
        "tta_crop_scale": args.tta_crop_scale,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    auxiliary_arrays = {
        f"probabilities_{name}": np.stack([values[int(index)] for index in ordered_indices])
        for name, values in auxiliary_by_name.items()
    } if args.save_auxiliary else {}
    embedding_arrays = {
        "embeddings": np.stack([embedding_by_index[int(index)] for index in ordered_indices])
    } if args.save_embeddings else {}
    np.savez(
        args.output,
        probabilities=probabilities,
        labels=labels,
        indices=ordered_indices,
        **auxiliary_arrays,
        **embedding_arrays,
    )
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
