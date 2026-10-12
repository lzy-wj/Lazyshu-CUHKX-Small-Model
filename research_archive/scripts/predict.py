import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset, get_records, prepare_batch
from src.models import (
    GroupedX3DTokenModel,
    MultiModalModel,
    SharedMViTModel,
    SharedSwin3DModel,
    SharedX3DTokenModel,
)
from src.r2plus1d_models import dequantize_state_dict
from src.scratch_models import ScratchMultiModalModel, validate_scratch_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--temperatures", nargs="+", type=float)
    parser.add_argument("--name", default="multimodal_final")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.5 <= args.tta_crop_scale <= 1:
        raise ValueError("TTA crop scale must be in [0.5, 1]")
    torch.set_num_threads(6)
    records = get_records()
    indices = [record["index"] for record in records if record["split"] == "test"]
    weights = np.asarray(args.weights if args.weights else [1] * len(args.checkpoints), dtype=np.float64)
    if len(weights) != len(args.checkpoints) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Supply one nonnegative weight per checkpoint, with positive total")
    weights /= weights.sum()
    temperatures = args.temperatures or [1.0] * len(args.checkpoints)
    if len(temperatures) != len(args.checkpoints) or any(value <= 0 for value in temperatures):
        raise ValueError("Supply one positive temperature per checkpoint")
    total_size = sum(path.stat().st_size for path in args.checkpoints)
    preprocessing = {}
    for path in args.checkpoints:
        config = torch.load(path, map_location="cpu", weights_only=False)["config"]
        if any(name.startswith("person_") for name in config["modalities"]):
            preprocessing["person_detector"] = ROOT / ".tools/torch/hub/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
        if any(name.endswith("_features") for name in config["modalities"]):
            preprocessing["feature_encoder"] = ROOT / ".tools/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth"
    preprocessing_size = sum(path.stat().st_size for path in preprocessing.values())
    total_size += preprocessing_size
    if total_size > 100_000_000:
        raise RuntimeError(f"Combined checkpoints exceed the conservative 100 MB limit: {total_size} bytes")
    predictions = []
    manifest = []
    for path, temperature in zip(args.checkpoints, temperatures):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config["fold"] != -1:
            print(f"Warning: {path} is a validation model, not trained on all training subjects", flush=True)
        if config.get("strict_scratch"):
            validate_scratch_config(config)
        model = (ScratchMultiModalModel(config["modalities"], fusion=config.get("fusion", "concat"),
                                       visual_multiplier=config.get("visual_multiplier", 1),
                                       visual_encoder=config.get("visual_encoder", "legacy"),
                                       visual_feedforward=config.get("visual_feedforward", 384),
                                       skeleton_representation=config.get("skeleton_representation", "coordinates"),
                                       scratch_skeleton_encoder=config.get("scratch_skeleton_encoder", "tcn"),
                                       skeleton_dimension=config.get("skeleton_dimension", 192),
                                       skeleton_depth=config.get("skeleton_depth", 3),
                                       skeleton_heads=config.get("skeleton_heads", 6),
                                       thermal_view=config.get("thermal_view", "full"), imu_representation=config.get("imu_representation", "raw"))
                 if config.get("strict_scratch") else
                 SharedMViTModel(config["modalities"], kinetics=False)
                 if config.get("backbone") == "mvit_v2_s_shared" else
                 SharedSwin3DModel(config["modalities"], kinetics=False)
                 if config.get("backbone") == "swin3d_t_shared" else
                 GroupedX3DTokenModel(config["modalities"], kinetics=False,
                                      kinetics_source=config.get("kinetics_source", "s"))
                 if config.get("backbone") == "x3d_grouped_tokens" else
                 SharedX3DTokenModel(config["modalities"], kinetics=False,
                                     kinetics_source=config.get("kinetics_source", "s"))
                 if config.get("backbone") == "x3d_shared_tokens" else
                 MultiModalModel(config["modalities"], width=config["width"],
                                spatial_pool=config.get("spatial_pool", 1), backbone=config.get("backbone", "scratch"),
                                skeleton_backbone=config.get("skeleton_backbone", "tcn"),
                                temporal_backbone=config.get("temporal_backbone", "tcn"),
                                temporal_shift=config.get("temporal_shift", False),
                                kinetics_source=config.get("kinetics_source", "s"),
                                fusion=config.get("fusion", "concat")))
        state_dict = (
            dequantize_state_dict(checkpoint["model_int4"])
            if "model_int4" in checkpoint else
            dequantize_state_dict(checkpoint["model_int7"])
            if "model_int7" in checkpoint else
            dequantize_state_dict(checkpoint["model_int6"])
            if "model_int6" in checkpoint else
            dequantize_state_dict(checkpoint["model_int8"])
            if "model_int8" in checkpoint else checkpoint["model"]
        )
        model.load_state_dict(state_dict)
        model = model.cuda().eval()
        dataset = ClipDataset(indices, config["modalities"], frames=config["frames"],
                               canonical_pose=config.get("canonical_pose", False), image_size=config.get("image_size", 96),
                               thermal_view=config.get("thermal_view", "full"), imu_representation=config.get("imu_representation", "raw"))
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        probabilities = []
        with torch.inference_mode():
            for batch in loader:
                batch = prepare_batch(batch, torch.device("cuda"))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(batch)
                    logit_views = [logits]
                    if args.tta_hflip:
                        flipped = {
                            name: torch.flip(values, dims=(-1,))
                            if name in config["modalities"] and values.ndim == 5 else values
                            for name, values in batch.items()
                        }
                        flipped_logits, _ = model(flipped)
                        logit_views.append(flipped_logits)
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
                                size=values.shape[-2:], mode="bilinear", align_corners=False,
                            ).reshape_as(values)
                        cropped_logits, _ = model(cropped)
                        logit_views.append(cropped_logits)
                        if args.tta_hflip:
                            cropped_flipped = {
                                name: torch.flip(values, dims=(-1,))
                                if name in config["modalities"] and values.ndim == 5 else values
                                for name, values in cropped.items()
                            }
                            cropped_flipped_logits, _ = model(cropped_flipped)
                            logit_views.append(cropped_flipped_logits)
                    logits = torch.stack(logit_views).mean(0)
                probabilities.append((logits.float() / temperature).softmax(1).cpu().numpy())
        probability = np.concatenate(probabilities)
        if probability.shape != (405, 40) or not np.isfinite(probability).all():
            raise RuntimeError("Invalid model output")
        predictions.append(probability)
        manifest.append({"checkpoint": str(path), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "config": config,
                         "epoch": checkpoint["epoch"]})
        del model
        torch.cuda.empty_cache()
        print(f"Predicted {path}", flush=True)
    combined = np.average(np.stack(predictions), axis=0, weights=weights)
    labels = combined.argmax(1)
    destination = ROOT / "outputs/submissions" / f"{args.name}.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "prediction"])
        writer.writeheader()
        writer.writerows({"path": records[index]["id"], "prediction": int(label)}
                        for index, label in zip(indices, labels))
    np.savez(ROOT / "outputs/predictions" / f"{args.name}.npz", probabilities=combined,
             per_model=np.stack(predictions), indices=np.asarray(indices), weights=weights)
    with destination.open() as handle:
        rows = list(csv.DictReader(handle))
    source = ROOT / "data/downloads/CUHK-X_Small_Model_Track/Small-Model-Track/Testing/test_file/test.csv"
    with source.open(encoding="utf-8-sig") as handle:
        expected = [row["path"] for row in csv.DictReader(handle)]
    actual = [row["path"] for row in rows]
    if actual != expected or len(set(actual)) != 405 or any(not 0 <= int(row["prediction"]) < 40 for row in rows):
        raise RuntimeError("Submission path order, uniqueness, or label-range validation failed")
    report = {"output": str(destination), "rows": len(rows), "weights": weights.tolist(), "temperatures": temperatures,
              "preprocessing_models": {name: {"path": str(path), "bytes": path.stat().st_size,
                                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                       for name, path in preprocessing.items()},
              "preprocessing_bytes": preprocessing_size,
              "checkpoint_bytes_total": total_size, "models": manifest,
              "tta_hflip": args.tta_hflip,
              "tta_crop_scale": args.tta_crop_scale,
              "class_counts": np.bincount(labels, minlength=40).tolist(),
              "test_accuracy": None, "note": "No test labels available. Predictions are not submitted to Kaggle."}
    (ROOT / "outputs/predictions" / f"{args.name}_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: value for name, value in report.items() if name != "models"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
