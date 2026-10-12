import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset, get_indices, get_records, prepare_batch
from src.models import (
    GroupedX3DTokenModel,
    MultiModalModel,
    SharedMViTModel,
    SharedSwin3DModel,
    SharedX3DTokenModel,
)
from src.scratch_models import ScratchMultiModalModel, validate_scratch_config
from src.scratch_protocol import ablation_metrics, modality_presence
from src.encoder_initialization import load_encoder_sources
from src.validation_splits import load_diagnostic_split
from src.scratch_augmentation import single_modality_mixup


def learning_rate_at_epoch(base, epoch, horizon, warmup):
    if horizon <= 0 or epoch <= 0 or warmup < 0:
        raise ValueError("Invalid learning-rate schedule")
    return base * min(1, epoch / max(warmup, 1e-6)) * (0.05 + 0.95 * (1 + math.cos(math.pi * min(epoch, horizon) / horizon)) / 2)


def evaluate(model, loader, device, selection_labels=None):
    model.eval()
    logits, labels, indices = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = prepare_batch(batch, device)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                output, _ = model(batch)
            logits.append(output.float().cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            indices.append(batch["index"].cpu().numpy())
    logits, labels, indices = np.concatenate(logits), np.concatenate(labels), np.concatenate(indices)
    probabilities = torch.from_numpy(logits).softmax(1).numpy()
    prediction = probabilities.argmax(1)
    accuracy = float((prediction == labels).mean())
    recalls = [float((prediction[labels == label] == label).mean()) for label in range(40) if (labels == label).any()]
    metrics = {"accuracy": accuracy, "balanced_accuracy": float(np.mean(recalls)),
               "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-8)).mean())}
    if selection_labels:
        selected = np.isin(labels, selection_labels)
        if not selected.any():
            raise ValueError("Validation split has no selected labels")
        metrics["selection_accuracy"] = float((prediction[selected] == labels[selected]).mean())
        metrics["selection_rows"] = int(selected.sum())
    return metrics, {
        "logits": logits, "labels": labels, "indices": indices, "probabilities": probabilities,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--modalities", default="skeleton,imu,radar")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--environment", choices=("all", "a", "b"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--swanlab-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--init", type=Path)
    parser.add_argument("--freeze-inherited", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--strong-augment", action="store_true")
    parser.add_argument("--canonical-pose", action="store_true")
    parser.add_argument("--spatial-pool", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument(
        "--backbone",
        choices=["scratch", "mobile_small", "efficientnet_b0", "resnet50", "x3d_s", "swin3d_t", "x3d_shared_tokens", "x3d_grouped_tokens", "mvit_v2_s_shared", "swin3d_t_shared"],
        default="scratch",
    )
    parser.add_argument("--imagenet", action="store_true")
    parser.add_argument("--kinetics", action="store_true")
    parser.add_argument("--kinetics-source", choices=["s", "m", "l"], default="s")
    parser.add_argument("--backbone-lr-factor", type=float, default=1)
    parser.add_argument("--schedule-epochs", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=float, default=5)
    parser.add_argument("--mixup", type=float, default=0)
    parser.add_argument("--single-modality-mixup", action="store_true")
    parser.add_argument("--spatial-augment", action="store_true")
    parser.add_argument("--synchronized-temporal-augment", action="store_true")
    parser.add_argument("--skeleton-backbone", choices=["tcn", "stgcn", "attention"], default="tcn")
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--freeze-backbone-bn", action="store_true")
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
    parser.add_argument("--temporal-backbone", choices=["tcn", "statistics", "pyramid"], default="tcn")
    parser.add_argument("--temporal-shift", action="store_true")
    parser.add_argument("--strict-scratch", action="store_true")
    parser.add_argument("--fusion", choices=["concat", "transformer", "residual_transformer", "cross_transformer"], default="concat")
    parser.add_argument("--visual-encoder", choices=["legacy", "cnn_pooled", "cnn_tokens", "patch_tokens"], default="legacy")
    parser.add_argument("--visual-feedforward", type=int, default=384)
    parser.add_argument("--visual-multiplier", type=int, choices=[1, 2], default=1)
    parser.add_argument("--skeleton-representation", choices=["coordinates", "centered", "invariant"], default="coordinates")
    parser.add_argument("--scratch-skeleton-encoder", choices=["tcn", "dstformer"], default="tcn")
    parser.add_argument("--skeleton-dimension", type=int, default=192)
    parser.add_argument("--skeleton-depth", type=int, default=3)
    parser.add_argument("--skeleton-heads", type=int, default=6)
    parser.add_argument("--thermal-view", choices=["full", "motion_crop"], default="full")
    parser.add_argument("--imu-representation", choices=["raw", "world"], default="raw")
    parser.add_argument("--imu-token-encoder", choices=["conv_lstm", "statistics", "transformer"], default="conv_lstm")
    parser.add_argument("--encoder-sources", nargs="+", default=[])
    parser.add_argument("--diagnostic-split", type=Path)
    parser.add_argument("--training-labels", default="")
    parser.add_argument("--selection-labels", default="")
    args = parser.parse_args()
    if args.strict_scratch:
        validate_scratch_config(vars(args))
    elif args.visual_encoder != "legacy":
        raise ValueError("Spatial encoders require --strict-scratch")
    elif args.fusion not in ("concat", "transformer", "residual_transformer"):
        raise ValueError("Cross-attention fusion is implemented only in the strict scratch model")
    if not args.strict_scratch and (args.encoder_sources or args.visual_multiplier != 1):
        raise ValueError("Encoder study options require --strict-scratch")
    if args.skeleton_representation != "coordinates" and (not args.strict_scratch or args.encoder_sources or args.init):
        raise ValueError("Representation experiments require fresh strict-scratch initialization")
    if args.single_modality_mixup and (not args.strict_scratch or args.encoder_sources or args.init):
        raise ValueError("Single-modality mixup experiments require fresh strict-scratch initialization")
    if (args.thermal_view != "full" or args.imu_representation != "raw") and not args.strict_scratch:
        raise ValueError("New input recipes require the strict-scratch model")
    if args.diagnostic_split and (not args.strict_scratch or args.init):
        raise ValueError("Nested validation requires strict scratch training without a generic warm start")
    if args.freeze_inherited and not args.init:
        raise ValueError("--freeze-inherited requires --init")
    if args.warmup_epochs < 0:
        raise ValueError("Warmup epochs must be nonnegative")
    if args.gradient_accumulation < 1:
        raise ValueError("Gradient accumulation must be positive")
    if args.kinetics and (args.imagenet or args.backbone not in ("x3d_s", "swin3d_t", "x3d_shared_tokens", "x3d_grouped_tokens", "mvit_v2_s_shared", "swin3d_t_shared")):
        raise ValueError("Kinetics initialization requires a supported video backbone, without --imagenet")
    if args.backbone in ("x3d_s", "swin3d_t", "x3d_shared_tokens", "x3d_grouped_tokens", "mvit_v2_s_shared", "swin3d_t_shared") and (args.imagenet or args.temporal_shift):
        raise ValueError("Video backbones do not use ImageNet initialization or extra temporal shift")
    if args.backbone in ("x3d_shared_tokens", "x3d_grouped_tokens") and args.fusion != "transformer":
        raise ValueError("Shared X3D temporal tokens require --fusion transformer")
    if args.backbone in ("mvit_v2_s_shared", "swin3d_t_shared") and args.fusion != "transformer":
        raise ValueError("Shared video Transformers require --fusion transformer")
    if args.backbone == "resnet50" and (not args.imagenet or args.temporal_shift):
        raise ValueError("ResNet50 requires ImageNet initialization and does not use temporal shift")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(6)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    modalities = args.modalities.split(",")
    training_labels = sorted({int(value) for value in args.training_labels.split(",") if value})
    selection_labels = sorted({int(value) for value in args.selection_labels.split(",") if value})
    if training_labels and not selection_labels:
        selection_labels = list(training_labels)
    if any(value < 0 or value >= 40 for value in training_labels + selection_labels):
        raise ValueError("Training and selection labels must be in [0, 39]")
    destination = ROOT / "outputs/checkpoints" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "best.pt").exists():
        raise RuntimeError("Experiment already has a checkpoint; use a new name to avoid overwriting")
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    if args.environment != "all":
        second_environment = args.environment == "b"
        train_indices = [index for index in train_indices
                         if (records[index]["user"] > 15) == second_environment]
        valid_indices = [index for index in valid_indices
                         if (records[index]["user"] > 15) == second_environment]
        if not train_indices or (args.fold >= 0 and not valid_indices):
            raise ValueError("Environment filter produced an empty split")
    outer_indices = []
    grouped_validation = False
    if args.diagnostic_split:
        train_indices, valid_indices, outer_indices = load_diagnostic_split(args.diagnostic_split, records, args.fold)
        grouped_validation = json.loads(args.diagnostic_split.read_text()).get("grouped_users", False)
    original_train_indices = list(train_indices)
    if training_labels:
        train_indices = [index for index in train_indices if records[index]["label"] in training_labels]
        if not train_indices:
            raise ValueError("Training-label filter produced an empty split")
    if args.strict_scratch:
        available = modality_presence(modalities).any(1)
        train_indices = [index for index in train_indices if available[index]]
        if len(train_indices) < args.batch_size:
            raise ValueError("Not enough available training clips for one full batch")
    input_options = {"thermal_view": args.thermal_view, "imu_representation": args.imu_representation}
    train_data = ClipDataset(train_indices, modalities, training=True, frames=args.frames,
                             canonical_pose=args.canonical_pose, image_size=args.image_size, **input_options)
    loader_options = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_generator = torch.Generator().manual_seed(args.seed) if args.strict_scratch else None
    valid_generator = torch.Generator().manual_seed(args.seed + 1) if args.strict_scratch else None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              generator=train_generator, **loader_options)
    valid_loader = DataLoader(ClipDataset(valid_indices, modalities, frames=args.frames,
                                         canonical_pose=args.canonical_pose, image_size=args.image_size, **input_options),
                              batch_size=args.batch_size, generator=valid_generator, **loader_options) if valid_indices else None
    diagnostic_loaders = {}
    if args.diagnostic_split:
        for position, (name, indices) in enumerate([("clean_train", train_indices), ("outer_valid", outer_indices)]):
            dataset = ClipDataset(indices, modalities, training=False, frames=args.frames,
                                  canonical_pose=args.canonical_pose, image_size=args.image_size, **input_options)
            diagnostic_loaders[name] = DataLoader(dataset, batch_size=args.batch_size,
                                                  generator=torch.Generator().manual_seed(args.seed + 2 + position), **loader_options)
    model = (ScratchMultiModalModel(modalities, fusion=args.fusion, visual_multiplier=args.visual_multiplier,
                                   skeleton_representation=args.skeleton_representation, visual_encoder=args.visual_encoder,
                                   visual_feedforward=args.visual_feedforward,
                                   scratch_skeleton_encoder=args.scratch_skeleton_encoder,
                                   skeleton_dimension=args.skeleton_dimension, skeleton_depth=args.skeleton_depth,
                                   skeleton_heads=args.skeleton_heads, imu_token_encoder=args.imu_token_encoder,
                                   **input_options) if args.strict_scratch else
             SharedMViTModel(modalities, kinetics=args.kinetics)
             if args.backbone == "mvit_v2_s_shared" else
             SharedSwin3DModel(modalities, kinetics=args.kinetics)
             if args.backbone == "swin3d_t_shared" else
             GroupedX3DTokenModel(modalities, kinetics=args.kinetics, kinetics_source=args.kinetics_source)
             if args.backbone == "x3d_grouped_tokens" else
             SharedX3DTokenModel(modalities, kinetics=args.kinetics, kinetics_source=args.kinetics_source)
             if args.backbone == "x3d_shared_tokens" else
             MultiModalModel(modalities, width=args.width, spatial_pool=args.spatial_pool,
                            backbone=args.backbone, imagenet=args.imagenet,
                            skeleton_backbone=args.skeleton_backbone, temporal_backbone=args.temporal_backbone,
                            temporal_shift=args.temporal_shift, kinetics=args.kinetics,
                            kinetics_source=args.kinetics_source, fusion=args.fusion)).to(device)
    if args.strict_scratch:
        prior_counts = np.bincount([records[index]["label"] for index in original_train_indices], minlength=40) + 1
        with torch.no_grad():
            model.class_prior_logits.copy_(torch.tensor(np.log(prior_counts / prior_counts.sum()), device=device))
    encoder_provenance = []
    if args.encoder_sources:
        encoder_provenance = load_encoder_sources(model, args.encoder_sources, records, train_indices, valid_indices, args.fold,
                                                  outer_indices if args.diagnostic_split else None)
        print("Initialized same-fold scratch encoders:", json.dumps(encoder_provenance), flush=True)
    if args.init:
        source = torch.load(args.init, map_location="cpu", weights_only=False)
        if source["config"].get("fold") != args.fold:
            raise ValueError("Warm starts must use the same fold, or full-data to full-data, to preserve the validation/refit protocol")
        source_split = args.init.parent / "split.json"
        if source_split.exists():
            source_indices = json.loads(source_split.read_text())["train"]
            if not set(source_indices).issubset(original_train_indices):
                raise ValueError("Warm-start training samples are outside this training split")
        current = model.state_dict()
        compatible = {name: value for name, value in source["model"].items()
                      if name in current and value.shape == current[name].shape}
        if args.backbone == "x3d_grouped_tokens" and source["config"].get("backbone") == "x3d_shared_tokens":
            for name, value in current.items():
                if name.startswith("visual_encoders."):
                    source_name = "visual." + name.split(".", 2)[2]
                    if source_name in source["model"] and source["model"][source_name].shape == value.shape:
                        compatible[name] = source["model"][source_name]
        source_fusion = source["config"].get("fusion", "concat")
        preserves_concat_head = source_fusion == "concat" and args.fusion == "residual_transformer"
        if source_fusion != args.fusion and not preserves_concat_head:
            compatible = {name: value for name, value in compatible.items() if not name.startswith("head.")}
        model.load_state_dict(compatible, strict=False)
        if args.freeze_inherited:
            for name, parameter in model.named_parameters():
                if name in compatible:
                    parameter.requires_grad_(False)
        print(f"Initialized {len(compatible)}/{len(current)} tensors from {args.init}", flush=True)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    backbone_parameters = [parameter for name, parameter in model.named_parameters()
                           if ".frames." in name and parameter.requires_grad]
    head_parameters = [parameter for name, parameter in model.named_parameters()
                       if ".frames." not in name and parameter.requires_grad]
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "factor": args.backbone_lr_factor})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "factor": 1})
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=0.015)
    labels = np.asarray([records[index]["label"] for index in train_indices])
    counts = np.bincount(labels, minlength=40).clip(1)
    weights = (counts.mean() / counts) ** 0.25
    weights = torch.tensor(weights / weights.mean(), device=device, dtype=torch.float32)
    config = vars(args).copy()
    config["init"] = str(args.init) if args.init else None
    config["diagnostic_split"] = str(args.diagnostic_split) if args.diagnostic_split else None
    config.update({"modalities": modalities, "train_clips": len(train_indices), "valid_clips": len(valid_indices),
                   "training_labels": training_labels, "selection_labels": selection_labels,
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "validation_users": sorted({records[index]["user"] for index in valid_indices}),
                   "pretrained": args.imagenet or args.kinetics or any(name.endswith("_features") for name in modalities),
                   "cached_feature_backbone": "efficientnet_b0 IMAGENET1K_V1" if any(name.endswith("_features") for name in modalities) else None,
                   "weights_source": ("torchvision resnet50 IMAGENET1K_V2" if args.backbone == "resnet50" else
                                      f"torchvision {args.backbone} IMAGENET1K_V1") if args.imagenet else None})
    if args.backbone == "resnet50" and args.imagenet:
        resnet_path = ROOT / ".tools/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
        config["pretrained_sha256"] = hashlib.sha256(resnet_path.read_bytes()).hexdigest()
    if args.strict_scratch:
        config.update({"architecture_policy": "CNN/RNN/Transformer only; no external pretrained weights",
                       "initialization_mode": "same_fold_scratch_encoders" if args.encoder_sources else "random",
                       "encoder_provenance": encoder_provenance,
                       "study": "regularization_study" if args.single_modality_mixup else "representation_study" if args.skeleton_representation != "coordinates" else "encoder_study" if args.encoder_sources or args.visual_multiplier != 1 or args.epochs > 60 else "scratch_ablation",
                       "train_clips_before_availability_filter": len(original_train_indices),
                       "train_coverage": len(train_indices) / len(original_train_indices),
                       "missing_prediction": "Laplace-smoothed label prior from this fold's training subjects",
                       "token_alignment": "relative clip positions, not verified physical synchronization",
                       "radar_representation": "existing 28-channel frame statistics; temporal CNN baseline, no PointNet",
                       "effective_encoder_width": 96, "effective_visual_spatial_pool": 2})
    if args.diagnostic_split:
        config.update({"study": "same_user_validation_diagnosis", "selection_partition": "same_user_holdout",
                       "outer_valid_clips": len(outer_indices),
                       "outer_validation_users": sorted({records[index]["user"] for index in outer_indices}),
                       "diagnostic_split_sha256": hashlib.sha256(args.diagnostic_split.read_bytes()).hexdigest()})
    if args.visual_encoder != "legacy":
        config.update({"effective_visual_spatial_pool": 2 if args.visual_encoder == "cnn_pooled" else None,
                       "visual_tokens_per_frame": 4 if args.visual_encoder == "cnn_pooled" else (args.image_size // 16) ** 2})
    if args.kinetics and args.backbone in ("x3d_s", "x3d_shared_tokens", "x3d_grouped_tokens"):
        target_size = {"s": 160, "m": 224, "l": 312}[args.kinetics_source]
        config["weights_source"] = f"pytorchvideo 0.1.5 X3D-{args.kinetics_source.upper()} Kinetics-400 initialization; target encoder uses adaptive pooling and {target_size}px inputs"
        config["pretrained_sha256"] = hashlib.sha256((ROOT / f".tools/torch/hub/checkpoints/X3D_{args.kinetics_source.upper()}.pyth").read_bytes()).hexdigest()
    elif args.kinetics and args.backbone in ("swin3d_t", "swin3d_t_shared"):
        config["weights_source"] = "torchvision Swin3D-T Kinetics-400 V1 initialization"
        swin_path = ROOT / ".tools/torch/hub/checkpoints/swin3d_t-7615ae03.pth"
        config["pretrained_sha256"] = hashlib.sha256(swin_path.read_bytes()).hexdigest()
    elif args.kinetics and args.backbone == "mvit_v2_s_shared":
        config["weights_source"] = "torchvision MViT-V2-S Kinetics-400 V1 initialization"
        mvit_path = ROOT / ".tools/torch/hub/checkpoints/mvit_v2_s-ae3be167.pth"
        config["pretrained_sha256"] = hashlib.sha256(mvit_path.read_bytes()).hexdigest()
    if args.init:
        config["init_sha256"] = hashlib.sha256(args.init.read_bytes()).hexdigest()
        config["init_source_name"] = source["config"].get("name")
        config["init_source_fold"] = source["config"].get("fold")
        config["init_loaded_tensors"] = len(compatible)
        config["init_total_tensors"] = len(current)
        config["pretrained"] = config["pretrained"] or source["config"].get("pretrained", False)
        config["inherited_weights_source"] = source["config"].get("weights_source")
    config["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ["scripts/train.py", "src/models.py", "src/data.py"]}
    if args.strict_scratch:
        for name in ["src/scratch_models.py", "src/scratch_protocol.py", "src/encoder_initialization.py", "data/metadata/samples.json"]:
            config["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if args.single_modality_mixup:
        config["source_sha256"]["src/scratch_augmentation.py"] = hashlib.sha256((ROOT / "src/scratch_augmentation.py").read_bytes()).hexdigest()
        config["train_accuracy_caveat"] = "Mixed-batch accuracy uses the primary label; use clean_train for comparable unaugmented accuracy"
    if args.imu_representation == "world":
        for name in ["scripts/prepare_imu_world.py", "data/processed/imu_world.npy"]:
            config["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for modality in ("imu_official", "imu_magnetic"):
        if modality in modalities:
            for name in ["scripts/prepare_imu_magnetic.py", f"data/processed/{modality}.npy"]:
                config["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if args.thermal_view == "motion_crop":
        for name in ["scripts/prepare_thermal_motion.py", "data/metadata/thermal_motion.json"]:
            config["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if args.thermal_view != "full" or args.imu_representation != "raw":
        config["study"] = "input_representation_study"
    if args.diagnostic_split:
        config["source_sha256"]["src/validation_splits.py"] = hashlib.sha256((ROOT / "src/validation_splits.py").read_bytes()).hexdigest()
    if args.visual_encoder != "legacy":
        config["source_sha256"]["src/spatial_tokens.py"] = hashlib.sha256((ROOT / "src/spatial_tokens.py").read_bytes()).hexdigest()
    if grouped_validation:
        config.update({"study": "spatial_token_grouped_study", "selection_partition": "inner_unseen_users",
                       "outer_evaluation": "once at the inner-selected checkpoint after training"})
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    saved_split = {"train": train_indices, "valid": valid_indices}
    if args.diagnostic_split:
        saved_split["outer_valid"] = outer_indices
    (destination / "split.json").write_text(json.dumps(saved_split) + "\n")
    import swanlab
    run = swanlab.init(project="cuhkx-small-multimodal", name=args.name, config=config, public=False,
                       mode=args.swanlab_mode, log_dir=str(ROOT / "outputs/swanlog"),
                       description=("Grouped inner users select checkpoints; outer users evaluated once after training" if grouped_validation else
                                    "Nested diagnosis: known-user held-out clips select checkpoints; unseen users are diagnostic only"
                                    if args.diagnostic_split else "CNN/RNN/Transformer, no external pretrained weights, user-isolated validation")
                       if args.strict_scratch else "Research experiment; disjoint-subject validation, no test labels.")
    print(json.dumps(config), flush=True)
    best_accuracy, best_epoch = -1, 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone_bn:
            for name, layer in model.named_modules():
                if ".frames." in name and isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        if args.freeze_inherited:
            for layer in model.modules():
                if (isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
                        and not any(parameter.requires_grad for parameter in layer.parameters())):
                    layer.eval()
        horizon = args.schedule_epochs or args.epochs
        learning_rate = learning_rate_at_epoch(args.lr, epoch, horizon, args.warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * group["factor"]
        loss_total, correct, seen = 0, 0, 0
        epoch_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        maximum_batches = min(len(train_loader), args.limit_batches or len(train_loader))
        for batch_index, batch in enumerate(train_loader):
            if args.strict_scratch:
                torch.manual_seed(args.seed + epoch * 100000 + batch_index)
            batch = prepare_batch(batch, device, training=True, strong_augment=args.strong_augment,
                                   spatial_augment=args.spatial_augment,
                                   synchronized_temporal_augment=args.synchronized_temporal_augment)
            mixing = float(np.random.beta(args.mixup, args.mixup)) if args.mixup > 0 and random.random() < 0.5 else 1.0
            if mixing < 1:
                permutation = torch.randperm(len(batch["label"]), device=device)
                alternate_labels = batch["label"][permutation]
                if args.single_modality_mixup:
                    batch = single_modality_mixup(batch, modalities[0], mixing, permutation)
                else:
                    for name in batch:
                        if name not in ["label", "index"]:
                            batch[name] = batch[name] * mixing + batch[name][permutation] * (1 - mixing)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, auxiliary = model(batch)
                loss = functional.cross_entropy(output, batch["label"], weight=weights, label_smoothing=0.05)
                if mixing < 1:
                    loss = loss * mixing + (1 - mixing) * functional.cross_entropy(
                        output, alternate_labels, weight=weights, label_smoothing=0.05)
                for name, branch in auxiliary.items():
                    branch_loss = functional.cross_entropy(branch, batch["label"], reduction="none", label_smoothing=0.05)
                    if mixing < 1:
                        branch_loss = branch_loss * mixing + (1 - mixing) * functional.cross_entropy(
                            branch, alternate_labels, reduction="none", label_smoothing=0.05)
                    mask = batch[name + "_mask"]
                    loss = loss + args.auxiliary_weight / len(auxiliary) * (branch_loss * mask).sum() / mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss / args.gradient_accumulation).backward()
            should_step = (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == maximum_batches
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    decay = min(args.ema_decay, 1 - 1 / (epoch * len(train_loader) + batch_index + 1))
                    for target, source in zip(ema.parameters(), model.parameters()):
                        target.lerp_(source, 1 - decay)
                    for target, source in zip(ema.buffers(), model.buffers()):
                        target.copy_(source)
            count = len(batch["label"])
            loss_total += loss.item() * count
            correct += int((output.argmax(1) == batch["label"]).sum())
            seen += count
            if args.limit_batches and batch_index + 1 >= args.limit_batches:
                break
        metrics = {"train/loss": loss_total / seen, "train/accuracy": correct / seen,
                   "train/lr": learning_rate, "system/epoch_seconds": time.monotonic() - epoch_started}
        if args.strict_scratch:
            metrics["system/peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 1e6
        if valid_loader and (epoch % args.eval_every == 0 or epoch == args.epochs):
            validation, predictions = evaluate(ema, valid_loader, device, selection_labels)
            metrics.update({f"valid/{name}": value for name, value in validation.items()})
            diagnostic_metrics, diagnostic_predictions = {}, {}
            if args.diagnostic_split:
                diagnostic_metrics["inner_valid"] = validation
                metrics.update({f"inner_valid/{name}": value for name, value in validation.items()})
                for name, loader in ([] if grouped_validation else diagnostic_loaders.items()):
                    scores, outputs = evaluate(ema, loader, device)
                    diagnostic_metrics[name], diagnostic_predictions[name] = scores, outputs
                    metrics.update({f"{name}/{key}": value for key, value in scores.items()})
                with (destination / "diagnostic_history.jsonl").open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch} | diagnostic_metrics) + "\n")
            if args.strict_scratch:
                detail = ablation_metrics(predictions, records, modalities)
                metrics.update({"valid/macro_f1": detail["all_validation"]["macro_f1"],
                                "valid/available_accuracy": detail["available_only"]["accuracy"],
                                "valid/coverage": detail["coverage"]})
            selection_accuracy = validation.get("selection_accuracy", validation["accuracy"])
            if selection_accuracy > best_accuracy:
                best_accuracy, best_epoch = selection_accuracy, epoch
                state = {"model": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
                         "config": config, "epoch": epoch, "metrics": validation}
                torch.save(state, destination / "best.pt")
                np.savez(destination / "validation.npz", **predictions)
                (destination / "metrics.json").write_text(json.dumps(validation | {"epoch": epoch}, indent=2) + "\n")
                if args.strict_scratch:
                    (destination / "ablation_metrics.json").write_text(json.dumps(detail, indent=2) + "\n")
                if args.diagnostic_split:
                    (destination / "diagnostic_best.json").write_text(json.dumps({"epoch": epoch, "selection_partition": "inner_valid"} | diagnostic_metrics, indent=2) + "\n")
                    for name, values in diagnostic_predictions.items():
                        np.savez(destination / f"{name}.npz", **values)
        elif not valid_loader and epoch == args.epochs:
            torch.save({"model": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
                        "config": config, "epoch": epoch, "metrics": {}}, destination / "best.pt")
        metrics["valid/best_accuracy"] = best_accuracy
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch} | metrics) + "\n")
        swanlab.log(metrics, step=epoch)
        print(f"epoch={epoch} " + " ".join(f"{name}={value:.4f}" for name, value in metrics.items()), flush=True)
        if valid_loader and epoch - best_epoch >= args.patience:
            print("Early stopping", flush=True)
            break
    if grouped_validation:
        state = torch.load(destination / "best.pt", map_location=device, weights_only=False)
        ema.load_state_dict(state["model"], strict=True)
        diagnostics = {"epoch": best_epoch, "selection_partition": "inner_unseen_users",
                       "inner_valid": state["metrics"]}
        for name, loader in diagnostic_loaders.items():
            scores, predictions = evaluate(ema, loader, device)
            diagnostics[name] = scores
            np.savez(destination / f"{name}.npz", **predictions)
            swanlab.log({f"final/{name}/{key}": value for key, value in scores.items()})
        (destination / "diagnostic_best.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"DONE best_accuracy={best_accuracy:.5f} best_epoch={best_epoch} minutes={(time.monotonic()-started)/60:.2f}", flush=True)
    if args.strict_scratch:
        size = (destination / "best.pt").stat().st_size
        if size > 100_000_000:
            raise RuntimeError(f"Scratch checkpoint exceeds 100 MB: {size}")
        summary = {"status": "complete", "checkpoint_bytes": size, "parameters": config["parameters"],
                   "best_accuracy": best_accuracy, "best_epoch": best_epoch,
                   "elapsed_seconds": time.monotonic() - started,
                   "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6}
        (destination / "completion.json").write_text(json.dumps(summary, indent=2) + "\n")
        swanlab.log({"system/checkpoint_bytes": size})
    swanlab.finish()


if __name__ == "__main__":
    main()
