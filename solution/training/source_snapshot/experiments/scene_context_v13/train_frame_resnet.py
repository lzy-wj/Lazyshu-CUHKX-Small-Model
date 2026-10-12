#!/usr/bin/env python3
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
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import ClipDataset, get_indices, get_records, prepare_batch


MODALITIES = ["scene_depth", "scene_ir", "thermal"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--modalities", default=",".join(MODALITIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


class SharedFrameResNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 40)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, videos, masks):
        batch, modalities, steps, channels, height, width = videos.shape
        images = videos.reshape(batch * modalities * steps, channels, height, width)
        images = ((images + 1) * 0.5 - self.mean) / self.std
        frame_logits = self.backbone(images).reshape(batch, modalities, steps, 40)
        modality_logits = frame_logits.mean(dim=2)
        weights = masks / masks.sum(dim=1, keepdim=True).clamp_min(1)
        clip_logits = (modality_logits * weights.unsqueeze(-1)).sum(dim=1)
        return clip_logits, modality_logits, frame_logits


def model_inputs(batch, training, modalities):
    videos = torch.stack([batch[name] for name in modalities], dim=1)
    masks = torch.stack([batch[name + "_mask"] for name in modalities], dim=1)
    videos = videos * masks[:, :, None, None, None, None]
    if training:
        flip = torch.rand(len(videos), device=videos.device) < 0.5
        videos[flip] = torch.flip(videos[flip], dims=(-1,))
    return videos, masks


def evaluate(model, loader, device, modalities):
    model.eval()
    probabilities, labels, indices = [], [], []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = prepare_batch(raw_batch, device)
            videos, masks = model_inputs(batch, False, modalities)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(videos, masks)[0]
            probabilities.append(logits.float().softmax(1).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            indices.append(batch["index"].cpu().numpy())
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)
    return {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-8)).mean()),
    }, {"probabilities": probabilities, "labels": labels, "indices": indices}


def main():
    args = parse_args()
    modalities = args.modalities.split(",")
    if not modalities or len(set(modalities)) != len(modalities) or set(modalities) - set(MODALITIES):
        raise ValueError("Frame ResNet modalities must be a unique subset of scene_depth, scene_ir, thermal")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        ClipDataset(
            train_indices, modalities, training=True, frames=args.frames, image_size=args.image_size
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    valid_loader = DataLoader(
        ClipDataset(valid_indices, modalities, frames=args.frames, image_size=args.image_size),
        batch_size=args.batch_size,
        **loader_options,
    )
    model = SharedFrameResNet(pretrained=True).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.fc")]
    head = list(model.backbone.fc.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr * 0.1}, {"params": head, "lr": args.lr}],
        weight_decay=0.015,
    )
    counts = np.bincount([records[index]["label"] for index in train_indices], minlength=40).clip(1)
    class_weights = torch.tensor(
        (counts.mean() / counts) ** 0.25, dtype=torch.float32, device=device
    )
    class_weights /= class_weights.mean()
    destination = ROOT / "outputs/checkpoints" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "best.pt").exists():
        raise RuntimeError("Experiment already has a checkpoint")
    pretrained_path = ROOT / ".tools/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
    config = vars(args) | {
        "modalities": modalities,
        "fold": args.fold,
        "train_clips": len(train_indices),
        "valid_clips": len(valid_indices),
        "validation_users": sorted({records[index]["user"] for index in valid_indices}),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "architecture": "Shared torchvision ResNet50 CNN with frame and clip supervision",
        "weights_source": "torchvision ResNet50 IMAGENET1K_V2",
        "pretrained_sha256": hashlib.sha256(pretrained_path.read_bytes()).hexdigest(),
        "precision": "BF16 autocast; FP32 optimizer and EMA",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "split.json").write_text(
        json.dumps({"train": train_indices, "valid": valid_indices}) + "\n"
    )
    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=config,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(ROOT / "outputs/swanlog"),
        description="Shared frame-level ResNet supervision across selected visual streams; unseen-user validation.",
    )
    print(json.dumps(config), flush=True)
    best_accuracy, best_epoch = -1.0, 0
    update_step = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.monotonic()
        lr_scale = 0.05 + 0.95 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        for group, base_lr in zip(optimizer.param_groups, (args.lr * 0.1, args.lr)):
            group["lr"] = base_lr * lr_scale
        loss_total = correct = seen = 0
        for raw_batch in train_loader:
            batch = prepare_batch(
                raw_batch,
                device,
                training=True,
                modality_dropout=0.1,
                strong_augment=True,
                synchronized_temporal_augment=True,
            )
            videos, masks = model_inputs(batch, True, modalities)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                clip_logits, modality_logits, frame_logits = model(videos, masks)
                loss = functional.cross_entropy(
                    clip_logits, batch["label"], weight=class_weights, label_smoothing=0.05
                )
                modality_loss = functional.cross_entropy(
                    modality_logits.flatten(0, 1),
                    batch["label"].repeat_interleave(len(modalities)),
                    weight=class_weights,
                    label_smoothing=0.05,
                    reduction="none",
                ).reshape(len(videos), len(modalities))
                loss += 0.5 * (modality_loss * masks).sum() / masks.sum().clamp_min(1)
                frame_loss = functional.cross_entropy(
                    frame_logits.flatten(0, 2),
                    batch["label"].repeat_interleave(len(modalities) * args.frames),
                    weight=class_weights,
                    label_smoothing=0.05,
                    reduction="none",
                ).reshape(len(videos), len(modalities), args.frames)
                loss += 0.25 * (frame_loss * masks.unsqueeze(-1)).sum() / (
                    masks.sum() * args.frames
                ).clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            update_step += 1
            with torch.no_grad():
                decay = min(0.99, 1 - 1 / update_step)
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - decay)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            batch_size = len(videos)
            loss_total += float(loss) * batch_size
            correct += int((clip_logits.argmax(1) == batch["label"]).sum())
            seen += batch_size
        metrics, predictions = evaluate(ema, valid_loader, device, modalities)
        log = {
            "train/loss": loss_total / seen,
            "train/accuracy": correct / seen,
            "valid/accuracy": metrics["accuracy"],
            "valid/nll": metrics["nll"],
            "valid/best_accuracy": max(best_accuracy, metrics["accuracy"]),
            "train/lr": optimizer.param_groups[-1]["lr"],
            "system/epoch_seconds": time.monotonic() - epoch_started,
            "system/peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = metrics["accuracy"], epoch
            torch.save(
                {
                    "model": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
                    "config": config,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                destination / "best.pt",
            )
            np.savez(destination / "validation.npz", **predictions)
            (destination / "metrics.json").write_text(
                json.dumps(metrics | {"epoch": epoch}, indent=2) + "\n"
            )
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch} | log) + "\n")
        swanlab.log(log, step=epoch)
        print(
            f"epoch={epoch} " + " ".join(f"{name}={value:.4f}" for name, value in log.items()),
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            print("Early stopping", flush=True)
            break
    summary = {
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
    }
    (destination / "completion.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DONE", json.dumps(summary), flush=True)
    swanlab.finish()


if __name__ == "__main__":
    main()
