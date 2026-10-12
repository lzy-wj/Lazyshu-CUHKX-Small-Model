#!/usr/bin/env python3
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import FOLDS, get_records
from src.r2plus1d_models import compact_quantized_state_dict, quantize_state_dict


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_paths(record, modality="Thermal"):
    folder = Path(record["modalities"].get(modality, ""))
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ThermalTrainDataset(Dataset):
    def __init__(self, indices, paths, views_per_clip):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.paths = paths
        self.views_per_clip = views_per_clip
        self.records = get_records()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.indices) * self.views_per_clip

    def __getitem__(self, position):
        index = int(self.indices[position % len(self.indices)])
        path = random.choice(self.paths[index])
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.records[index]["label"])


class ThermalClipTrainDataset(Dataset):
    def __init__(self, indices, paths, views):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.paths = paths
        self.views = views
        self.records = get_records()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        paths = self.paths[index]
        boundaries = np.linspace(0, len(paths), self.views + 1).round().astype(int)
        selected = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            start = min(int(start), len(paths) - 1)
            end = max(start + 1, min(int(end), len(paths)))
            selected.append(random.randrange(start, end))
        flipped = random.random() < 0.5
        images = []
        for frame in selected:
            with Image.open(paths[frame]) as image:
                tensor = self.transform(image.convert("RGB"))
            if flipped:
                tensor = torch.flip(tensor, dims=(-1,))
            images.append(tensor)
        return torch.stack(images), int(self.records[index]["label"])


class ThermalValidationDataset(Dataset):
    def __init__(self, indices, paths, views):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.paths = paths
        self.views = views
        self.records = get_records()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        paths = self.paths[index]
        selected = np.linspace(0, len(paths) - 1, self.views).round().astype(int)
        images = []
        for frame in selected:
            with Image.open(paths[int(frame)]) as image:
                images.append(self.transform(image.convert("RGB")))
        record = self.records[index]
        return {
            "images": torch.stack(images),
            "label": int(record["label"]),
            "index": index,
            "user": int(record["user"]),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--validation-users", default="")
    parser.add_argument("--output-root", default=str(HERE / "outputs"))
    parser.add_argument(
        "--modality", choices=("Thermal", "IR", "Depth_Color"), default="Thermal"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-views", type=int, default=4)
    parser.add_argument("--clip-training", action="store_true")
    parser.add_argument("--validation-views", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    indices = []
    users = []
    losses = []
    for batch in loader:
        images = batch["images"]
        batch_size, views = images.shape[:2]
        images = images.flatten(0, 1).to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images).reshape(batch_size, views, 40).mean(dim=1)
            loss = functional.cross_entropy(logits, target)
        probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
        labels.append(batch["label"].numpy())
        indices.append(batch["index"].numpy())
        users.append(batch["user"].numpy())
        losses.append(float(loss) * batch_size)
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)
    users = np.concatenate(users)
    prediction = probabilities.argmax(axis=1)
    user_accuracy = {
        str(user): float((prediction[users == user] == labels[users == user]).mean())
        for user in sorted(set(users.tolist()))
    }
    metrics = {
        "loss": sum(losses) / len(labels),
        "accuracy": float((prediction == labels).mean()),
        "correct": int((prediction == labels).sum()),
        "rows": len(labels),
        "user_macro_accuracy": float(np.mean(list(user_accuracy.values()))),
        "user_accuracy": user_accuracy,
    }
    return metrics, probabilities, labels, indices, users


def main():
    args = parse_args()
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    torch.cuda.manual_seed_all(args.seed + args.fold)
    records = get_records()
    paths = [image_paths(record, args.modality) for record in records]
    validation_users = {
        int(value) for value in args.validation_users.split(",") if value.strip()
    }
    if not validation_users:
        validation_users = set(FOLDS[args.fold])
    train_indices = [
        record["index"]
        for record in records
        if record["split"] == "train"
        and record["user"] not in validation_users
        and paths[record["index"]]
    ]
    valid_indices = [
        record["index"]
        for record in records
        if record["split"] == "train"
        and record["user"] in validation_users
        and paths[record["index"]]
    ]
    train_users = {records[index]["user"] for index in train_indices}
    valid_users = {records[index]["user"] for index in valid_indices}
    if train_users & valid_users:
        raise RuntimeError("Subject leakage")
    if {records[index]["label"] for index in train_indices} != set(range(40)):
        raise RuntimeError("Thermal training split is missing classes")

    generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_dataset = (
        ThermalClipTrainDataset(train_indices, paths, args.train_views)
        if args.clip_training
        else ThermalTrainDataset(train_indices, paths, args.train_views)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=generator,
    )
    valid_loader = DataLoader(
        ThermalValidationDataset(valid_indices, paths, args.validation_views),
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 40)
    model.to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    head_parameters = list(model.fc.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "factor": args.backbone_lr_factor},
            {"params": head_parameters, "factor": 1.0},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = len(train_loader)
    global_step = 0
    destination = Path(args.output_root) / args.name
    destination.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "validation_users": sorted(validation_users),
        "train_clips": len(train_indices),
        "valid_clips": len(valid_indices),
        "train_frames_per_epoch": len(train_loader.dataset) * (
            args.train_views if args.clip_training else 1
        ),
        "validation_frames": len(valid_indices) * args.validation_views,
        "train_users": sorted(train_users),
        "valid_users": sorted(valid_users),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": f"ImageNet-pretrained ResNet34 CNN for {args.modality}, clip-level multi-frame mean logits",
        "precision": "BF16 autocast; FP32 optimizer and EMA; INT8 serialized weights",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=config,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(Path(args.output_root).parent / "swanlog"),
        description=f"Official-style 224px {args.modality} frame CNN with strict clip-level cross-user evaluation.",
    )
    best_correct = -1
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if global_step < warmup_steps:
                rate = args.lr * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                rate = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = rate * group["factor"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if images.ndim == 5:
                    clip_batch, clip_views = images.shape[:2]
                    logits = model(images.flatten(0, 1)).reshape(
                        clip_batch, clip_views, 40
                    ).mean(dim=1)
                else:
                    logits = model(images)
                loss = functional.cross_entropy(
                    logits, labels, label_smoothing=args.label_smoothing
                )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - args.ema_decay)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += len(labels)
            global_step += 1
        metrics, probabilities, labels, indices, users = evaluate(
            ema, valid_loader, device
        )
        values = {
            "epoch": epoch,
            "train/loss": loss_sum / seen,
            "train/accuracy": correct / seen,
            "train/lr": optimizer.param_groups[-1]["lr"],
            **{f"valid/{key}": value for key, value in metrics.items() if key != "user_accuracy"},
        }
        swanlab.log(values, step=epoch)
        print(" ".join(f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}" for key, value in values.items()), flush=True)
        if metrics["correct"] > best_correct:
            best_correct = metrics["correct"]
            quantized, tensor_bytes = quantize_state_dict(ema.state_dict())
            checkpoint = {
                "model_int8": compact_quantized_state_dict(quantized),
                "config": config,
                "epoch": epoch,
                "metrics": metrics,
                "quantized_tensor_bytes": tensor_bytes,
            }
            torch.save(checkpoint, destination / "best_int8.pt")
            np.savez(
                destination / "validation.npz",
                probabilities=probabilities,
                labels=labels,
                indices=indices,
                users=users,
            )
            (destination / "metrics.json").write_text(
                json.dumps(metrics | {"epoch": epoch}, indent=2) + "\n"
            )
    completion = {
        "status": "complete",
        "best_correct": best_correct,
        "valid_rows": len(valid_indices),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_bytes": (destination / "best_int8.pt").stat().st_size,
        "swanlab_url": run.url if hasattr(run, "url") else None,
    }
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print("DONE " + json.dumps(completion), flush=True)
    swanlab.finish()


if __name__ == "__main__":
    main()
