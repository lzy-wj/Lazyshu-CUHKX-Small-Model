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
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v176_fullscene_multimodal.model import FullSceneMultiModal
from src.data import FOLDS, get_records
from src.r2plus1d_models import compact_quantized_state_dict, quantize_state_dict


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FullSceneDataset(Dataset):
    def __init__(self, indices, frames, image_size, training):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.frames = frames
        self.image_size = image_size
        self.training = training
        self.records = get_records()
        self.skeleton = np.load(
            ROOT / "data/processed/skeleton.npy", mmap_mode="r"
        )
        self.imu = np.load(ROOT / "data/processed/imu_world.npy", mmap_mode="r")
        self.sensor_mask = np.load(
            ROOT / "data/processed/sensor_mask.npy", mmap_mode="r"
        )
        self.video_mask = np.load(
            ROOT / "data/processed/video_mask.npy", mmap_mode="r"
        )
        self.mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(STD).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.indices)

    def select_frames(self, length):
        boundaries = np.linspace(0, length, self.frames + 1).round().astype(int)
        selected = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            start = min(int(start), length - 1)
            end = max(start + 1, min(int(end), length))
            selected.append(
                random.randrange(start, end) if self.training else (start + end - 1) // 2
            )
        return selected

    def __getitem__(self, position):
        index = int(self.indices[position])
        video = np.load(
            ROOT / f"data/processed/video_24_{self.image_size}/{index:05d}.npy",
            mmap_mode="r",
        )
        selected = self.select_frames(len(video))
        video = torch.from_numpy(np.asarray(video[selected]).copy()).float().div_(255)
        streams = torch.stack(
            [video[:, :3], video[:, 4:7], video[:, 3:4].expand(-1, 3, -1, -1)]
        )
        streams = (streams - self.mean) / self.std
        skeleton = torch.from_numpy(np.asarray(self.skeleton[index]).copy())
        imu = torch.from_numpy(np.asarray(self.imu[index]).copy())
        if self.training:
            skeleton += torch.randn_like(skeleton) * 0.006
            imu += torch.randn_like(imu) * 0.006
        record = self.records[index]
        return {
            "images": streams,
            "skeleton": skeleton,
            "imu": imu,
            "visual_mask": torch.from_numpy(
                np.asarray(self.video_mask[index, [0, 2, 1]]).copy()
            ),
            "sensor_mask": torch.from_numpy(
                np.asarray(self.sensor_mask[index, :2]).copy()
            ),
            "label": int(record["label"]),
            "index": index,
            "user": int(record["user"]),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in ("images", "skeleton", "imu", "visual_mask", "sensor_mask", "label")
    }


def loss_value(logits, auxiliary, batch, smoothing, auxiliary_weight):
    loss = functional.cross_entropy(
        logits, batch["label"], label_smoothing=smoothing
    )
    visual = auxiliary["visual"]
    visual_losses = functional.cross_entropy(
        visual.flatten(0, 1),
        batch["label"].repeat_interleave(visual.shape[1]),
        reduction="none",
        label_smoothing=smoothing,
    ).reshape(len(logits), visual.shape[1])
    visual_weight = batch["visual_mask"]
    visual_loss = (visual_losses * visual_weight).sum() / visual_weight.sum().clamp_min(1)
    sensor_losses = []
    for position, name in enumerate(("skeleton", "imu")):
        values = functional.cross_entropy(
            auxiliary[name], batch["label"], reduction="none", label_smoothing=smoothing
        )
        mask = batch["sensor_mask"][:, position]
        sensor_losses.append((values * mask).sum() / mask.sum().clamp_min(1))
    return loss + auxiliary_weight * (visual_loss + sum(sensor_losses)) / 3


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    indices = []
    users = []
    loss_sum = 0.0
    for raw in loader:
        batch = move_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, auxiliary = model(
                batch["images"], batch["skeleton"], batch["imu"],
                batch["visual_mask"], batch["sensor_mask"]
            )
            loss = loss_value(logits, auxiliary, batch, 0.0, 0.15)
        probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
        labels.append(raw["label"].numpy())
        indices.append(raw["index"].numpy())
        users.append(raw["user"].numpy())
        loss_sum += float(loss) * len(logits)
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
        "loss": loss_sum / len(labels),
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
    train_indices = [record["index"] for record in records if record["split"] == "train" and record["user"] not in FOLDS[args.fold]]
    valid_indices = [record["index"] for record in records if record["split"] == "train" and record["user"] in FOLDS[args.fold]]
    device = torch.device("cuda")
    train_loader = DataLoader(
        FullSceneDataset(train_indices, args.frames, args.image_size, True),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed + args.fold),
    )
    valid_loader = DataLoader(
        FullSceneDataset(valid_indices, args.frames, args.image_size, False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    model = FullSceneMultiModal(pretrained=True).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    backbone_ids = {id(parameter) for parameter in model.visual.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": model.visual.parameters(), "factor": args.backbone_lr_factor},
            {"params": [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids], "factor": 1.0},
        ], lr=args.lr, weight_decay=args.weight_decay,
    )
    destination = HERE / "outputs" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "train_clips": len(train_indices), "valid_clips": len(valid_indices),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": "shared ImageNet ResNet18 full-scene CNN + Skeleton/IMU CNN-RNN + Transformer fusion",
        "precision": "BF16 autocast; FP32 optimizer and EMA; INT8 serialized weights",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    import swanlab
    run = swanlab.init(
        project="cuhkx-small-multimodal", name=args.name, config=config,
        public=False, mode=args.swanlab_mode, log_dir=str(HERE / "swanlog"),
        description="Full-scene object context with five-modality temporal Transformer fusion.",
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = len(train_loader)
    global_step = 0
    best_correct = -1
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for raw in train_loader:
            batch = move_batch(raw, device)
            if global_step < warmup_steps:
                rate = args.lr * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                rate = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = rate * group["factor"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, auxiliary = model(
                    batch["images"], batch["skeleton"], batch["imu"],
                    batch["visual_mask"], batch["sensor_mask"]
                )
                loss = loss_value(
                    logits, auxiliary, batch, args.label_smoothing, args.auxiliary_weight
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - args.ema_decay)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            loss_sum += float(loss.detach()) * len(logits)
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            seen += len(logits)
            global_step += 1
        metrics, probabilities, labels, indices, users = evaluate(ema, valid_loader, device)
        values = {
            "epoch": epoch, "train/loss": loss_sum / seen,
            "train/accuracy": correct / seen, "train/lr": optimizer.param_groups[-1]["lr"],
            **{f"valid/{key}": value for key, value in metrics.items() if key != "user_accuracy"},
        }
        swanlab.log(values, step=epoch)
        print(" ".join(f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}" for key, value in values.items()), flush=True)
        if metrics["correct"] > best_correct:
            best_correct = metrics["correct"]
            quantized, tensor_bytes = quantize_state_dict(ema.state_dict())
            torch.save(
                {"model_int8": compact_quantized_state_dict(quantized), "config": config,
                 "epoch": epoch, "metrics": metrics, "quantized_tensor_bytes": tensor_bytes},
                destination / "best_int8.pt",
            )
            np.savez(destination / "validation.npz", probabilities=probabilities, labels=labels, indices=indices, users=users)
            (destination / "metrics.json").write_text(json.dumps(metrics | {"epoch": epoch}, indent=2) + "\n")
    completion = {
        "status": "complete", "best_correct": best_correct,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_bytes": (destination / "best_int8.pt").stat().st_size,
        "swanlab_url": run.url if hasattr(run, "url") else None,
    }
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print("DONE " + json.dumps(completion), flush=True)
    swanlab.finish()


if __name__ == "__main__":
    main()
