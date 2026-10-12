#!/usr/bin/env python3
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.adapt_paired_source import (
    checkpoint_state,
    model_inputs,
    select_trainable,
)
from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, build_model, make_loader
from src.data import get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset


WINDOWS = {
    "full": (0.0, 1.0),
    "early_0875": (0.0, 0.875),
    "late_0875": (0.125, 1.0),
    "center_0875": (0.0625, 0.9375),
    "early_0750": (0.0, 0.75),
    "late_0750": (0.25, 1.0),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=32)
    return parser.parse_args()


def temporal_resample(values, start, end, output_steps, video=False):
    if start == 0 and end == 1 and values.shape[2 if video else 1] == output_steps:
        return values
    steps = values.shape[2 if video else 1]
    first = round(start * (steps - 1))
    stop = max(first + 2, round(end * (steps - 1)) + 1)
    stop = min(stop, steps)
    if video:
        return functional.interpolate(
            values[:, :, first:stop],
            size=(output_steps, values.shape[-2], values.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )
    return functional.interpolate(
        values[:, first:stop].transpose(1, 2),
        size=output_steps,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)


def temporal_view(video, keyword, start, end):
    view_video = temporal_resample(video, start, end, video.shape[2], video=True)
    view_keyword = dict(keyword)
    for name in ("skeleton", "imu", "radar"):
        if name in view_keyword:
            values = view_keyword[name]
            view_keyword[name] = temporal_resample(values, start, end, values.shape[1])
    return view_video, view_keyword


def spatial_logits(model, video, keyword):
    views = []
    logits, _ = model(video, **keyword)
    views.append(logits)
    flipped, _ = model(torch.flip(video, dims=(-1,)), **keyword)
    views.append(flipped)
    side = round(video.shape[-1] * 0.75)
    top = (video.shape[-2] - side) // 2
    left = (video.shape[-1] - side) // 2
    cropped = functional.interpolate(
        video[..., top:top + side, left:left + side],
        size=video.shape[-3:],
        mode="trilinear",
        align_corners=False,
    )
    cropped_logits, _ = model(cropped, **keyword)
    views.append(cropped_logits)
    cropped_flipped, _ = model(torch.flip(cropped, dims=(-1,)), **keyword)
    views.append(cropped_flipped)
    return torch.stack(views).mean(0)


@torch.no_grad()
def predict_views(model, loader, mode):
    rows = {name: [] for name in WINDOWS}
    labels = []
    indices = []
    model.eval()
    for batch in tqdm(loader, desc="temporal views"):
        video, keyword = model_inputs(batch, mode)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for name, (start, end) in WINDOWS.items():
                view_video, view_keyword = temporal_view(video, keyword, start, end)
                logits = spatial_logits(model, view_video, view_keyword)
                rows[name].append(logits.float().softmax(1).cpu().numpy())
        labels.append(batch["label"].numpy())
        indices.append(batch["index"].numpy())
    return (
        {name: np.concatenate(parts) for name, parts in rows.items()},
        np.concatenate(labels),
        np.concatenate(indices),
    )


def adapt(model, teacher, loader, mode, pseudo_labels, selected_rows, epochs=2):
    selected_parameters = select_trainable(model, "heads")
    parameters = [parameter for _, parameter in selected_parameters]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
    model.eval()
    teacher.eval()
    history = []
    for epoch in range(epochs):
        loss_sum = 0.0
        rows = 0
        for batch in loader:
            indices = batch["index"].cuda(non_blocking=True)
            active = selected_rows[indices]
            if not active.any():
                continue
            video, keyword = model_inputs(batch, mode)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits, _ = teacher(video, **keyword)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(video, **keyword)
                pseudo_loss = functional.cross_entropy(logits[active], pseudo_labels[indices][active])
                retention = functional.kl_div(
                    functional.log_softmax(logits.float(), dim=1),
                    functional.softmax(teacher_logits.float(), dim=1),
                    reduction="batchmean",
                )
                loss = pseudo_loss + 0.25 * retention
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            optimizer.step()
            active_rows = int(active.sum())
            loss_sum += float(loss.detach()) * active_rows
            rows += active_rows
        history.append({"epoch": epoch + 1, "loss": loss_sum / max(rows, 1), "rows": rows})
        print(json.dumps(history[-1]), flush=True)
    return history


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mode = config["mode"]
    state = checkpoint_state(checkpoint)
    model = build_model(config, copy.deepcopy(state))
    teacher = build_model(config, copy.deepcopy(state))

    records = get_records()
    _, target_indices = get_indices(records, args.fold)
    archive = np.load(args.pseudo_labels)
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    pseudo_labels = torch.full((len(records),), -1, dtype=torch.long, device="cuda")
    selected_rows = torch.zeros(len(records), dtype=torch.bool, device="cuda")
    for index in target_indices:
        position = positions[index]
        pseudo_labels[index] = int(archive["pseudo_labels"][position])
        selected_rows[index] = bool(archive["paired"][position] or archive["confidence"][position] >= 0.6)

    training = R2Plus1DDataset(
        target_indices,
        args.cache_dir,
        training=True,
        include_sensors=True,
        include_radar=bool(config.get("use_radar", False)),
        canonical_pose=bool(config.get("canonical_pose", False)),
        frames=int(config.get("frames", 32)),
        image_size=int(config.get("image_size", 128)),
        photometric_jitter=0.05,
        horizontal_flip_probability=0.5,
    )
    training_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    history = adapt(model, teacher, training_loader, mode, pseudo_labels, selected_rows)
    del teacher
    torch.cuda.empty_cache()

    target_loader = make_loader(target_indices, args, config, mode)
    adapt_batch_norm(model, target_loader, mode, passes=1, reset=True)
    probabilities, labels, indices = predict_views(model, target_loader, mode)
    ordered = np.argsort(indices)
    labels = labels[ordered]
    indices = indices[ordered]
    probabilities = {name: values[ordered] for name, values in probabilities.items()}
    expected_indices = np.asarray(target_indices, dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError("Prediction indices do not match the fold order")

    metrics = {
        name: {
            "correct": int((values.argmax(1) == labels).sum()),
            "accuracy": float((values.argmax(1) == labels).mean()),
        }
        for name, values in probabilities.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, labels=labels, indices=indices, **{f"probabilities_{k}": v for k, v in probabilities.items()})
    report = {
        "fold": args.fold,
        "selected_pseudo_rows": int(selected_rows.sum()),
        "history": history,
        "windows": WINDOWS,
        "metrics": metrics,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
