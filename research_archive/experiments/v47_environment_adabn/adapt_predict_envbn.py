#!/usr/bin/env python3
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.adapt_paired_source import checkpoint_state
from experiments.v46_temporal_tta.adapt_predict_views import adapt
from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, build_model, make_loader, predict
from src.data import get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
    train_indices, target_indices = get_indices(records, args.fold)
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
        frames=args.frames,
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
    adapted_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    del model, teacher
    torch.cuda.empty_cache()

    rows = []
    groups = []
    for environment in (0, 1):
        source = [index for index in train_indices if int(records[index]["user"] > 15) == environment]
        target = [index for index in target_indices if int(records[index]["user"] > 15) == environment]
        environment_model = build_model(config, copy.deepcopy(adapted_state))
        adaptation_loader = make_loader(source + target, args, config, mode)
        adapt_batch_norm(environment_model, adaptation_loader, mode, passes=1, reset=True)
        target_loader = make_loader(target, args, config, mode)
        rows.extend(predict(environment_model, target_loader, mode, True, 0.75))
        groups.append(
            {
                "environment": environment,
                "source_rows": len(source),
                "target_rows": len(target),
                "adaptation_rows": len(source) + len(target),
            }
        )
        del environment_model
        torch.cuda.empty_cache()

    ordered = {index: (label, values) for index, label, values, _ in rows}
    indices = np.asarray(target_indices, dtype=np.int64)
    labels = np.asarray([ordered[int(index)][0] for index in indices])
    probabilities = np.stack([ordered[int(index)][1] for index in indices])
    report = {
        "fold": args.fold,
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "correct": int((probabilities.argmax(1) == labels).sum()),
        "selected_pseudo_rows": int(selected_rows.sum()),
        "history": history,
        "adaptation_set": "source_target_environment",
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=indices)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

