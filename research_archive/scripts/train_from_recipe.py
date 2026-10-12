import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    folder = ROOT / "outputs/checkpoints" / args.source
    config = json.loads((folder / "config.json").read_text())
    if config.get("diagnostic_split"):
        raise ValueError("Nested-validation diagnosis is not an automatic full-data training recipe")
    if config.get("init") or config.get("encoder_sources"):
        raise ValueError("Warm-start recipes require an explicit corresponding full-data source; do not silently drop the initialization stage")
    metrics = json.loads((folder / "metrics.json").read_text())
    final_clips = 3036
    if config.get("strict_scratch"):
        sys.path.insert(0, str(ROOT))
        from src.data import get_indices, get_records
        from src.scratch_protocol import modality_presence
        available = modality_presence(config["modalities"]).any(1)
        final_indices, _ = get_indices(get_records(), -1)
        final_clips = sum(bool(available[index]) for index in final_indices)
    final_steps = final_clips // config["batch_size"]
    fold_steps = config["train_clips"] // config["batch_size"]
    epochs = args.epochs or max(1, round(metrics["epoch"] * fold_steps / final_steps))
    horizon = max(epochs, round(config["epochs"] * fold_steps / final_steps))
    warmup = config.get("warmup_epochs", 5) * fold_steps / final_steps
    command = [sys.executable, str(ROOT / "scripts/train.py"), "--name", args.name, "--fold", "-1",
               "--epochs", str(epochs), "--schedule-epochs", str(horizon), "--warmup-epochs", str(warmup)]
    for key in ["visual_encoder", "visual_feedforward", "modalities", "seed", "batch_size", "lr", "width", "workers", "frames", "spatial_pool",
                "image_size", "backbone", "backbone_lr_factor", "mixup", "skeleton_backbone", "ema_decay", "auxiliary_weight", "temporal_backbone", "kinetics_source", "fusion", "visual_multiplier", "skeleton_representation", "thermal_view", "imu_representation"]:
        if key in config:
            value = ",".join(config[key]) if key == "modalities" else str(config[key])
            command.extend(["--" + key.replace("_", "-"), value])
    for key in ["imagenet", "kinetics", "canonical_pose", "strong_augment", "spatial_augment", "freeze_backbone_bn", "temporal_shift", "strict_scratch", "single_modality_mixup"]:
        if config.get(key):
            command.append("--" + key.replace("_", "-"))
    recipe = {"source": args.source, "source_best_epoch": metrics["epoch"], "final_epochs": epochs,
              "schedule_epochs": horizon, "warmup_epochs": warmup, "command": command,
              "rationale": "Approximately match fold-0-selected optimizer updates, including warmup, when increasing training-set size."}
    (ROOT / "configs" / f"{args.name}.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(json.dumps(recipe), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
