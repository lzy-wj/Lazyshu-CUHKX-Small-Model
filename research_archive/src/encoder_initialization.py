import hashlib
import json
from pathlib import Path

import torch

from src.data import ROOT, get_indices
from src.scratch_models import validate_scratch_config
from src.scratch_protocol import modality_presence
from src.validation_splits import validate_diagnostic_split


def load_encoder_sources(model, paths, records, train_indices, valid_indices, fold, outer_indices=None):
    if len(paths) != len(model.modalities):
        raise ValueError("Provide exactly one independently trained source per target modality")
    expected_train, expected_valid = get_indices(records, fold)
    grouped = outer_indices is not None
    if grouped:
        if outer_indices != expected_valid or set(train_indices) & set(valid_indices) or set(train_indices) | set(valid_indices) != set(expected_train):
            raise ValueError("Grouped target partitions do not reconstruct the declared fold")
    elif valid_indices != expected_valid or not set(train_indices).issubset(expected_train):
        raise ValueError("Target split does not match the declared fold")
    metadata_hash = hashlib.sha256((ROOT / "data/metadata/samples.json").read_bytes()).hexdigest()
    validated = []
    names = set()
    for source_path in paths:
        path = Path(source_path).resolve()
        if not path.is_relative_to((ROOT / "outputs/checkpoints").resolve()):
            raise ValueError("Encoder sources must be local project checkpoints")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        validate_scratch_config(config)
        if not config.get("strict_scratch") or config.get("encoder_sources") or config.get("weights_source"):
            raise ValueError("Source must be an independently trained scratch model with no initialization chain")
        if config.get("fold") != fold:
            raise ValueError("Cross-fold or full-data-to-validation initialization is forbidden")
        if config.get("source_sha256", {}).get("data/metadata/samples.json") != metadata_hash:
            raise ValueError("Source dataset metadata does not match the current dataset")
        if len(config["modalities"]) != 1:
            raise ValueError("Only single-modality source checkpoints are accepted")
        name = config["modalities"][0]
        if name not in model.modalities or name in names:
            raise ValueError("Encoder sources must uniquely cover the target modalities")
        if name == "skeleton" and config.get("skeleton_representation", "coordinates") != model.encoders[name].representation:
            raise ValueError("Source and target skeleton representations must match")
        if name == "imu" and config.get("imu_representation", "raw") != model.imu_representation:
            raise ValueError("Source and target IMU representations must match")
        if name == "thermal" and config.get("thermal_view", "full") != model.thermal_view:
            raise ValueError("Source and target thermal views must match")
        split_path = path.parent / "split.json"
        split = json.loads(split_path.read_text())
        available = modality_presence([name]).any(1)
        expected_source_train = [index for index in train_indices if available[index]]
        if split["train"] != expected_source_train or split["valid"] != valid_indices:
            raise ValueError("Source split differs from the audited availability-filtered fold")
        if grouped:
            diagnostic_path = config.get("diagnostic_split")
            if not diagnostic_path:
                raise ValueError("Grouped encoder sources require a frozen diagnostic split")
            diagnostic_path = ROOT / diagnostic_path
            source_train, source_valid, source_outer = validate_diagnostic_split(json.loads(diagnostic_path.read_text()), records, fold)
            if source_train != train_indices or source_valid != valid_indices or source_outer != outer_indices or split.get("outer_valid") != outer_indices:
                raise ValueError("Grouped source does not match the target train/inner/outer users")
        if not set(split["train"]).issubset(train_indices):
            raise ValueError("Source training samples are outside the target training set")
        completion = json.loads((path.parent / "completion.json").read_text())
        if completion["status"] != "complete" or completion["checkpoint_bytes"] != path.stat().st_size:
            raise ValueError("Source must be a complete, unmodified training artifact")
        if json.loads((path.parent / "config.json").read_text()) != config:
            raise ValueError("Source checkpoint and sidecar configurations disagree")
        prefix = f"encoders.{name}."
        weights = {key.removeprefix(prefix): value for key, value in checkpoint["model"].items() if key.startswith(prefix)}
        target = model.encoders[name].state_dict()
        if weights.keys() != target.keys() or any(weights[key].shape != target[key].shape for key in target):
            raise ValueError("Source and target encoder structures do not match exactly")
        validated.append((name, weights, {
            "modality": name, "checkpoint": str(path.relative_to(ROOT)),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "source_name": config["name"], "fold": fold, "epoch": checkpoint["epoch"],
            "train_clips": len(split["train"]), "loaded_tensors": len(weights), "grouped": grouped,
        }))
        names.add(name)
    for name, weights, _ in validated:
        model.encoders[name].load_state_dict(weights, strict=True)
    return [entry for _, _, entry in validated]
